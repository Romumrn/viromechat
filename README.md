# 🦠 Viromech@t — Virus Dataset AI Agent

A conversational agent for exploring viral taxonomy and virus–host data, built on a strict
tool-calling architecture: the LLM never sees raw data directly, it can only act through a
small set of audited tools exposed by a separate [MCP server](README_MCP.md).

## Project Context

This project is developed within the framework of **SHAPE-Med@Lyon** and contributes to the
structuring research initiative [**Virome@tlas**](https://www.shape-med-lyon.fr/projets/structurants-vague-1/virometlas/).

*Virome@tlas* aims to build an integrated digital platform for large-scale exploration and
surveillance of the global virosphere, leveraging publicly available sequencing data to analyze
virus diversity, virus–host interactions, and ecological distribution patterns within a
transdisciplinary **One Health** framework spanning human, animal, and environmental health.

Viromech@t supports this effort as a research companion tool combining deterministic dataset
querying, transparent visualization, controlled external knowledge retrieval, and strict
grounding of every biological statement in tool output.

---

## Architecture

The system is split into **two independent processes** that only talk to each other over MCP/HTTP:

```
┌─────────────────────┐   MCP over HTTP    ┌───────────────────────────┐
│   app.py             │ ─────────────────► │   server_mcp.py            │
│   Streamlit client   │ ◄───────────────── │   FastMCP server            │
│   (chat UI, Albert   │   tools + resources │   (owns all data access:   │
│    API tool-calling) │                     │   taxonomy CSV + S3 host  │
└─────────────────────┘                     │   Parquet via DuckDB)     │
                                             └───────────────────────────┘
```

* **`app.py`** never touches a dataframe or a credential for the S3 bucket. It lists the tools
  the MCP server currently exposes, forwards them to the [Albert API](https://albert.api.etalab.gouv.fr)
  (French government sovereign LLM infrastructure, OpenAI-compatible) for tool-calling, and
  dispatches each call back to the MCP server. It is deliberately generic: it reads a tool's
  JSON schema to decide which UI-configured defaults apply, rather than hardcoding tool names.
* **`server_mcp.py`** owns the datasets, the DuckDB/S3 connection, and every tool's business
  logic and guardrails. See **[README_MCP.md](README_MCP.md)** for the full tool/resource
  reference.

Both processes read their own, separate secrets file — see [Configuration](#configuration).

---

## Features

* Natural-language querying of viral taxonomy and virus–host relationships
* Authoritative taxonomy/acronym resolution via NCBI Taxonomy (e.g. `HIV` → `Lentivirus humimdef1`)
* SQL queries against a multi-GB virus–host Parquet dataset on S3, without ever loading it into memory (DuckDB + `httpfs`/`spatial`)
* Interactive Plotly charts and geographic maps
* Wikipedia and PubMed lookups for biological/clinical background, with mandatory inline citations
* 🎤 Voice input — record a question, transcribed via Albert API's Whisper endpoint
* Conversation memory across up to 10 questions (full tool-call trace, not just Q&A) with an explicit in-chat reset notice once the limit is hit
* PMID hallucination guard: any PMID not returned by an actual `pubmed_search` call is stripped from the answer
* Per-tool-call status line (search keyword only — no clutter for dataset/map calls) with full detail logged to disk
* In-app error reporting button (question, answer, executed code, and recent logs bundled into a report file)

---

## Conversation Memory

Each new question is answered with full context of the previous ones — including every tool call
the model made and the results it got back, not just a summary of the final answers. This lets the
model resolve follow-ups like *"and which family is that genus part of?"* without the subject
being restated.

This context accumulates for up to `MAX_CONTEXT_TURNS` questions (`config.py`, default **10**).
Once the limit is reached, the conversation memory resets — a message is posted in the chat
("🔄 Conversation context reset after 10 questions — starting fresh…") and the next question starts
with no memory of what came before. A failed turn (API timeout/error) is never added to memory and
doesn't count against the limit.

Given the model's 131k-token context window and that full agent traces are kept (not just
Q&A), a heavy question involving several tool calls can add a few thousand tokens to memory — the
10-question cap keeps this comfortably bounded rather than growing unbounded for the life of a
session.

---

## Datasets

| Dataset | Storage | Description |
|---|---|---|
| **Taxonomy** (`df_taxo`) | Local CSV (`data/TAXONOMY.csv`), loaded fully into memory | NCBI Taxonomy, enriched with genome assembly availability, SRA sequencing activity, and GBIF biodiversity observations. One row per taxon. |
| **Virus–host occurrences** (`host` / `df_host`) | Parquet on S3, queried on demand via DuckDB — never fully loaded | SRA/GenBank/BioSample samples linked to host & virus taxonomy, geographic location (as a `GEOMETRY` point column), and disease status. |

Column-by-column descriptions of both datasets are **not hardcoded in the client** — they are
published by the MCP server as resources (`resource://datasets/taxonomy/schema` and
`resource://datasets/host/schema`) and read once per conversation by `app.py`, which folds them
into the system prompt. This means the two datasets' schemas can change server-side without any
client code change.

---

## Setup

### Requirements

* Python 3.10+
* An **Albert API key** ([albert.api.etalab.gouv.fr](https://albert.api.etalab.gouv.fr))
* Read access to the S3-compatible bucket hosting the virus–host Parquet dataset
* Python packages (`streamlit>=1.53` is required for the native microphone button in the chat input):

```bash
pip install -r requirements.txt
```

### Configuration

Secrets are split into **two separate `.env` files, one per process** — never shared, never
imported by the other process:

* **`.env.app`** — read by `app.py` (copy from [`.env.app.example`](.env.app.example)):

  ```bash
  ALBERT_API_KEY=sk-...

  # Optional access protection
  # PASSWORD_ENABLED=true
  # ACCESS_CODE=your_access_code
  ```

* **`.env.mcp`** — read by `server_mcp.py` (copy from [`.env.mcp.example`](.env.mcp.example)):

  ```bash
  ENDPOINT=your-s3-endpoint
  ACCESS_KEY=...
  SECRET_KEY=...
  BUCKET=...
  VIRAL_HOST_DATASET=your_dataset.parquet

  # Optional, default shown:
  # REGION=fr
  # S3_URL_STYLE=path
  ```

Both files are gitignored. When deploying to **Streamlit Community Cloud**, use
`.streamlit/secrets.toml` instead for the app-side secrets — it takes priority over `.env.app`
whenever both are present (see `_get_secret`/`_get_bool_secret` in `app.py`).

Non-secret configuration (model defaults, sampling parameters, timeouts, the MCP server URL, …)
lives in `config.py`, shared by both processes.

### Running

Two terminals, in order:

```bash
# 1. Start the MCP server (loads the taxonomy CSV, connects to S3)
python server_mcp.py

# 2. Start the Streamlit app (connects to the MCP server at MCP_SERVER_URL)
streamlit run app.py
```

`app.py` checks the MCP server is reachable on startup and refuses to proceed otherwise.

### Running with Docker

A single image runs both processes (`server_mcp.py` in the background, then `streamlit run app.py`
via `entrypoint.sh`, once the MCP server is reachable):

```bash
docker build -t viromechat .
docker run -p 8501:8501 --env-file .env.app --env-file .env.mcp viromechat
```

Secrets are excluded from the image by `.dockerignore` — always pass them at `docker run` time
(`--env-file`, or individual `-e KEY=value` flags), never bake them into the image.

---

## Scientific Guardrails

Enforced through the system prompt, tool-level validation, and post-processing on the client:

* No invention of taxa, species counts, coordinates, or any biological fact — every statement must trace back to a tool call.
* Acronyms (HIV, MPOX, SARS, …) must be resolved via `ncbi_taxonomy_search` before being used in any other tool.
* `query_host_sql` rejects bare `SELECT *` (it would pull ~65 columns, including a heavy geometry blob, across the whole S3 dataset) and only allows read-only `SELECT` statements.
* `create_map` rejects any map that doesn't include the sample identifier (`primary_id`) in its hover data — every plotted point must be traceable back to its exact BioSample sample.
* **PMID hallucination guard**: a whitelist of real PMIDs is built from actual `pubmed_search` calls in the conversation; any PMID outside that whitelist is stripped from the final answer and logged.
* Bracket-style citation artifacts (e.g. `【4†L13-L17】`, a known gpt-oss-120b browsing-tool artifact) are stripped — citations must be real Markdown links to a URL actually returned by a tool.
* If information is absent from the datasets and tools, the agent must say so explicitly rather than guess.

---

## Example Queries

* "Give me information about Orthopoxvirus — is it a genus or a family, and how many species does it include?"
* "Show a pie chart of genus distribution within Poxviridae."
* "World distribution of Poxviridae."
* "Tell me more about Polyomavirus infection pathway."
* "What is HBV, exactly, taxonomically?"

---

## Transparency & Logging

* Executed SQL/pandas code is shown in the "📚 Sources" expander of each response.
* Wikipedia, PubMed, and NCBI Taxonomy links used to build an answer are listed separately in the same expander.
* Every tool call is numbered and fully traced in `logs/agent_YYYY-MM.log`, including a preview of the actual response content (not just success/failure).
* Error reports submitted via the in-app "🚩 Report an error" button are saved to `logs/error_reports/`, bundled with the question, answer, executed code, and recent log lines.

---

## ⚠️ Disclaimer

This system is intended for exploratory and research support purposes only. All outputs should
be independently verified before use in scientific or medical contexts. Dataset coverage reflects
what has been sequenced and deposited in public repositories — it does not reflect epidemiological
prevalence or clinical severity.

---

## License

GNU General Public License v3.0 (GPL-3.0) — see [LICENSE](LICENSE).
