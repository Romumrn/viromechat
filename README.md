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

## Architecture

The system is split into **two independent processes**, `app.py` and `server_mcp.py`, plus the
external **Albert API**. Every request starts and ends at `app.py`: it round-trips to Albert (top)
to decide what to do, then round-trips to `server_mcp.py` (bottom) to actually do it, feeding the
result back into the next round-trip to Albert — repeating until Albert returns a final answer
instead of more tool calls:

```
┌─────────────────────────┐
│       Albert API        │
│                         │
└────────────────────────┘
             ▼
             │  chat history + tool specs
             │  tool_calls / final answer
             ▼
┌─────────────────────┐   HTTP request      ┌───────────────────────────┐
│   app.py            │ ─────────────────►  │   server_mcp.py           │
│   Streamlit client  │ ◄─────────────────  │   FastMCP server          │
│   (chat UI, agent   │   tools + resources │   (owns all data access:  │
│    loop)            │                     │   taxonomy CSV + S3 host  │
└─────────────────────┘                     │   Parquet via DuckDB)     │
                                            └───────────────────────────┘
```

* **`app.py`** never touches a dataframe or a credential for the S3 bucket, and never talks to
  Albert and `server_mcp.py` in the same round-trip — it's strictly the middleman relaying
  between the two. It lists the tools the MCP server currently exposes, forwards them to the
  [Albert API](https://albert.api.etalab.gouv.fr) (French government sovereign LLM infrastructure,
  OpenAI-compatible) for tool-calling, and dispatches each call back to the MCP server. It is
  deliberately generic: it reads a tool's JSON schema to decide which UI-configured defaults
  apply, rather than hardcoding tool names.
* **`server_mcp.py`** owns the datasets, the DuckDB/S3 connection, and every tool's business
  logic and guardrails. See [README_MCP.md](README_MCP.md) for the full tool/resource reference. It never talks to Albert directly.

`app.py` and `server_mcp.py` read their own, separate secrets file, see [Configuration](#configuration). Albert needs only the API key in `.env.app`.


## Features

* Natural-language querying of viral taxonomy and virus–host relationships
* Authoritative taxonomy/acronym resolution via NCBI Taxonomy (e.g. `HIV` → `Lentivirus humimdef1`)
* SQL queries against a multi-GB virus–host Parquet dataset on S3, without ever loading it into memory (DuckDB + `httpfs`/`spatial`)
* Interactive Plotly charts and geographic maps
* Wikipedia and PubMed lookups for biological/clinical background, with mandatory inline citations
* Voice input — record a question, transcribed via Albert API's Whisper endpoint
* Conversation memory across up to 5 questions (Q&A text only — tool-call traces are stripped after each turn) with an explicit in-chat reset notice once the limit is hit
* Required local user accounts (login + self-service registration, no external IdP) with a per-user chat history that persists across sessions — see [Accounts](#accounts)
* PMID hallucination guard: any PMID not returned by an actual `pubmed_search` call is stripped from the answer
* Per-tool-call status line (search keyword only — no clutter for dataset/map calls) with full detail logged to disk
* In-app error reporting button (question, answer, executed code, and recent logs bundled into a report file)


## Scientific Guardrails

Enforced through the system prompt, tool-level validation, and post-processing on the client:

* No invention of taxa, species counts, coordinates, or any biological fact — every statement must trace back to a tool call.
* Acronyms (HIV, MPOX, SARS, …) must be resolved via `ncbi_taxonomy_search` before being used in any other tool.
* `query_host_sql` rejects bare `SELECT *` (it would pull ~65 columns, including a heavy geometry blob, across the whole S3 dataset) and only allows read-only `SELECT` statements.
* `create_map` rejects any map that doesn't include the sample identifier (`primary_id`) in its hover data — every plotted point must be traceable back to its exact BioSample sample.
* **PMID hallucination guard**: a whitelist of real PMIDs is built from actual `pubmed_search` calls in the conversation; any PMID outside that whitelist is stripped from the final answer and logged.
* Bracket-style citation artifacts (e.g. `【4†L13-L17】`, a known gpt-oss-120b browsing-tool artifact) are stripped — citations must be real Markdown links to a URL actually returned by a tool.
* If information is absent from the datasets and tools, the agent must say so explicitly rather than guess.

## Example Queries

* "Give me information about Orthopoxvirus — is it a genus or a family, and how many species does it include?"
* "Show a pie chart of genus distribution within Poxviridae."
* "World distribution of Poxviridae."
* "Tell me more about Polyomavirus infection pathway."
* "What is HBV, exactly, taxonomically?"


## Transparency & Logging

* Executed SQL/pandas code is shown in the "📚 Sources" expander of each response.
* Wikipedia, PubMed, and NCBI Taxonomy links used to build an answer are listed separately in the same expander.
* Every tool call is numbered and fully traced in `logs/agent_YYYY-MM.log`, including a preview of the actual response content (not just success/failure).
* Error reports submitted via the in-app "🚩 Report an error" button are saved to `logs/error_reports/`, bundled with the question, answer, executed code, and recent log lines.



## Conversation Memory

Each new question is answered with the previous questions and the model's final text answers as
context. Tool calls and their raw results are dropped from history right after each turn (see
`_clean_history_messages` in `app.py`) — only the user/assistant text is kept. This still lets the
model resolve follow-ups like *"and which family is that genus part of?"* without the subject
being restated, without replaying every past tool call/result to Albert on every new question.

This context accumulates for up to `MAX_CONTEXT_TURNS` questions (`config.py`, default **5**,
adjustable at runtime from the ⚙️ Expert mode sidebar). Once the limit is reached, the conversation
memory resets — a message is posted in the chat ("🔄 Conversation context reset after N
questions — starting fresh…") and the next question starts with no memory of what came before. A
failed turn (API timeout/error) is never added to memory and doesn't count against the limit.

When [accounts](#accounts) are enabled, this same history is also saved to disk per user (see
below), so it survives page reloads — not just the current browser session.

## Accounts

A user account is required to use the app — there is no guest mode. Accounts are fully local — no
external identity provider, no email service — built on top of
[`streamlit-authenticator`](https://github.com/mkhorasani/Streamlit-Authenticator) for login,
session cookies, and credential storage, with a custom minimal registration form:

* **Registration** — first name, last name, institutional email, password, and a captcha (to deter
  bot sign-ups). No separate username (the email doubles as one), no repeat-password field.
  Validated at submit: the password must be **at least 6 characters and contain 1 special
  character** (shown as a hint under the field), and the email domain must **not** be a free
  webmail provider (Gmail, Outlook, Yahoo, iCloud, Proton, Orange, … — see `_BLOCKED_EMAIL_DOMAINS`
  in `app.py`). This is a quick blocklist, not a real institution allowlist — someone with their
  own custom domain still gets through, the goal is just to steer people to their work address.
  Self-service and collapsed by default: anyone who can reach the app can register, but the form
  only expands once someone clicks "Create an account".
* **Optional invite code** — set `REGISTRATION_CODE` in `.env.app` (or `.streamlit/secrets.toml`)
  to gate registration behind a shared code: an extra "Registration code" field appears and must
  match before an account is created. Leave it unset to keep registration open. The expected value
  is only ever read from the secret, never stored in the source.
* **Login** — asks for email + password. A signed cookie then keeps the user logged in across page
  reloads for `cookie.expiry_days` (30 by default).
* **Storage** — emails, bcrypt-hashed passwords, first/last names live in
  `auth_data/.streamlit_auth.yaml` (gitignored, auto-created on first run with a random
  cookie-signing key; under Docker it's a host bind-mount, see [Running with Docker](#running-with-docker)).
  There is no email verification step — account creation is immediate, gated only by email
  uniqueness.
* **Per-user chat history** — each account's conversation (questions, answers, and the Albert-format
  context used for follow-ups) is saved to its own file under `logs/user_histories/` and reloaded
  on login, so it persists across sessions. A "🗑️ Clear my history" button in the sidebar resets it.


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
  ```

  User accounts (see [Accounts](#accounts)) need no configuration here — they're always on and
  self-provisioning (`.streamlit_auth.yaml` is created automatically on first run).

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
whenever both are present (see `_get_secret` in `app.py`).

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

Each process gets its own image and container, orchestrated by `docker-compose.yml`:

* **`Dockerfile.mcp`** → `mcp` service (`server_mcp.py`, port 8000)
* **`Dockerfile.app`** → `app` service (`app.py`, port 8501) — waits for `mcp`'s healthcheck to
  pass before starting (`depends_on: condition: service_healthy`), then reaches it at
  `http://mcp:8000/mcp` (`MCP_SERVER_URL`, set via Compose — the two containers no longer share
  `localhost`, unlike when both processes ran in one container)

```bash
docker compose up --build
```

Secrets are excluded from both images by `.dockerignore` — each service loads only its own
`.env.app` / `.env.mcp` at runtime via Compose's `env_file:`, never baked into the image. Local
accounts (`auth_data/.streamlit_auth.yaml`, i.e. `AUTH_CONFIG_PATH`) live in a **host bind-mount**
(`./auth_data`), so the file sits directly on the host/VM and can be read, edited, or backed up
without going through the container. The `auth_data/` directory is tracked in git (via `.gitkeep`,
its contents gitignored) so it exists before the first run.

Because a bind-mount keeps the host directory's ownership — which may not match the container's
non-root `app` user (uid 1000) and would otherwise cause `PermissionError: … auth_data/…` — the
app container starts from an entrypoint (`entrypoint-app.sh`) that runs briefly as root to
`chown` the mounted `auth_data/` and `logs/` to the app user, then drops privileges (via `gosu`)
to run Streamlit as `app`. So account creation works regardless of host-side ownership, no manual
`chown` needed.


## ⚠️ Disclaimer

This system is intended for exploratory and research support purposes only. All outputs should
be independently verified before use in scientific or medical contexts. Dataset coverage reflects
what has been sequenced and deposited in public repositories — it does not reflect epidemiological
prevalence or clinical severity.

## License

GNU General Public License v3.0 (GPL-3.0) — see [LICENSE](LICENSE).
