# `server_mcp.py` — the ViromeChatMCP server

A [FastMCP](https://gofastmcp.com/) server that owns **all** dataset access, external API calls,
and business logic for Viromech@t. The Streamlit client ([app.py](app.py), see the main
[README](README.md)) never touches a dataframe, an S3 credential, or a column name directly — it
only talks to this server over MCP/HTTP, generically, by reading whatever tools and resources it
currently publishes.

This document is the reference for everyone working on the server itself: what each tool/resource
does, the response contract every tool must follow, and how to extend it safely.

---

## Running it

```bash
cp .env.mcp.example .env.mcp   # fill in your S3 credentials
python server_mcp.py
```

Starts an HTTP server on `0.0.0.0:8000`, MCP endpoint at `/mcp` (`http://localhost:8000/mcp`,
matching `MCP_SERVER_URL` in `config.py`). On startup it:

1. Loads `data/TAXONOMY.csv` fully into memory as `df_taxo`.
2. Loads the two column-description files (`data/v@_columns_description.csv` and
   `data/TAXONOMY_columns_description.json`) that back the two MCP resources below.
3. Opens an in-memory DuckDB connection, installs the `httpfs` and `spatial` extensions, and
   registers a `host` view over the S3 Parquet dataset — **the Parquet file is never loaded into
   memory**; every `query_host_sql` call is pushed down to S3 by DuckDB (column/row-group
   pruning).

---

## Resources

Resources are static, read-once knowledge — not something the LLM "calls" like a tool. The client
reads them once per conversation and folds their content into the system prompt.

| URI | Content | Source |
|---|---|---|
| `resource://datasets/host/schema` | JSON map `{column_name: {description, Type}}` for every column of the `host` table | `data/v@_columns_description.csv` |
| `resource://datasets/taxonomy/schema` | Full JSON schema (name, description, columns, primary key, row definition) of `df_taxo` | `data/TAXONOMY_columns_description.json` |

Adding a new resource (e.g. a third dataset) requires no client-side change: `app.py` discovers
resources via `list_resources()` and reads each one generically.

---

## The response contract

**Every tool returns exactly this shape**, regardless of what it does:

```jsonc
{
  "success": true,           // or false
  "content": "human-readable text — this is what the LLM reads back as the tool result",
  "artifacts": [ ... ]        // structured extras the client can render; [] if none
}
```

On failure, `content` holds the error message (with retry guidance where possible) and
`artifacts` is empty. The two helpers `_ok(content, artifacts)` / `_fail(content)` at the top of
`server_mcp.py` build this shape — always use them instead of hand-rolling a dict.

### Artifact types

| `type` | Emitted by | Shape | Consumed by the client as |
|---|---|---|---|
| `url` | `wikipedia_search` | `{"type": "url", "url": "..."}` | Wikipedia link in the "Sources" panel |
| `pubmed` | `pubmed_search` | `{"type": "pubmed", "pmids": [123, 456]}` | PubMed links + PMID whitelist for the hallucination guard |
| `ncbi_taxonomy` | `ncbi_taxonomy_search` | `{"type": "ncbi_taxonomy", "url": "...", "tax_id": "..."}` | NCBI Taxonomy link in the "Sources" panel |
| `table` | `query_host_sql`, `query_dataframe` | `{"type": "table", "rows": [...], "columns": [...], "total_rows": N}` | Tracked as executed SQL/code in "Sources"; `rows` capped to `preview_rows` |
| `plotly` | `create_visualization`, `create_map` | `{"type": "plotly", "figure": {...}}` (from `fig.to_json()`, parsed back to a dict) | Rendered with `st.plotly_chart` after `pio.from_json(...)` |

The client (`app.py`) dispatches purely on `artifact["type"]` — never on the tool's name. Adding a
tool that reuses an existing artifact type (e.g. another `"table"`-returning tool) requires **no
client change at all**.

---

## Tools

### `wikipedia_search(search_term: str, wikipedia_limit: int = 4000) -> dict`

Looks up a page on Wikipedia; falls back to the closest full-text search match if there's no exact
title match (flagged as a "fuzzy match" note in the content). Returns a `url` artifact.

### `pubmed_search(query: str, max_results: int = 5) -> dict`

Searches PubMed (NCBI E-utilities `esearch` + `efetch`, db=`pubmed`) and returns title, authors,
journal, year, abstract, DOI, and PMID for each hit. Returns a `pubmed` artifact with every real
PMID found — this is the sole source of truth for the client's PMID hallucination guard.

### `ncbi_taxonomy_search(name: str) -> dict`

Resolves any organism name — acronym, common name, or scientific name — against the **NCBI
Taxonomy** database (E-utilities, db=`taxonomy`). Returns, for every match: scientific name, rank
(species/genus/family/…), division, full lineage, and known synonyms/acronyms. This is the
authoritative way to turn `HIV` into `Human immunodeficiency virus 1` / genus `Lentivirus`, or to
check whether a name is a genus or a family, without depending on Wikipedia's phrasing. Returns an
`ncbi_taxonomy` artifact for the top match.

> Implementation note: NCBI's `efetch` XML nests one `<Taxon>` per ancestor rank inside each
> result's `<LineageEx>`. The parser only iterates `root.findall("Taxon")` (direct children) —
> using `.//Taxon` would also pick up every ancestor as if it were a separate match.

### `query_host_sql(sql: str, preview_rows: int = 50) -> dict`

Runs a read-only `SELECT` against the `host` view (the S3 Parquet dataset), returning a `table`
artifact. This is the **required first step** before `query_dataframe`, `create_visualization`, or
`create_map` can use `df_host` — those tools operate on the result of the *last* `query_host_sql`
call (`ctx.last_host_result`), never on the full dataset.

Guardrails enforced before execution:
* Only a single `SELECT` statement — `INSERT/UPDATE/DELETE/DDL/PRAGMA/...` are rejected by
  `_FORBIDDEN_SQL_KEYWORDS`.
* **Bare `SELECT *` is rejected outright.** `host` has ~65 columns including a heavy `geometry`
  blob; pulling every column for every matching row over S3 is what caused multi-minute timeouts
  before this guard existed. Callers must project only the columns they need.
* Coordinates live in a native `GEOMETRY` point column, not plain `lat`/`lon` — extract them with
  `ST_X(geometry) AS lon, ST_Y(geometry) AS lat` (the `spatial` extension is loaded at startup).

### `query_dataframe(code: str, preview_rows: int = 50) -> dict`

Executes pandas code with `df_taxo`, `df_host` (= `ctx.last_host_result`, or a clear error if
`query_host_sql` hasn't been called yet), `pd`, and `np` in scope. Must assign a DataFrame to
`result`. Returns a `table` artifact.

### `create_visualization(code: str) -> dict`

Same execution environment as `query_dataframe`, plus `px`/`go`. Must assign a Plotly figure to
`fig`. Rejects empty figures (0 data points) with a guidance message rather than silently returning
a blank chart. Returns a `plotly` artifact.

### `create_map(code: str) -> dict`

Same as `create_visualization`, but enforces `px.scatter_mapbox(...)` (never `scatter_map`) and
that the preceding `query_host_sql` call already extracted `lon`/`lat` from `geometry`. Returns a
`plotly` artifact.

---

## Extending the server

To add a new tool:

1. Write it as a plain function decorated with `@mcp.tool`, returning `_ok(content, artifacts)` or
   `_fail(content)` — never a hand-built dict.
2. If it produces something the client should render specially (a link, a table, a figure), reuse
   an existing artifact `type` from the table above whenever the shape fits — this means zero
   client changes. Only invent a new `type` (and wire it into `app.py`'s dispatch loop) if the
   shape is genuinely new.
3. Put every usage rule, caveat, and example **in the tool's docstring**. It is sent verbatim to
   the LLM as the tool's description — this is the only place dataset-specific guidance should
   live; `app.py` and `prompt.py` are deliberately generic and must not encode knowledge about
   individual tools or column names.
4. If the tool needs a UI-configurable default (like `preview_rows` or `wikipedia_limit`), just
   name the parameter that; `app.py` applies the matching sidebar setting to any tool whose JSON
   schema declares a parameter with that name — no code change needed there either.

---

## Configuration

`server_mcp.py` reads `.env.mcp` (see [`.env.mcp.example`](.env.mcp.example)) at import time,
via `load_env_file()` from `config.py`:

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `ENDPOINT` | yes | — | S3-compatible endpoint hostname |
| `ACCESS_KEY` | yes | — | S3 access key |
| `SECRET_KEY` | yes | — | S3 secret key |
| `BUCKET` | yes | — | S3 bucket name |
| `VIRAL_HOST_DATASET` | yes | `*.parquet` | Object key of the Parquet dataset inside the bucket |
| `REGION` | no | `fr` | S3 region |
| `S3_URL_STYLE` | no | `path` | DuckDB `s3_url_style` setting |

Non-secret settings (`TAXO_DB_PATH`, default preview row count, default Wikipedia extract length)
are in `config.py`, shared with `app.py`.
