# AGENTS.md — Viromech@t

## Purpose

Viromech@t is an AI application for exploring viral taxonomy and virus–host relationships from the Virome@tlas datasets.

The application is deliberately tool-grounded: the LLM does **not** access the raw datasets directly. It decides which audited MCP tools to call, receives their results, and produces an answer from those results. Do not weaken this boundary or introduce a code path that lets the model invent dataset facts.

## Repository layout

```text
.
├── backend/
│   ├── agent.py          # Main async agent/tool-calling loop
│   ├── albert.py         # ALBERT API client and MCP/tool helpers
│   ├── auth.py           # Passwords, JWTs, roles, registration policy
│   ├── main.py           # FastAPI application
│   ├── reports.py        # Error-report persistence
│   ├── schemas.py        # API/Pydantic schemas
│   └── routers/          # HTTP route modules
├── frontend/
│   ├── src/              # React + TypeScript application
│   └── package.json
├── auth_data/
│   └── viromechat.db     # Local SQLite persistence
├── config.py              # Central configuration/defaults
├── db.py                  # SQLite persistence layer
├── prompt.py              # System-prompt construction
├── logging_utils.py       # Logging setup
├── requirements/
├── docker/
├── docker-compose.yml
└── tests/
```

The MCP data server is **not part of this repository**. It lives in the separate `viromeatlas_mcp` repository and is reached over HTTP through `MCP_SERVER_URL`.

## Architecture

The runtime has three main components:

1. **React frontend** — user interface, conversations, account/admin/dev views.
2. **FastAPI backend** — authentication, persistence, SSE streaming, and the agent loop.
3. **MCP data server** — datasets, tool implementations, validation, and data-access guardrails.

The backend sends the available MCP tool schemas to the ALBERT API. ALBERT chooses whether to answer or request tools. The backend executes requested tools through MCP, feeds the results back to ALBERT, and repeats until a final answer is produced.

The normal request flow is:

```text
Browser
   │ REST + SSE
   ▼
FastAPI backend
   │
   ├── JWT/authentication
   ├── conversation persistence
   └── run_agent()
          │
          ├── ALBERT API
          │      └── tool-call decisions
          │
          └── MCP server
                 ├── NCBI Taxonomy
                 ├── PubMed
                 ├── Wikipedia
                 └── Virome@tlas data / SQL tools
```

In development:

- Frontend: `http://localhost:5173`
- FastAPI: `http://localhost:8080`
- MCP server: normally `http://localhost:8000/mcp`

In production, the backend can serve the built React application as a single origin.

## Core development rules

### 1. Preserve the data boundary

The backend must not directly access Virome@tlas dataframes, S3 credentials, or MCP-owned business logic.

If a new data operation is needed:

1. implement it in the MCP server;
2. expose it through a properly documented MCP tool/resource;
3. consume it from this repository through the existing MCP client path.

Do not duplicate dataset logic in the FastAPI backend.

### 2. Keep scientific answers tool-grounded

The application has explicit anti-hallucination requirements:

- Never invent taxa, species counts, coordinates, biological facts, or dataset values.
- If the available tools/datasets do not contain the requested information, say so.
- Resolve biological acronyms through `ncbi_taxonomy_search` before using them in other tools when required by the system prompt.
- Keep PubMed identifiers restricted to PMIDs actually returned by `pubmed_search`.
- Keep citation URLs tied to URLs actually returned by tools.
- Preserve the post-processing guards in `backend/albert.py`.

Do not remove or bypass these protections merely to make an answer look more complete.

### 3. Treat MCP schemas as the source of truth

`backend/agent.py` is intentionally generic. It discovers the available MCP tools and their JSON schemas instead of hardcoding the full dataset schema.

When adding or changing an MCP tool, prefer schema-driven behavior over adding tool-name-specific assumptions in the backend.

### 4. Keep the agent loop streaming

`backend/agent.py::run_agent()` is an async generator. It yields SSE events during execution.

Important event types include:

```json
{"type": "status", "text": "..."}
{"type": "tool_call", "...": "..."}
{"type": "tool_result", "...": "..."}
{"type": "figure", "figure": {...}}
{"type": "sources", "...": "..."}
{"type": "final", "content": "..."}
{"type": "error", "message": "..."}
{"type": "done", "...": "..."}
```

When changing event payloads, update both the backend producer and the frontend consumer. Do not silently change event names or required fields.

### 5. Keep conversation history bounded

The application strips tool-call bookkeeping before replaying conversation history.

`MAX_CONTEXT_TURNS` currently defaults to `5`. Tool calls/results should not be persisted into the model context as ordinary user/assistant conversation turns.

Use the existing helpers in `backend/albert.py` rather than implementing a second history-cleaning strategy.

## Agent implementation

### `backend/agent.py`

The main entry point is:

```python
async def run_agent(
    model: str,
    api_key: str,
    user_query: str,
    username: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    preview_rows: int = DEFAULT_PREVIEW_ROWS,
    wikipedia_limit: int = DEFAULT_WIKIPEDIA_LIMIT,
    max_tool_content: int = DEFAULT_MAX_TOOL_CONTENT,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
    seed: int = DEFAULT_SEED,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    parallel_tool_calls: bool = DEFAULT_PARALLEL_TOOL_CALLS,
    history_messages: list | None = None,
)
```

`check_mcp_connection()` performs a short-lived `list_tools()` probe and returns a boolean. Connection failures are logged with `MCP_CONNECTION_FAIL`.

`run_agent()` keeps per-turn state for:

- Wikipedia URLs
- PubMed URLs
- NCBI URLs
- executed SQL/code
- generated Plotly figures
- authentic PMIDs
- tool-call count

When the tool-call limit is reached, the agent must stop requesting additional tools and synthesize from the information already collected.

The agent also contains a fallback for empty ALBERT responses after tool calls. Preserve that fallback unless the underlying ALBERT behavior changes and the replacement is verified.

## ALBERT client

### `backend/albert.py`

`albert_chat()` sends OpenAI-compatible chat-completion requests to ALBERT.

Current defaults are centralized in `config.py`:

| Setting | Default |
|---|---:|
| `ALBERT_BASE_URL` | `https://albert.api.etalab.gouv.fr/v1` |
| `ALBERT_TIMEOUT` | `120` seconds |
| `ALBERT_MODEL_DEFAULT` | `openai/gpt-oss-120b` |
| `ALBERT_MAX_RETRIES` | `5` |
| `ALBERT_RETRY_BACKOFF_CAP` | `30` seconds |

For HTTP 429 responses:

1. honor `Retry-After` when it is a numeric value;
2. otherwise use exponential backoff;
3. cap the wait at `ALBERT_RETRY_BACKOFF_CAP`;
4. raise `AlbertRateLimitError` after retries are exhausted.

Do not log API keys or other secrets.

### Tool conversion and result normalization

Keep these helpers as the canonical translation layer:

- `mcp_tools_to_openai_spec()`
- `unwrap_mcp_result()`
- `describe_available_datasets()`
- `parse_tool_arguments()`
- `clean_history_messages()`
- `build_context_window()`
- `strip_hallucinated_pmids()`
- `strip_fake_citation_markers()`

`parse_tool_arguments()` intentionally tolerates malformed partial JSON because some models may emit incomplete tool arguments. Do not turn malformed model output into an unvalidated tool invocation.

## Authentication and authorization

Authentication lives in `backend/auth.py`.

Passwords are stored as bcrypt hashes. Sessions use stateless JWTs with:

```text
algorithm: HS256
default lifetime: 720 minutes (12 hours)
```

Roles are ordered:

```text
user < dev < admin
```

- `user`: chat, own conversations, password changes.
- `dev`: user capabilities plus Expert mode, MCP tool testing, and agent-log access.
- `admin`: dev capabilities plus user administration, role changes, deletion, platform statistics, and access to users' conversations.

Role checks must go through `require_role()` / `get_current_user()` rather than being duplicated in individual handlers.

The first admin can be bootstrapped through `ADMIN_EMAILS` in `.env.app`.

### Registration policy

The current password policy requires:

- at least 12 characters;
- at least one lowercase letter;
- at least one uppercase letter;
- at least one digit;
- at least one special character.

The registration flow blocks a configured list of common consumer/free-mail domains and is intended for institutional accounts.

Do not weaken authentication or authorization checks to simplify development.

## Configuration and secrets

Configuration is centralized in `config.py`.

Secrets belong in `.env.app` or the deployment environment and must never be committed.

Important variables include:

```text
ALBERT_API_KEY
JWT_SECRET
ADMIN_EMAILS
REGISTRATION_CODE       # when enabled by the application
MCP_SERVER_URL
JWT_EXPIRE_MIN
```

`JWT_SECRET` must be replaced with a long random secret in any real deployment. The development fallback `dev-insecure-change-me` is intentionally unsafe.

The SQLite database path defaults to:

```text
auth_data/viromechat.db
```

Logs are written under:

```text
logs/
```

## Database and persistence

`db.py` owns SQLite persistence for:

- users;
- conversations;
- messages;
- application metadata;
- error reports.

Keep database access in the existing persistence layer. Routes and agent code should not create ad-hoc SQLite schemas or bypass the helper functions.

Conversation payloads can include generated figures, sources, and executed code metadata. Preserve backward compatibility when changing stored message structures.

## Error reporting and logging

`backend/reports.py::save_error_report()` stores a report in the database and attempts a JSON backup under:

```text
logs/error_reports/
```

Reports can contain:

- the user's question;
- the generated answer;
- executed SQL/code;
- user comment;
- user email when available;
- recent agent-log lines.

Agent logs use the monthly naming pattern:

```text
logs/agent_YYYY-MM.log
```

Keep logs useful for debugging, but never add secrets, API keys, JWTs, or credentials to them.

## Prompt changes

The system prompt is built by:

```python
prompt.build_system_prompt()
```

The dataset description is supplied dynamically from MCP resources. The backend must not silently assume column names or dataset semantics owned by the MCP server.

When changing the system prompt:

- preserve scientific grounding requirements;
- preserve acronym-resolution rules;
- preserve citation requirements;
- preserve the instruction to explicitly report missing information;
- avoid adding facts about datasets that are not guaranteed by the MCP server.

## Frontend rules

The frontend is a React + Vite + TypeScript + Tailwind application.

Main UI areas include:

- Login
- Register
- Chat
- Account
- Admin
- Dev

The frontend consumes the backend's REST/SSE API. If an SSE event changes, update the frontend parser and rendering logic together.

Do not expose backend secrets or MCP credentials to browser code.

## Adding a new MCP tool

When a new data capability is required:

1. Implement the tool in `viromeatlas_mcp`.
2. Give it an accurate JSON input schema.
3. Document its behavior and constraints in the MCP server.
4. Verify that the backend can discover it through `list_tools()`.
5. Verify argument parsing and sanitization.
6. Verify the tool result normalization path.
7. Add artifact handling only if the tool returns a new artifact type.
8. Add/update frontend rendering if the tool introduces a new SSE event or artifact.
9. Add tests for the integration.

Avoid hardcoding the tool into the agent unless there is a genuine backend-specific reason.

## Running locally

The repository currently expects three processes for development.

### 1. MCP server

From the separate `viromeatlas_mcp` repository:

```bash
cd ../viromeatlas_mcp
python3 server_mcp.py
```

The MCP server needs its own `.env` and data-service credentials.

### 2. FastAPI backend

Create `.env.app` from `.env.app.example`, then install:

```bash
pip install -r requirements/api.txt
```

Run:

```bash
uvicorn backend.main:app --reload --port 8080
```

API documentation:

```text
http://localhost:8080/docs
```

### 3. React frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

If the backend is on another port:

```bash
API_PROXY_TARGET=http://localhost:<port> npm run dev
```

## Docker

The repository's Compose setup starts the API container, which also serves the built React SPA.

Start it with:

```bash
docker compose up --build
```

The API is exposed on:

```text
http://localhost:8080
```

The MCP server remains a separate service/repository. Configure `MCP_SERVER_URL` appropriately for the deployment.

## Testing and verification

Before considering a backend change complete:

1. Run the relevant Python tests.
2. Exercise authentication/authorization paths when they are affected.
3. Exercise the affected API endpoint.
4. If the agent loop changed, test:
   - normal final answers;
   - tool calls;
   - tool errors;
   - MCP unavailability;
   - ALBERT rate limiting;
   - empty ALBERT responses after tools;
   - PMID/citation cleanup.
5. If SSE payloads changed, test the frontend event handling.
6. If database structures changed, verify existing conversation/user data remains readable.

For frontend changes, run the repository's configured npm checks/build in addition to the relevant tests.

## Troubleshooting

### ALBERT rate limiting

Symptoms:

- HTTP 429 responses;
- `RATE_LIMIT` logs;
- `AlbertRateLimitError`.

Check the ALBERT API status/limits and let the existing retry logic run before adding custom retries.

### Empty answer after tool calls

The agent has a fallback synthesis path for this situation. Check the tool results and the ALBERT response before changing the fallback.

### MCP unavailable

Check:

- `MCP_SERVER_URL`;
- whether the MCP server is running;
- container/host networking;
- MCP server credentials and its own `.env`;
- `MCP_CONNECTION_FAIL` logs.

### Unexpected scientific claims

Trace the claim back through:

```text
final answer
→ agent tool results
→ MCP tool
→ MCP dataset/resource
```

Do not fix a grounding problem by simply adding a stronger prose instruction. Verify the tool/data path and preserve the output guards.

## Security checklist

Before merging changes:

- No secrets committed.
- No API keys/JWTs/passwords in logs.
- JWT secret is non-default in production.
- Role checks remain enforced server-side.
- User-owned conversations remain isolated unless the caller has the required admin privilege.
- MCP data access remains behind the MCP server.
- SQL/data tools remain subject to their server-side read-only/validation guardrails.
- Scientific anti-hallucination checks remain enabled.
- User-controlled input is validated through the existing Pydantic/FastAPI schemas.

## Source of truth

This file describes the current repository architecture and development constraints.

For API endpoints and role behavior, consult `README_API.md`.

For dataset/tool semantics, consult the separate `viromeatlas_mcp` repository and its tool/resource documentation.

For runtime defaults, consult `config.py`.

For the model's scientific behavior, consult `prompt.py`.

Do not copy old architecture documentation forward when it conflicts with the current source code.
