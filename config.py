"""
config.py — Centralized configuration for the Virus Dataset AI Agent.
Import in any module with: from config import *  or  from config import TAXO_DB_PATH, ...

Secrets/credentials are NOT stored here — they live in two separate,
gitignored .env files, one per process:
  - .env.app  (loaded by app.py)         → ALBERT_API_KEY
  - .env.mcp  (loaded by server_mcp.py)  → S3 credentials (ENDPOINT, ACCESS_KEY, ...)
See .env.app.example / .env.mcp.example for the expected keys.

A user account is required to use the app — there is no guest mode. Accounts
are handled entirely locally by streamlit-authenticator, with credentials
stored in AUTH_CONFIG_PATH — also gitignored, see the ACCOUNTS section in
app.py.
"""

import os

# ==================== PATHS ==================== #
TAXO_DB_PATH = "data/TAXONOMY.csv"
LOG_DIR      = "logs"

APP_ENV_PATH = ".env.app"
MCP_ENV_PATH = ".env.mcp"

# Kept in its own subdirectory (rather than directly under the app's working
# directory) so it can be given its own persistent volume in Docker — see
# docker-compose.yml — without also having to bind-mount the whole app
# directory just to make this one file survive a container rebuild.
AUTH_CONFIG_PATH  = os.path.join("auth_data", ".streamlit_auth.yaml")  # legacy local accounts (migrated into DB_PATH on first run)
USER_HISTORY_DIR  = os.path.join(LOG_DIR, "user_histories")  # legacy per-user chat JSON (migrated into DB_PATH on first run)

# Single SQLite database holding users (email, bcrypt hash, role), their
# conversations and messages, plus a little app_meta (the streamlit-authenticator
# cookie config). Lives alongside AUTH_CONFIG_PATH in the already-persistent
# auth_data/ Docker volume. The legacy YAML + per-user JSON files above are
# imported into it once (see db.maybe_migrate_legacy_data) and then left alone.
DB_PATH = os.path.join("auth_data", "viromechat.db")


def _admin_emails() -> set[str]:
    """Emails granted the 'admin' role, from the ADMIN_EMAILS secret/env
    (comma-separated). Read lazily so tests and the app can set it via env.
    Matching is case-insensitive — emails double as usernames and are stored
    lower-cased (see _register_user in app.py)."""
    raw = os.environ.get("ADMIN_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def load_env_file(env_path: str) -> None:
    """
    Load KEY=VALUE pairs from a .env-style file into os.environ, without
    overriding variables already set in the real environment (so a real
    deployment's env vars always win over a local .env file).
    """
    if not os.path.exists(env_path):
        return
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()

# ── Albert API Configuration ──────────────────────────────────────────────────
ALBERT_BASE_URL      = "https://albert.api.etalab.gouv.fr/v1"
ALBERT_TIMEOUT       = 120          # seconds — large models can be slow
ALBERT_MODEL_DEFAULT = "AgentPublic/gptoss120b"  # fallback if model list fails
ALBERT_WHISPER_MODEL = "openai/whisper-large-v3"  # speech-to-text for the mic input

# The free Albert API is heavily rate-limited (HTTP 429), especially on the
# large models — retry a few times, honoring the server's Retry-After header
# when present and otherwise backing off exponentially up to this cap (seconds).
ALBERT_MAX_RETRIES       = 5
ALBERT_RETRY_BACKOFF_CAP = 30

# ==================== APP ==================== #
PAGE_TITLE   = "Virus Dataset AI Agent 🦠"
PAGE_ICON    = "🦠"
GITHUB_URL   = "https://github.com/Romumrn/chat-virus-AI"

# Overridable via env var: app.py and server_mcp.py run in separate Docker
# containers (see docker-compose.yml), where "localhost" no longer points
# to the other container — compose sets this to http://mcp:8000/mcp there.
# Local, non-Docker dev (both processes on the same host) keeps the default.
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000/mcp")

# ==================== AGENT DEFAULTS ==================== #
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9

DEFAULT_PRESENCE_PENALTY = -0.2
DEFAULT_FREQUENCY_PENALTY = 0.2

DEFAULT_SEED = 42

DEFAULT_MAX_COMPLETION_TOKENS = 4096
DEFAULT_PARALLEL_TOOL_CALLS = False

DEFAULT_MAX_TOOL_CALLS = 7
DEFAULT_MAX_TOOL_CONTENT = 6000

# How many user questions the model keeps context for before the
# conversation memory resets. Tool calls/results are stripped from history
# after each turn (see _clean_history_messages in app.py), so this only
# bounds the number of user/assistant text exchanges kept.
MAX_CONTEXT_TURNS = 5

# ==================== UI DEFAULTS ==================== #
DEFAULT_PREVIEW_ROWS   = 50
DEFAULT_WIKIPEDIA_LIMIT = 4000