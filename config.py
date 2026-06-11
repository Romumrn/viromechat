"""
config.py — Centralized configuration for the Virus Dataset AI Agent.
Import in any module with: from config import *  or  from config import TAXO_DB_PATH, ...
"""

# ==================== PATHS ==================== #
TAXO_DB_PATH = "data/TAXONOMY.csv"
HOST_DB_PATH = "data/viral_host_clean_llm.csv"
LOG_DIR      = "logs"

# ── Albert API Configuration ──────────────────────────────────────────────────
ALBERT_BASE_URL      = "https://albert.api.etalab.gouv.fr/v1"
ALBERT_TIMEOUT       = 180          # seconds — large models can be slow
ALBERT_MODEL_DEFAULT = "AgentPublic/gptoss120b"  # fallback if model list fails

# ==================== APP ==================== #
PAGE_TITLE   = "Virus Dataset AI Agent 🦠"
PAGE_ICON    = "🦠"
GITHUB_URL   = "https://github.com/Romumrn/chat-virus-AI"

# ==================== AGENT DEFAULTS ==================== #
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

# ==================== UI DEFAULTS ==================== #
DEFAULT_PREVIEW_ROWS   = 50
DEFAULT_WIKIPEDIA_LIMIT = 4000