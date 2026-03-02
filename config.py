"""
config.py — Centralized configuration for the Virus Dataset AI Agent.
Import in any module with: from config import *  or  from config import TAXO_DB_PATH, ...
"""

# ==================== PATHS ==================== #
TAXO_DB_PATH = "data/TAXONOMY.csv"
HOST_DB_PATH = "data/viral_host_clean_llm.csv"
LOG_DIR      = "logs"

# ==================== OLLAMA ==================== #
OLLAMA_BASE_URL    = "http://localhost:11434"
OLLAMA_TIMEOUT     = 120          # seconds before giving up on a model response
OLLAMA_DEFAULT_MODEL_PREFIX = "gpt-oss"  # used to pre-select model in the UI

# ==================== APP ==================== #
PAGE_TITLE   = "Virus Dataset AI Agent 🦠"
PAGE_ICON    = "🦠"
GITHUB_URL   = "https://github.com/Romumrn/chat-virus-AI"

# ==================== AGENT DEFAULTS ==================== #
DEFAULT_TEMPERATURE    = 0.0
DEFAULT_TOP_P          = 1.0
DEFAULT_REPEAT_PENALTY = 1.0
DEFAULT_SEED           = 42
DEFAULT_MAX_TOOL_CALLS = 7
DEFAULT_MAX_TOOL_CONTENT = 6000   # chars — truncation limit sent back to the model

# ==================== UI DEFAULTS ==================== #
DEFAULT_PREVIEW_ROWS   = 50
DEFAULT_WIKIPEDIA_LIMIT = 5000

# ==================== TAXONOMY COLUMNS ==================== #
# Expected column names in df_taxo — update here if the CSV schema changes
TAXO_COL_FAMILY  = "family"
TAXO_COL_GENUS   = "genus"
TAXO_COL_SPECIES = "species"

# ==================== HOST COLUMNS ==================== #
# Expected column names in df_host — update here if the CSV schema changes
HOST_COL_VIRUS   = "virus_name"
HOST_COL_HOST    = "host_name"
HOST_COL_COUNTRY = "country"