#!/usr/bin/python3
"""
Viromech@t: A conversational interface for exploring viral metagenomic data.
Uses Albert API (OpenAI-compatible) with tool-calling capabilities for:
- Dataset queries (taxonomy and host databases)
- Wikipedia and PubMed searches
- Data visualization and mapping

All tools (wikipedia_search, pubmed_search, query_host_sql, query_dataframe,
create_visualization, create_map) and dataset schema resources now live in a
separate FastMCP server (server_mcp.py) and are reached exclusively through
the MCP protocol over HTTP — this app no longer imports them or holds the
DataFrames in memory.
"""

import asyncio
import streamlit as st
import requests
import os
import json
import logging
import time
import re
from datetime import datetime

import bcrypt
import plotly.io as pio
from fastmcp import Client
import streamlit_authenticator as stauth

# Local imports
import db
from prompt import build_system_prompt
from config import (
    ALBERT_BASE_URL, ALBERT_TIMEOUT, ALBERT_MODEL_DEFAULT, ALBERT_WHISPER_MODEL,
    ALBERT_MAX_RETRIES, ALBERT_RETRY_BACKOFF_CAP,
    LOG_DIR, MCP_SERVER_URL, APP_ENV_PATH,
    _admin_emails,
    PAGE_TITLE, PAGE_ICON, GITHUB_URL,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_PRESENCE_PENALTY,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_PARALLEL_TOOL_CALLS, DEFAULT_SEED,
    DEFAULT_MAX_TOOL_CALLS, DEFAULT_MAX_TOOL_CONTENT,
    DEFAULT_PREVIEW_ROWS, DEFAULT_WIKIPEDIA_LIMIT,
    MAX_CONTEXT_TURNS,
    load_env_file,
)
from logging_utils import setup_logger

# Local dev secrets (ALBERT_API_KEY) — see .env.app.example. On Streamlit
# Community Cloud, st.secrets is used instead and always takes priority (see
# _get_secret below). Account credentials themselves live in the SQLite
# database (config.DB_PATH), not in this file — see the ACCOUNTS section.
load_env_file(APP_ENV_PATH)


# ── Logging & Error Reporting ─────────────────────────────────────────────────
REPORT_DIR = os.path.join(LOG_DIR, "error_reports")
logger = setup_logger(LOG_DIR)


# ── UI labels for tool status display (kept local — no need to reach the
#    server just to show an icon). Must stay in sync with the tool names
#    exposed by server_mcp.py. ─────────────────────────────────────────
TOOL_LABELS = {
    "wikipedia_search":     ("📖", "Wikipedia search"),
    "pubmed_search":        ("🔬", "PubMed search"),
    "ncbi_taxonomy_search": ("🧬", "NCBI Taxonomy lookup"),
    "query_host_sql":       ("🪣", "Querying S3 host table"),
    "query_dataframe":      ("🔬", "Dataset query"),
    "create_visualization": ("📊", "Creating chart"),
    "create_map":           ("🗺️",  "Creating map"),
}


def _ui_search_keyword(call_args: dict) -> str:
    """Pick out just the search keyword (search_term / query / name) for the
    UI status line, generically — no tool-name special-casing. Dataset/map
    tools (sql, code params) have no such keyword and return "", so their
    status line shows only the tool label, as requested.
    """
    for key in ("search_term", "query", "name"):
        val = call_args.get(key)
        if isinstance(val, str) and val.strip():
            snippet = " ".join(val.strip().split())
            return snippet[:80] + ("…" if len(snippet) > 80 else "")
    return ""


def _snippet(text: str, max_len: int = 120) -> str:
    """Collapse arbitrary tool output to a single-line preview (used in logs)."""
    snippet = " ".join(text.strip().split())
    return snippet[:max_len] + ("…" if len(snippet) > max_len else "")


# ==================== MCP HELPERS (inline, no separate wrapper module) ====================

def _mcp_tools_to_openai_spec(tools) -> list[dict]:
    """Convert fastmcp Tool objects (from client.list_tools()) to the OpenAI
    `tools=[...]` format expected by Albert API."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def _unwrap_mcp_result(result) -> dict:
    """Normalize a fastmcp CallToolResult into the plain dict our tools return."""
    if getattr(result, "data", None) is not None:
        return result.data
    if getattr(result, "structured_content", None) is not None:
        return result.structured_content
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", None) == "text":
            try:
                return json.loads(block.text)
            except json.JSONDecodeError:
                return {"success": False, "content": block.text, "artifacts": []}
    return {"success": False, "content": "Empty MCP tool response", "artifacts": []}


async def _describe_available_datasets(mcp: Client) -> str:
    """
    Fetch every resource the MCP server currently publishes and render them
    as a single text block for the system prompt.

    This app never assumes which resources exist, how many there are, or
    what shape their content takes — the MCP server is the sole owner of
    dataset knowledge (schemas, column semantics, etc.). Adding, renaming,
    or restructuring a resource server-side requires no client change.
    """
    try:
        resources = await mcp.list_resources()
    except Exception as e:
        logger.warning(f"MCP_RESOURCES_LIST_FAIL | {e}")
        return ""

    blocks = []
    for r in resources:
        try:
            contents = await mcp.read_resource(r.uri)
            text = next((getattr(c, "text", None) for c in contents if getattr(c, "text", None)), None)
            if text:
                blocks.append(f"### {r.name or r.uri}\n{text}")
        except Exception as e:
            logger.warning(f"MCP_RESOURCE_READ_FAIL | {r.uri} | {e}")

    return "\n\n".join(blocks)


# ==================== PMID HALLUCINATION GUARD ====================

def _strip_hallucinated_pmids(text: str, real_pmids: set) -> tuple[str, list]:
    """
    Remove PMID references that were hallucinated by the model (not from actual PubMed searches).

    Args:
        text: The model's output text to clean
        real_pmids: Set of PMIDs that were actually returned by pubmed_search calls

    Returns:
        Tuple of (cleaned_text, list_of_removed_pmids)
    """
    pattern = re.compile(r'\bPMID[:\s#]*([0-9]{5,9})\b', re.IGNORECASE)
    removed = []

    def _replace(match):
        pmid = match.group(1)
        if pmid in real_pmids:
            return match.group(0)
        removed.append(pmid)
        return ""

    cleaned = pattern.sub(_replace, text)
    cleaned = re.sub(r'\(e\.g\.\s*,?\s*on\s+[^)]{0,80}\)', '', cleaned)
    cleaned = re.sub(r'\(see\s*\)', '', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    return cleaned, removed


_FAKE_CITATION_PATTERN = re.compile(r'【[^【】]*】')


def _strip_fake_citation_markers(text: str) -> tuple[str, int]:
    """
    Strip bracket-style citation markers like 【4†L13-L17】 that gpt-oss-120b
    sometimes emits — an artifact of browsing-tool citation formats seen in
    its training data. This MCP setup has no such citation system, so these
    markers never point to a real, resolvable source; they must be removed
    rather than shown to the user as if they were real references.
    """
    cleaned, n = _FAKE_CITATION_PATTERN.subn('', text)
    if n:
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = cleaned.strip()
    return cleaned, n


# ==================== SECRETS (st.secrets, falling back to .env.app) ====================

def _get_secret(key: str, default: str = "") -> str:
    """Read a secret with priority: st.secrets (Streamlit Cloud deployment)
    then the environment (populated locally from .env.app)."""
    val = st.secrets.get(key)
    if val is None:
        val = os.environ.get(key, default)
    return val


# ==================== ACCOUNTS (local, no external IdP) ====================
#
# A user account is required to use this app — there is no guest mode.
# Accounts live in the SQLite database (db.py): bcrypt-hashed passwords and a
# signed re-auth cookie via streamlit-authenticator, no external identity
# provider and no email service. streamlit-authenticator stays the crypto /
# cookie / session engine — it's fed a credentials dict built from the users
# table (db.build_credentials_dict) rather than reading a YAML file, and the
# cookie signing key lives in the app_meta table (db.get_cookie_config).
# The email address doubles as the username (login asks for "Email", not a
# separate username) — registration itself is a minimal, purpose-built form
# (see _register_user below), not streamlit-authenticator's own register_user()
# widget, which insists on a separate username and a repeat-password field this
# app doesn't want. Bot sign-ups are deterred by the optional shared
# REGISTRATION_CODE gate rather than a captcha.


# Free / consumer email providers rejected at registration: this app is meant
# for institutional (university, research lab, ...) accounts, so we blocklist
# the common public webmail domains rather than trying to maintain an
# impossible allowlist of every institution's domain. A blocklist is quick and
# good enough — a determined user with their own domain still gets through,
# which is fine, the goal is just to steer people to their work address.
_BLOCKED_EMAIL_DOMAINS = {
    "gmail.com", "googlemail.com",
    "outlook.com", "outlook.fr", "hotmail.com", "hotmail.fr", "live.com", "live.fr", "msn.com",
    "yahoo.com", "yahoo.fr", "ymail.com",
    "icloud.com", "me.com", "mac.com",
    "aol.com",
    "proton.me", "protonmail.com",
    "gmx.com", "gmx.fr", "gmx.de",
    "orange.fr", "wanadoo.fr", "free.fr", "sfr.fr", "laposte.net", "bbox.fr",
    "yandex.com", "yandex.ru", "mail.com", "zoho.com",
}

# At least 12 characters, with a lowercase letter, an uppercase letter, a digit
# and one non-alphanumeric (special) character.
_PASSWORD_HELP = (
    "At least 12 characters, including 1 lowercase letter, 1 uppercase letter, "
    "1 digit and 1 special character (e.g. ! ? @ # …)."
)


# Each rule: (predicate, error message for _password_problem, short label for
# the live checklist shown while the user types — see _password_checklist).
_PASSWORD_RULES = [
    (lambda p: len(p) >= 12,
     "Password must be at least 12 characters long.",
     "At least 12 characters"),
    (lambda p: re.search(r"[a-z]", p) is not None,
     "Password must contain at least 1 lowercase letter.",
     "1 lowercase letter"),
    (lambda p: re.search(r"[A-Z]", p) is not None,
     "Password must contain at least 1 uppercase letter.",
     "1 uppercase letter"),
    (lambda p: re.search(r"[0-9]", p) is not None,
     "Password must contain at least 1 digit.",
     "1 digit"),
    (lambda p: re.search(r"[^A-Za-z0-9]", p) is not None,
     "Password must contain at least 1 special character (e.g. ! ? @ # …).",
     "1 special character (e.g. ! ? @ # …)"),
]


def _password_problem(password: str) -> str | None:
    """Return a human-readable reason the password is unacceptable, or None if OK."""
    for check, message, _label in _PASSWORD_RULES:
        if not check(password):
            return message
    return None


def _password_checklist(password: str) -> list[tuple[str, bool]]:
    """Per-rule (label, is_satisfied) pairs, for live feedback while the user
    types — see _password_problem for the same rules used as a hard gate."""
    return [(label, check(password)) for check, _msg, label in _PASSWORD_RULES]


# Session flag that keeps the "Create an account" expander open across the
# submit rerun, so validation errors/success (rendered inside it) stay visible
# instead of vanishing when the expander collapses back to its default.
_REGISTER_OPEN_KEY = "_register_open"


def _keep_register_open() -> None:
    st.session_state[_REGISTER_OPEN_KEY] = True


def _register_user(conn) -> None:
    """
    Minimal registration form: first name, last name, institutional email
    (used as the login username — free webmail domains are rejected, see
    _BLOCKED_EMAIL_DOMAINS), password (min 12 chars, mixed case + digit +
    1 special character),
    and — when a REGISTRATION_CODE secret is configured — a shared invite
    code (the main anti-bot gate). No separate username, no repeat-password.

    Writes the new account straight into the users table (db.create_user) with
    a bcrypt-hashed password. authenticate() rebuilds its credentials dict from
    the DB on the next rerun, so a login right after registering sees it.
    """
    # Shared registration code (invite gate): when REGISTRATION_CODE is set in
    # the secrets/.env.app, a matching code is required to create an account,
    # so only people who were given it can register. When it's unset, the gate
    # is simply off and registration stays open. The expected value is never
    # in the source — always read from the secret.
    expected_code = _get_secret("REGISTRATION_CODE", "").strip()

    st.subheader("Create an account")

    # The password field deliberately lives OUTSIDE any st.form: widgets
    # inside a form only trigger a rerun (and thus only refresh whatever
    # depends on them) when the form is submitted — so a live "requirements
    # met" checklist below it would just sit frozen until the user clicks
    # Register. Left as a plain widget, Streamlit reruns on every blur/Enter,
    # updating the checklist as the user types instead of only after a failed
    # submit. The rest of the fields stay in a form purely for the Enter-to-
    # submit convenience, which doesn't depend on live feedback the same way.
    password = st.text_input("Password", type="password", key="register_password", help=_PASSWORD_HELP)
    if password:
        for label, ok in _password_checklist(password):
            st.caption(f"{'✅' if ok else '❌'} {label}")
    else:
        st.caption(f"🔒 {_PASSWORD_HELP}")

    with st.form("register_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        first_name = col1.text_input("First name")
        last_name = col2.text_input("Last name")
        email = st.text_input(
            "Institutional email",
            help="Use your university / research institution address — free webmail "
                 "(Gmail, Outlook, Yahoo, …) is not accepted.",
        )
        entered_code = ""
        if expected_code:
            entered_code = st.text_input(
                "Registration code", type="password",
                help="The invite code shared by the Virome@t team.",
            )
        # on_click fires before the rerun's script runs, so the expander is
        # already open (see authenticate()) by the time the error/success
        # message below renders inside it — otherwise the expander would
        # collapse back to its default on submit and hide the message.
        submitted = st.form_submit_button("Register", on_click=_keep_register_open)

    if not submitted:
        return

    email = email.strip().lower()
    required = [first_name, last_name, email, password]
    if expected_code:
        required.append(entered_code)
    if not all(required):
        st.error("All fields are required.")
        return
    if expected_code and entered_code.strip() != expected_code:
        st.error("Invalid registration code — ask the Virome@t team for the current one.")
        return
    if "@" not in email or "." not in email.split("@")[-1]:
        st.error("Please enter a valid email address.")
        return

    domain = email.split("@")[-1]
    if domain in _BLOCKED_EMAIL_DOMAINS:
        st.error(
            "Please register with an institutional email address "
            "(university or research lab) — free webmail providers are not accepted."
        )
        return

    pwd_problem = _password_problem(password)
    if pwd_problem:
        st.error(pwd_problem)
        return

    if db.get_user(conn, email) is not None:
        st.error("An account with this email already exists.")
        return

    # Grant admin to emails listed in the ADMIN_EMAILS secret; everyone else is
    # a regular user. Same rule the legacy-data migration applies (see db.py).
    role = "admin" if email in _admin_emails() else "user"
    db.create_user(conn, email, first_name, last_name, stauth.Hasher.hash(password), role)
    st.success(f"Account created for {email} — you can log in above now.")
    logger.info(f"AUTH | New account registered | email={email}")
    # Registration done — let the expander collapse again on the next rerun.
    st.session_state[_REGISTER_OPEN_KEY] = False
    # Password lives outside register_form (see above), so clear_on_submit
    # doesn't reach it — clear it explicitly. This run already rendered the
    # field with its old value before we got here, so it visually empties on
    # the next rerun, same one-run lag as the checklist itself.
    st.session_state.pop("register_password", None)


def authenticate(conn) -> tuple[str, str, str, "stauth.Authenticate"]:
    """
    Render the login / account-creation UI and block the rest of the app
    (st.stop()) until the user is authenticated — every user of this app
    needs an account, there is no guest mode.

    Credentials come from the SQLite users table (db.build_credentials_dict)
    and the cookie config from app_meta (db.get_cookie_config); auto_hash is
    off because passwords are always stored already bcrypt-hashed.

    Returns (username, display_name, role, authenticator) for the logged-in
    user — username is the user's email address. The authenticator is handed
    back so the caller can place the logout button wherever it belongs in its
    own layout.
    """
    credentials = db.build_credentials_dict(conn)
    cookie_name, cookie_key, cookie_expiry = db.get_cookie_config(conn)
    authenticator = stauth.Authenticate(
        credentials, cookie_name, cookie_key, cookie_expiry, auto_hash=False
    )

    if not st.session_state.get("authentication_status"):
        # A brand-new browser session doesn't know yet whether a valid
        # re-auth cookie exists — resolve that silently (location=
        # "unrendered": no title, no form) before deciding whether to show
        # the login screen at all. Already-authenticated sessions (the
        # common case, every rerun after the first) skip this entirely.
        authenticator.login(location="unrendered")

    auth_status = st.session_state.get("authentication_status")

    if not auth_status:
        st.title("Access Required")

        try:
            authenticator.login(fields={"Username": "Email"})
        except stauth.LoginError as e:
            st.error(str(e))

        auth_status = st.session_state.get("authentication_status")

    if not auth_status:
        if auth_status is False:
            st.error("Incorrect email or password")
            logger.warning("AUTH | Failed login attempt")

        with st.expander("Create an account", expanded=st.session_state.get(_REGISTER_OPEN_KEY, False)):
            _register_user(conn)

        st.stop()

    username = st.session_state["username"]
    display_name = st.session_state["name"]

    # Stamp last_login once per browser session (not on every rerun).
    if not st.session_state.get("_last_login_recorded"):
        db.set_last_login(conn, username)
        st.session_state["_last_login_recorded"] = True

    role = (db.get_user(conn, username) or {}).get("role", "user")
    return username, display_name, role, authenticator


# ==================== CONVERSATION CONTEXT WINDOW ====================

def build_context_window(messages: list, max_turns: int) -> list:
    """
    From a conversation's full on-screen history, build the bounded message
    list replayed to Albert: strip tool bookkeeping (via _clean_history_messages)
    then keep only the last `max_turns` user/assistant exchanges.

    This replaces the old hard reset (which wiped all memory at once when the
    turn limit was hit): older turns now simply fall out of the window while
    the complete history stays visible on screen and stored in the database —
    the same "unbounded scrollback, bounded prompt" behaviour as ChatGPT. Keeps
    Albert's token usage bounded regardless of how long the conversation grows.
    """
    cleaned = _clean_history_messages(
        [{"role": m["role"], "content": m.get("content", "")} for m in messages]
    )
    if max_turns > 0:
        # Each turn is a user question + its assistant answer → 2 messages.
        cleaned = cleaned[-(max_turns * 2):]
    return cleaned


# ==================== ALBERT API HELPERS ====================

def _get_api_key() -> str:
    """
    Retrieve Albert API key with priority:
    1. st.secrets["ALBERT_API_KEY"] (Streamlit Cloud deployment)
    2. Environment variable ALBERT_API_KEY (populated locally from .env.app)
    """
    key = _get_secret("ALBERT_API_KEY")
    if not key:
        st.error(
            "Albert API key not found. "
            "Add ALBERT_API_KEY to your .env.app (local) or Streamlit secrets (deployment)."
        )
        st.stop()
    return key


def _albert_headers(api_key: str) -> dict:
    """Build authorization headers for Albert API requests."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _list_albert_models(api_key: str) -> list:
    """
    Fetch available text-generation models from Albert API.
    Filters out embedding, audio, and reranking models.
    Falls back to ALBERT_MODEL_DEFAULT if the API call fails.
    """
    try:
        r = requests.get(
            f"{ALBERT_BASE_URL}/models",
            headers=_albert_headers(api_key),
            timeout=15,
        )
        r.raise_for_status()
        data = r.json().get("data", [])

        excluded_keywords = ("embed", "whisper", "rerank")
        names = [
            m["id"] for m in data
            if m.get("object") == "model"
            and not any(kw in m["id"].lower() for kw in excluded_keywords)
        ]
        return names if names else [ALBERT_MODEL_DEFAULT]

    except Exception as e:
        logger.warning(f"MODEL_LIST_FAIL | {e} — using default model")
        return [ALBERT_MODEL_DEFAULT]


def _transcribe_audio(audio_bytes: bytes, api_key: str) -> str:
    """
    Transcribe a recorded question to text via Albert API's Whisper endpoint
    (OpenAI-compatible `/audio/transcriptions`). Returns "" on failure so the
    caller can fall back to asking the user to type instead.
    """
    try:
        r = requests.post(
            f"{ALBERT_BASE_URL}/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("recording.wav", audio_bytes, "audio/wav")},
            data={"model": ALBERT_WHISPER_MODEL},
            timeout=ALBERT_TIMEOUT,
        )
        r.raise_for_status()
        text = (r.json().get("text") or "").strip()
        if not text:
            logger.warning("WHISPER_EMPTY | transcription returned empty text")
        return text
    except Exception as e:
        logger.error(f"WHISPER_FAIL | {e}")
        return ""


def _parse_tool_arguments(raw_args) -> dict:
    """
    Parse tool call arguments from Albert API response.

    Handles multiple formats that Albert/vLLM may return:
    - dict (ideal case)
    - JSON string (most common)
    - Malformed/partial JSON (known gpt-oss-120b bug)

    Falls back to {"_raw": raw_args} if all parsing fails.
    """
    if isinstance(raw_args, dict):
        return raw_args

    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            pass

        recovered = {}
        for m in re.finditer(
            r'"(\w+)"\s*:\s*("(?:[^"\\]|\\.)*"|\d+(?:\.\d+)?|true|false|null)',
            raw_args,
        ):
            try:
                recovered[m.group(1)] = json.loads(m.group(2))
            except Exception:
                recovered[m.group(1)] = m.group(2)

        if recovered:
            logger.warning(f"TOOL_ARG_PARTIAL_PARSE | recovered={recovered}")
            return recovered

        logger.error(f"TOOL_ARG_PARSE_FAIL | raw={raw_args[:300]}")
        return {"_raw": raw_args}

    return {}


class AlbertRateLimitError(RuntimeError):
    """Raised when Albert keeps returning HTTP 429 after all retries are
    exhausted — the (free) API is rate-limiting/saturating requests. Kept
    distinct from a genuine connectivity or server error so the agent loop can
    show a clearer, actionable message."""


def _albert_chat(
    messages: list,
    tools: list,
    model: str,
    api_key: str,
    temperature,
    top_p,
    presence_penalty=0,
    frequency_penalty=0,
    seed=42,
    max_completion_tokens=4096,
    parallel_tool_calls=False,
    retry: int = ALBERT_MAX_RETRIES,
) -> dict:
    """
    Send a chat completion request to Albert API with retry logic.

    Retries up to `retry` times on HTTP 429 (rate limiting). Honors the
    server's Retry-After header when present, otherwise backs off exponentially
    (2**attempt seconds, capped at ALBERT_RETRY_BACKOFF_CAP). Raises
    AlbertRateLimitError if every attempt is rate-limited — note this cannot
    lift a hard quota, it only rides out transient throttling.
    """
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "seed": seed,
        "max_completion_tokens": max_completion_tokens,
        "parallel_tool_calls": parallel_tool_calls,
        "stream": False,
    }

    for attempt in range(1, retry + 1):
        try:
            r = requests.post(
                f"{ALBERT_BASE_URL}/chat/completions",
                headers=_albert_headers(api_key),
                json=payload,
                timeout=ALBERT_TIMEOUT,
            )

            if r.status_code == 429:
                # Prefer the server's own Retry-After (seconds) when it sends
                # one; otherwise exponential backoff, both capped so we never
                # sleep absurdly long.
                retry_after = (r.headers.get("Retry-After") or "").strip()
                if retry_after.isdigit():
                    wait = min(int(retry_after), ALBERT_RETRY_BACKOFF_CAP)
                else:
                    wait = min(2 ** attempt, ALBERT_RETRY_BACKOFF_CAP)
                logger.warning(f"RATE_LIMIT | attempt {attempt}/{retry} — waiting {wait}s")
                if attempt < retry:  # no point sleeping after the last attempt
                    time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json()

        except requests.exceptions.Timeout:
            logger.error(f"ALBERT_TIMEOUT | attempt {attempt}")
            if attempt == retry:
                raise
        except requests.exceptions.RequestException as e:
            logger.error(f"ALBERT_HTTP_ERROR | {e}")
            raise

    raise AlbertRateLimitError(
        f"Albert rate-limited the request after {retry} attempts (HTTP 429)."
    )


# ==================== ERROR REPORTING ====================

def save_error_report(question: str, answer: str, executed_codes: list, comment: str = ""):
    """
    Save an error report with context for debugging.

    Includes:
    - User's question and model's answer
    - Executed code snippets
    - User's comment about what went wrong
    - Recent log entries for context
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.now()
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"report_{ts_str}.json")

    log_filename = os.path.join(LOG_DIR, f"agent_{timestamp.strftime('%Y-%m')}.log")
    related_logs = []
    if os.path.exists(log_filename):
        try:
            with open(log_filename, "r", encoding="utf-8") as f:
                related_logs = [line.rstrip() for line in f.readlines()[-200:]]
        except Exception:
            related_logs = ["[Could not read log file]"]

    report = {
        "timestamp": timestamp.isoformat(),
        "user_comment": comment,
        "question": question,
        "answer": answer,
        "executed_codes": executed_codes,
        "recent_logs": related_logs,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.warning(f"ERROR_REPORT | Saved to {report_path} | comment={comment!r}")
    return report_path


# ==================== UI COMPONENTS ====================

def render_sources(wikipedia_urls: list, pubmed_urls: list, ncbi_urls: list, executed_codes: list):
    """Display expandable sources section with Wikipedia, PubMed, NCBI Taxonomy, and code references."""
    if not wikipedia_urls and not pubmed_urls and not ncbi_urls and not executed_codes:
        return

    with st.expander("📚 Sources"):
        if wikipedia_urls:
            st.markdown("**📘 Wikipedia**")
            for url in wikipedia_urls:
                title = url.split("/")[-1].replace("_", " ")
                st.markdown(f"- [{title}]({url})")

        if pubmed_urls:
            st.markdown("**🔬 PubMed**")
            for url in pubmed_urls:
                pmid = url.split("/")[-2] if url.endswith("/") else url.split("/")[-1]
                st.markdown(f"- [PMID: {pmid}]({url})")

        if ncbi_urls:
            st.markdown("**🧬 NCBI Taxonomy**")
            for url in ncbi_urls:
                tax_id = url.split("id=")[-1]
                st.markdown(f"- [TaxID: {tax_id}]({url})")

        if executed_codes:
            st.markdown("**📊 Dataset Query & Visualization**")
            full_code = "\n\n---\n\n".join(
                f"# Code {i}\n{code}" for i, code in enumerate(executed_codes, 1)
            )
            st.code(full_code, language="python")


def render_report_button(msg_idx: int, question: str, answer: str, executed_codes: list):
    """
    Render an error reporting button with dialog for user feedback.

    Tracks whether a report has already been submitted for this message
    to prevent duplicate reports.
    """
    report_key = f"reported_{msg_idx}"
    dialog_key = f"show_dialog_{msg_idx}"

    if st.session_state.get(report_key):
        st.caption("⚠️ Error reported — thank you for your feedback.")
        return

    if st.button("🚩 Report an error", key=f"btn_report_{msg_idx}",
                 help="Signal a wrong or misleading answer"):
        st.session_state[dialog_key] = not st.session_state.get(dialog_key, False)

    if st.session_state.get(dialog_key):
        with st.container(border=True):
            st.markdown("**What went wrong?** *(optional)*")
            comment = st.text_area(
                "Your comment", key=f"comment_{msg_idx}",
                placeholder="e.g. Wrong species name, incorrect count, hallucinated data…",
                label_visibility="collapsed",
            )

            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("Send report", key=f"send_report_{msg_idx}", type="primary"):
                    save_error_report(question, answer, executed_codes, comment)
                    st.session_state[report_key] = True
                    st.session_state[dialog_key] = False
                    st.success("Report saved ✅")
                    st.rerun()
            with col2:
                if st.button("Cancel", key=f"cancel_report_{msg_idx}"):
                    st.session_state[dialog_key] = False
                    st.rerun()


# ==================== AGENT LOOP (Core Logic) ====================

def _clean_history_messages(messages: list) -> list:
    """
    Strip tool-call bookkeeping (assistant tool_calls + tool results, and
    any injected system reminders) from a turn's messages before storing it
    as conversation history. Only the user question and the model's final
    text answer carry over to the next turn — otherwise every tool call and
    tool result from every past turn gets replayed to Albert on each new
    question, and the context blows past the token limit.
    """
    return [
        m for m in messages
        if m["role"] == "user"
        or (m["role"] == "assistant" and not m.get("tool_calls"))
    ]


async def albert_agent_loop(
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
    status_container=None,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
    seed: int = DEFAULT_SEED,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    parallel_tool_calls: bool = DEFAULT_PARALLEL_TOOL_CALLS,
    history_messages: list | None = None,
):
    """
    Main agentic loop: iteratively calls Albert API with tool-calling capability.
    Tools are executed on the remote FastMCP server — this function never
    touches the DataFrames, it only talks MCP over one shared connection
    (opened once for the whole loop via `mcp_session()`).

    history_messages, if given, is the full message history (user/assistant/
    tool messages — everything except the system prompt) from previous
    turns in this conversation, replayed before the new user_query so the
    model has full context of earlier questions, its own tool calls, and
    their results.

    username (the user's email) is only used to tag the USER_QUERY log line
    so every question can be traced back to who asked it — this log line is
    written to disk unconditionally, independent of the user's own session
    history, so it survives even after that user clears their history (see
    the "Clear my history" button in main()).

    The loop continues until:
    - The model returns a final answer (no tool calls)
    - Maximum tool call limit is reached
    - An error occurs

    Returns:
        Tuple of (answer_text, figures, wikipedia_urls, pubmed_urls, ncbi_urls,
        executed_codes, updated_history). updated_history is the full message
        history (system prompt excluded) including this turn, ready to pass
        as history_messages on the next call — or None if the call failed,
        in which case the caller should keep the previous history unchanged.
    """

    used_wikipedia_urls = []
    used_pubmed_urls = []
    used_ncbi_urls = []
    executed_codes = []
    generated_figures = []
    real_pmids = set()

    tool_call_count = 0

    def _update_status(icon, label, ok=None, detail=""):
        if status_container is None:
            return
        prefix = {None: "⏳", True: "✅", False: "❌"}[ok]
        suffix = f" — {detail}" if detail else ""
        status_container.write(f"{prefix} {icon} {label}{'…' if ok is None else ''}{suffix}")

    logger.info("=" * 50)
    logger.info(f"USER_QUERY | user={username or 'unknown'} | {user_query}")
    logger.info(
        f"CONFIG | model={model} temp={temperature} top_p={top_p} "
        f"max_calls={max_tool_calls} preview={preview_rows}rows "
        f"max_tool_content={max_tool_content}"
    )

    _update_status("🧠", "Thinking")

    # ── Open ONE MCP connection for the whole loop ───────────────────────────
    async with Client(MCP_SERVER_URL) as mcp:
        tools = await mcp.list_tools()
        tools_spec = _mcp_tools_to_openai_spec(tools)
        # Parameter schema per tool — used to generically decide which UI
        # settings (preview_rows, wikipedia_limit, ...) apply to a given call,
        # instead of hardcoding a per-tool-name argument mapping here.
        tool_schemas = {t.name: (t.inputSchema or {}) for t in tools}

        # Dataset schemas are published as MCP resources (not tools) — read
        # once per conversation and baked into the system prompt so the model
        # always knows the exact column names without a dedicated tool call.
        datasets_description = await _describe_available_datasets(mcp)
        system_prompt = build_system_prompt(datasets_description)

        messages = [{"role": "system", "content": system_prompt}]
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": user_query})

        while True:
            logger.info(
                f"LLM_CALL | sending {len(messages)} messages | "
                f"roles={[m['role'] for m in messages]}"
            )

            try:
                resp = _albert_chat(
                    messages=messages,
                    tools=tools_spec,
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    seed=seed,
                    max_completion_tokens=max_completion_tokens,
                    parallel_tool_calls=parallel_tool_calls,
                )
            except requests.exceptions.Timeout:
                logger.error("ALBERT_TIMEOUT")
                return (
                    "The model took too long to respond (>180 s). Please try again.",
                    generated_figures, [], [], [], executed_codes, None,
                )
            except AlbertRateLimitError:
                logger.error("ALBERT_RATE_LIMITED")
                return (
                    "⏳ Albert is rate-limiting requests right now — the free API is "
                    "saturated, especially on the large models. Please wait a moment "
                    "and try again, or pick a lighter model in ⚙️ Expert mode.",
                    generated_figures, [], [], [], executed_codes, None,
                )
            except Exception as e:
                logger.error(f"ALBERT_ERROR | {e}")
                return (
                    f"Could not reach Albert API: {e}",
                    generated_figures, [], [], [], executed_codes, None,
                )

            choice = resp["choices"][0]
            msg = choice["message"]
            finish = choice.get("finish_reason", "")

            logger.info(
                f"LLM_RESPONSE | finish_reason={finish!r} | "
                f"has_tool_calls={bool(msg.get('tool_calls'))} | "
                f"content_len={len(msg.get('content') or '')}"
            )

            # ── CASE 1: Final answer (no tool calls) ─────────────────────────
            if finish != "tool_calls" or not msg.get("tool_calls"):
                final_text = (msg.get("content") or "").strip()

                if not final_text and tool_call_count > 0 and tool_call_count < min(3, max_tool_calls):
                    logger.warning(
                        f"EARLY_STOP | finish=stop, content empty, only {tool_call_count} tool calls — "
                        f"injecting continuation prompt"
                    )
                    messages.append(msg)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You have only searched {tool_call_count} time(s) so far and haven't "
                            f"found enough information yet. Please continue your research: "
                            f"search Wikipedia and PubMed for viruses infecting the requested host, "
                            f"then provide a complete scientific answer."
                        ),
                    })
                    continue

                if not final_text and tool_call_count > 0:
                    logger.warning(
                        "EMPTY_FINAL_ANSWER | content blank after tool calls — "
                        "rebuilding clean context for synthesis (gpt-oss-120b workaround)"
                    )

                    tool_results_text = []
                    for m in messages:
                        if m.get("role") == "tool":
                            tool_name = m.get("name", "tool")
                            tool_results_text.append(
                                f"=== Result from {tool_name} ===\n{m.get('content', '')}"
                            )

                    context_block = "\n\n".join(tool_results_text)

                    error_count = sum(
                        1 for r in tool_results_text
                        if r.startswith("=== Result") and "Error:" in r
                    )
                    success_count = len(tool_results_text) - error_count

                    if success_count == 0 and error_count > 0:
                        logger.warning("SYNTHESIS_FALLBACK | all tool results are errors")
                        context_note = (
                            "Note: the dataset did not contain data for this query "
                            "(search returned no results). Answer from scientific knowledge only."
                        )
                    else:
                        context_note = f"Here is all the information gathered:\n\n{context_block}"

                    clean_messages = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"Original question: {user_query}\n\n"
                                f"{context_note}\n\n"
                                f"Write a detailed, well-structured scientific answer. "
                                f"Requirements:\n"
                                f"- Cover each relevant virus with its taxonomy (family, genus, species), "
                                f"transmission route, pathogenesis, and key clinical signs\n"
                                f"- Use proper scientific nomenclature\n"
                                f"- Structure your answer with one section per virus\n"
                                f"- Be thorough and complete — do not summarize\n"
                                f"- Do not mention tools, datasets, or data retrieval\n"
                                f"- CRITICAL: do NOT invent or mention any PMID. "
                                f"Only reference PMIDs that appear verbatim in the context above. "
                                f"If no PMIDs are in the context, write none."
                            ),
                        },
                    ]

                    try:
                        retry_resp = _albert_chat(
                            messages=clean_messages,
                            tools=[],
                            model=model,
                            api_key=api_key,
                            temperature=temperature,
                            top_p=top_p,
                            max_completion_tokens=6144,
                        )
                        final_text = (
                            retry_resp["choices"][0]["message"].get("content") or ""
                        ).strip()

                        final_text, stripped = _strip_hallucinated_pmids(final_text, real_pmids)
                        if stripped:
                            logger.warning(
                                f"PMID_HALLUCINATION_SYNTHESIS | stripped {len(stripped)} fake PMID(s): {stripped}"
                            )
                        final_text, n_fake = _strip_fake_citation_markers(final_text)
                        if n_fake:
                            logger.warning(f"FAKE_CITATION_SYNTHESIS | stripped {n_fake} bracket marker(s)")

                        logger.info(f"SYNTHESIS_OK | content_len={len(final_text)}")
                        if not final_text:
                            logger.error("SYNTHESIS_FAIL | still empty after clean context")
                            final_text = (
                                "⚠️ The model retrieved information but failed to generate "
                                "a final answer. Please try rephrasing your question."
                            )
                    except Exception as e:
                        logger.error(f"SYNTHESIS_RETRY_FAIL | {e}")
                        final_text = "The model did not produce a final answer. Please rephrase your question."

                final_text, stripped = _strip_hallucinated_pmids(final_text, real_pmids)
                if stripped:
                    logger.warning(
                        f"PMID_HALLUCINATION | stripped {len(stripped)} fake PMID(s): {stripped}"
                    )
                final_text, n_fake = _strip_fake_citation_markers(final_text)
                if n_fake:
                    logger.warning(f"FAKE_CITATION | stripped {n_fake} bracket marker(s)")

                logger.info(
                    f"RESULT | len={len(final_text)} | "
                    f"{final_text[:500]}{'...' if len(final_text) > 500 else ''}"
                )
                messages.append(msg)

                return (
                    final_text,
                    generated_figures,
                    used_wikipedia_urls,
                    used_pubmed_urls,
                    used_ncbi_urls,
                    executed_codes,
                    _clean_history_messages(messages[1:]),  # no system prompt, no tool traffic
                )

            # ── CASE 2: Tool calls ───────────────────────────────────────────
            messages.append(msg)

            if tool_call_count >= max_tool_calls:
                logger.warning(f"MAX_TOOL_CALLS | Limit reached ({max_tool_calls})")
                messages.append({
                    "role": "system",
                    "content": (
                        f"Tool call limit reached ({max_tool_calls}). "
                        f"Synthesize a final answer to: '{user_query}'"
                    ),
                })
                continue

            for call in msg["tool_calls"]:
                tool_call_count += 1
                name = call["function"]["name"]
                raw_args = call["function"].get("arguments", {})
                args = _parse_tool_arguments(raw_args)

                logger.info(
                    f"TOOL_CALL #{tool_call_count}/{max_tool_calls} | {name} | "
                    f"call_id={call.get('id','?')} | "
                    f"parsed_args={json.dumps(args, ensure_ascii=False)[:400]}"
                )

                icon, label = TOOL_LABELS.get(name, ("🔧", name))

                # ── Build call args generically from the tool's own schema —
                #    no per-tool-name special-casing. ─────────────────────────
                schema = tool_schemas.get(name) or {}
                props = schema.get("properties", {})
                required = schema.get("required", [])

                call_args = dict(args)
                # Recover from malformed-JSON args (only "_raw" survived) by
                # mapping it onto the tool's sole required parameter, if any.
                if set(call_args.keys()) == {"_raw"} and len(required) == 1:
                    call_args[required[0]] = call_args.pop("_raw")

                # Only apply a UI-configured default for parameters the tool
                # actually declares — this app never assumes which tools exist.
                if "preview_rows" in props:
                    call_args.setdefault("preview_rows", preview_rows)
                if "wikipedia_limit" in props:
                    call_args.setdefault("wikipedia_limit", wikipedia_limit)

                # Drop any argument the tool doesn't declare — the model
                # sometimes carries a parameter over from an earlier call in
                # the same turn (e.g. adding preview_rows to a tool that has
                # no such parameter), which the MCP server rejects outright.
                if props:
                    dropped = set(call_args) - set(props)
                    if dropped:
                        logger.warning(f"TOOL_ARG_DROPPED | {name} | unsupported keys: {sorted(dropped)}")
                        call_args = {k: v for k, v in call_args.items() if k in props}

                output = _unwrap_mcp_result(await mcp.call_tool(name, call_args))
                content = output.get("content", "Unknown error")

                if output.get("success"):
                    for artifact in output.get("artifacts", []):
                        a_type = artifact.get("type")
                        if a_type == "url":
                            if artifact["url"] not in used_wikipedia_urls:
                                used_wikipedia_urls.append(artifact["url"])
                        elif a_type == "pubmed":
                            for pmid in artifact.get("pmids", []):
                                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                                if pubmed_url not in used_pubmed_urls:
                                    used_pubmed_urls.append(pubmed_url)
                                real_pmids.add(str(pmid))
                        elif a_type == "ncbi_taxonomy":
                            if artifact["url"] not in used_ncbi_urls:
                                used_ncbi_urls.append(artifact["url"])
                        elif a_type == "plotly":
                            generated_figures.append(pio.from_json(json.dumps(artifact["figure"])))

                    if "code" in call_args:
                        executed_codes.append(call_args["code"])
                    elif "sql" in call_args:
                        executed_codes.append(f"-- SQL ({name})\n{call_args['sql']}")

                    logger.info(f"TOOL_OK | {name} | {_snippet(content, 500)}")
                else:
                    logger.warning(f"TOOL_FAIL | {name} | {content}")

                # UI stays minimal: just the search keyword for
                # wikipedia/pubmed/ncbi lookups, nothing for dataset/map
                # tools (sql/code) — full detail goes to the logs above.
                _update_status(
                    icon, label, ok=output.get("success"), detail=_ui_search_keyword(call_args)
                )

                # ── Truncate oversized tool output ───────────────────────────
                if len(content) > max_tool_content:
                    original_len = len(content)
                    content = (
                        content[:max_tool_content]
                        + f"\n\n[...truncated — {original_len - max_tool_content} chars omitted]"
                    )
                    logger.warning(
                        f"TOOL_CONTENT_TRUNCATED | {name} | "
                        f"trimmed from {original_len} to {max_tool_content} chars"
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": content,
                })

            logger.info(
                f"LOOP_STATE | messages_in_history={len(messages)} | "
                f"tool_calls_used={tool_call_count}/{max_tool_calls} | "
                f"figures={len(generated_figures)} | codes={len(executed_codes)}"
            )

async def check_mcp_connection() -> bool:
    """Vérifie si le serveur MCP est accessible en listant les outils."""
    try:
        async with Client(MCP_SERVER_URL) as client:
            await client.list_tools()
            return True
    except Exception as e:
        logger.error(f"MCP_CONNECTION_FAIL | {e}")
        return False
    
# ==================== CONVERSATION SIDEBAR ====================

def _open_conversation(conn, conversation_id: int) -> None:
    """Switch the active conversation: load its stored messages into the
    session so the chat area and the context window reflect that thread."""
    st.session_state.current_conversation_id = conversation_id
    st.session_state.messages = db.list_messages(conn, conversation_id)


def _render_conversation_sidebar(conn, username: str) -> None:
    """ChatGPT-style conversation panel: start a new thread, switch between
    past ones, rename or delete each. The active thread is highlighted."""
    st.subheader("💬 Conversations")

    if st.button("➕ New conversation", use_container_width=True, key="new_conv_btn"):
        st.session_state.current_conversation_id = None
        st.session_state.messages = []
        st.rerun()

    conversations = db.list_conversations(conn, username)
    current = st.session_state.get("current_conversation_id")

    if not conversations:
        st.caption("No conversation yet — ask a question to start one.")

    for c in conversations:
        is_active = (c["id"] == current)
        title = c["title"] or "Untitled"
        row = st.columns([5, 1])
        if row[0].button(
            ("🟢 " if is_active else "") + title,
            key=f"open_conv_{c['id']}",
            help=title,
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            _open_conversation(conn, c["id"])
            st.rerun()
        with row[1].popover("⋯", use_container_width=True):
            new_title = st.text_input(
                "Rename", value=c["title"] or "", key=f"rename_input_{c['id']}"
            )
            if st.button("💾 Save", key=f"rename_save_{c['id']}", use_container_width=True):
                db.rename_conversation(conn, c["id"], new_title.strip() or title)
                st.rerun()
            if st.button(
                "🗑 Delete", key=f"del_conv_{c['id']}",
                type="primary", use_container_width=True,
            ):
                db.delete_conversation(conn, c["id"])
                if current == c["id"]:
                    st.session_state.current_conversation_id = None
                    st.session_state.messages = []
                logger.info(f"CONVERSATION_DELETED | user={username} | id={c['id']}")
                st.rerun()


# ==================== CHAT PAGE ====================

def chat_page(conn, username: str, display_name: str) -> None:
    """The main chat interface — one page of the multipage app (see main())."""
    api_key = _get_api_key()

    st.title(f"Welcome {display_name}!")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; max-width: 1400px; }
        .stChatMessage {
            border-radius: 14px; padding: 0.75rem;
            cursor: pointer; transition: background-color 0.2s;
        }
        .stChatMessage:hover { background-color: rgba(100,126,234,0.05); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.caption("""
        For best results:
        - Use English and provide as much relevant detail as you can
        - Ask precise and clearly formulated questions, avoid acronym or abbreviation
        - Stupid question leads to stupid answer ;)

        Example questions:
        - "Give me information about Orthopoxvirus. Is it a family or a genus? How many species does it include?"
        - "Show me a summary in piechart of the dataframe in term of viral family repartition"
        - "World repartition of poxviridae"
        - "Tell me more about Polyomavirus infection way"
    """)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        _render_conversation_sidebar(conn, username)

        st.divider()

        st.title("Viromech@t 🦠")
        st.caption("""
A chatbot to explore viral metagenomic data from the [Virome@tlas project](http://shape-med-lyon.fr/projets/structurants-vague-1/virometlas/)

**Datasets**
- **Taxonomy** : NCBI Taxonomy, enriched with genome assembly availability, SRA sequencing activity, and GBIF biodiversity observations
- **Virus-host occurrences** : SRA/GenBank/BioSample samples linked to host & virus taxonomy, geographic location, and disease status

**Tools**
- SQL & pandas queries over both datasets
- Interactive maps and charts
- Wikipedia & PubMed search for biological background
""")

        st.header("Settings")

        # ── MCP server reachability check ──────────────────────────────────────
        with st.spinner("Connecting to data server…"):
            mcp_ok = asyncio.run(check_mcp_connection())
            if not mcp_ok:
                st.error(
                    "Could not reach the MCP data server. "
                    "Check that server_mcp.py is running and MCP_SERVER_URL is correct."
                )
                st.stop()

        with st.spinner("Loading models…"):
            model_names = _list_albert_models(api_key)

        with st.expander("⚙️ Expert mode", expanded=False):
            default_idx = next(
                (i for i, m in enumerate(model_names)
                 if ALBERT_MODEL_DEFAULT.lower() in m.lower()),
                0,
            )
            model = st.selectbox(
                "LLM Model", options=model_names, index=default_idx,
                help="Text-generation models available on Albert API",
            )

            st.markdown("**Sampling**")

            temperature = st.slider("Temperature", 0.0, 2.0, DEFAULT_TEMPERATURE, 0.05)
            top_p = st.slider("Top-p", 0.0, 1.0, DEFAULT_TOP_P, 0.05)

            presence_penalty = st.slider(
                "Presence penalty", -2.0, 2.0, DEFAULT_PRESENCE_PENALTY, 0.1,
                help="Encourage le modèle à explorer de nouveaux sujets."
            )
            frequency_penalty = st.slider(
                "Frequency penalty", -2.0, 2.0, DEFAULT_FREQUENCY_PENALTY, 0.1,
                help="Réduit les répétitions."
            )

            seed = st.number_input("Seed", value=DEFAULT_SEED, step=1)
            max_completion_tokens = st.number_input(
                "Max completion tokens",
                min_value=512, max_value=32768,
                value=DEFAULT_MAX_COMPLETION_TOKENS, step=512
            )
            parallel_tool_calls = st.checkbox(
                "Parallel tool calls", value=DEFAULT_PARALLEL_TOOL_CALLS
            )

            st.markdown("**Agent**")
            max_tool_calls = st.slider(
                "Max tool calls", 1, 20, DEFAULT_MAX_TOOL_CALLS,
                help="Increase for complex multi-step questions"
            )
            max_context_turns = st.slider(
                "Max context turns", 1, 20, MAX_CONTEXT_TURNS,
                help="How many recent question/answer exchanges are sent to Albert as "
                     "context — a sliding window: older turns drop off the prompt but "
                     "stay on screen and in your saved history. Lower this if Albert "
                     "runs out of tokens."
            )
            preview_rows = st.slider("Preview rows", 5, 200, DEFAULT_PREVIEW_ROWS, 5)
            wikipedia_limit = st.slider(
                "Wiki limit (chars/article)", 500, 30000, DEFAULT_WIKIPEDIA_LIMIT, 500
            )
            max_tool_content = st.slider(
                "Max tool content (chars)", 2000, 30000, DEFAULT_MAX_TOOL_CONTENT, 1000
            )

        st.markdown("---")
        st.caption(f"🔗 GitHub: {GITHUB_URL}")

    # ── Conversation state — a fresh login starts on a new, empty thread; past
    #    conversations are one click away in the sidebar. Messages are loaded
    #    from the database when a thread is opened (see _open_conversation). ──
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
        st.session_state.messages = []

    # ── Display chat history ─────────────────────────────────────────────────
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if not msg.get("content"):
                continue
            st.markdown(msg["content"])

            for fig_idx, fig in enumerate(msg.get("figures", [])):
                st.plotly_chart(
                    fig,
                    key=f"fig_{msg_idx}_{fig_idx}",
                    config={"displayModeBar": True, "scrollZoom": True},
                )

            render_sources(
                msg.get("wikipedia_urls", []),
                msg.get("pubmed_urls", []),
                msg.get("ncbi_urls", []),
                msg.get("executed_codes", []),
            )

            if msg["role"] == "assistant":
                question = ""
                if msg_idx > 0 and st.session_state.messages[msg_idx - 1]["role"] == "user":
                    question = st.session_state.messages[msg_idx - 1].get("content", "")
                render_report_button(
                    msg_idx=msg_idx,
                    question=question,
                    answer=msg["content"],
                    executed_codes=msg.get("executed_codes", []),
                )

    # ── Query handling (shared by text input and mic input) ─────────────────
    def _handle_query(query: str):
        # Bound the prompt to the last `max_context_turns` exchanges — built
        # BEFORE the new question is appended (the agent loop appends it
        # itself). Older turns fall out of the window but stay on screen and in
        # the database, so the memory degrades gracefully instead of resetting.
        history_window = build_context_window(st.session_state.messages, max_context_turns)

        # Create the conversation lazily on its first message, so empty threads
        # never clutter the sidebar; title it from the opening question.
        if st.session_state.current_conversation_id is None:
            title = " ".join(query.strip().split())[:40] or "New conversation"
            st.session_state.current_conversation_id = db.create_conversation(
                conn, username, title
            )
        conversation_id = st.session_state.current_conversation_id

        st.session_state.messages.append({"role": "user", "content": query})
        db.add_message(conn, conversation_id, "user", query)
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            status_container = status_placeholder.status("Processing…", expanded=True)

            answer, figures, wikipedia_urls, pubmed_urls, ncbi_urls, executed_codes, _updated_history = asyncio.run(
                albert_agent_loop(
                    model=model,
                    api_key=api_key,
                    user_query=query,
                    username=username,
                    temperature=temperature,
                    top_p=top_p,
                    max_tool_calls=max_tool_calls,
                    preview_rows=preview_rows,
                    wikipedia_limit=wikipedia_limit,
                    max_tool_content=max_tool_content,
                    status_container=status_container,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    seed=seed,
                    max_completion_tokens=max_completion_tokens,
                    parallel_tool_calls=parallel_tool_calls,
                    history_messages=history_window,
                )
            )

            status_placeholder.empty()
            st.markdown(answer)

            new_msg_idx = len(st.session_state.messages)
            for fig_idx, fig in enumerate(figures):
                st.plotly_chart(
                    fig,
                    key=f"fig_{new_msg_idx}_{fig_idx}",
                    config={"displayModeBar": True, "scrollZoom": True},
                )

            render_sources(wikipedia_urls, pubmed_urls, ncbi_urls, executed_codes)
            render_report_button(
                msg_idx=new_msg_idx,
                question=query,
                answer=answer,
                executed_codes=executed_codes,
            )

        assistant_msg = {
            "role": "assistant",
            "content": answer,
            "figures": figures,
            "wikipedia_urls": wikipedia_urls,
            "pubmed_urls": pubmed_urls,
            "ncbi_urls": ncbi_urls,
            "executed_codes": executed_codes,
        }
        st.session_state.messages.append(assistant_msg)
        db.add_message(conn, conversation_id, "assistant", answer, assistant_msg)
        db.touch_conversation(conn, conversation_id)

        # Rerun so a just-created conversation (and its title) appears in the
        # sidebar immediately; the full exchange re-renders from session state.
        st.rerun()

    # ── Chat input (text + mic, native recording button next to send) ───────
    submission = st.chat_input("Ask about viruses...", accept_audio=True)
    if submission is not None:
        if submission.audio is not None:
            with st.spinner("Transcribing your question…"):
                transcribed = _transcribe_audio(submission.audio.getvalue(), api_key)
            if transcribed:
                _handle_query(transcribed)
            else:
                st.warning("Could not transcribe the recording — please try again or type your question.")
        elif submission.text:
            _handle_query(submission.text)

    # ── Footer disclaimer ────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "⚠️ **AI is not magic** — Results may contain errors and "
        "should be verified for scientific or medical use."
    )


# ==================== ACCOUNT PAGE (change password) ====================

def account_page(conn, username: str, display_name: str) -> None:
    """Let the signed-in user change their own password. The current password
    is verified against the stored bcrypt hash; the new one must satisfy the
    same rules as registration (_password_problem)."""
    st.title("👤 My account")
    st.caption(f"Signed in as **{display_name}** — {username}")
    st.divider()
    st.subheader("Change my password")

    # New password lives outside the form so the requirements checklist below
    # updates live as the user types (same rationale as _register_user).
    new_password = st.text_input(
        "New password", type="password", key="account_new_password", help=_PASSWORD_HELP
    )
    if new_password:
        for label, ok in _password_checklist(new_password):
            st.caption(f"{'✅' if ok else '❌'} {label}")
    else:
        st.caption(f"🔒 {_PASSWORD_HELP}")

    with st.form("change_password_form"):
        current_password = st.text_input("Current password", type="password")
        confirm_password = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Update password", type="primary")

    if not submitted:
        return

    user = db.get_user(conn, username)
    if not user or not bcrypt.checkpw(
        current_password.encode("utf-8"), user["password_hash"].encode("utf-8")
    ):
        st.error("Current password is incorrect.")
        return
    problem = _password_problem(new_password)
    if problem:
        st.error(problem)
        return
    if new_password != confirm_password:
        st.error("The new password and its confirmation do not match.")
        return
    if bcrypt.checkpw(new_password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        st.error("The new password must be different from the current one.")
        return

    db.update_password(conn, username, stauth.Hasher.hash(new_password))
    logger.info(f"AUTH | Password changed | email={username}")
    st.session_state.pop("account_new_password", None)
    st.success("Password updated. Use it the next time you sign in.")


# ==================== ADMIN PAGE (read-only) ====================

def admin_page(conn, role: str) -> None:
    """Read-only overview for admins: the user list (never passwords) and a
    browser to inspect any user's conversations. No destructive actions."""
    st.title("🛠 Admin")

    # Defence in depth: st.navigation already hides this page from non-admins,
    # but guard the render too in case someone reaches it by URL.
    if role != "admin":
        st.error("Administrators only.")
        st.stop()

    users = db.list_users_with_counts(conn)
    st.subheader(f"Users ({len(users)})")
    st.dataframe(
        [
            {
                "Email": u["email"],
                "First name": u["first_name"],
                "Last name": u["last_name"],
                "Role": u["role"],
                "Created": (u["created_at"] or "")[:19],
                "Last login": (u["last_login"] or "")[:19],
                "Conversations": u["n_conversations"],
            }
            for u in users
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Browse a user's conversations")

    emails = [u["email"] for u in users]
    if not emails:
        return
    selected = st.selectbox("User", options=emails, key="admin_user_select")

    conversations = db.list_conversations(conn, selected)
    if not conversations:
        st.info("This user has no conversation yet.")
        return

    labels = {
        f"{c['title'] or 'Untitled'}  ·  {(c['updated_at'] or '')[:16]}  (#{c['id']})": c["id"]
        for c in conversations
    }
    chosen = st.selectbox("Conversation", options=list(labels.keys()), key="admin_conv_select")
    conversation_id = labels[chosen]

    st.caption("Read-only view.")
    for m_idx, m in enumerate(db.list_messages(conn, conversation_id)):
        with st.chat_message(m["role"]):
            st.markdown(m.get("content") or "")
            for f_idx, fig in enumerate(m.get("figures", [])):
                st.plotly_chart(
                    fig,
                    key=f"admin_fig_{conversation_id}_{m_idx}_{f_idx}",
                    config={"displayModeBar": False},
                )
            render_sources(
                m.get("wikipedia_urls", []),
                m.get("pubmed_urls", []),
                m.get("ncbi_urls", []),
                m.get("executed_codes", []),
            )


# ==================== MAIN APPLICATION (router) ====================

def main():
    """Entry point: authenticate, then route between the chat, account and
    (for admins) admin pages via Streamlit's native multipage navigation.
    Role-gating is done by only listing the admin page for admins."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

    conn = db.init_db()
    username, display_name, role, authenticator = authenticate(conn)

    def _chat():
        chat_page(conn, username, display_name)

    def _account():
        account_page(conn, username, display_name)

    def _admin():
        admin_page(conn, role)

    pages = [
        st.Page(_chat, title="Chat", icon="💬", url_path="chat", default=True),
        st.Page(_account, title="My account", icon="👤", url_path="account"),
    ]
    if role == "admin":
        pages.append(st.Page(_admin, title="Admin", icon="🛠", url_path="admin"))

    nav = st.navigation(pages)

    # Signed-in user + logout, rendered on every page (below the page switcher).
    with st.sidebar:
        head = st.columns([2, 1])
        head[0].caption(f"👤 **{display_name}**")
        with head[1]:
            authenticator.logout("Logout", key="logout_btn", use_container_width=True)
        st.divider()

    nav.run()


if __name__ == "__main__":
    main()