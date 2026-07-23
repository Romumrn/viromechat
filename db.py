"""
db.py — SQLite persistence for Viromech@t.

A single database (config.DB_PATH) holds everything that must survive restarts:

  - app_meta      : key/value — the streamlit-authenticator cookie config
                    (name/key/expiry) and the schema version.
  - users         : email (PK), first/last name, bcrypt password_hash, role
                    ('user' | 'admin'), created_at, last_login.
  - conversations : one row per chat thread, owned by a user.
  - messages      : the individual turns of a conversation (role, content, and
                    a JSON payload carrying figures / source URLs / executed code).

streamlit-authenticator stays the crypto/cookie/session engine — this module
only stores and retrieves. Passwords are ALWAYS bcrypt-hashed: never written or
returned in clear, and never surfaced to the admin view.

The first time the DB is empty, maybe_migrate_legacy_data() imports the previous
YAML accounts (AUTH_CONFIG_PATH) and the per-user chat JSON files
(USER_HISTORY_DIR), preserving bcrypt hashes and the cookie key so existing
logins and cookies keep working. The import is idempotent — it only runs while
the users table is empty, so calling it on every startup is safe.

No Streamlit import here on purpose: db.py must be usable from plain pytest.
One process-wide connection per database path is shared across Streamlit reruns
and sessions; writes are serialized with a lock and WAL keeps reads concurrent.
"""

import json
import os
import re
import secrets as _secrets
import sqlite3
import threading
from datetime import datetime, timezone

import plotly.io as pio

from config import (
    DB_PATH,
    AUTH_CONFIG_PATH,
    USER_HISTORY_DIR,
    _admin_emails,
)

SCHEMA_VERSION = "1"

_conns: dict[str, sqlite3.Connection] = {}
_conn_lock = threading.Lock()   # guards _conns creation
_write_lock = threading.Lock()  # serializes writes — SQLite is single-writer


def _now() -> str:
    """UTC timestamp, ISO-8601 — stored as TEXT (SQLite has no native datetime)."""
    return datetime.now(timezone.utc).isoformat()


# ==================== CONNECTION ====================

def get_conn(db_path: str | None = None) -> sqlite3.Connection:
    """
    Return a process-wide shared connection for db_path (default: config.DB_PATH),
    creating it — and its schema — on first use. Streamlit imports this module
    once, so the cached connection persists across reruns and is shared by every
    session; writes go through _write_lock and WAL mode keeps reads concurrent.

    Passing an explicit db_path (a tmp file, or ":memory:") is how tests get an
    isolated database.
    """
    path = db_path or DB_PATH
    with _conn_lock:
        conn = _conns.get(path)
        if conn is None:
            if path != ":memory:":
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            _init_schema(conn)
            _conns[path] = conn
        return conn


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """Open (creating if needed) the database and run the one-time legacy
    migration. Call once at app startup."""
    conn = get_conn(db_path)
    maybe_migrate_legacy_data(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS app_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS users (
            email         TEXT PRIMARY KEY,
            first_name    TEXT,
            last_name     TEXT,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'user',
            created_at    TEXT,
            last_login    TEXT
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL REFERENCES users(email) ON DELETE CASCADE,
            title      TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role            TEXT NOT NULL,
            content         TEXT,
            payload_json    TEXT,
            created_at      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_email);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
        """
    )
    conn.commit()
    if get_meta(conn, "schema_version") is None:
        set_meta(conn, "schema_version", SCHEMA_VERSION)


# ==================== APP META (cookie config, versions) ====================

def get_meta(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value FROM app_meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value) -> None:
    with _write_lock:
        conn.execute(
            "INSERT INTO app_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )
        conn.commit()


def get_cookie_config(conn: sqlite3.Connection) -> tuple[str, str, float]:
    """(cookie_name, cookie_key, expiry_days) for streamlit-authenticator.
    On a brand-new DB with no migrated YAML cookie, a random signing key is
    generated once and persisted so sessions stay valid across restarts."""
    name = get_meta(conn, "cookie_name") or "viromechat_auth"
    key = get_meta(conn, "cookie_key")
    if not key:
        key = _secrets.token_hex(32)
        set_meta(conn, "cookie_name", name)
        set_meta(conn, "cookie_key", key)
        set_meta(conn, "cookie_expiry_days", "30")
    expiry = get_meta(conn, "cookie_expiry_days") or "30"
    return name, key, float(expiry)


# ==================== USERS ====================

def get_user(conn: sqlite3.Connection, email: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email.lower(),)
    ).fetchone()
    return dict(row) if row else None


def create_user(
    conn: sqlite3.Connection,
    email: str,
    first_name: str,
    last_name: str,
    password_hash: str,
    role: str = "user",
) -> None:
    with _write_lock:
        conn.execute(
            "INSERT INTO users(email, first_name, last_name, password_hash, role, created_at, last_login) "
            "VALUES(?, ?, ?, ?, ?, ?, NULL)",
            (email.lower(), first_name, last_name, password_hash, role, _now()),
        )
        conn.commit()


def update_password(conn: sqlite3.Connection, email: str, password_hash: str) -> None:
    with _write_lock:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE email = ?",
            (password_hash, email.lower()),
        )
        conn.commit()


def set_last_login(conn: sqlite3.Connection, email: str) -> None:
    with _write_lock:
        conn.execute(
            "UPDATE users SET last_login = ? WHERE email = ?", (_now(), email.lower())
        )
        conn.commit()


def set_role(conn: sqlite3.Connection, email: str, role: str) -> None:
    with _write_lock:
        conn.execute("UPDATE users SET role = ? WHERE email = ?", (role, email.lower()))
        conn.commit()


def build_credentials_dict(conn: sqlite3.Connection) -> dict:
    """Build the credentials dict streamlit-authenticator expects, from the
    users table. Rebuilt fresh on each run — streamlit-authenticator mutates
    its own copy (logged_in, failed_login_attempts) but never persists here."""
    creds: dict = {"usernames": {}}
    for row in conn.execute("SELECT * FROM users"):
        creds["usernames"][row["email"]] = {
            "email": row["email"],
            "first_name": row["first_name"] or "",
            "last_name": row["last_name"] or "",
            "password": row["password_hash"],
            "logged_in": False,
            "roles": None,
            "failed_login_attempts": 0,
        }
    return creds


def list_users_with_counts(conn: sqlite3.Connection) -> list[dict]:
    """All users with their conversation count, for the admin view. Never
    includes password_hash — admins have no business seeing it."""
    rows = conn.execute(
        """
        SELECT u.email, u.first_name, u.last_name, u.role, u.created_at, u.last_login,
               (SELECT COUNT(*) FROM conversations c WHERE c.user_email = u.email) AS n_conversations
        FROM users u
        ORDER BY u.created_at
        """
    ).fetchall()
    return [dict(r) for r in rows]


# ==================== CONVERSATIONS ====================

def list_conversations(conn: sqlite3.Connection, email: str) -> list[dict]:
    """A user's conversations, most-recently-updated first (sidebar order)."""
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations "
        "WHERE user_email = ? ORDER BY updated_at DESC, id DESC",
        (email.lower(),),
    ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conn: sqlite3.Connection, conversation_id: int) -> dict | None:
    row = conn.execute(
        "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
    ).fetchone()
    return dict(row) if row else None


def create_conversation(conn: sqlite3.Connection, email: str, title: str) -> int:
    now = _now()
    with _write_lock:
        cur = conn.execute(
            "INSERT INTO conversations(user_email, title, created_at, updated_at) VALUES(?, ?, ?, ?)",
            (email.lower(), title, now, now),
        )
        conn.commit()
        return cur.lastrowid


def rename_conversation(conn: sqlite3.Connection, conversation_id: int, title: str) -> None:
    with _write_lock:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), conversation_id),
        )
        conn.commit()


def touch_conversation(conn: sqlite3.Connection, conversation_id: int) -> None:
    """Bump updated_at so this conversation sorts back to the top of the list."""
    with _write_lock:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_now(), conversation_id),
        )
        conn.commit()


def delete_conversation(conn: sqlite3.Connection, conversation_id: int) -> None:
    with _write_lock:
        # messages are removed by the ON DELETE CASCADE foreign key.
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()


# ==================== MESSAGES ====================
#
# Everything on a message besides role/content is stored as JSON in
# messages.payload_json. Plotly figures are serialized with fig.to_json() and
# rehydrated with pio.from_json — mirroring the previous per-user-JSON
# persistence (see the old _load/_save_user_history in app.py).

_PAYLOAD_KEYS = ("figures", "wikipedia_urls", "pubmed_urls", "ncbi_urls", "executed_codes")


def _serialize_payload(msg: dict) -> str:
    payload: dict = {}
    for k in _PAYLOAD_KEYS:
        if k not in msg:
            continue
        if k == "figures":
            payload[k] = [fig.to_json() for fig in msg[k]]
        else:
            payload[k] = msg[k]
    return json.dumps(payload)


def _rehydrate_payload(payload_json: str | None) -> dict:
    data = json.loads(payload_json) if payload_json else {}
    if "figures" in data:
        data["figures"] = [pio.from_json(fj) for fj in data["figures"]]
    return data


def add_message(
    conn: sqlite3.Connection,
    conversation_id: int,
    role: str,
    content: str,
    msg: dict | None = None,
) -> None:
    """Append one turn. `msg` may carry the rich fields in _PAYLOAD_KEYS
    (figures as Plotly Figure objects, URL lists, executed code)."""
    payload_json = _serialize_payload(msg or {})
    with _write_lock:
        conn.execute(
            "INSERT INTO messages(conversation_id, role, content, payload_json, created_at) "
            "VALUES(?, ?, ?, ?, ?)",
            (conversation_id, role, content, payload_json, _now()),
        )
        conn.commit()


def list_messages(conn: sqlite3.Connection, conversation_id: int) -> list[dict]:
    """Messages in UI shape ({role, content, **payload}), figures rehydrated
    to Figure objects — ready to drop into st.session_state.messages."""
    rows = conn.execute(
        "SELECT role, content, payload_json FROM messages WHERE conversation_id = ? ORDER BY id",
        (conversation_id,),
    ).fetchall()
    out = []
    for r in rows:
        m = {"role": r["role"], "content": r["content"]}
        m.update(_rehydrate_payload(r["payload_json"]))
        out.append(m)
    return out


# ==================== ONE-TIME LEGACY MIGRATION ====================

def _sanitize_email(email: str) -> str:
    """Same sanitization the old _user_history_path used to turn an email into
    a filename (e.g. romuald.marin@cnrs.fr -> romuald.marin_cnrs.fr)."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", email)


def _match_history_file_to_user(fname: str, emails: list[str]) -> str | None:
    stem = fname[:-5] if fname.endswith(".json") else fname
    for e in emails:
        if _sanitize_email(e) == stem:
            return e
    return None


def maybe_migrate_legacy_data(
    conn: sqlite3.Connection,
    auth_config_path: str | None = None,
    user_history_dir: str | None = None,
) -> bool:
    """
    Import the pre-DB YAML accounts and per-user chat JSON the first time the
    users table is empty. Preserves bcrypt hashes and the cookie key/name so
    existing logins and re-auth cookies keep working. Idempotent: a no-op once
    any user exists. Returns True if a migration actually ran.
    """
    n_users = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
    if n_users > 0:
        return False

    auth_path = auth_config_path or AUTH_CONFIG_PATH
    hist_dir = user_history_dir or USER_HISTORY_DIR
    admins = _admin_emails()

    imported_emails: list[str] = []

    if os.path.exists(auth_path):
        try:
            import yaml
            with open(auth_path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        # Preserve the existing cookie so current re-auth cookies stay valid.
        cookie = cfg.get("cookie") or {}
        if cookie.get("name"):
            set_meta(conn, "cookie_name", cookie["name"])
        if cookie.get("key"):
            set_meta(conn, "cookie_key", cookie["key"])
        if cookie.get("expiry_days") is not None:
            set_meta(conn, "cookie_expiry_days", str(cookie["expiry_days"]))

        usernames = (cfg.get("credentials") or {}).get("usernames") or {}
        for email, u in usernames.items():
            email_l = str(email).lower()
            if not u.get("password"):
                continue  # nothing usable to migrate for this account
            role = "admin" if email_l in admins else "user"
            create_user(
                conn, email_l,
                u.get("first_name", ""), u.get("last_name", ""),
                u.get("password", ""), role,
            )
            imported_emails.append(email_l)

    # Import each user's single legacy conversation (one JSON file per user).
    if os.path.isdir(hist_dir):
        for fname in sorted(os.listdir(hist_dir)):
            if not fname.endswith(".json"):
                continue
            email_l = _match_history_file_to_user(fname, imported_emails)
            if email_l is None:
                continue
            try:
                with open(os.path.join(hist_dir, fname)) as f:
                    data = json.load(f)
            except Exception:
                continue
            messages = data.get("messages") or []
            if not messages:
                continue
            cid = create_conversation(conn, email_l, "Historique importé")
            for m in messages:
                mm = dict(m)
                # Legacy files store figures as JSON strings — turn them back
                # into Figure objects so add_message serializes them uniformly.
                if "figures" in mm:
                    mm["figures"] = [pio.from_json(fj) for fj in mm["figures"]]
                add_message(conn, cid, mm.get("role", "user"), mm.get("content", ""), mm)

    return True
