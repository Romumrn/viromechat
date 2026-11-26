"""
db.py — SQLite persistence for Viromech@t.

A single database (config.DB_PATH) holds everything that must survive restarts:

  - app_meta      : key/value — the schema version (and any future settings).
  - users         : email (PK), first/last name, bcrypt password_hash, role
                    ('user' | 'dev' | 'admin'), created_at, last_login.
  - conversations : one row per chat thread, owned by a user.
  - messages      : the individual turns of a conversation (role, content, and
                    a JSON payload carrying figures / source URLs / executed code).
  - error_reports : user-submitted "Report an error" feedback for dev triage.

This module only stores and retrieves — the FastAPI backend (backend/auth.py) is
the crypto/session engine. Passwords are ALWAYS bcrypt-hashed: never written or
returned in clear, and never surfaced to the admin view.

One process-wide connection per database path is shared across requests; writes
are serialized with a lock and WAL keeps reads concurrent.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

import plotly.io as pio

from config import DB_PATH

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
    creating it — and its schema — on first use. The cached connection is shared
    across requests; writes go through _write_lock and WAL mode keeps reads
    concurrent.

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
    """Open (creating if needed) the database. Call once at app startup."""
    return get_conn(db_path)


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
        CREATE TABLE IF NOT EXISTS error_reports (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email     TEXT REFERENCES users(email) ON DELETE SET NULL,
            question       TEXT,
            answer         TEXT,
            executed_codes TEXT,   -- JSON array of code snippets
            comment        TEXT,
            recent_logs    TEXT,   -- JSON array — tail of the agent log for context
            status         TEXT NOT NULL DEFAULT 'open',  -- open | in_progress | done
            created_at     TEXT,
            updated_at     TEXT
        );
        CREATE TABLE IF NOT EXISTS helper_scores (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email   TEXT REFERENCES users(email) ON DELETE SET NULL,
            virus        TEXT,
            days         INTEGER,
            infected_pct REAL,      -- 0..1 population-weighted world coverage
            dead         INTEGER,   -- absolute death toll (people)
            won          INTEGER NOT NULL DEFAULT 0,  -- 0/1
            score        INTEGER NOT NULL,
            created_at   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_email);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_error_reports_status ON error_reports(status);
        CREATE INDEX IF NOT EXISTS idx_helper_scores_score ON helper_scores(score DESC);
        """
    )
    conn.commit()
    if get_meta(conn, "schema_version") is None:
        set_meta(conn, "schema_version", SCHEMA_VERSION)


# ==================== APP META (schema version, settings) ====================

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


def search_users(conn: sqlite3.Connection, query: str) -> list[dict]:
    """Same shape as list_users_with_counts, filtered to users whose email or
    name matches `query` (case-insensitive substring). Empty query → all."""
    q = (query or "").strip().lower()
    users = list_users_with_counts(conn)
    if not q:
        return users
    return [
        u for u in users
        if q in (u["email"] or "").lower()
        or q in (u["first_name"] or "").lower()
        or q in (u["last_name"] or "").lower()
    ]


def delete_user(conn: sqlite3.Connection, email: str) -> None:
    """Delete a user and — via ON DELETE CASCADE — all their conversations and
    messages. Admin-only operation (enforced at the API layer)."""
    with _write_lock:
        conn.execute("DELETE FROM users WHERE email = ?", (email.lower(),))
        conn.commit()


def get_stats(conn: sqlite3.Connection) -> dict:
    """Platform-wide counts for the admin dashboard: users per role, plus
    total conversations and messages."""
    by_role = {
        r["role"]: r["n"]
        for r in conn.execute(
            "SELECT role, COUNT(*) AS n FROM users GROUP BY role"
        ).fetchall()
    }
    n_users = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
    n_conv = conn.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()["n"]
    n_msg = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
    return {
        "users_total": n_users,
        "users_by_role": by_role,
        "conversations_total": n_conv,
        "messages_total": n_msg,
    }


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
# rehydrated with pio.from_json.

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


# Plotly 7 removed the Mapbox-based traces. Maps saved before the upgrade
# (Plotly 6, scatter_mapbox) are stored with the old names and would make
# pio.from_json raise, so they're renamed to their MapLibre equivalents on load.
_LEGACY_MAP_TRACES = {
    "scattermapbox": "scattermap",
    "densitymapbox": "densitymap",
    "choroplethmapbox": "choroplethmap",
}


def _upgrade_legacy_map_layout(layout: dict) -> None:
    for key in [k for k in layout if k.startswith("mapbox")]:
        sub = layout.pop(key)
        if isinstance(sub, dict):
            sub.pop("accesstoken", None)
            if str(sub.get("style", "")).startswith("stamen"):
                sub["style"] = "open-street-map"  # Stamen tiles don't exist in MapLibre
        layout["map" + key[len("mapbox"):]] = sub


def _upgrade_legacy_map_figure(fig: dict) -> dict:
    """Rename Plotly <7 Mapbox traces/layout keys to Plotly 7 MapLibre ones (in place)."""
    for trace in fig.get("data") or []:
        if trace.get("type") in _LEGACY_MAP_TRACES:
            trace["type"] = _LEGACY_MAP_TRACES[trace["type"]]
            if str(trace.get("subplot", "")).startswith("mapbox"):
                trace["subplot"] = "map" + trace["subplot"][len("mapbox"):]
    layout = fig.get("layout") or {}
    _upgrade_legacy_map_layout(layout)
    template = layout.get("template") or {}
    tdata = template.get("data") or {}
    for old, new in _LEGACY_MAP_TRACES.items():
        if old in tdata:
            legacy = tdata.pop(old)
            tdata.setdefault(new, legacy)
    _upgrade_legacy_map_layout(template.get("layout") or {})
    return fig


def _load_figure_dict(fig_json: str) -> dict:
    return _upgrade_legacy_map_figure(json.loads(fig_json))


def _rehydrate_payload(payload_json: str | None) -> dict:
    data = json.loads(payload_json) if payload_json else {}
    if "figures" in data:
        data["figures"] = [
            pio.from_json(json.dumps(_load_figure_dict(fj))) for fj in data["figures"]
        ]
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


def list_messages_json(conn: sqlite3.Connection, conversation_id: int) -> list[dict]:
    """Messages in JSON-serializable shape for the REST/React API: identical to
    list_messages but figures stay as Plotly JSON dicts (parsed from the stored
    string) instead of Figure objects, so FastAPI can serialize them directly."""
    rows = conn.execute(
        "SELECT role, content, payload_json FROM messages WHERE conversation_id = ? ORDER BY id",
        (conversation_id,),
    ).fetchall()
    out = []
    for r in rows:
        m = {"role": r["role"], "content": r["content"]}
        data = json.loads(r["payload_json"]) if r["payload_json"] else {}
        if "figures" in data:
            # Stored as Plotly JSON strings; hand back parsed dicts for React.
            data["figures"] = [_load_figure_dict(fj) for fj in data["figures"]]
        m.update(data)
        out.append(m)
    return out


# ==================== ERROR REPORTS ====================

ERROR_REPORT_STATUSES = ("open", "in_progress", "done")


def create_error_report(
    conn: sqlite3.Connection,
    user_email: str | None,
    question: str,
    answer: str,
    executed_codes: list | None,
    comment: str,
    recent_logs: list | None = None,
) -> int:
    """Persist a user-submitted error report and return its id. Starts 'open'."""
    now = _now()
    with _write_lock:
        cur = conn.execute(
            """INSERT INTO error_reports
               (user_email, question, answer, executed_codes, comment,
                recent_logs, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?)""",
            (
                user_email,
                question,
                answer,
                json.dumps(executed_codes or [], ensure_ascii=False),
                comment,
                json.dumps(recent_logs or [], ensure_ascii=False),
                now,
                now,
            ),
        )
        conn.commit()
        return cur.lastrowid


def list_error_reports(
    conn: sqlite3.Connection, status: str | None = None
) -> list[dict]:
    """Error reports, newest first, with JSON columns parsed. Optionally filter
    by status ('open' | 'in_progress' | 'done')."""
    if status:
        rows = conn.execute(
            "SELECT * FROM error_reports WHERE status = ? ORDER BY id DESC",
            (status,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM error_reports ORDER BY id DESC"
        ).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "user_email": r["user_email"],
                "question": r["question"],
                "answer": r["answer"],
                "executed_codes": json.loads(r["executed_codes"] or "[]"),
                "comment": r["comment"],
                "recent_logs": json.loads(r["recent_logs"] or "[]"),
                "status": r["status"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
        )
    return out


def update_error_report_status(
    conn: sqlite3.Connection, report_id: int, status: str
) -> bool:
    """Set a report's status. Returns False if the id doesn't exist."""
    if status not in ERROR_REPORT_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    with _write_lock:
        cur = conn.execute(
            "UPDATE error_reports SET status = ?, updated_at = ? WHERE id = ?",
            (status, _now(), report_id),
        )
        conn.commit()
        return cur.rowcount > 0


# ==================== HELPER SCORES (contagion mini-game) =====================

def add_helper_score(
    conn: sqlite3.Connection,
    user_email: str | None,
    virus: str,
    days: int,
    infected_pct: float,
    dead: int,
    won: bool,
    score: int,
) -> int:
    """Record one finished game and return its row id."""
    with _write_lock:
        cur = conn.execute(
            "INSERT INTO helper_scores(user_email, virus, days, infected_pct, dead, won, score, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_email.lower() if user_email else None,
                virus,
                days,
                infected_pct,
                dead,
                1 if won else 0,
                score,
                _now(),
            ),
        )
        conn.commit()
        return cur.lastrowid


def top_helper_scores(conn: sqlite3.Connection, limit: int = 5) -> list[sqlite3.Row]:
    """Highest scores across all users, best first. Joins the player's name."""
    return conn.execute(
        "SELECT h.score, h.virus, h.days, h.infected_pct, h.dead, h.created_at, "
        "       h.user_email, u.first_name, u.last_name "
        "FROM helper_scores h LEFT JOIN users u ON u.email = h.user_email "
        "ORDER BY h.score DESC, h.days ASC LIMIT ?",
        (limit,),
    ).fetchall()


