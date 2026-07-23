"""Unit tests for the SQLite persistence layer (db.py).

Each test uses its own temp database file (db.get_conn caches connections per
path, so a unique path per test guarantees isolation). No Streamlit runtime is
needed — db.py is deliberately Streamlit-free.
"""

import json
import os

import plotly.graph_objects as go
import pytest

import db


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(str(tmp_path / "test.db"))


# ==================== users ====================

def test_create_and_get_user_lowercases_email(conn):
    db.create_user(conn, "Alice@Lab.FR", "Alice", "Doe", "HASH", "user")
    user = db.get_user(conn, "alice@lab.fr")
    assert user is not None
    assert user["email"] == "alice@lab.fr"
    assert user["role"] == "user"
    # email is looked up case-insensitively
    assert db.get_user(conn, "ALICE@LAB.FR")["first_name"] == "Alice"


def test_get_user_missing_returns_none(conn):
    assert db.get_user(conn, "nobody@lab.fr") is None


def test_update_password_changes_hash_only(conn):
    db.create_user(conn, "a@lab.fr", "A", "B", "OLD", "user")
    db.update_password(conn, "a@lab.fr", "NEW")
    assert db.get_user(conn, "a@lab.fr")["password_hash"] == "NEW"


def test_set_last_login_sets_timestamp(conn):
    db.create_user(conn, "a@lab.fr", "A", "B", "H", "user")
    assert db.get_user(conn, "a@lab.fr")["last_login"] is None
    db.set_last_login(conn, "a@lab.fr")
    assert db.get_user(conn, "a@lab.fr")["last_login"] is not None


def test_build_credentials_dict_shape(conn):
    db.create_user(conn, "a@lab.fr", "Al", "Ice", "HASH", "user")
    creds = db.build_credentials_dict(conn)
    assert set(creds.keys()) == {"usernames"}
    entry = creds["usernames"]["a@lab.fr"]
    assert entry["password"] == "HASH"
    assert entry["email"] == "a@lab.fr"
    assert entry["first_name"] == "Al"
    # fields streamlit-authenticator expects to exist
    assert entry["logged_in"] is False
    assert entry["failed_login_attempts"] == 0


def test_list_users_with_counts_excludes_password_and_counts_conversations(conn):
    db.create_user(conn, "a@lab.fr", "A", "B", "HASH", "admin")
    db.create_conversation(conn, "a@lab.fr", "c1")
    db.create_conversation(conn, "a@lab.fr", "c2")
    rows = db.list_users_with_counts(conn)
    assert len(rows) == 1
    row = rows[0]
    assert "password_hash" not in row          # never exposed to admins
    assert row["n_conversations"] == 2
    assert row["role"] == "admin"


# ==================== conversations ====================

def test_conversations_listed_most_recent_first(conn):
    db.create_user(conn, "a@lab.fr", "A", "B", "H", "user")
    c1 = db.create_conversation(conn, "a@lab.fr", "first")
    c2 = db.create_conversation(conn, "a@lab.fr", "second")
    # bump c1 so it sorts back to the top
    db.touch_conversation(conn, c1)
    ids = [c["id"] for c in db.list_conversations(conn, "a@lab.fr")]
    assert ids == [c1, c2]


def test_conversations_scoped_to_user(conn):
    db.create_user(conn, "a@lab.fr", "A", "B", "H", "user")
    db.create_user(conn, "b@lab.fr", "B", "B", "H", "user")
    db.create_conversation(conn, "a@lab.fr", "mine")
    assert db.list_conversations(conn, "b@lab.fr") == []


def test_rename_conversation(conn):
    db.create_user(conn, "a@lab.fr", "A", "B", "H", "user")
    cid = db.create_conversation(conn, "a@lab.fr", "old")
    db.rename_conversation(conn, cid, "new")
    assert db.get_conversation(conn, cid)["title"] == "new"


def test_delete_conversation_cascades_messages(conn):
    db.create_user(conn, "a@lab.fr", "A", "B", "H", "user")
    cid = db.create_conversation(conn, "a@lab.fr", "c")
    db.add_message(conn, cid, "user", "hi")
    db.delete_conversation(conn, cid)
    assert db.get_conversation(conn, cid) is None
    # messages are gone with the conversation (ON DELETE CASCADE)
    remaining = conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?", (cid,)
    ).fetchone()["n"]
    assert remaining == 0


# ==================== messages ====================

def test_message_roundtrip_preserves_payload(conn):
    db.create_user(conn, "a@lab.fr", "A", "B", "H", "user")
    cid = db.create_conversation(conn, "a@lab.fr", "c")
    db.add_message(conn, cid, "user", "question?")
    db.add_message(
        conn, cid, "assistant", "answer",
        {
            "wikipedia_urls": ["https://en.wikipedia.org/wiki/Rabies"],
            "pubmed_urls": ["https://pubmed.ncbi.nlm.nih.gov/123/"],
            "ncbi_urls": [],
            "executed_codes": ["print('x')"],
        },
    )
    msgs = db.list_messages(conn, cid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "question?"
    assert msgs[1]["wikipedia_urls"] == ["https://en.wikipedia.org/wiki/Rabies"]
    assert msgs[1]["executed_codes"] == ["print('x')"]


def test_message_roundtrip_rehydrates_plotly_figures(conn):
    db.create_user(conn, "a@lab.fr", "A", "B", "H", "user")
    cid = db.create_conversation(conn, "a@lab.fr", "c")
    fig = go.Figure(data=[go.Bar(x=["a", "b"], y=[1, 2])])
    db.add_message(conn, cid, "assistant", "chart", {"figures": [fig]})
    msgs = db.list_messages(conn, cid)
    figs = msgs[0]["figures"]
    assert len(figs) == 1
    assert isinstance(figs[0], go.Figure)
    assert list(figs[0].data[0].y) == [1, 2]


# ==================== cookie config / meta ====================

def test_get_cookie_config_generates_and_persists_key(conn):
    name, key, expiry = db.get_cookie_config(conn)
    assert name == "viromechat_auth"
    assert len(key) >= 32
    assert isinstance(expiry, float)
    # stable across calls (persisted, not regenerated)
    assert db.get_cookie_config(conn)[1] == key


# ==================== legacy migration ====================

def _write_legacy_yaml(path, email="romuald@cnrs.fr", cookie_key="COOKIEKEY123"):
    import yaml
    cfg = {
        "cookie": {"name": "viromechat_auth", "key": cookie_key, "expiry_days": 30},
        "credentials": {"usernames": {email: {
            "email": email, "first_name": "Ro", "last_name": "Ma",
            "password": "$2b$12$abcdefghijklmnopqrstuv", "roles": None,
        }}},
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f)


def test_migration_imports_users_history_and_preserves_cookie(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", "romuald@cnrs.fr")
    yaml_path = tmp_path / "auth.yaml"
    hist_dir = tmp_path / "hist"
    hist_dir.mkdir()
    _write_legacy_yaml(str(yaml_path))
    # legacy history file: name is the sanitized email (@ -> _)
    with open(hist_dir / "romuald_cnrs.fr.json", "w") as f:
        json.dump({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "wikipedia_urls": ["u"]},
        ]}, f)

    conn = db.get_conn(str(tmp_path / "mig.db"))
    ran = db.maybe_migrate_legacy_data(conn, str(yaml_path), str(hist_dir))
    assert ran is True

    user = db.get_user(conn, "romuald@cnrs.fr")
    assert user["role"] == "admin"                       # from ADMIN_EMAILS
    assert user["password_hash"].startswith("$2b$")      # bcrypt hash preserved
    assert db.get_meta(conn, "cookie_key") == "COOKIEKEY123"  # cookie preserved

    convs = db.list_conversations(conn, "romuald@cnrs.fr")
    assert len(convs) == 1
    msgs = db.list_messages(conn, convs[0]["id"])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["wikipedia_urls"] == ["u"]


def test_migration_is_idempotent(tmp_path):
    yaml_path = tmp_path / "auth.yaml"
    _write_legacy_yaml(str(yaml_path))
    conn = db.get_conn(str(tmp_path / "mig2.db"))

    assert db.maybe_migrate_legacy_data(conn, str(yaml_path), str(tmp_path / "nohist")) is True
    # second run is a no-op because the users table is no longer empty
    assert db.maybe_migrate_legacy_data(conn, str(yaml_path), str(tmp_path / "nohist")) is False
    assert len(db.list_users_with_counts(conn)) == 1


def test_migration_non_admin_default_role(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", "someone.else@cnrs.fr")
    yaml_path = tmp_path / "auth.yaml"
    _write_legacy_yaml(str(yaml_path), email="plain@cnrs.fr")
    conn = db.get_conn(str(tmp_path / "mig3.db"))
    db.maybe_migrate_legacy_data(conn, str(yaml_path), str(tmp_path / "nohist"))
    assert db.get_user(conn, "plain@cnrs.fr")["role"] == "user"
