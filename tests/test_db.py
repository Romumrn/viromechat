"""Unit tests for the SQLite persistence layer (db.py).

Each test uses its own temp database file (db.get_conn caches connections per
path, so a unique path per test guarantees isolation).
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


# A map as Plotly 6 serialized it (px.scatter_mapbox) — trimmed real output.
_LEGACY_MAPBOX_FIG = {
    "data": [{
        "type": "scattermapbox", "subplot": "mapbox", "mode": "markers",
        "lat": [11.8], "lon": [99.8], "customdata": [["SAMN1"]],
        "hovertemplate": "sample_id=%{customdata[0]}<extra></extra>",
    }],
    "layout": {
        "mapbox": {"center": {"lat": 11.8, "lon": 99.8}, "zoom": 1, "style": "open-street-map"},
        "template": {
            "data": {"scattermapbox": [{"type": "scattermapbox", "marker": {"colorbar": {"ticks": ""}}}]},
            "layout": {"mapbox": {"style": "light"}},
        },
        "title": {"text": "LSDV"},
    },
}


def _store_raw_figure(conn, fig_dict):
    """Insert a message whose payload holds a figure exactly as an older
    Plotly version wrote it (bypassing add_message, which needs a Figure)."""
    db.create_user(conn, "a@lab.fr", "A", "B", "H", "user")
    cid = db.create_conversation(conn, "a@lab.fr", "c")
    conn.execute(
        "INSERT INTO messages(conversation_id, role, content, payload_json, created_at) "
        "VALUES(?, 'assistant', 'map', ?, 'now')",
        (cid, json.dumps({"figures": [json.dumps(fig_dict)]})),
    )
    conn.commit()
    return cid


def test_legacy_mapbox_figure_is_upgraded_on_rehydrate(conn):
    cid = _store_raw_figure(conn, _LEGACY_MAPBOX_FIG)
    fig = db.list_messages(conn, cid)[0]["figures"][0]
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "scattermap"
    assert fig.layout.map.style == "open-street-map"
    assert list(fig.data[0].customdata[0]) == ["SAMN1"]


def test_legacy_mapbox_figure_is_upgraded_in_json_api(conn):
    cid = _store_raw_figure(conn, _LEGACY_MAPBOX_FIG)
    fig = db.list_messages_json(conn, cid)[0]["figures"][0]
    assert fig["data"][0]["type"] == "scattermap"
    assert fig["data"][0]["subplot"] == "map"
    assert "mapbox" not in fig["layout"] and fig["layout"]["map"]["zoom"] == 1
    assert "scattermapbox" not in fig["layout"]["template"]["data"]


# ==================== error reports ====================

def test_error_report_lifecycle(conn):
    db.create_user(conn, "a@lab.fr", "A", "B", "H", "user")
    rid = db.create_error_report(
        conn, "a@lab.fr", "why?", "because", ["print(1)"], "wrong count", ["log"]
    )
    reports = db.list_error_reports(conn)
    assert len(reports) == 1
    r = reports[0]
    assert r["id"] == rid
    assert r["user_email"] == "a@lab.fr"
    assert r["status"] == "open"           # starts open
    assert r["executed_codes"] == ["print(1)"]
    assert r["comment"] == "wrong count"

    # move it through the triage statuses
    assert db.update_error_report_status(conn, rid, "in_progress") is True
    assert db.list_error_reports(conn, "in_progress")[0]["status"] == "in_progress"
    assert db.list_error_reports(conn, "done") == []

    # unknown id / invalid status are handled
    assert db.update_error_report_status(conn, 999, "done") is False
    with pytest.raises(ValueError):
        db.update_error_report_status(conn, rid, "bogus")


def test_error_report_user_set_null_on_user_delete(conn):
    db.create_user(conn, "a@lab.fr", "A", "B", "H", "user")
    db.create_error_report(conn, "a@lab.fr", "q", "a", [], "", [])
    db.delete_user(conn, "a@lab.fr")
    # the report survives the user's deletion, with user_email cleared
    assert db.list_error_reports(conn)[0]["user_email"] is None
