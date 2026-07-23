import os
from types import SimpleNamespace

import pytest

import app


# ==================== _ui_search_keyword ====================

def test_ui_search_keyword_picks_search_term():
    assert app._ui_search_keyword({"search_term": "rabies virus"}) == "rabies virus"


def test_ui_search_keyword_falls_back_to_query_then_name():
    assert app._ui_search_keyword({"query": "influenza"}) == "influenza"
    assert app._ui_search_keyword({"name": "Poxviridae"}) == "Poxviridae"


def test_ui_search_keyword_prefers_search_term_over_query_and_name():
    args = {"search_term": "A", "query": "B", "name": "C"}
    assert app._ui_search_keyword(args) == "A"


def test_ui_search_keyword_collapses_internal_whitespace():
    assert app._ui_search_keyword({"search_term": "  hello    world  "}) == "hello world"


def test_ui_search_keyword_truncates_long_values():
    long_term = "a" * 100
    result = app._ui_search_keyword({"search_term": long_term})
    assert result == "a" * 80 + "…"


def test_ui_search_keyword_no_match_returns_empty_string():
    assert app._ui_search_keyword({"sql": "SELECT 1"}) == ""


def test_ui_search_keyword_ignores_non_string_values():
    assert app._ui_search_keyword({"search_term": 42}) == ""


def test_ui_search_keyword_ignores_blank_string():
    assert app._ui_search_keyword({"search_term": "   "}) == ""


# ==================== _snippet ====================

def test_snippet_collapses_whitespace_and_truncates():
    text = "line one\n   line two\t\ttrailing"
    assert app._snippet(text, max_len=13) == "line one line…"


def test_snippet_short_text_is_unchanged():
    assert app._snippet("hello world") == "hello world"


def test_snippet_exact_max_len_no_ellipsis():
    text = "a" * 120
    assert app._snippet(text) == text


def test_snippet_over_max_len_adds_ellipsis():
    text = "a" * 130
    result = app._snippet(text)
    assert result == "a" * 120 + "…"
    assert len(result) == 121


# ==================== _mcp_tools_to_openai_spec ====================

def test_mcp_tools_to_openai_spec_converts_fields():
    fake_tool = SimpleNamespace(
        name="wikipedia_search",
        description="Search Wikipedia",
        inputSchema={"type": "object", "properties": {"search_term": {"type": "string"}}},
    )

    spec = app._mcp_tools_to_openai_spec([fake_tool])

    assert spec == [{
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search Wikipedia",
            "parameters": {"type": "object", "properties": {"search_term": {"type": "string"}}},
        },
    }]


def test_mcp_tools_to_openai_spec_defaults_missing_description_and_schema():
    fake_tool = SimpleNamespace(name="create_map", description=None, inputSchema=None)

    spec = app._mcp_tools_to_openai_spec([fake_tool])

    assert spec[0]["function"]["description"] == ""
    assert spec[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_mcp_tools_to_openai_spec_empty_list():
    assert app._mcp_tools_to_openai_spec([]) == []


# ==================== _unwrap_mcp_result ====================

def test_unwrap_mcp_result_prefers_data_attribute():
    result = SimpleNamespace(data={"success": True, "content": "ok"}, structured_content=None, content=[])
    assert app._unwrap_mcp_result(result) == {"success": True, "content": "ok"}


def test_unwrap_mcp_result_falls_back_to_structured_content():
    result = SimpleNamespace(data=None, structured_content={"success": True}, content=[])
    assert app._unwrap_mcp_result(result) == {"success": True}


def test_unwrap_mcp_result_parses_json_text_block():
    block = SimpleNamespace(type="text", text='{"success": true, "content": "hi"}')
    result = SimpleNamespace(data=None, structured_content=None, content=[block])
    assert app._unwrap_mcp_result(result) == {"success": True, "content": "hi"}


def test_unwrap_mcp_result_non_json_text_block_becomes_failure_dict():
    block = SimpleNamespace(type="text", text="not json at all")
    result = SimpleNamespace(data=None, structured_content=None, content=[block])
    unwrapped = app._unwrap_mcp_result(result)
    assert unwrapped["success"] is False
    assert unwrapped["content"] == "not json at all"


def test_unwrap_mcp_result_empty_response_returns_failure_dict():
    result = SimpleNamespace(data=None, structured_content=None, content=[])
    unwrapped = app._unwrap_mcp_result(result)
    assert unwrapped["success"] is False
    assert "Empty MCP tool response" in unwrapped["content"]


# ==================== _strip_hallucinated_pmids ====================

def test_strip_hallucinated_pmids_keeps_real_pmid():
    text = "This is confirmed (PMID 12345678)."
    cleaned, removed = app._strip_hallucinated_pmids(text, real_pmids={"12345678"})
    assert "PMID 12345678" in cleaned
    assert removed == []


def test_strip_hallucinated_pmids_removes_fake_pmid():
    text = "This is confirmed (PMID 99999999)."
    cleaned, removed = app._strip_hallucinated_pmids(text, real_pmids={"12345678"})
    assert "99999999" not in cleaned
    assert removed == ["99999999"]


def test_strip_hallucinated_pmids_handles_multiple_pmids_mixed():
    text = "First fact (PMID 11111111). Second fact (PMID 22222222)."
    cleaned, removed = app._strip_hallucinated_pmids(text, real_pmids={"22222222"})
    assert "11111111" not in cleaned
    assert "22222222" in cleaned
    assert removed == ["11111111"]


def test_strip_hallucinated_pmids_no_pmids_in_text():
    text = "No citations here at all."
    cleaned, removed = app._strip_hallucinated_pmids(text, real_pmids=set())
    assert cleaned == text
    assert removed == []


# ==================== _strip_fake_citation_markers ====================

def test_strip_fake_citation_markers_removes_bracket_marker():
    text = "Binds sialic acid receptors【4†L13-L17】."
    cleaned, count = app._strip_fake_citation_markers(text)
    assert "【" not in cleaned
    assert count == 1
    # no stray space introduced before the period
    assert cleaned == "Binds sialic acid receptors."


def test_strip_fake_citation_markers_no_marker_returns_unchanged():
    text = "Nothing to strip here."
    cleaned, count = app._strip_fake_citation_markers(text)
    assert cleaned == text
    assert count == 0


def test_strip_fake_citation_markers_multiple_markers():
    text = "Fact one【1】 and fact two【2】."
    cleaned, count = app._strip_fake_citation_markers(text)
    assert count == 2
    assert "【" not in cleaned


# ==================== _password_problem ====================

@pytest.mark.parametrize("password,expected_substring", [
    ("short1A!", "at least 12 characters"),
    ("nouppercase123!", "1 uppercase letter"),
    ("NOLOWERCASE123!", "1 lowercase letter"),
    ("NoDigitsHere!!", "1 digit"),
    ("NoSpecialChar123", "1 special character"),
])
def test_password_problem_flags_each_rule(password, expected_substring):
    problem = app._password_problem(password)
    assert problem is not None
    assert expected_substring in problem


def test_password_problem_accepts_valid_password():
    assert app._password_problem("Valid-Password123") is None


# ==================== _password_checklist ====================

def test_password_checklist_all_false_for_empty_password():
    checklist = app._password_checklist("")
    assert len(checklist) == len(app._PASSWORD_RULES)
    assert all(ok is False for _label, ok in checklist)


def test_password_checklist_all_true_for_valid_password():
    checklist = app._password_checklist("Valid-Password123")
    assert all(ok is True for _label, ok in checklist)


def test_password_checklist_mixed_results_match_password_problem():
    password = "nouppercase123!"  # missing only the uppercase rule
    checklist = dict(app._password_checklist(password))
    assert checklist["1 uppercase letter"] is False
    assert checklist["At least 12 characters"] is True
    assert checklist["1 lowercase letter"] is True
    assert checklist["1 digit"] is True
    # _password_problem stops at the first failing rule (uppercase, in
    # rule order) — the checklist must agree on which one that is
    assert "uppercase" in app._password_problem(password)


def test_password_checklist_labels_are_stable_and_ordered():
    labels = [label for label, _ok in app._password_checklist("x")]
    assert labels == [
        "At least 12 characters",
        "1 lowercase letter",
        "1 uppercase letter",
        "1 digit",
        "1 special character (e.g. ! ? @ # …)",
    ]


# ==================== _parse_tool_arguments ====================

def test_parse_tool_arguments_passes_through_dict():
    args = {"search_term": "HIV"}
    assert app._parse_tool_arguments(args) is args


def test_parse_tool_arguments_parses_json_string():
    assert app._parse_tool_arguments('{"search_term": "HIV"}') == {"search_term": "HIV"}


def test_parse_tool_arguments_recovers_partial_json():
    # missing closing brace, as seen from the known gpt-oss-120b bug
    raw = '{"search_term": "HIV", "max_results": 5'
    result = app._parse_tool_arguments(raw)
    assert result == {"search_term": "HIV", "max_results": 5}


def test_parse_tool_arguments_total_garbage_falls_back_to_raw():
    raw = "not json and no key-value pairs either"
    result = app._parse_tool_arguments(raw)
    assert result == {"_raw": raw}


def test_parse_tool_arguments_non_dict_non_str_returns_empty_dict():
    assert app._parse_tool_arguments(None) == {}
    assert app._parse_tool_arguments(42) == {}


# ==================== _clean_history_messages ====================

def test_clean_history_messages_keeps_user_and_final_assistant_messages():
    messages = [
        {"role": "user", "content": "How many species?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "42", "tool_call_id": "1"},
        {"role": "assistant", "content": "There are 42 species."},
    ]

    cleaned = app._clean_history_messages(messages)

    assert cleaned == [
        {"role": "user", "content": "How many species?"},
        {"role": "assistant", "content": "There are 42 species."},
    ]


def test_clean_history_messages_empty_list():
    assert app._clean_history_messages([]) == []


def test_clean_history_messages_drops_system_messages():
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "hi"},
    ]
    cleaned = app._clean_history_messages(messages)
    assert cleaned == [{"role": "user", "content": "hi"}]


# ==================== build_context_window ====================

def _exchanges(n):
    """n user/assistant exchanges as display-shape messages."""
    msgs = []
    for i in range(1, n + 1):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


def test_build_context_window_keeps_last_n_turns():
    window = app.build_context_window(_exchanges(5), max_turns=2)
    assert [(m["role"], m["content"]) for m in window] == [
        ("user", "q4"), ("assistant", "a4"),
        ("user", "q5"), ("assistant", "a5"),
    ]


def test_build_context_window_shorter_than_limit_returns_all():
    window = app.build_context_window(_exchanges(2), max_turns=5)
    assert len(window) == 4


def test_build_context_window_zero_turns_returns_all():
    # max_turns <= 0 disables the slice (full history sent)
    assert len(app.build_context_window(_exchanges(3), max_turns=0)) == 6


def test_build_context_window_strips_rich_fields_and_figures():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "figures": ["<fig>"], "wikipedia_urls": ["u"]},
    ]
    window = app.build_context_window(messages, max_turns=5)
    # only role/content survive into the prompt — no figures/urls
    assert window == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]


# ==================== _albert_chat retry / rate limiting ====================

class _FakeResp:
    def __init__(self, status_code, headers=None, body=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body or {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def test_albert_chat_returns_json_on_success(monkeypatch):
    monkeypatch.setattr(app.requests, "post", lambda *a, **k: _FakeResp(200, body={"ok": True}))
    assert app._albert_chat([], [], "m", "k", 0.2, 0.9)["ok"] is True


def test_albert_chat_raises_rate_limit_error_after_retries(monkeypatch):
    monkeypatch.setattr(app.requests, "post", lambda *a, **k: _FakeResp(429))
    sleeps = []
    monkeypatch.setattr(app.time, "sleep", lambda s: sleeps.append(s))
    with pytest.raises(app.AlbertRateLimitError):
        app._albert_chat([], [], "m", "k", 0.2, 0.9, retry=3)
    # sleeps only between attempts, never after the final one
    assert len(sleeps) == 2


def test_albert_chat_honors_retry_after_header(monkeypatch):
    monkeypatch.setattr(
        app.requests, "post", lambda *a, **k: _FakeResp(429, headers={"Retry-After": "5"})
    )
    waited = []
    monkeypatch.setattr(app.time, "sleep", lambda s: waited.append(s))
    with pytest.raises(app.AlbertRateLimitError):
        app._albert_chat([], [], "m", "k", 0.2, 0.9, retry=2)
    assert waited == [5]  # honored Retry-After (< cap), one sleep before the last attempt


def test_albert_chat_backoff_is_capped(monkeypatch):
    monkeypatch.setattr(app.requests, "post", lambda *a, **k: _FakeResp(429))
    waited = []
    monkeypatch.setattr(app.time, "sleep", lambda s: waited.append(s))
    with pytest.raises(app.AlbertRateLimitError):
        app._albert_chat([], [], "m", "k", 0.2, 0.9, retry=8)
    # exponential 2**attempt, each capped at ALBERT_RETRY_BACKOFF_CAP
    assert max(waited) <= app.ALBERT_RETRY_BACKOFF_CAP
