import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest

import server_mcp as sm


# ==================== _ok / _fail ====================

def test_ok_builds_success_dict():
    result = sm._ok("all good", [{"type": "table"}])
    assert result == {"success": True, "content": "all good", "artifacts": [{"type": "table"}]}


def test_ok_defaults_to_empty_artifacts():
    result = sm._ok("all good")
    assert result["artifacts"] == []


def test_fail_builds_failure_dict_with_no_artifacts():
    result = sm._fail("something broke")
    assert result == {"success": False, "content": "something broke", "artifacts": []}


# ==================== _check_figure_has_data ====================

def test_check_figure_has_data_true_for_bar_chart_with_points():
    fig = px.bar(x=["a", "b"], y=[1, 2])
    assert sm._check_figure_has_data(fig) is True


def test_check_figure_has_data_false_for_empty_dataframe():
    empty_df = pd.DataFrame({"x": [], "y": []})
    fig = px.bar(empty_df, x="x", y="y")
    assert sm._check_figure_has_data(fig) is False


def test_check_figure_has_data_true_for_map_with_lat_lon():
    fig = go.Figure(go.Scattergeo(lat=[10.0, 20.0], lon=[30.0, 40.0]))
    assert sm._check_figure_has_data(fig) is True


def test_check_figure_has_data_true_for_pie_chart_values():
    fig = go.Figure(go.Pie(labels=["a", "b"], values=[1, 2]))
    assert sm._check_figure_has_data(fig) is True


def test_check_figure_has_data_false_for_figure_with_no_traces():
    fig = go.Figure()
    assert sm._check_figure_has_data(fig) is False


# ==================== _check_hover_has_column ====================

def test_check_hover_has_column_true_when_column_in_hovertemplate():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "COUNTRY": ["FR", "US"]})
    fig = px.scatter(df, x="x", y="y", hover_data=["COUNTRY"])
    assert sm._check_hover_has_column(fig, "COUNTRY") is True


def test_check_hover_has_column_false_when_column_missing():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    fig = px.scatter(df, x="x", y="y")
    assert sm._check_hover_has_column(fig, "COUNTRY") is False


# ==================== _check_data_not_empty ====================

def test_check_data_not_empty_returns_none_when_no_dataframe_present():
    assert sm._check_data_not_empty({"unrelated": 123}) is None


def test_check_data_not_empty_returns_none_when_dataframe_has_rows():
    env = {"data": pd.DataFrame({"a": [1, 2]})}
    assert sm._check_data_not_empty(env) is None


def test_check_data_not_empty_flags_empty_dataframe():
    env = {"data": pd.DataFrame()}
    message = sm._check_data_not_empty(env)
    assert message is not None
    assert "'data'" in message
    assert "0 rows" in message


def test_check_data_not_empty_checks_known_varnames_only():
    # a differently-named empty dataframe isn't inspected
    env = {"some_other_name": pd.DataFrame()}
    assert sm._check_data_not_empty(env) is None


# ==================== _figure_artifact / _figure_title ====================

def test_figure_artifact_has_plotly_type_and_json_figure():
    fig = px.bar(x=["a"], y=[1])
    artifact = sm._figure_artifact(fig)
    assert artifact["type"] == "plotly"
    assert "data" in artifact["figure"]
    assert "layout" in artifact["figure"]


def test_figure_title_returns_none_when_no_title_set():
    fig = go.Figure()
    assert sm._figure_title(fig) is None


def test_figure_title_returns_title_text():
    fig = go.Figure()
    fig.update_layout(title="Virus distribution")
    assert sm._figure_title(fig) == "Virus distribution"


# ==================== _table_result ====================

def test_table_result_reports_total_rows_and_columns():
    df = pd.DataFrame({"species": ["a", "b", "c"], "count": [1, 2, 3]})
    result = sm._table_result(df, preview_rows=50)

    assert result["success"] is True
    assert "3 rows" in result["content"]
    artifact = result["artifacts"][0]
    assert artifact["type"] == "table"
    assert artifact["total_rows"] == 3
    assert artifact["columns"] == ["species", "count"]
    assert len(artifact["rows"]) == 3


def test_table_result_singular_row_wording():
    df = pd.DataFrame({"species": ["a"]})
    result = sm._table_result(df, preview_rows=50)
    assert "1 row." in result["content"]


def test_table_result_truncates_preview_and_mentions_remaining_rows():
    df = pd.DataFrame({"n": list(range(10))})
    result = sm._table_result(df, preview_rows=3)

    assert "10 rows" in result["content"]
    assert "and 7 more rows" in result["content"]
    # the table artifact itself still only carries the preview rows
    assert len(result["artifacts"][0]["rows"]) == 3
    assert result["artifacts"][0]["total_rows"] == 10


# ==================== query_host_sql input validation ====================
# These checks all happen before ctx.host_con is touched, so they're safe to
# exercise without a live S3/DuckDB connection.

def test_query_host_sql_rejects_non_select_statements():
    result = sm.query_host_sql("DROP TABLE host")
    assert result["success"] is False
    assert "only SELECT statements are allowed" in result["content"]


@pytest.mark.parametrize("keyword", ["INSERT", "UPDATE", "DELETE", "ALTER", "PRAGMA"])
def test_query_host_sql_rejects_forbidden_keywords_even_inside_a_select(keyword):
    sql = f"SELECT 1; {keyword} host SET x=1"
    result = sm.query_host_sql(sql)
    assert result["success"] is False
    assert "forbidden keyword" in result["content"]


def test_query_host_sql_rejects_bare_select_star():
    result = sm.query_host_sql("SELECT * FROM host WHERE COUNTRY = 'FR'")
    assert result["success"] is False
    assert "SELECT *" in result["content"]


def test_query_host_sql_allows_explicit_column_select_star_check_only_blocks_bare_star():
    # sanity check that the bare-star regex doesn't false-positive on a
    # column named e.g. "star_rating" or similar edge case with explicit cols
    sql = "SELECT VIRUS_SPECIES FROM host WHERE COUNTRY = 'FR' LIMIT 10"
    result = sm.query_host_sql(sql)
    # this will go on to hit ctx.host_con (not connected in tests), so we
    # only assert it got past the * validation, not that it fully succeeds
    assert "SELECT *" not in (result.get("content") or "")
