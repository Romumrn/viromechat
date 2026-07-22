from prompt import build_system_prompt


def test_build_system_prompt_no_dataset_description_uses_placeholder():
    prompt = build_system_prompt()
    assert "(no dataset description was returned by the MCP server)" in prompt


def test_build_system_prompt_empty_string_also_uses_placeholder():
    # empty string is falsy, same as omitting the argument
    prompt = build_system_prompt("")
    assert "(no dataset description was returned by the MCP server)" in prompt


def test_build_system_prompt_includes_given_description_verbatim():
    description = "### Host dataset schema\ncolumn foo: bar"
    prompt = build_system_prompt(description)
    assert description in prompt
    assert "(no dataset description was returned by the MCP server)" not in prompt


def test_build_system_prompt_contains_key_rules():
    prompt = build_system_prompt()
    # a couple of load-bearing rules that the rest of the app depends on
    assert "NEVER invent species, families, counts, coordinates" in prompt
    assert "create_map" in prompt
    assert "ANSWER IN MARKDOWN" in prompt
