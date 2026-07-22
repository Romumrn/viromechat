import os

import pytest

from config import load_env_file


@pytest.fixture(autouse=True)
def _clean_env():
    # make sure the keys we use in this file don't leak between tests
    keys = ["FOO", "BAR", "ALREADY_SET"]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v


def test_load_env_file_sets_variables(tmp_path):
    env_file = tmp_path / ".env.test"
    env_file.write_text("FOO=hello\nBAR=world\n")

    load_env_file(str(env_file))

    assert os.environ["FOO"] == "hello"
    assert os.environ["BAR"] == "world"


def test_load_env_file_ignores_comments_and_blank_lines(tmp_path):
    env_file = tmp_path / ".env.test"
    env_file.write_text("# this is a comment\n\nFOO=hello\n   \n# BAR=should_not_be_set\n")

    load_env_file(str(env_file))

    assert os.environ["FOO"] == "hello"
    assert "BAR" not in os.environ


def test_load_env_file_does_not_override_existing_env_var(tmp_path):
    os.environ["ALREADY_SET"] = "from_real_env"
    env_file = tmp_path / ".env.test"
    env_file.write_text("ALREADY_SET=from_dotenv\n")

    load_env_file(str(env_file))

    # real environment always wins over the .env file
    assert os.environ["ALREADY_SET"] == "from_real_env"


def test_load_env_file_strips_whitespace_around_key_and_value(tmp_path):
    env_file = tmp_path / ".env.test"
    env_file.write_text("  FOO  =   hello world  \n")

    load_env_file(str(env_file))

    assert os.environ["FOO"] == "hello world"


def test_load_env_file_missing_file_is_a_noop(tmp_path):
    missing = tmp_path / "does_not_exist.env"

    # should not raise
    load_env_file(str(missing))

    assert "FOO" not in os.environ


def test_load_env_file_ignores_lines_without_equals(tmp_path):
    env_file = tmp_path / ".env.test"
    env_file.write_text("this line has no equals sign\nFOO=hello\n")

    load_env_file(str(env_file))

    assert os.environ["FOO"] == "hello"
