import logging

from logging_utils import C, ColoredFormatter, PlainFormatter, setup_logger


def _make_record(msg, level=logging.INFO):
    return logging.LogRecord(
        name="test", level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )


def test_plain_formatter_does_not_add_ansi_codes():
    formatter = PlainFormatter("%(levelname)s | %(message)s")
    record = _make_record("Error: something broke")

    output = formatter.format(record)

    assert "\033[" not in output
    assert "Error: something broke" in output


def test_colored_formatter_wraps_known_keyword():
    formatter = ColoredFormatter("%(message)s")
    record = _make_record("TOOL_CALL query_dataframe")

    output = formatter.format(record)

    assert C["BOLD"] + C["BRIGHT_MAGENTA"] + "TOOL_CALL" + C["RESET"] in output


def test_colored_formatter_leaves_levelname_unmodified_after_formatting():
    formatter = ColoredFormatter("%(levelname)s")
    record = _make_record("some message")

    formatter.format(record)

    # format() must restore the original levelname on the record afterwards,
    # otherwise the next handler (e.g. the plain file handler) would also
    # see the colorized version
    assert record.levelname == "INFO"


def test_colored_formatter_highlights_error_prefix():
    formatter = ColoredFormatter("%(message)s")
    record = _make_record("Error: connection refused")

    output = formatter.format(record)

    assert C["BRIGHT_RED"] + C["BOLD"] + "Error:" + C["RESET"] in output


def test_colored_formatter_highlights_call_index():
    formatter = ColoredFormatter("%(message)s")
    record = _make_record("running tool call #3")

    output = formatter.format(record)

    assert f"{C['BRIGHT_YELLOW']}{C['BOLD']}#3{C['RESET']}" in output


def test_setup_logger_creates_log_dir_and_two_handlers(tmp_path):
    log_dir = tmp_path / "logs"

    logger = setup_logger(str(log_dir))

    assert log_dir.exists()
    assert len(logger.handlers) == 2
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers)


def test_setup_logger_called_twice_does_not_duplicate_handlers(tmp_path):
    log_dir = tmp_path / "logs"

    setup_logger(str(log_dir))
    logger = setup_logger(str(log_dir))

    assert len(logger.handlers) == 2
