# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] - 2026-07-22

First tagged release.

### Added
- Local account system (`streamlit-authenticator`), replacing the earlier no-auth/captcha setup.
- Per-user persisted chat history, with a `USER_QUERY` log line (tagged with the user's email)
  written independently of session state, so it survives a user clearing their own history.
- Unit test suite (`tests/`, `pytest`) covering the pure helper logic in `app.py` and
  `server_mcp.py`.
- GitHub Actions CI running the test suite on every push and pull request to `main`.

### Changed
- Split runtime into two independent processes/Docker images (`app.py` client, `server_mcp.py`
  FastMCP server), communicating exclusively over MCP/HTTP.
