#!/bin/sh
# Entrypoint for the app container. Runs as root only long enough to fix the
# ownership of the host bind-mounts (./auth_data, ./logs) — whose owner on the
# host may not match the container's app user (uid 1000), which is what caused
# "PermissionError: ... 'auth_data/.streamlit_auth.yaml'". Then it drops
# privileges and runs Streamlit as the non-root app user via gosu.
set -e

chown -R app:app /app/auth_data /app/logs 2>/dev/null || true

exec gosu app streamlit run app.py --server.address=0.0.0.0 --server.port=8501
