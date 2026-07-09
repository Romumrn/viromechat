#!/bin/sh
set -e

# Start the MCP server in the background — it loads the taxonomy CSV and
# connects to S3, which takes a few seconds.
python server_mcp.py &

# Wait for it to be reachable before starting the Streamlit app, which
# checks MCP connectivity once on boot and refuses to proceed otherwise.
for i in $(seq 1 30); do
    if curl -s -o /dev/null http://localhost:8000/mcp; then
        break
    fi
    sleep 1
done

exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501
