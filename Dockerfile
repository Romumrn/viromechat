# Utilise une image Python légère et officielle
FROM python:3.11-slim

# Définit le répertoire de travail
WORKDIR /app

# Installe les dépendances système nécessaires (curl : healthcheck du MCP dans entrypoint.sh)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installe les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Crée un utilisateur non-root pour des raisons de sécurité
RUN addgroup --gid 1000 app && adduser --uid 1000 --gid 1000 --system app

# Copie le reste de l'application (secrets exclus via .dockerignore —
# passer .env.app / .env.mcp au runtime via --env-file ou -e)
COPY --chown=app:app . .
RUN chmod +x entrypoint.sh

# 8501 : Streamlit (app.py) — 8000 : serveur MCP (server_mcp.py)
EXPOSE 8501 8000

# Passe à l'utilisateur non-root
USER app

# Lance les deux process : server_mcp.py en arrière-plan, puis streamlit
CMD ["./entrypoint.sh"]
