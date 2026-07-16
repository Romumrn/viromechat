# Streamlit client (app.py) — talks to Albert API and to server_mcp.py over
# MCP/HTTP. See Dockerfile.mcp for the other half of the system.

FROM python:3.11-slim

WORKDIR /app

# Installe les dépendances Python (app.py uniquement — voir requirements-app.txt)
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Crée un utilisateur non-root pour des raisons de sécurité
RUN addgroup --gid 1000 app && adduser --uid 1000 --gid 1000 --system app

COPY --chown=app:app app.py config.py prompt.py logging_utils.py ./

# Répertoires écrits au runtime : logs de chat / historiques utilisateurs
# (./logs, monté en volume) et le fichier de comptes locaux (./auth_data,
# monté en volume nommé — voir docker-compose.yml). WORKDIR /app est créé
# par root avant le COPY --chown ci-dessus, qui ne rétroagit pas sur son
# propriétaire : sans ce chown explicite, l'utilisateur non-root ne peut
# créer aucun nouveau fichier/dossier directement sous /app.
RUN mkdir -p logs auth_data && chown app:app /app logs auth_data

EXPOSE 8501

USER app

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
