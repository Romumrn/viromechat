# Streamlit client (app.py) — talks to Albert API and to server_mcp.py over
# MCP/HTTP. See Dockerfile.mcp for the other half of the system.

FROM python:3.11-slim

WORKDIR /app

# gosu : bascule root -> app dans l'entrypoint (voir entrypoint-app.sh) après
# avoir corrigé les permissions des bind-mounts. Tiny, dispo dans les dépôts
# Debian bookworm dont dérive python:3.11-slim.
RUN apt-get update && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/*

# Installe les dépendances Python (app.py uniquement — voir requirements-app.txt)
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Crée un utilisateur non-root pour des raisons de sécurité, avec un vrai
# HOME (writable) : "adduser --system" seul donne HOME=/nonexistent, et
# Streamlit veut pouvoir écrire sous ~/.streamlit (config/credentials cache).
RUN addgroup --gid 1000 app \
    && adduser --uid 1000 --gid 1000 --system --home /home/app --shell /usr/sbin/nologin app \
    && mkdir -p /home/app && chown app:app /home/app
ENV HOME=/home/app

COPY --chown=app:app app.py config.py prompt.py logging_utils.py ./
COPY entrypoint-app.sh /usr/local/bin/entrypoint-app.sh
RUN chmod +x /usr/local/bin/entrypoint-app.sh

# Répertoires écrits au runtime : logs de chat / historiques utilisateurs
# (./logs) et le fichier de comptes locaux (./auth_data). Tous deux sont des
# bind-mounts host au runtime (voir docker-compose.yml) ; l'entrypoint
# re-chown ces points de montage au démarrage pour que l'utilisateur app
# puisse y écrire quel que soit le propriétaire côté hôte.
RUN mkdir -p logs auth_data && chown app:app /app logs auth_data

EXPOSE 8501

# L'entrypoint démarre en root (pour le chown), puis exécute Streamlit en tant
# qu'utilisateur app via gosu — donc PAS de "USER app" ici.
ENTRYPOINT ["/usr/local/bin/entrypoint-app.sh"]
