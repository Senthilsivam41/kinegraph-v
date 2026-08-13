#!/usr/bin/env bash

# Start the Kinegraph Docker services from any working directory.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
COMPOSE_FILE="$ROOT_DIR/infra/docker-compose.yml"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing $ENV_FILE. Copy .env.example to .env and configure it first." >&2
    exit 1
fi

# Compose performs interpolation before it starts any service. Match the
# application's minimum length so startup and runtime validation agree.
NEO4J_PASSWORD_VALUE="$(sed -n 's/^[[:space:]]*NEO4J_PASSWORD=[[:space:]]*//p' "$ENV_FILE" | head -n 1)"
if [[ "$NEO4J_PASSWORD_VALUE" == '"'*'"' ]]; then
    NEO4J_PASSWORD_VALUE="${NEO4J_PASSWORD_VALUE:1:${#NEO4J_PASSWORD_VALUE}-2}"
fi
if (( ${#NEO4J_PASSWORD_VALUE} < 16 )); then
    echo "NEO4J_PASSWORD in $ENV_FILE must be at least 16 characters." >&2
    exit 1
fi

if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
else
    echo "Docker Compose v2 or docker-compose is required." >&2
    exit 1
fi

exec "${COMPOSE[@]}" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up --build -d "$@"
