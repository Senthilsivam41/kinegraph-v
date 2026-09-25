#!/usr/bin/env bash

# Backward-compatible entry point; keep startup logic in one maintained script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/scripts/start_services.sh" "$@"
