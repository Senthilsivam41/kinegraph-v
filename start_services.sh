#!/bin/bash

# Exit immediately if any step fails
set -e

# Clear log files
rm -f celery.log uvicorn.log

echo "🚀 Starting KineticGraph-Vectra Services..."

# 1. Start Docker Databases
echo "📦 Starting Docker containers (Redis, ChromaDB, Neo4j)..."
docker compose -f infra/docker-compose.yml up -d redis chroma neo4j

# 2. Wait for databases to be healthy
echo "⏳ Waiting for ChromaDB to initialize on port 8001..."
until curl -s http://localhost:8001/api/v1/heartbeat > /dev/null; do
    sleep 1
done
echo "✅ ChromaDB is online!"

echo "⏳ Waiting for Neo4j to initialize on port 7474..."
until curl -s http://localhost:7474 > /dev/null; do
    sleep 1
done
echo "✅ Neo4j is online!"

# 3. Setup cleanup handler to cleanly shutdown background processes on exit
cleanup() {
    echo -e "\n🛑 Stopping background servers..."
    # Kill background jobs spawned by this script
    kill $(jobs -p) 2>/dev/null || true
    echo "👋 Cleanup complete."
}
trap cleanup EXIT

# 4. Start Celery worker in the background
echo "⚙️ Starting Celery worker in background (logging to celery.log)..."
./venv311/bin/celery -A backend.workers.celery_app worker --loglevel=info > celery.log 2>&1 &

# 5. Start FastAPI Backend in the background
echo "🔥 Starting FastAPI Backend on http://localhost:8000 (logging to uvicorn.log)..."
./venv311/bin/uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &

# 6. Wait for FastAPI to start responding
echo "⏳ Waiting for FastAPI API to respond..."
until curl -s http://localhost:8000/health > /dev/null; do
    sleep 1
done
echo "✅ FastAPI Backend is ready!"

# 7. Start Frontend Server in the foreground
echo "🌐 Starting Frontend Chat UI on http://localhost:8080..."
python3 frontend/serve.py
