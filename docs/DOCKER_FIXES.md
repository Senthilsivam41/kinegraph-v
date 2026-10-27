# Docker Compose Fixes - February 11, 2026

> **Historical record:** The fixes and command output in this document were
> captured on February 11, 2026. They are not a current smoke-test result.
> For current KineGraph operations from the repository root, explicitly select
> `infra/docker-compose.yml` and the repository `.env` file as shown below.

## Issues Found

1. **Obsolete `version` attribute** - Docker Compose warning
2. **Neo4j healthcheck failing** - Container exiting with code 1
3. **Deprecated Neo4j environment variables** - Using old `dbms.*` naming

## Fixes Applied

### 1. Removed Obsolete Version Line
```diff
- version: '3.8'
-
services:
```

### 2. Fixed Neo4j Healthcheck
The original healthcheck used `neo4j status` which was failing because it detected Neo4j was already running.

**Old healthcheck:**
```yaml
healthcheck:
  test: ["CMD", "neo4j", "status"]
  interval: 30s
  timeout: 10s
  retries: 5
```

**New healthcheck:**
```yaml
healthcheck:
  test: ["CMD-SHELL", "wget --no-verbose --tries=1 --spider http://localhost:7474 || exit 1"]
  interval: 10s
  timeout: 5s
  retries: 5
  start_period: 40s
```

**Changes:**
- Uses HTTP endpoint check instead of internal status command
- Added `start_period: 40s` to give Neo4j time to fully initialize
- More frequent checks (10s interval) for faster startup
- More reliable as it tests the actual HTTP interface

### 3. Updated Neo4j Environment Variables
Changed from deprecated `dbms.*` to current `server.*` naming:

```diff
- NEO4J_dbms_memory_pagecache_size=512M
+ NEO4J_server_memory_pagecache_size=512M

- NEO4J_dbms_memory_heap_initial__size=512M
+ NEO4J_server_memory_heap_initial__size=512M

- NEO4J_dbms_memory_heap_max__size=2G
+ NEO4J_server_memory_heap_max__size=2G
```

## Historical Verification Snapshot

The following output was recorded when these fixes were applied on February 11,
2026; run the current commands below to establish present service health.

```bash
$ docker-compose ps
NAME             STATUS
kinetic-api      Up (healthy)
kinetic-chroma   Up (healthy)
kinetic-neo4j    Up (healthy)
kinetic-redis    Up (healthy)
kinetic-worker   Up (healthy)
```

```bash
$ curl -s http://localhost:8000/health/
{
    "status": "healthy",
    "services": {
        "api": true,
        "chroma": true,
        "neo4j": true,
        "redis": true
    },
    "version": "1.0.0"
}
```

## Service Endpoints in the Historical Snapshot

The February 11 snapshot recorded these service endpoints:

- **API**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health/
- **Neo4j Browser**: http://localhost:7474
- **ChromaDB**: http://localhost:8001
- **Redis**: localhost:6379

## Starting/Stopping Services

Run these commands from the repository root. The root `docker-compose.yml`
defines the separate LiteLLM quickstart stack, not the KineGraph application
services described in this document.

```bash
# Start all services
docker compose --env-file .env -f infra/docker-compose.yml up -d

# Stop all services
docker compose --env-file .env -f infra/docker-compose.yml down

# View logs
docker compose --env-file .env -f infra/docker-compose.yml logs -f

# View specific service logs
docker compose --env-file .env -f infra/docker-compose.yml logs -f app
docker compose --env-file .env -f infra/docker-compose.yml logs -f neo4j

# Check status
docker compose --env-file .env -f infra/docker-compose.yml ps

# Rebuild and restart
docker compose --env-file .env -f infra/docker-compose.yml up -d --build
```

## Troubleshooting

### If containers fail to start:

1. **Check logs:**
   ```bash
   docker compose --env-file .env -f infra/docker-compose.yml logs [service-name]
   ```

2. **Completely reset:**
   ```bash
   docker compose --env-file .env -f infra/docker-compose.yml down -v  # Remove volumes
   docker compose --env-file .env -f infra/docker-compose.yml up -d --build
   ```

3. **Check resource usage:**
   ```bash
   docker stats
   ```

### Health check failures:

- Neo4j takes ~30-40 seconds to fully start (now handled by `start_period`)
- ChromaDB usually starts in ~10-15 seconds
- Redis starts almost immediately

## Next Steps

After the scoped Compose status command confirms a healthy backend, you can:

1. **Start the Chat UI:**
   ```bash
   ./scripts/start-ui.sh
   # Or manually:
   cd frontend && python3 serve.py
   ```

2. **Upload documents:**
   - Use the chat UI at http://localhost:8080
   - Or use the API directly

3. **Query the system:**
   - Ask questions via the chat UI
   - Or use the API endpoints

## Summary

- ✅ Removed obsolete `version` attribute.
- ✅ Fixed the Neo4j healthcheck to use its HTTP endpoint.
- ✅ Updated Neo4j environment variables to current naming.
- ✅ The February 11, 2026 snapshot recorded successful startup.
- ✅ The February 11, 2026 snapshot recorded passing health checks.

Run the scoped Compose commands above before treating the services as ready
today.
