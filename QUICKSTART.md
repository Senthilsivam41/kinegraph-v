# Quick Start Guide

## Getting Started in 5 Minutes

This guide will help you get KineticGraph-Vectra up and running quickly.

### Step 1: Prerequisites Check

```bash
# Check Docker installation
docker --version
docker-compose --version

# Ensure you have at least 8GB RAM available
docker system info | grep "Total Memory"
```

### Step 2: Environment Setup

```bash
# Navigate to project directory
cd /Users/sendils/work/repo/kinetic-v/kinegraph-v

# Create environment file
cp .env.example .env

# Edit .env file - IMPORTANT: Add your OpenAI API key
# Use nano, vim, or any editor
nano .env
```

**Required Change in .env:**
```bash
OPENAI_API_KEY=sk-your-actual-key-here
```

### Step 3: Start Services

```bash
# Build and start all containers (this may take 5-10 minutes the first time)
docker-compose up --build -d

# Watch the startup process
docker-compose logs -f
```

Wait until you see:
- `kinetic-api    | Uvicorn running on http://0.0.0.0:8000`
- `kinetic-worker | celery@... ready`

### Step 4: Verify Installation

```bash
# Check all services are running
docker-compose ps

# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "services": {
#     "api": true,
#     "chroma": true,
#     "neo4j": true,
#     "redis": true
#   },
#   "version": "1.0.0"
# }
```

### Step 5: Access the UI

Open your browser and visit:

- **API Documentation:** http://localhost:8000/docs
- **Neo4j Browser:** http://localhost:7474
  - Username: `neo4j`
  - Password: (check your `.env` file)

### Step 6: Ingest Your First Document

Using the **Swagger UI** at http://localhost:8000/docs:

1. Navigate to **POST /api/v1/ingest/document**
2. Click "Try it out"
3. Upload a PDF file
4. Click "Execute"
5. Copy the `task_id` from the response

Monitor progress:
```bash
# Replace abc123 with your actual task_id
curl http://localhost:8000/api/v1/ingest/task/abc123
```

### Step 7: Query the System

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Your question here",
    "mode": "hybrid",
    "max_results": 5
  }'
```

---

## Common Issues

### Issue: Port Already in Use

```bash
# Check what's using the port
lsof -i :8000

# Stop the conflicting service or change ports in docker-compose.yml
```

### Issue: Out of Memory

```bash
# Increase Docker memory limit in Docker Desktop settings
# Minimum recommended: 8GB

# Or reduce service replicas temporarily
docker-compose up -d --scale worker=1
```

### Issue: ChromaDB Connection Failed

```bash
# Restart ChromaDB service
docker-compose restart chroma

# Check logs
docker-compose logs chroma
```

### Issue: Neo4j Authentication Failed

```bash
# Reset Neo4j password
docker-compose down
docker volume rm kinegraph-v_neo4j_data
docker-compose up -d
```

---

## Next Steps

1. **Explore the API:** Visit http://localhost:8000/docs
2. **Read the Full Documentation:** See [README.md](README.md)
3. **Customize Configuration:** Edit `.env` and `docker-compose.yml`
4. **Deploy to Production:** Follow the [Kubernetes Guide](KUBERNETES.md)

---

## Stopping the System

```bash
# Stop all services
docker-compose down

# Stop and remove all data
docker-compose down -v
```

---

Need help? Check the [README.md](README.md) for detailed documentation.
