# Quick Start Testing Guide

This guide provides step-by-step instructions for testing the KineticGraph-Vectra Hybrid RAG System.

## 🚀 Prerequisites

- Docker and Docker Compose installed
- OpenAI API Key
- At least 8GB RAM available
- curl or any HTTP client
- Python with pytest (for unit tests)

## 📋 Testing Steps

### 1. Start the Application

```bash
# Clone and navigate to the repository
cd /Users/sendils/work/repo/kinetic-v/kinegraph-v

# Copy environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use your favorite editor

# Build and start all containers
docker-compose up --build -d

# Check service health
docker-compose ps

# View logs (optional)
docker-compose logs -f app
```

### 2. Verify Service Health

```bash
# Overall health check
curl http://localhost:8000/health

# Liveness probe
curl http://localhost:8000/health/liveness

# Readiness probe
curl http://localhost:8000/health/readiness
```

### 3. Access Individual Services

- **FastAPI:** http://localhost:8000
- **Neo4j Browser:** http://localhost:7474 (user: `neo4j`, password: see `.env`)
- **ChromaDB:** http://localhost:8001
- **API Documentation:** http://localhost:8000/docs

### 4. Test Document Ingestion

```bash
# Create a test PDF (if you don't have one)
# Then upload it to the system
curl -X POST "http://localhost:8000/api/v1/ingest/document" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.pdf" \
  -F 'metadata={"author":"Test User","category":"test"}'

# Check the processing status (replace with actual task_id)
curl http://localhost:8000/api/v1/ingest/task/abc123
```

Expected Response:
```json
{
  "task_id": "abc123",
  "status": "PENDING",
  "message": "Document 'test.pdf' queued for processing"
}
```

### 5. Test Query Endpoints

#### 5.1 Hybrid Search (Vector + Graph)
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings about climate change?",
    "mode": "hybrid",
    "max_results": 10
  }'
```

#### 5.2 Vector-Only Search
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "climate change impact",
    "mode": "vector",
    "max_results": 5
  }'
```

#### 5.3 Graph-Only Search
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show relationships between climate and weather",
    "mode": "graph",
    "max_results": 5
  }'
```

Expected Response:
```json
{
  "query": "climate change impact",
  "mode": "hybrid",
  "results": [
    {
      "content": "Climate change has significant impacts...",
      "metadata": {
        "document_id": "doc_abc123",
        "file_name": "climate_report.pdf",
        "chunk_index": 3
      },
      "score": 0.92,
      "source": "vector"
    }
  ],
  "total_results": 10,
  "execution_time_ms": 234.56
}
```

### 6. Run Python Unit Tests

```bash
# Install test dependencies (already in requirements.txt)
pip install pytest pytest-asyncio

# Run all tests
pytest tests/ -v

# Run specific test file (if exists)
pytest tests/test_health.py -v

# Run with coverage
pytest tests/ -v --cov=app --cov=core --cov=services
```

### 7. Test Performance and Scaling

```bash
# Scale workers for testing
docker-compose up -d --scale worker=5

# Test load with multiple concurrent requests
for i in {1..10}; do
  curl -X POST "http://localhost:8000/api/v1/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "test query", "mode": "hybrid", "max_results": 5}' &
done
wait
```

## 🛠️ Service Management

### Docker Compose Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f [service_name]

# Restart a service
docker-compose restart app

# Rebuild after code changes
docker-compose up --build app

# Scale workers
docker-compose up -d --scale worker=5
```

### Service Ports

| Service | Internal Port | External Port |
|---------|---------------|---------------|
| FastAPI | 8000 | 8000 |
| ChromaDB | 8000 | 8001 |
| Neo4j HTTP | 7474 | 7474 |
| Neo4j Bolt | 7687 | 7687 |
| Redis | 6379 | 6379 |

## 🔍 Common Test Scenarios

### Scenario 1: Basic Health Check
1. Start services
2. Verify health endpoint returns 200 OK
3. Check all individual services are accessible

### Scenario 2: Document Processing Pipeline
1. Upload a PDF document
2. Monitor task status
3. Verify document is processed and indexed
4. Query the uploaded content

### Scenario 3: Query Performance
1. Test different query modes (vector, graph, hybrid)
2. Compare response times
3. Validate result quality and relevance

### Scenario 4: Error Handling
1. Test with invalid file formats
2. Test with missing parameters
3. Test service failures

## 🐛 Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   # Check logs
   docker-compose logs -f
   
   # Check port conflicts
   lsof -i :8000
   ```

2. **OpenAI API errors**
   - Verify API key in `.env` file
   - Check API quota and billing

3. **Memory issues**
   - Ensure at least 8GB RAM available
   - Monitor container resource usage: `docker stats`

4. **Test failures**
   ```bash
   # Check test dependencies
   pip install -r requirements.txt
   
   # Verify test environment
   pytest --version
   ```

## 📊 Expected Performance Metrics

- **Health Check Response:** < 100ms
- **Simple Query:** < 500ms
- **Document Processing:** 30-60 seconds (depending on size)
- **Concurrent Queries:** 10+ requests simultaneously

## 📝 Test Checklist

- [ ] All services start successfully
- [ ] Health endpoints return 200 OK
- [ ] Document ingestion works
- [ ] Query endpoints return expected results
- [ ] Different query modes work correctly
- [ ] API documentation is accessible
- [ ] Unit tests pass
- [ ] Performance meets expectations
- [ ] Error handling works as expected

## 🔧 Development Testing

For local development without Docker:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run Celery worker (separate terminal)
celery -A workers.celery_app worker --loglevel=info
```

## 📞 Support

For issues and questions:
- Check the main documentation: [README.md](./README.md)
- Open an issue on GitHub
- Review API docs at http://localhost:8000/docs