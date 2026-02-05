# API Documentation

## Base URL

```
http://localhost:8000
```

---

## Endpoints

### Health Check

#### `GET /health`

Check the health of all services.

**Response:**
```json
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

#### `GET /health/liveness`

Kubernetes liveness probe.

**Response:**
```json
{
  "status": "alive"
}
```

#### `GET /health/readiness`

Kubernetes readiness probe.

**Response:**
```json
{
  "status": "ready"
}
```

---

### Document Ingestion

#### `POST /api/v1/ingest/document`

Upload and process a PDF document.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/ingest/document" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F 'metadata={"author":"John Doe","category":"research"}'
```

**Parameters:**
- `file` (required): PDF file
- `metadata` (optional): JSON string with document metadata

**Response:**
```json
{
  "task_id": "abc123-def456",
  "status": "PENDING",
  "message": "Document 'document.pdf' queued for processing"
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid file type or metadata
- `500`: Server error

---

#### `GET /api/v1/ingest/task/{task_id}`

Check the status of a document processing task.

**Request:**
```bash
curl http://localhost:8000/api/v1/ingest/task/abc123-def456
```

**Response (Processing):**
```json
{
  "task_id": "abc123-def456",
  "status": "PROGRESS",
  "result": null,
  "error": null
}
```

**Response (Success):**
```json
{
  "task_id": "abc123-def456",
  "status": "SUCCESS",
  "result": {
    "document_id": "doc_xyz789",
    "file_name": "document.pdf",
    "total_chunks": 42,
    "entities_count": 15,
    "relationships_count": 8,
    "status": "success"
  },
  "error": null
}
```

**Response (Failure):**
```json
{
  "task_id": "abc123-def456",
  "status": "FAILURE",
  "result": null,
  "error": "Error message here"
}
```

**Task Statuses:**
- `PENDING`: Task queued but not started
- `STARTED`: Task is running
- `PROGRESS`: Task in progress (with updates)
- `SUCCESS`: Task completed successfully
- `FAILURE`: Task failed
- `RETRY`: Task being retried

---

### Query System

#### `POST /api/v1/query`

Query the hybrid RAG system.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings?",
    "mode": "hybrid",
    "max_results": 10,
    "filters": {"category": "research"}
  }'
```

**Body Parameters:**
- `query` (required): Natural language query string
- `mode` (optional): Query mode - `vector`, `graph`, or `hybrid` (default: `hybrid`)
- `max_results` (optional): Maximum results to return (1-100, default: 10)
- `filters` (optional): Metadata filters as key-value pairs

**Response:**
```json
{
  "query": "What are the main findings?",
  "mode": "hybrid",
  "results": [
    {
      "content": "The main findings indicate...",
      "metadata": {
        "document_id": "doc_xyz789",
        "file_name": "document.pdf",
        "chunk_index": 5,
        "total_chunks": 42,
        "author": "John Doe",
        "category": "research"
      },
      "score": 0.89,
      "source": "vector"
    }
  ],
  "total_results": 10,
  "execution_time_ms": 234.56
}
```

**Query Modes:**

1. **Vector Mode (`vector`):**
   - Semantic similarity search only
   - Fast, good for general queries
   - Uses ChromaDB embeddings

2. **Graph Mode (`graph`):**
   - Cypher query generation
   - Good for relationship queries
   - Uses Neo4j graph traversal

3. **Hybrid Mode (`hybrid`):**
   - Combines both vector and graph
   - Uses Reciprocal Rank Fusion (RRF)
   - Best for complex queries
   - Recommended default

**Status Codes:**
- `200`: Success
- `400`: Invalid request
- `500`: Server error

---

#### `GET /api/v1/query/test`

Test endpoint to verify query system is operational.

**Response:**
```json
{
  "status": "operational",
  "chroma_connected": true,
  "neo4j_connected": true
}
```

---

## Query Examples

### Example 1: Simple Semantic Search

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "mode": "vector",
    "max_results": 5
  }'
```

### Example 2: Relationship Query

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show relationships between neural networks and deep learning",
    "mode": "graph",
    "max_results": 5
  }'
```

### Example 3: Hybrid Search with Filters

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the impact of climate change?",
    "mode": "hybrid",
    "max_results": 10,
    "filters": {
      "author": "John Doe",
      "category": "research"
    }
  }'
```

---

## Response Schemas

### DocumentChunk

```json
{
  "content": "string",
  "metadata": {
    "document_id": "string",
    "file_name": "string",
    "chunk_index": 0,
    "total_chunks": 0,
    "custom_field": "any"
  },
  "score": 0.0,
  "source": "vector | graph"
}
```

---

## Error Handling

All endpoints return standard error responses:

```json
{
  "detail": "Error message description"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error
- `503`: Service Unavailable

---

## Rate Limiting

Currently, there is no rate limiting implemented. For production deployments, consider implementing rate limiting at the API gateway or load balancer level.

---

## Authentication

This implementation does not include authentication. For production use, implement:

- **JWT tokens** for API authentication
- **API keys** for service-to-service communication
- **OAuth 2.0** for user authentication

Example with JWT (to be implemented):

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

---

## Swagger UI

Interactive API documentation is available at:

**http://localhost:8000/docs**

This provides:
- Interactive API testing
- Request/response examples
- Schema documentation
- Try-it-out functionality

---

## ReDoc

Alternative API documentation is available at:

**http://localhost:8000/redoc**

---

## WebSocket Support

WebSocket endpoints are not currently implemented. For real-time streaming responses, this could be added in future versions.

---

## Batch Operations

For batch document ingestion or queries, iterate through the single-item endpoints or implement batch endpoints as needed.

Example batch ingestion script:

```python
import requests
from pathlib import Path

api_url = "http://localhost:8000/api/v1/ingest/document"

for pdf_file in Path("./documents").glob("*.pdf"):
    with open(pdf_file, "rb") as f:
        files = {"file": f}
        response = requests.post(api_url, files=files)
        print(f"Uploaded {pdf_file.name}: {response.json()}")
```

---

## SDK Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Ingest document
def ingest_document(file_path, metadata=None):
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {"metadata": str(metadata)} if metadata else {}
        response = requests.post(f"{BASE_URL}/api/v1/ingest/document", files=files, data=data)
        return response.json()

# Query system
def query(question, mode="hybrid", max_results=10):
    payload = {
        "query": question,
        "mode": mode,
        "max_results": max_results
    }
    response = requests.post(f"{BASE_URL}/api/v1/query", json=payload)
    return response.json()

# Example usage
task = ingest_document("document.pdf", {"author": "John"})
print(f"Task ID: {task['task_id']}")

results = query("What are the main findings?")
for result in results["results"]:
    print(result["content"][:100])
```

### JavaScript/Node.js

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const BASE_URL = 'http://localhost:8000';

// Ingest document
async function ingestDocument(filePath, metadata = {}) {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath));
  form.append('metadata', JSON.stringify(metadata));
  
  const response = await axios.post(
    `${BASE_URL}/api/v1/ingest/document`,
    form,
    { headers: form.getHeaders() }
  );
  return response.data;
}

// Query system
async function query(question, mode = 'hybrid', maxResults = 10) {
  const response = await axios.post(`${BASE_URL}/api/v1/query`, {
    query: question,
    mode: mode,
    max_results: maxResults
  });
  return response.data;
}

// Example usage
(async () => {
  const task = await ingestDocument('document.pdf', { author: 'John' });
  console.log(`Task ID: ${task.task_id}`);
  
  const results = await query('What are the main findings?');
  results.results.forEach(r => console.log(r.content.substring(0, 100)));
})();
```

---

## Version History

- **v1.0.0** (2026-02-05): Initial release
  - Core RAG functionality
  - Vector and graph search
  - RRF fusion
  - Document ingestion

---

For more information, see the [README.md](README.md) or visit http://localhost:8000/docs
