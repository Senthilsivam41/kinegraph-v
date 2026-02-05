# KineticGraph-Vectra

<div align="center">

**A Production-Ready Hybrid RAG System**

*Combining Vector Search (ChromaDB) and Graph Reasoning (Neo4j) with LangGraph Orchestration*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-326CE5.svg)](https://kubernetes.io)

</div>

---

## 🎯 Overview

**KineticGraph-Vectra** is a scalable, container-first hybrid RAG (Retrieval-Augmented Generation) system that orchestrates searches across:

- **Vector Database (ChromaDB):** Fast semantic similarity search on document embeddings
- **Graph Database (Neo4j):** Deep relational reasoning with entities and relationships
- **Fusion Layer (RRF):** Reciprocal Rank Fusion to intelligently merge results

The system uses **LangGraph** to orchestrate complex query workflows and **Celery** for asynchronous document processing.

---

## 🏗️ Architecture

### System Components

```
┌─────────────────┐
│   FastAPI App   │  ← REST API for queries and ingestion
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌───▼───┐
│Vector│  │ Graph │
│Agent │  │ Agent │
└───┬──┘  └───┬───┘
    │         │
┌───▼─────────▼───┐
│  Fusion Node    │  ← Reciprocal Rank Fusion (RRF)
│      (RRF)      │
└─────────────────┘
         │
    ┌────▼────┐
    │ Results │
    └─────────┘
```

```
<img width="1024" height="962" alt="image" src="https://github.com/user-attachments/assets/810d3540-c0e8-43c8-b223-dd0eb00f7376" />
```

### Infrastructure Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Framework** | FastAPI | Asynchronous REST API |
| **Orchestration** | LangGraph | Query workflow management |
| **Vector DB** | ChromaDB | Semantic search |
| **Graph DB** | Neo4j | Entity relationships |
| **Task Queue** | Celery + Redis | Async document processing |
| **Containerization** | Docker Compose | Local development |
| **Orchestration** | Kubernetes | Production deployment |

---

## 🚀 Quick Start

### Prerequisites

- **Docker** and **Docker Compose** installed
- **OpenAI API Key**
- At least **8GB RAM** available for containers

### 1. Clone and Setup

```bash
# Clone the repository
cd /Users/sendils/work/repo/kinetic-v/kinegraph-v

# Copy environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use your favorite editor
```

### 2. Start All Services

```bash
# Build and start all containers
docker-compose up --build -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f app
```

### 3. Verify System Health

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs
```

### 4. Access Individual Services

- **FastAPI:** http://localhost:8000
- **Neo4j Browser:** http://localhost:7474 (user: `neo4j`, password: see `.env`)
- **ChromaDB:** http://localhost:8001

---

## 📚 Usage

### Document Ingestion

Upload a PDF document for processing:

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/document" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf" \
  -F 'metadata={"author":"John Doe","category":"research"}'
```

**Response:**
```json
{
  "task_id": "abc123",
  "status": "PENDING",
  "message": "Document 'document.pdf' queued for processing"
}
```

### Check Processing Status

```bash
curl http://localhost:8000/api/v1/ingest/task/abc123
```

### Query the System

#### Hybrid Search (Vector + Graph)

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings about climate change?",
    "mode": "hybrid",
    "max_results": 10
  }'
```

#### Vector-Only Search

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "climate change impact",
    "mode": "vector",
    "max_results": 5
  }'
```

#### Graph-Only Search

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show relationships between climate and weather",
    "mode": "graph",
    "max_results": 5
  }'
```

**Response Example:**
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

---

## 🏗️ Project Structure

```
kinegraph-v/
├── app/                        # FastAPI application
│   ├── main.py                 # Application entry point
│   ├── models.py               # Pydantic models
│   └── api/
│       └── routes/
│           ├── health.py       # Health check endpoints
│           ├── ingest.py       # Document ingestion
│           └── query.py        # Query endpoints
├── core/                       # Core logic
│   ├── config.py               # Configuration management
│   ├── langgraph_workflow.py  # LangGraph orchestration
│   └── rrf.py                  # Reciprocal Rank Fusion
├── services/                   # Database services
│   ├── chroma_service.py       # ChromaDB client
│   └── neo4j_service.py        # Neo4j client
├── workers/                    # Celery workers
│   ├── celery_app.py           # Celery configuration
│   ├── tasks.py                # Processing tasks
│   └── document_processor.py   # Document utilities
├── k8s/                        # Kubernetes manifests
│   ├── configmap.yaml
│   ├── api-deployment.yaml
│   ├── worker-deployment.yaml
│   ├── chroma-deployment.yaml
│   ├── neo4j-statefulset.yaml
│   ├── redis-deployment.yaml
│   └── ingress.yaml
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Container image
├── requirements.txt            # Python dependencies
└── .env.example                # Environment template
```

---

## 🐳 Docker Services

### Service Ports

| Service | Internal Port | External Port |
|---------|---------------|---------------|
| FastAPI | 8000 | 8000 |
| ChromaDB | 8000 | 8001 |
| Neo4j HTTP | 7474 | 7474 |
| Neo4j Bolt | 7687 | 7687 |
| Redis | 6379 | 6379 |

### Managing Services

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

---

## ☸️ Kubernetes Deployment

### Prerequisites

- **kubectl** configured
- **Kubernetes cluster** (local or cloud)
- **Docker image** built and pushed to registry

### 1. Build and Push Docker Image

```bash
# Build image
docker build -t your-registry/kinetic-vectra:latest .

# Push to registry
docker push your-registry/kinetic-vectra:latest
```

### 2. Update Image References

Edit `k8s/*.yaml` files and replace `kinetic-vectra:latest` with your image.

### 3. Deploy to Kubernetes

```bash
# Create namespace and configmaps
kubectl apply -f k8s/configmap.yaml

# Deploy databases
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/chroma-deployment.yaml
kubectl apply -f k8s/neo4j-statefulset.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=redis -n kinetic-v --timeout=300s
kubectl wait --for=condition=ready pod -l app=chroma -n kinetic-v --timeout=300s
kubectl wait --for=condition=ready pod -l app=neo4j -n kinetic-v --timeout=300s

# Deploy application and workers
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/worker-deployment.yaml

# (Optional) Deploy ingress
kubectl apply -f k8s/ingress.yaml
```

### 4. Access the Application

```bash
# Port forward to access locally
kubectl port-forward -n kinetic-v svc/kinetic-api-service 8000:8000

# Or get LoadBalancer IP
kubectl get svc -n kinetic-v kinetic-api-service
```

### 5. Monitor Deployments

```bash
# Check pods
kubectl get pods -n kinetic-v

# View logs
kubectl logs -n kinetic-v -l app=kinetic-api --tail=100 -f

# Check HPA status
kubectl get hpa -n kinetic-v
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Application
APP_NAME=KineticGraph-Vectra
ENVIRONMENT=development
LOG_LEVEL=INFO

# OpenAI (Required)
OPENAI_API_KEY=sk-your-key-here

# ChromaDB
CHROMA_HOST=chroma
CHROMA_PORT=8000
CHROMA_COLLECTION_NAME=kinetic_vectors

# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password

# Redis & Celery
REDIS_HOST=redis
REDIS_PORT=6379
CELERY_BROKER_URL=redis://redis:6379/0

# RRF Settings
RRF_K=60
MAX_RESULTS=10
```

---

## 🧪 Testing

### Test Health Endpoints

```bash
# Overall health
curl http://localhost:8000/health

# Liveness probe
curl http://localhost:8000/health/liveness

# Readiness probe
curl http://localhost:8000/health/readiness
```

### Test Document Processing

```bash
# Create a test PDF and upload
curl -X POST "http://localhost:8000/api/v1/ingest/document" \
  -F "file=@test.pdf"
```

### Run Python Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v
```

---

## 📊 Monitoring and Scaling

### Horizontal Pod Autoscaling (HPA)

The system includes HPA configurations for both API and workers:

```bash
# Check autoscaler status
kubectl get hpa -n kinetic-v

# Manually scale
kubectl scale deployment kinetic-worker -n kinetic-v --replicas=10
```

### Resource Limits

Each component has defined resource requests and limits:

- **API Pods:** 512Mi-1Gi RAM, 500m-1000m CPU
- **Worker Pods:** 1Gi-2Gi RAM, 1000m-2000m CPU
- **ChromaDB:** 1Gi-2Gi RAM, 500m-1000m CPU
- **Neo4j:** 2Gi-4Gi RAM, 1000m-2000m CPU

---

## 🔐 Security Considerations

### Production Checklist

- [ ] Change default Neo4j password
- [ ] Use secrets management (e.g., Kubernetes Secrets, HashiCorp Vault)
- [ ] Enable TLS/SSL for all services
- [ ] Implement API authentication (JWT, OAuth)
- [ ] Configure CORS appropriately
- [ ] Set up network policies in Kubernetes
- [ ] Enable audit logging
- [ ] Regular security updates for base images

---

## 🛠️ Development

### Local Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run Celery worker
celery -A workers.celery_app worker --loglevel=info
```

### Code Formatting

```bash
# Install dev tools
pip install black isort flake8

# Format code
black .
isort .

# Lint
flake8 .
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **FastAPI** for the excellent async framework
- **LangChain/LangGraph** for orchestration capabilities
- **ChromaDB** for vector storage
- **Neo4j** for graph database
- **Celery** for distributed task processing

---

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the [documentation](./docs/)
- Review API docs at `/docs`

---

## 🗺️ Roadmap

### Current Features ✅
- [x] Hybrid RAG with Vector + Graph
- [x] Reciprocal Rank Fusion
- [x] Async document processing
- [x] Docker Compose setup
- [x] Kubernetes manifests
- [x] Horizontal Pod Autoscaling

### Future Enhancements 🚀
- [ ] Multi-modal support (images, tables)
- [ ] Advanced caching layer
- [ ] Query result explanations
- [ ] Feedback loop for quality improvement
- [ ] Integration with more LLM providers
- [ ] Real-time streaming responses
- [ ] Admin dashboard
- [ ] Advanced security features

---

**Built with ❤️ for scalable, production-ready RAG systems**
