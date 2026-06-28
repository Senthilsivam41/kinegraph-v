# KineticGraph-Vectra

<div align="center">

**A Production-Ready Hybrid RAG System**

*Combining Vector Search (ChromaDB) and Graph Reasoning (Neo4j) with LangGraph Orchestration*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluated-7c3aed.svg)](https://docs.ragas.io)

</div>

---

## 🎯 Overview

**KineticGraph-Vectra** is a lean, highly optimized hybrid RAG (Retrieval-Augmented Generation) system that coordinates searches across:

- **Vector Database (ChromaDB):** Fast semantic similarity search on document embeddings.
- **Graph Database (Neo4j):** Deep relational reasoning with entities and relationships.
- **Vectorless Search (BM25 Local Cache):** Ultra-fast lexical retrieval directly from disk.
- **Fusion Layer (RRF):** Reciprocal Rank Fusion to intelligently merge results from all active retrieval pathways.

The system uses **LangGraph** to orchestrate complex query workflows and **Celery** for asynchronous document processing.

---

## ✨ Codebase Highlights & Optimization

### 1. Robust RAGAS Evaluation & Diagnostics
- **Live Failure Interception**: Updated `RAGASEvaluator` ([ragas_evaluator.py](file:///Users/sendils/work/repo/kinetic-v/kinegraph-v/eval/ragas_evaluator.py)) to catch evaluation failures, logging detailed warnings showing the exact query and the specific metrics that failed (e.g. returning `NaN` or raising exceptions) while falling back cleanly to keyword heuristics.
- **Concurrent Batch Eval**: Added `evaluate_live_workflow` and `evaluate_live_single` supporting concurrent, rate-limited live workflow evaluations asynchronously, resulting in faster and safer benchmark runs.
- **Model-Agnostic Critic Settings**: Configures separate evaluation LLMs (critic/judge) via the `critic_model` parameter, enabling stable benchmark tracking (e.g., using Claude Haiku) even while testing various generation engines.
- **OpenRouter Compatibility**: Detects OpenRouter environments and automatically configures base URLs accordingly, enabling direct usage of stable and extremely cheap paid OpenRouter endpoints (e.g., `gpt-4o-mini` or `meta-llama/llama-3.3-70b-instruct`) without hitting free-tier congestion or 429 rate-limiting.

### 2. "Ponytail" YAGNI Optimization
- **Purged Over-Engineering**: Cleaned up the repository by deleting speculative Kubernetes manifests, custom telemetry databases, Streamlit dashboard instances, and LangSmith tracing wrappers.
- **Pruned Dependencies & Vulnerabilities**: Removed unused heavy libraries (`streamlit`, `plotly`, `langsmith`, `psycopg2-binary`) from `requirements.txt` to minimize codebase bloat, dependency conflicts, and security vulnerability surface areas.
- **Ingestion Streamlining**: Replaced Celery's billiard-based parallel PDF text extraction pool with a fast, sequential PyMuPDF (fitz) iteration loop inside `document_processor.py`, eliminating process-forking overhead and file-locking bottlenecks.

---

## 🏗️ Architecture

### System Components

```
┌────────────────────────────────────────────────────────────────┐
│                          FastAPI App                           │  ← REST API (query, ingest)
└───────────────────────────────┬────────────────────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │   Intent Router     │  ← Query classification + routing
                     └──┬───┬──────────┬───┘
                        │   │          │
        ┌───────────────┘   │          └────────────────┐
        │                   │                           │
  ┌─────▼─────┐       ┌─────▼──────────┐          ┌─────▼──────┐
  │  Vector   │       │ Parallel Fetch │          │ Vectorless │  ← Local lexical BM25
  │  Agent    │       │ (Vector∥Graph) │          │ Agent      │    search
  └─────┬─────┘       └─────┬──────────┘          └─────┬──────┘
        │                   │                           │
  ┌─────▼─────┐             │                           │
  │   Graph   │             │                           │
  │   Agent   │             │                           │
  └─────┬─────┘             │                           │
        │                   │                           │
        └─────────────────┐ │ ┌─────────────────────────┘
                          │ │ │
                        ┌──▼─▼─▼────────┐
                        │  Fusion Node  │  ← Reciprocal Rank Fusion (RRF)
                        └───────┬───────┘
                                │
                        ┌───────▼───────┐
                        │  Rerank Node  │  ← Context relevance filter
                        └───────┬───────┘
                                │
                        ┌───────▼───────┐
                        │ Generate Node │  ← Faithfulness-first LLM answer
                        └───────────────┘
```

### Architecture Overview

![KineticGraph-Vectra Architecture](architecture.png)

### Infrastructure Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Framework** | FastAPI | Asynchronous REST API |
| **Orchestration** | LangGraph | Intent routing, parallel retrieval, rerank, generate |
| **Vector DB** | ChromaDB | Semantic search |
| **Graph DB** | Neo4j | Entity relationships |
| **Task Queue** | Celery + Redis | Async document processing (sequential PyMuPDF extraction) |
| **Containerization** | Docker Compose | Local deployment and testing |
| **Evaluation** | RAGAS | Offline/live evaluation (faithfulness, relevancy, recall, correctness) |

---

## 🚀 Quick Start

### Prerequisites

- **Docker** and **Docker Compose**
- **OpenAI API Key**

### 1. Clone and Setup

```bash
# Clone the repository and copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env
```

### 2. Start Services

```bash
# Build and start all services via Docker Compose (resides in infra/)
cd infra
docker compose up --build -d
```

### 3. Verify Health

- **Swagger Documentation:** http://localhost:8000/docs
- **Neo4j Browser:** http://localhost:7474 (user: `neo4j`, password: see `.env`)
- **ChromaDB API:** http://localhost:8001
- **Chat UI:** http://localhost:8080 (Start via `cd frontend && python3 serve.py`)

---

## 📚 Usage

### Document Ingestion

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/document" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"
```

### Querying the System (Hybrid Search)

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings about Reciprocal Rank Fusion?",
    "mode": "hybrid",
    "max_results": 5
  }'
```

---

## 🔬 Evaluation & Diagnostics

The evaluation layer is built around [RAGAS](https://docs.ragas.io) to monitor system performance offline and live.

- **Offline Evaluation**: The Jupyter notebook at [rag_evaluation.ipynb](file:///Users/sendils/work/repo/kinetic-v/kinegraph-v/notebooks/rag_evaluation.ipynb) performs full RAGAS batch assessments, rendering mode comparison plots and a polar radar chart mapping baseline scores.
- **Live Diagnostics**: The `RAGASEvaluator` ([ragas_evaluator.py](file:///Users/sendils/work/repo/kinetic-v/kinegraph-v/eval/ragas_evaluator.py)) automatically catches evaluation errors/NaN scores and warns on the console with the exact query and failed metrics details, falling back cleanly to keyword heuristics.
- **Concurrent Batch Eval**: Supports asynchronous concurrent live evaluation of the RAG workflow via the `evaluate_live_workflow` method.

#### RAGAS Evaluation Radar Chart
![RAGAS Evaluation Radar Chart](reports/spider_graph_ragas_score.png)

---

## 🛠️ Local Development

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI app locally
uvicorn backend.app.main:app --reload --port 8000

# Run Celery worker
celery -A backend.workers.celery_app worker --loglevel=info
```

---

**Built with ❤️ for lean, highly performant hybrid RAG systems**
