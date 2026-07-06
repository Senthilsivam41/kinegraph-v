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

### 3. Decoupled PropertyGraphIndex Dual-Store (Neo4j + ChromaDB)
KineticGraph-Vectra features a **state-of-the-art decoupled Graph RAG index** using LlamaIndex's `PropertyGraphIndex` but uniquely engineered to maintain **strict storage isolation** between the relational graph database (**Neo4j**) and the dense vector database (**ChromaDB**). 

Unlike default implementations that collapse both vectors and graph entities into Neo4j's native vector indices, this decoupled architecture guarantees:
- **Independent Scaling**: Scale your vector query capacity (ChromaDB memory/nodes) independently from your graph traversal complexity (Neo4j memory/CPU).
- **Vendor Lock-in Avoidance**: Swap out ChromaDB or Neo4j for other specialized engines without rebuilding the entire extraction or ingestion pipelines.
- **Granular Embedding Fusing**: Compute and store embeddings for chunks, entities, and relationship summaries, enabling precise Reciprocal Rank Fusion (RRF) at query time.

#### Ingestion Flow & Extractor Stack
Every ingested document goes through a pipeline of concurrent, specialized extractors aligned with a constrained ontology schema:

```mermaid
graph LR
    Doc[Ingested Document] --> Parse[Text Chunking]
    Parse --> Ext1[SchemaLLMPathExtractor]
    Parse --> Ext2[TaggedSimpleLLMPathExtractor]
    Parse --> Ext3[EntityResolutionExtractor]
    Parse --> Ext4[ImplicitPathExtractor]
    
    Ext1 & Ext2 & Ext3 & Ext4 --> GraphStore[CustomNeo4jPropertyGraphStore]
    Ext1 & Ext2 & Ext3 & Ext4 --> VectorStore[ChromaVectorStore]
```

1. **`SchemaLLMPathExtractor`**: Constrains extraction to a strict YAML-defined ontology (e.g., `Component`, `Service`, `Command`, `Status`) and relations (e.g., `Uses`, `Implements`, `Minimizes`).
2. **`TaggedSimpleLLMPathExtractor`**: Serves as a high-recall fallback extractor to capture open-domain facts when schema constraints are too narrow.
3. **`EntityResolutionExtractor`**: Runs inline semantic deduplication (using exact token match and high-similarity Levenshtein/Rau-de-Smet clustering) to collapse alias nodes (e.g. `LangGraph` and `LangGraph orchestrator`) before write time.
4. **`ImplicitPathExtractor`**: Injects structural knowledge paths (e.g., `Chunk` nodes linked via `MENTIONS` to their extracted entities) to preserve local source context.

---

## 🏗️ Architecture

### System Components

```mermaid
graph TD
    User([User Query]) --> Router{Intent Router}
    
    Router -->|Vector Mode| VA[Vector Agent]
    Router -->|Hybrid/Parallel Mode| PF[Parallel Fetch]
    Router -->|Vectorless Mode| VLA[Vectorless Agent]
    
    VA --> Chroma[(ChromaDB Vector Store)]
    
    PF --> Chroma
    PF --> Neo4j[(Neo4j Graph Database)]
    
    VLA --> Lexical[BM25 Local Lexical Cache]
    
    Chroma --> Fusion[Fusion Node - Reciprocal Rank Fusion]
    Neo4j --> Fusion
    Lexical --> Fusion
    
    Fusion --> Rerank[Rerank Node - sentence-transformers]
    Rerank --> Generate[Generate Node - LLM Answer]
    Generate --> Output([Final Answer])
    
    style User fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff
    style Router fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff
    style Fusion fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff
    style Output fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff
```

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

- **Synthetic Testset Generation**: Create rich multi-hop and specific factual benchmark questions directly from the project's documentation using:
  ```bash
  python scripts/generate_testset.py --size 5 --output eval/kinegraph_benchmark_v1.csv
  ```
- **Direct Database Ingestion**: Populate ChromaDB and Neo4j locally with documentation:
  ```bash
  python scripts/ingest_docs.py
  ```
- **Live Pipeline Benchmarking**: Execute the live RAG pipeline on your benchmark questions and compute RAGAS metrics:
  ```bash
  python scratch/run_evaluation.py
  ```
- **Live Diagnostics**: The `RAGASEvaluator` ([ragas_evaluator.py](file:///Users/sendils/work/repo/kinetic-v/kinegraph-v/eval/ragas_evaluator.py)) automatically catches evaluation errors/NaN scores and warns on the console with the exact query and failed metrics details, falling back cleanly to keyword heuristics.
- **Concurrent Batch Eval**: Supports asynchronous concurrent live evaluation of the RAG workflow via the `evaluate_live_workflow` method.

#### Baseline Evaluation Scores

| Metric | Baseline Score | Composed Graph Score | Status / Recommendation |
| :--- | :---: | :---: | :--- |
| **Faithfulness** | 0.3292 | **0.6000** | Fraction of answer claims supported by retrieved context. |
| **Answer Relevancy** | 0.1016 | **0.5659** | How well the answer addresses the question. |
| **Context Precision** | 1.0000 | **0.4500** | Signal-to-noise: most relevant chunks ranked first. |
| **Context Recall** | 0.3476 | **0.6000** | Fraction of ground-truth info present in context. |
| **Answer Correctness** | 0.3745 | **0.4082** | Answer accuracy vs reference ground truth. |
| **Overall Composite** | **0.4306** | **0.5248** | Overall composite score over benchmark dataset. |

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
