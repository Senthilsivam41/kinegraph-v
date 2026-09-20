# KineticGraph-Vectra

<div align="center">

**A Production-Ready Hybrid RAG System**

*Combining Vector Search (ChromaDB) and Graph Reasoning (Neo4j) with LangGraph Orchestration*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluated-7c3aed.svg)](https://docs.ragas.io)

</div>

---

## Overview

KineticGraph-Vectra answers questions from your documents while showing the
evidence used for each answer. It combines three complementary search methods:

- **Semantic search** finds passages with similar meaning, even when the wording differs.
- **Graph search** follows named entities and relationships stored in Neo4j.
- **Keyword search** finds exact names, identifiers, and phrases in a local index.

Most requests use semantic and graph search together. The system merges the
results, removes duplicates, ranks the strongest evidence, and asks a configured
language model to answer only from that evidence.

### What users get

- Answers with stable source identifiers such as `[chunk-42]`.
- The exact evidence chunks sent to the language model.
- A plain data record explaining how each chunk was selected.
- Visible routing, recovery, citation checks, and processing time.
- An explicit fallback response when the available evidence is insufficient.

## Architecture

![KineticGraph-Vectra ingestion and query architecture](architecture.png)

The top half shows how uploaded documents become vector, graph, and keyword
indexes. The bottom half shows how a question travels through those indexes to
a grounded answer. The language model is configurable; the `GPT-4` label in the
diagram represents the generation step, not a required vendor or model.

### Query path

```text
Question received by the API
  -> choose semantic, graph, combined, or direct-document search
  -> search the selected indexes
  -> retry with safe query expansion only when the first search is weak
  -> merge rankings and remove duplicate evidence
  -> rank the most relevant chunks against the original question
  -> generate claims that cite exact chunk IDs
  -> reject unknown citations and unsupported claims
  -> return the answer, evidence, explanations, and timing
```

### Ingestion path

PDFs are parsed locally by LiteParse. If LiteParse is unavailable, the system
falls back to PyMuPDF. Text is split into chunks, embedded into ChromaDB, and
used to extract entities and relationships for Neo4j. Stable chunk IDs connect
the two stores so graph results can be traced back to source text.

### Components

| Component | Responsibility |
|---|---|
| FastAPI | Versioned query, ingestion, health, and Swagger endpoints |
| LangGraph | Routing and end-to-end query orchestration |
| ChromaDB | Chunk embeddings and semantic retrieval |
| Neo4j | Entities, relationships, and bounded multi-hop traversal |
| Local BM25 cache | Vectorless and opt-in lexical retrieval |
| LiteParse / PyMuPDF | Local PDF parsing with a resilient fallback |
| Celery / Redis | Asynchronous document processing |
| RAGAS | Optional quality evaluation for maintainers |

## Current capabilities and defaults

| Setting | Default behavior |
|---|---|
| Search mode | Combined semantic and graph search (`hybrid`) |
| Retrieval pool | Up to 25 candidates from each active search channel |
| Answer context | The best 6 chunks after merging, deduplication, and ranking |
| Graph depth | Breadth-first traversal up to 2 relationships; requests may choose 1-5 |
| Search recovery | Enabled only when the first retrieval is weak |
| Keyword channel | Off for normal Hybrid requests; available explicitly |
| Reranking | Lightweight keyword and graph-signal ranking |
| Answer checks | Exact citation validation and grounded-claim review enabled |
| Chunking | Recursive text splitting |

Optional research controls—adaptive routing, structural chunking, HyDE,
cross-encoder reranking, lexical fusion, and experimental confidence scoring—
are disabled by default. They are not required to run the normal system.

### Project status

The ingestion, retrieval, explanation, and citation-validation paths are
implemented and covered by automated tests. The repository also contains a
20-question quality benchmark, but there is currently no accepted post-v3 run
proving that one experimental configuration is better than the defaults. The
README therefore describes implemented behavior without claiming benchmark
superiority.

## Quick start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- An OpenAI-compatible API key; OpenRouter is supported
- A unique Neo4j password of at least 16 characters

### Configure and start

```bash
cp .env.example .env
# In .env, set:
# OPENAI_API_KEY=<OpenAI key or OpenRouter key used by the application>
# NEO4J_PASSWORD=<unique password with at least 16 characters>
./scripts/start_services.sh
```

The launcher reads the repository `.env`, validates the Neo4j password, and
starts the stack from [`infra/docker-compose.yml`](infra/docker-compose.yml).
The equivalent direct command is:

```bash
docker compose --env-file .env -f infra/docker-compose.yml up --build -d
```

### Service endpoints

| Service | URL |
|---|---|
| Swagger UI | <http://localhost:8000/docs> |
| Neo4j Browser | <http://localhost:7474> |
| ChromaDB heartbeat | <http://localhost:8001/api/v1/heartbeat> |
| LiteParse | <http://localhost:5707> |
| Chat UI | <http://localhost:8080> |

Start the separate chat UI with:

```bash
cd frontend
python3 serve.py
```

Open <http://localhost:8000/docs> to try the API interactively. A healthy stack
does not mean documents have been indexed; ingest at least one document before
expecting query results.

## Usage

### Ingest a document

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/document" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"
```

### Query the system

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do Neo4j and ChromaDB work together?",
    "mode": "hybrid",
    "max_results": 6,
    "candidate_pool_size": 25,
    "max_hops": 2
  }'
```

The response includes the generated answer, requested and effective modes,
grounded claims, citation validation, grounding critique, relevancy coverage,
recovery details, fusion settings, stage latency, and retrieved chunks. Each
chunk includes a deterministic `explanation` object:

```json
{
  "candidate_id": "chunk-42",
  "source_channels": ["vector", "graph"],
  "channel_ranks": {"vector": 1, "graph": 3},
  "original_scores": {"vector": 0.91, "graph": 0.78},
  "rrf_contributions": {"vector": 0.0164, "graph": 0.0159},
  "reranking": {
    "mode": "graph_aware_keyword",
    "score": 0.89,
    "semantic_score": 0.84,
    "graph_signal_score": 0.2,
    "graph_signals_applied": true,
    "components": {}
  },
  "graph_paths": {}
}
```

The explanation fields mean:

| Field | Meaning |
|---|---|
| `source_channels` | Search methods that found this chunk |
| `channel_ranks` | Position within each search method before results were merged |
| `original_scores` | Relevance score reported by each source |
| `rrf_contributions` | How much each source contributed to the merged ranking |
| `reranking` | Final ranking mode and component scores |
| `graph_paths` | Relationship path evidence when the graph contributed |

These values come directly from the retrieval pipeline; no extra LLM call is
used to invent an explanation. Full schemas are available in Swagger and
[`docs/API.md`](docs/API.md).

### Upgrade an existing v2 graph

New installations do not need this step. Existing deployments can preview the
additive graph enrichment before applying it:

```bash
# Preview without changing Neo4j.
PYTHONPATH=. venv311/bin/python scripts/enrich_kinetic_v_nodes.py --dry-run --batch-size 200

# Apply the additive migration.
PYTHONPATH=. venv311/bin/python scripts/enrich_kinetic_v_nodes.py
```

## Local development

```bash
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt

uvicorn backend.app.main:app --reload --port 8000
celery -A backend.workers.celery_app worker --loglevel=info

PYTHONPATH=. venv311/bin/pytest -q
```

Useful focused checks:

```bash
PYTHONPATH=. venv311/bin/pytest -q \
  tests/test_query_explanations.py \
  tests/test_traversal_sweep.py \
  tests/test_provenance.py

PYTHONPATH=. venv311/bin/python scripts/audit_schema_coverage.py
```

## Quality evaluation for contributors

Evaluation is separate from normal application use. The checked-in dataset has
20 reviewed questions and expected answers. Validation fails closed: a run is
not accepted when model judging fails, a metric is missing or non-finite, the
code or dataset changed unexpectedly, or retrieval provenance is incomplete.

The following checks do not run the full provider-backed benchmark:

```bash
# Validate the reviewed benchmark sources.
PYTHONPATH=. venv311/bin/python scripts/audit_benchmark_references.py

# Validate judge configuration and the local embedding cache without a paid call.
PYTHONPATH=. venv311/bin/python scripts/run_ragas_evaluation.py \
  --model qwen/qwen3.6-27b \
  --judge-provider openrouter \
  --preflight-only
```

Full benchmarks and graph-depth comparisons require working databases, model
capacity, and explicit provider usage. They are intentionally not part of the
quick start. See [experiment validation](docs/EXPERIMENT_VALIDATION.md), the
[regression gate](docs/REGRESSION_GATE.md), and the persisted
[benchmark status](reports/1.0/README.md).

## Troubleshooting

| Symptom | Check |
|---|---|
| API fails during startup | Confirm `.env` contains `OPENAI_API_KEY` and a 16+ character `NEO4J_PASSWORD` |
| Query returns no evidence | Ingest a document and confirm ChromaDB and Neo4j are reachable |
| PDF parsing falls back | Check the LiteParse container at port `5707`; PyMuPDF fallback is expected when it is unavailable |
| Graph results are missing | Check Neo4j at port `7687` and verify graph ingestion completed |
| Semantic results are missing | Check the ChromaDB heartbeat at port `8001` and the configured collection name |
| A model request fails | Verify the key, model name, base URL, and provider quota configured in `.env` |

## Learn more

For application users:

- [API requests and responses](docs/API.md)
- [Security and deployment settings](docs/SECURITY.md)
- [Detailed local setup](docs/QUICKSTART.md)

For contributors evaluating architecture changes:

- [Architecture principles](docs/ARCHITECTURE_PRINCIPLES.md)
- [Architecture roadmap](docs/architecture/ROADMAP.md)
- [Adaptive routing decision](docs/architecture/ADR-001-Adaptive-Routing.md)
- [Adaptive chunking decision](docs/architecture/ADR-002-Adaptive-Chunking.md)
- [Retrieval orchestration decision](docs/architecture/ADR-003-Retrieval-Orchestration.md)
- [Controlled experiment policy](docs/EXPERIMENT_VALIDATION.md)
- [Benchmark reference audit](docs/BENCHMARK_REFERENCE_AUDIT.md)

---

**Built for evidence-first, explainable hybrid retrieval.**
