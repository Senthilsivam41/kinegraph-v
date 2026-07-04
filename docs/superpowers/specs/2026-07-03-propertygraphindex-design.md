# Design Specification: PropertyGraphIndex Dual-Store Integration (Neo4j + ChromaDB)

## 1. Goal & Context
Integrate LlamaIndex's `PropertyGraphIndex` into the existing `kinegraph-v` hybrid RAG pipeline. This integration replaces the custom entity/relationship extraction and graph-write logic with a schema-constrained extractor stack, a semantic entity deduplication pipeline, and a composed retriever, while keeping graph storage (Neo4j) and vector storage (ChromaDB) decoupled as two separate systems (without using Neo4j's native vector support).

---

## 2. Architecture & Directory Layout

We introduce new modules for ingestion and retrieval, separated into isolated components:

```
kinegraph_v/
  config/
    ontology_schema.yaml    # Ontology and strict schema constraints
  graph_ingestion/
    schema.py               # Schema loader & parser
    extractors.py           # Combined Schema + Simple Path Extractors
    dedup.py                 # Entity resolution and alias clustering
    stores.py                # Neo4j and Chroma connection and wiring
    ingest.py                # Ingestion entrypoint with content hashing
  graph_retrieval/
    retrievers.py           # Composed retriever (Vector + Synonym + Cypher)
    langgraph_node.py       # LangGraph integration tool/node wrapper
  eval/
    graph_eval_harness.py   # RAGAS benchmark runner & reporter
```

---

## 3. Ingestion Module (`graph_ingestion`)

### A. Ontology Schema Definition
Stored in `config/ontology_schema.yaml`:
- Allowed entity types (e.g., `Person`, `Organization`, `Component`, `Concept`, `Technology`).
- Allowed relation types (e.g., `DEVELOPED`, `ORCHESTRATES`, `COMBINES`, `USES`, `MENTIONS`, `INTEGRATES`).
- Allowed source-target-relation triples to restrict the domain model.

### B. Extractor Stack (`extractors.py`)
- Stack `SchemaLLMPathExtractor` and `SimpleLLMPathExtractor`.
- `SchemaLLMPathExtractor` enforces the exact ontology schema in strict mode.
- `SimpleLLMPathExtractor` handles out-of-schema mentions as a fallback. Relationships extracted by this fallback are marked with metadata `source: simple_fallback` to differentiate them from schema-conformant relations.

### C. Entity Resolution (`dedup.py`)
To prevent alias fragmentation (e.g., "Sendil K." vs "Sendil Kumar"):
1. First pass: Case-insensitive exact name matching.
2. Second pass: Cosine similarity using the wrapped embedding model.
3. If similarity exceeds a threshold (e.g., `0.85`), the entity is resolved to the canonical form.
4. Collisions are logged to a JSON file or trace system.

### D. Explicit Dual Stores (`stores.py`)
- Initialize `Neo4jPropertyGraphStore` for graph storage.
- Initialize `ChromaVectorStore` for node embeddings, explicitly pointing to a `kg_nodes` collection in ChromaDB.
- A code comment will explicitly verify that Neo4j's native vector index is bypassed in favor of ChromaDB.

### E. Ingestion Idempotency (`ingest.py`)
- Computes SHA-256 content hashes of each processed document chunk.
- Hashes are stored as node properties or tracked in a local registry.
- Re-running the ingestion on identical documents skips processing to avoid duplicate nodes/relationships.

---

## 4. Retrieval Module (`graph_retrieval`)

### A. Composed Retriever (`retrievers.py`)
Constructs a LlamaIndex custom retriever combining:
1. `VectorContextRetriever` against the `kg_nodes` collection in ChromaDB.
2. `LLMSynonymRetriever` for synonym-based expansions.
3. `TextToCypherRetriever` behind a feature flag, running on a read-only sandboxed Neo4j connection.

Results are fused and deduplicated before returning.

### B. LangGraph Node Wrapper (`langgraph_node.py`)
Wraps the composed retriever to match the interface contract expected by the existing `HybridRAGWorkflow` orchestration. Returns `DocumentChunk` models containing retrieved text and metadata.

---

## 5. Evaluation Harness & Verification Plan

### A. RAGAS Evaluation
- `eval/graph_eval_harness.py` runs evaluations on the baseline vector-only pipeline and the new composed graph pipeline.
- Uses `eval/kinegraph_benchmark_v1.csv` to calculate `context_precision`, `context_recall`, and `answer_relevancy` for comparison.
- Generates a markdown report summarizing the metrics.

### B. Testing Matrix
- **Unit Tests**:
  - `test_extractors.py`: Verifies schema path output structure and fallback tag.
  - `test_dedup.py`: Verifies exact-match and similarity clustering, including a forced collision case.
  - `test_ingest_idempotency.py`: Checks that re-ingestion does not duplicate nodes.
  - `test_retrievers.py`: Validates retriever output composition.
- **Integration Tests**:
  - `test_integration_graph.py`: Validates end-to-end ingestion and retrieval against active Neo4j and Chroma Docker instances.
