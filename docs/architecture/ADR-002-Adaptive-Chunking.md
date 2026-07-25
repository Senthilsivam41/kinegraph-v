# ADR-002: Adaptive Chunking

- **Status:** Implemented behind experimental flag
- **Date:** 2026-07-23
- **Decision:** Replace one-size-fits-all chunking with a versioned structural-first policy.

## Context

Fixed recursive chunks are useful fallbacks but may separate a heading from its
content, flatten tables, and lose image/caption relationships. These losses
create a retrieval ceiling before routing or reranking can help.

## Decision

Ingestion parses structure first and chooses the least-complex chunk type:
structural section chunks, semantic refinements, recursive fallback chunks,
table chunks, and image/OCR chunks. Every chunk includes document provenance,
page/section location, policy version, and a stable SHA-256-derived ID.

Graph entities link only to verified chunk IDs. Data migrations are additive,
idempotent, and report incomplete enrichment rather than inventing context.

## Implementation notes

- Policy module: `backend/graph_ingestion/adaptive_chunking.py`
- Policy version: `kinegraph.adaptive-chunking.v1`
- Feature flag: `ADAPTIVE_CHUNKING_ENABLED` (default `false` — recursive-only)
- Wired into Celery ingest (`backend/workers/tasks.py`), graph ingest
  (`backend/graph_ingestion/ingest.py`), and `scripts/ingest_docs.py`
- Semantic boundary refinement remains experimental and is not applied by default
- Page coordinates depend on richer parser output; markdown path records section
  locality and reports missing enrichment instead of inventing context

## Consequences

The index gains richer metadata and migration cost, while retrieval gains
explainable structural locality. Semantic chunking remains experimental until it
shows a measurable gain over structural-plus-recursive baselines.

## Implementation

- `backend/graph_ingestion/adaptive_chunking.py` owns the versioned
  `kinegraph.adaptive-chunking.v1` chunk contract and Markdown structural parser.
- `ADAPTIVE_CHUNKING_ENABLED` activates structural-first policy. Legacy recursive
  chunking remains the default until a frozen-corpus comparison accepts the
  policy. `ADAPTIVE_CHUNKING_ENABLE_SEMANTIC` keeps sentence-boundary refinement
  opt-in and experimental.
- Chunk records carry `chunk_type`, `section_path`, page hints, policy/parser
  versions, table/image provenance, and SHA-256-stable `chunk_id` values.
- `IdempotentGraphIngester` and Celery `process_document` emit the contract as
  LlamaIndex/Chroma/Neo4j metadata, keep content-hash idempotency, and surface
  an `ingestion_validation` completeness report that never invents missing
  vector links.
- Image chunks persist only source alt/caption/OCR text; they never invent a
  visual description.

This implementation does not promote adaptive chunking to the production
default and does not claim a metric improvement. Promotion still requires the
acceptance comparisons below.

## Alternatives rejected

- Embed entire documents: weak precision and citation locality.
- Only semantic chunking: hard to reproduce and can cross source boundaries.
- Store embeddings in Neo4j by default: violates distinct-store responsibility without measured benefit.

## Acceptance

Compare chunk policies on a frozen corpus with document, table, and image-heavy
slices. Verify stable links, ingestion completeness, Recall@K, context
precision, latency, and storage impact. Recursive remains the production default
until an accepted benchmark gate promotes adaptive chunking.
