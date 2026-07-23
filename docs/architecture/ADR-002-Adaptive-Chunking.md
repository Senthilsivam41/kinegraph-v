# ADR-002: Adaptive Chunking

- **Status:** Proposed
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

## Consequences

The index gains richer metadata and migration cost, while retrieval gains
explainable structural locality. Semantic chunking remains experimental until it
shows a measurable gain over structural-plus-recursive baselines.

## Alternatives rejected

- Embed entire documents: weak precision and citation locality.
- Only semantic chunking: hard to reproduce and can cross source boundaries.
- Store embeddings in Neo4j by default: violates distinct-store responsibility without measured benefit.

## Acceptance

Compare chunk policies on a frozen corpus with document, table, and image-heavy
slices. Verify stable links, ingestion completeness, Recall@K, context
precision, latency, and storage impact.
