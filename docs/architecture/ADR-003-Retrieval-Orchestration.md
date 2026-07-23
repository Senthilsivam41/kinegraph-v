# ADR-003: Retrieval Orchestration

- **Status:** Proposed
- **Date:** 2026-07-23
- **Decision:** Use a provenance-preserving multi-stage pipeline.

## Context

RRF combines rankings but does not determine evidence completeness or truth. A
necessary candidate can be strong in one channel and weak in another, while
duplicate chunks consume the context budget.

## Decision

Kinetic-V 2.0 executes metadata pre-filtering, conditional recovery, wide
channel retrieval, weighted RRF, identity deduplication, semantic-first
reranking, and context optimization. Candidates carry stable IDs, source
channels, original scores, graph paths, and survival or drop reasons.

Reranking uses the original question. Graph centrality, edge evidence,
community membership, and traversal depth are bounded secondary signals. A
cross-encoder is optional until it proves a slice-level gain; any keyword
fallback is declared in provenance.

## Consequences

The system becomes more observable and tunable. Retrieval stays modular: any
channel can be disabled without breaking the rest of the pipeline.

## Alternatives rejected

- Fuse then generate directly: makes noisy or incomplete context more likely.
- Let graph centrality dominate: ranks popularity rather than relevance.
- Learned black-box fusion immediately: lacks labels and weakens explainability.

## Acceptance

Sweep one lever at a time using versioned Hybrid, Hybrid+BM25, and Vectorless
slices. Measure Precision@K, Recall@K, nDCG, context metrics, p95 latency, and
candidate provenance completeness.
