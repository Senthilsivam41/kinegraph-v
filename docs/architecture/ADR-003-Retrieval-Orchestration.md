# ADR-003: Retrieval Orchestration

- **Status:** Implemented behind an experimental flag; benchmark promotion pending
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

## Implementation

- `backend/core/retrieval_orchestration.py` owns the versioned
  `kinegraph.retrieval-orchestration.v1` contract.
- Stable store IDs replace truncated-content identity in RRF. Fusion retains
  every contributing channel's original score and rank, plus graph paths.
- Identity/semantic deduplication, reranking, and context optimization emit a
  survival or drop decision with a reason for every candidate.
- Context optimization applies configurable source/community caps after
  semantic-first reranking and before generation.
- Cross-encoder reranking is request-scoped, explicitly reported, and disabled
  by default. Keyword fallback and its cause remain visible.
- `eval/retrieval_acceptance.py` rejects incomplete or heuristic benchmark
  evidence and enforces a one-lever cross-encoder comparison.
- Stable identity and provenance are additive correctness controls. The
  behavior-changing context optimizer remains gated by
  `RETRIEVAL_ORCHESTRATION_ENABLED=false` until the required slices pass.

## Alternatives rejected

- Fuse then generate directly: makes noisy or incomplete context more likely.
- Let graph centrality dominate: ranks popularity rather than relevance.
- Learned black-box fusion immediately: lacks labels and weakens explainability.

## Acceptance

Sweep one lever at a time using versioned Hybrid, Hybrid+BM25, and Vectorless
slices. Measure Precision@K, Recall@K, nDCG, context metrics, p95 latency, and
candidate provenance completeness.

The acceptance gate is implemented, but no post-implementation run is recorded
by this ADR. Promotion therefore remains pending.
