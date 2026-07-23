# ADR-004: Verification Framework

- **Status:** Proposed
- **Date:** 2026-07-23
- **Decision:** Return only citation-validated claims and expose evidence confidence through a Kinetic Score.

## Context

Generation can sound complete when retrieval misses a required facet.
Faithfulness must be structural rather than dependent on model self-rating.
Users also need a concise, interpretable indication of evidence support.

## Decision

Generation produces atomic claims with citations to retrieved chunk IDs.
Deterministic validation rejects unknown IDs. A grounding critic may remove or
hedge unsupported or irrelevant claims; it cannot add facts or citations.
Insufficient, conflicting, or unavailable evidence produces a bounded partial
answer or refusal.

The response includes a versioned 0–100 Kinetic Score based on coverage,
verification success, relevance, reranking, diversity, and metadata/link
consistency, less conflict and missing-facet penalties. It is evidence
confidence, not a guarantee of factual correctness.

## Consequences

Answers may be shorter when evidence is incomplete, intentionally. Responses
gain citations, gaps, score components, and a provenance contract. Thresholds
must be calibrated and never suppress a refusal.

## Alternatives rejected

- Unconstrained self-rated confidence: uncalibrated and unauditable.
- Post-hoc citation decoration: does not prove a claim used context.
- Critic rewrites: can introduce uncited information.

## Acceptance

Measure citation validity, unsupported-claim rate, refusal precision,
faithfulness, answer relevancy, and score calibration by route and query type.
No score policy becomes user-facing by default without these results.
