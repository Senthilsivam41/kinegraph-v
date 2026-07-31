# ADR-004: Verification Framework

- **Status:** Implemented in shadow mode behind an experimental flag; calibration pending
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

## Implementation

- Existing structured generation emits atomic cited claims, and deterministic
  validation rejects missing or unknown context IDs before a claim can return.
- The critic remains filter-only: it cannot rewrite a claim, add a fact, or add
  a citation.
- `backend/core/verification.py` owns the versioned
  `kinegraph.verification.v1` response policy and
  `kinegraph.kinetic-score.v1` shadow score.
- Explicit outcomes are `verified`, `partial`, or `refused`. Missing facets and
  explicit retrieval conflict markers are returned as evidence gaps.
- The 0–100 score uses observable coverage, citation, critic, relevance,
  reranking, diversity, and link-consistency signals. It is labeled evidence
  confidence and never changes a refusal into an answer.
- Calibration requires labeled samples and produces a separate versioned
  artifact; it cannot promote the score automatically.
- `VERIFICATION_FRAMEWORK_ENABLED=false` keeps the policy and score out of the
  default response pending benchmark calibration.

## Alternatives rejected

- Unconstrained self-rated confidence: uncalibrated and unauditable.
- Post-hoc citation decoration: does not prove a claim used context.
- Critic rewrites: can introduce uncited information.

## Acceptance

Measure citation validity, unsupported-claim rate, refusal precision,
faithfulness, answer relevancy, and score calibration by route and query type.
No score policy becomes user-facing by default without these results.

The calibration hook is implemented. No calibrated post-v3 score or promotion
decision is claimed by this ADR.
