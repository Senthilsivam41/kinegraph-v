# ADR-001: Adaptive Routing

- **Status:** Implemented behind an experimental flag; benchmark promotion pending
- **Date:** 2026-07-23
- **Decision:** Use a confidence-aware, reversible execution-plan router.

## Context

Kinetic-V supports Vector, Graph, Hybrid, Hybrid+BM25, and Vectorless paths.
The root-cause analysis shows a requested Hybrid query can be silently
downgraded to a single channel, which is unsafe for compound or exact-token
questions. A route must be explainable and preserve caller intent.

## Decision

The router honors explicit caller mode, classifies facets, entities, exact
tokens, attachments, and route confidence, and retains Hybrid for compound,
comparison, coverage-sensitive, or low-confidence queries. It uses Vectorless
only for explicit mode or eligible attachment/local-document cases. It triggers
recovery only after measurable initial weakness and emits requested/effective
mode, alternatives, signals, and rationale.

## Consequences

Routing is slightly more expensive to observe and test, but no longer hides a
lossy downgrade. Adaptive routing remains behind a feature flag and legacy
routing remains the default until the adaptive profile passes its benchmark
gate.

## Implementation

- `backend/core/adaptive_routing.py` owns the deterministic, versioned
  `kinegraph.adaptive-routing.v1` execution-plan contract.
- `enable_adaptive_routing` activates the policy. The deprecated
  `enable_conservative_routing` flag remains a rollback-compatible alias.
- Every plan records numeric confidence, literal facets, exact tokens, entity
  candidates, attachment eligibility, required and recommended channels,
  rejected alternatives, rationale, and an optional Hybrid fallback trigger.
- Exact-token queries retain Hybrid and recommend the lexical channel; BM25 is
  executed only when the separate lexical experiment flag is enabled.
- A high-confidence single-channel plan escalates to Hybrid before RRF when its
  initial retrieval has insufficient results, low score, missing graph
  evidence, low source diversity, or incomplete facet coverage.
- API responses expose requested mode, effective mode, and the complete routing
  details. Benchmark provenance persists the same execution plan.

This implementation does not promote adaptive routing to the production
default and does not claim a metric improvement. Promotion still requires the
accepted benchmark slices below.

## Alternatives rejected

- Always Hybrid: wastes graph calls and can add noise for simple questions.
- LLM-only routing: opaque confidence and weak reproducibility.
- HyDE before routing: conflicts with original-query authority.

## Acceptance

Evaluate slices for single-facet, compound, relationship, exact-token,
misspelled, attachment, and Vectorless questions. Promote only if recall
improves without exceeding the configured precision and latency budget.
