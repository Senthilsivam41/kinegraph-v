# ADR-001: Adaptive Routing

- **Status:** Proposed
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
lossy downgrade. Legacy routing remains available behind a feature flag until
the adaptive profile passes its benchmark gate.

## Alternatives rejected

- Always Hybrid: wastes graph calls and can add noise for simple questions.
- LLM-only routing: opaque confidence and weak reproducibility.
- HyDE before routing: conflicts with original-query authority.

## Acceptance

Evaluate slices for single-facet, compound, relationship, exact-token,
misspelled, attachment, and Vectorless questions. Promote only if recall
improves without exceeding the configured precision and latency budget.
