# Kinetic-V 2.0 Roadmap

- **Status:** Proposed
- **Governance:** Every phase follows the architecture decision gate and produces an immutable benchmark manifest.

## Phase 1 — Evidence and observability foundation

- Complete per-query route, candidate, graph-path, fusion, rerank, citation, and judge provenance.
- Freeze and audit benchmark references; maintain separate Hybrid, Hybrid+BM25, and Vectorless slices.
- Enforce exact graph-to-vector link verification and ingestion validation summaries.
- Establish baseline Precision@K, Recall@K, nDCG, RAGAS metrics, latency, and cost.

**Exit criteria:** every accepted run is reproducible, rejects failed RAGAS rows,
and reports an effective route plus candidate survival reasons.

## Phase 2 — Routing and retrieval controls

- Implement ADR-001 behind flags.
- Add metadata pre-filtering and exact-token detection.
- Sweep graph hop depth and lexical fusion weights one lever at a time.
- Complete ADR-003 candidate contract, source-preserving RRF, and declared rerank fallback.

**Exit criteria:** an adaptive route or lexical setting improves its intended
slice without unacceptable precision, p95 latency, or cost regression.

## Phase 3 — Adaptive ingestion and graph evidence

- Implement ADR-002 structural-first chunk contract.
- Add table/image chunk policies with source coordinates.
- Migrate additively; verify IDs, enrich touched nodes automatically, and version policy metadata.
- Compare structural, recursive, and semantic policies on a frozen corpus.

**Exit criteria:** migration is idempotent, incomplete links are visible, and
the selected policy improves retrieval measures without breaking provenance.

## Phase 4 — Verification and calibrated trust

- Implement ADR-004 citation validation and claim-level verification contract.
- Add missing-facet and conflict reporting.
- Introduce Kinetic Score in shadow mode; calibrate before UI exposure.
- Add spider graphs only for accepted, versioned runs.

**Exit criteria:** citation validity and score calibration are demonstrated by
slice, refusal behavior is tested, and no unsupported claim is introduced by a
verification stage.

## Deferred research track

HyDE, learned fusion, deep community traversal, multimodal grounding, and
automatic query decomposition remain experiments. Each needs a weakness trigger,
provenance, a feature flag, rollback, and a benchmark result before default use.

## Release checklist

- [ ] Architecture decisions accepted or explicitly superseded.
- [ ] Benchmark reference version accepted.
- [ ] One-change experiment manifests available.
- [ ] Quality targets evaluated on intended slices.
- [ ] Security, data migration, and rollback plans reviewed.
- [ ] User-facing Kinetic Score language validated as evidence confidence.
