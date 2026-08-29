# Benchmark v1 Coverage Audit

- Audit date: 2026-08-29
- Frozen dataset: `eval/kinegraph_benchmark_v1.csv`
- Frozen dataset SHA-256: `d874acd6beb529ba8e9114b094f2102962bf79b8ba30b22632c9624fc5ad6ab4`
- Accepted reference audit: `eval/kinegraph_benchmark_v1.audit.json`, dataset version `1.1.0`
- Decision: retain v1 unchanged; review targeted additions separately before assembling or accepting v2

## Outcome

The 20 accepted v1 rows are useful for the existing baseline but are concentrated on architecture explanation, hybrid retrieval, RRF, API usage, and setup guidance. The score distribution from an evaluation run is not a dataset-balance measure, so no row was removed or rewritten because it scored poorly.

Nine source-backed draft questions were added at `eval/drafts/kinegraph_benchmark_v2-targeted-additions-draft.csv`. They close semantic QA gaps for implemented-but-gated ADR contracts. They are not part of the v1 baseline and are not accepted evaluation evidence until a named reviewer approves their references and an audit sidecar is frozen.

## Existing v1 distribution

| Dimension | v1 count | Finding |
|---|---:|---|
| Single-hop | 8 | Covered |
| Multi-hop | 12 | Covered |
| Compound | 13 | Covered |
| Exact-token | 5 | Covered |
| Misspelled | 3 | Covered |
| Two-reference facets | 6 | Covered |
| Hybrid | 11 | Strongly represented |
| API | 9 | Strongly represented |
| RRF | 4 | Represented |
| Security | 4 | Represented, mainly setup/key handling |
| Vectorless | 2 | Thin |
| Resource limits | 1 | Thin but not an active DeepEval gate concern |
| Refusal / insufficient evidence | 0 | Missing |
| Conflicting evidence | 0 | Missing |
| Citation rejection | 0 | Missing |
| Route preservation / recovery | 0 direct cases | Missing |
| Candidate/path provenance | 0 direct cases | Missing |
| Chunk provenance | 0 direct cases | Missing |
| Declared reranker fallback | 0 | Missing |

Persona and style metadata are incomplete for the 12 multi-hop synthetic rows. This is a reporting limitation, not a reason to rewrite frozen v1.

## Targeted additions

| Draft row | Gap closed | ADR / contract | Expected evaluation layer |
|---:|---|---|---|
| 1 | Explicit-mode preservation | ADR-001 | RAGAS semantics plus deterministic route assertion |
| 2 | Weak-retrieval recovery | ADR-001 | RAGAS semantics plus deterministic trigger assertion |
| 3 | Table structure and stable provenance | ADR-002 | RAGAS semantics plus ingestion metadata assertion |
| 4 | No invented image description | ADR-002 | RAGAS faithfulness plus deterministic forbidden-claim check |
| 5 | Candidate, channel, score, rank, and path provenance | ADR-003 | RAGAS semantics plus provenance-schema assertion |
| 6 | Declared reranker fallback | ADR-003 | RAGAS semantics plus deterministic fallback assertion |
| 7 | Visible graph-channel failure with base-result retention | Architecture Principle 15 / ADR-003 | Failure-path integration test |
| 8 | Missing or unknown citation rejection | ADR-004 | DeepEval citation gate plus deterministic citation-ID assertion |
| 9 | Partial answer or refusal for insufficient/conflicting evidence | ADR-004 | DeepEval rubric plus deterministic outcome assertion |

## Evidence sources

- `docs/ARCHITECTURE_PRINCIPLES.md`
- `docs/architecture/ADR-001-Adaptive-Routing.md`
- `docs/architecture/ADR-002-Adaptive-Chunking.md`
- `docs/architecture/ADR-003-Retrieval-Orchestration.md`
- `docs/architecture/ADR-004-Verification-Framework.md`

These ADRs describe implemented experimental contracts whose benchmark promotion remains pending. The additions test the declared contracts; they do not claim that the experimental flags already pass their gates.

## Acceptance boundary

Do not merge these rows into the accepted dataset or compare their scores with the v1 Laguna baseline yet. The next dataset step is:

1. Human-review each new reference against the checked-in evidence source.
2. Add stable IDs `KGV2-021` through `KGV2-029` and a versioned audit sidecar.
3. Assemble v2 as the exact 20 v1 rows plus the nine accepted additions.
4. Record both the inherited v1 hash and the v2 effective dataset hash.
5. Run the new rows as an isolated diagnostic before collecting a full v2 baseline.

DeepEval must still inspect structured runtime output. A natural-language QA row cannot prove that a citation ID resolved, a route was actually preserved, or candidate provenance was emitted; the deterministic assertions listed above remain required.
