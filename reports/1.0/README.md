# Kinegraph v1.0 Qwen 3.6 benchmark status

## Decision

The artifacts in this directory are a **captured diagnostic run**, but they are
**not an accepted v1.0 comparison baseline**.

They are useful for debugging retrieval and evaluation failures. They must not
be used to claim that a future architecture branch improved or regressed RAGAS
metrics.

## What was captured

| Item | Captured value |
|---|---|
| Run label | `kinegraph-1.0-qwen36-diagnostic` |
| Queries executed | 20 |
| Dataset | `eval/kinegraph_benchmark_v1.csv` |
| Dataset SHA-256 | `d874acd6beb529ba8e9114b094f2102962bf79b8ba30b22632c9624fc5ad6ab4` |
| Dataset version | `1.1.0-draft` |
| Declared code revision | `d86ab43991f683c8b2a95fd8f33fe90ab5faaea6` |
| Artifact commit | `ffdaf7af342589367f2988e326f78a46a58a50eb` |
| Declared generation model | `qwen/qwen3.6-27b` |
| Declared judge model | `qwen/qwen3.6-27b` |
| Retrieval profile | Fixed Hybrid requested; lexical fusion disabled |
| Workflow errors | 0 |
| RAGAS failures | 20 of 20 |
| Evaluation mode | Heuristic or mixed fallback |
| Accepted as RAGAS | No |
| Reference audit accepted | No |
| Profile valid | No |

The workflow completed for all 20 questions with Qwen 3.6 declared as the
generation model. However, every row records
`ragas_error="RAGAS not available or not configured"`. Therefore the Qwen model
was requested as the judge but did not produce an accepted RAGAS evaluation.

## Diagnostic scores

These values are preserved exactly as produced by the fallback evaluator.
They are **diagnostic-only**, not the frozen baseline for a before/after
architecture comparison.

| Metric | Diagnostic mean | Architecture target | Baseline status |
|---|---:|---:|---|
| Faithfulness | 0.2035 | >= 0.75 | Not accepted |
| Answer relevancy | 0.0294 | >= 0.65 | Not accepted |
| Context precision | 0.7500 | >= 0.90 | Not accepted |
| Context recall | 0.3454 | >= 0.65 | Not accepted |
| Answer correctness | 0.0292 | >= 0.60 | Not accepted |
| Weighted composite | 0.3697 | N/A | Not accepted |

Mean workflow latency was 42.839 seconds per query, ranging from 16.530 to
89.618 seconds. Five queries returned zero contexts. The candidate diagnostics
report 286 observed candidates, 49 sent to generation, and 232 removed during
reranking.

## Why this run is not a valid baseline

1. **RAGAS did not run successfully.** All 20 rows have
   `ragas_failed=True`; the stored metric values came from fallback heuristics.
2. **The benchmark references are not accepted.** The audit is marked
   `accepted_for_evaluation=false`, with multiple references requiring
   correction or technical review.
3. **The fixed-Hybrid profile was violated.** The run produced 14 Hybrid,
   5 Vector, and 1 Graph effective routes. Six rows therefore failed the
   declared profile contract.
4. **The run revision is not fully reproducible.** Provenance names `d86ab43`,
   while the artifacts were committed with `ffdaf7a`, which also contains a
   workflow node-name fix. The run did not persist a clean/dirty working-tree
   flag.
5. **The results mix evaluation semantics.** The report correctly identifies
   `heuristic_or_mixed`, so it cannot be compared numerically with a future
   accepted RAGAS run.

## Current project stage

| Stage | Status |
|---|---|
| 20-query benchmark dataset exists | Complete |
| Per-query results and provenance captured | Complete for diagnostic use |
| Qwen 3.6 generation run captured | Complete according to run provenance |
| Qwen 3.6 accepted RAGAS judge run | Not complete |
| Reference answers technically reviewed and accepted | Not complete |
| Fixed retrieval profile enforced for every row | Not complete |
| Immutable accepted v1.0 baseline manifest/tag | Not complete |
| Kinetic-V 2.0 proposal and ADRs | Proposed and merged |
| Separate architecture implementation/release branch | Not created |
| Accepted post-architecture comparison | Not started |

The diagnostic was run against the existing v3-era workflow before any future
Kinetic-V 2.0 implementation. It is not a pristine “original v1” code baseline:
Phase 1–4 retrieval, reranking, recovery, and grounding capabilities already
existed in that workflow.

## Required baseline gate

Before starting the architecture implementation branch:

1. Finish technical review of all 20 references and set the audit to accepted.
2. Re-run the baseline from the current hardened `main`, not `d86ab43`; security
   and runtime hardening merged after the diagnostic and now forms the real
   pre-architecture code boundary.
3. Run from a clean, committed revision and persist both the commit and
   `working_tree_dirty=false`.
4. Fix the profile contract so a fixed-Hybrid run cannot silently become Vector
   or Graph.
5. Configure the RAGAS judge successfully through the intended OpenRouter/Qwen
   endpoint and reject the run if any row has `ragas_failed=True`.
6. Persist model identifiers, base URLs/providers, embedding strategy,
   retrieval settings, per-query provenance, latency, and cost.
7. Create an immutable accepted manifest and tag that exact commit.
8. Create the architecture implementation branch from the accepted baseline
   tag, then evaluate changes one lever at a time.

Recommended naming after the gate succeeds:

- Baseline tag: `benchmark/v1.0-qwen36-accepted`
- Implementation branch: `release/kinetic-v2-architecture`
- Post-change run label: `kinegraph-2.0-qwen36-<experiment>`

## Artifacts

- `kinegraph-1.0-qwen36-diagnostic_results.csv` — per-query answers, contexts,
  provenance, fallback scores, and failure flags.
- `kinegraph-1.0-qwen36-diagnostic_provenance.jsonl` — detailed per-query
  retrieval and generation trace.
- `kinegraph-1.0-qwen36-diagnostic_diagnostics.json` — aggregate profile,
  category, and failure-stage counts.
- `kinegraph-1.0-qwen36-diagnostic_report.json` — aggregate diagnostic report.
- `kinegraph-1.0-qwen36-diagnostic_spider.png` — diagnostic visualization only;
  it is not an accepted benchmark chart.
