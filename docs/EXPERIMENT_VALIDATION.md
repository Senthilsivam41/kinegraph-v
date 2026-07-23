# Controlled Benchmark Experiments

Kinegraph's benchmark runner applies an OpenResearch-inspired ratchet policy:
each candidate changes one attributable lever, runs against the same frozen
dataset and judge, and is kept only when its weighted score improves (or ties
within noise) without hiding a material single-metric regression.

## Acceptance contract

A run is valid only when:

- All 20 benchmark rows complete through the selected explicit profile.
- Every effective mode remains within the profile's declared modes.
- Every row has `ragas_failed=False` and no workflow error.
- Every required metric is present, numeric, finite, and between 0 and 1.
- The working tree was clean when the run started.
- The versioned reference audit is accepted, hash-matched, and human-reviewed.
- Its manifest records the Git revision, dataset SHA-256, complete retrieval
  configuration, generation model, grounding critic, judge, and judge embedding.

Heuristic fallback values remain diagnostic only and never enter an accepted
manifest.

## Weighted objective and guardrails

The experiment score is:

```text
0.35 × faithfulness
+ 0.30 × context precision
+ 0.20 × context recall
+ 0.15 × answer relevancy
```

Answer correctness remains visible in the report but is not part of this
ratchet objective. Each report includes a deterministic 95% bootstrap interval
for the weighted per-query score.

A candidate is:

- `keep` when its weighted score improves or is within `0.01` of the baseline,
  and no metric falls by more than `0.05`.
- `revert` when the composite regresses beyond `0.01`, or any individual metric
  falls by more than `0.05`.
- `invalid` when the dataset or judge changed, either run is not accepted RAGAS,
  the working tree was dirty, or the candidate changed anything other than one
  pipeline/model/code lever.

The runner reports the decision but never modifies Git automatically.
Using the same model for generation and judging is recorded as
`judge_model_matches_generation_model`; prefer a stable, separate judge to reduce
self-enhancement bias.

## Running a controlled cycle

Create the accepted baseline:

```bash
PYTHONPATH=. venv/bin/python eval/ragas_evaluator.py \
  --profile hybrid \
  --max-hops 2 \
  --max-results 6 \
  --candidate-pool-size 25 \
  --generation-model gpt-4o-mini \
  --judge-model gpt-4o-mini \
  --run-label baseline
```

Change exactly one lever and compare it:

```bash
PYTHONPATH=. venv/bin/python eval/ragas_evaluator.py \
  --profile hybrid \
  --max-hops 3 \
  --max-results 6 \
  --candidate-pool-size 25 \
  --generation-model gpt-4o-mini \
  --judge-model gpt-4o-mini \
  --baseline-manifest reports/ragas_baseline_manifest.json \
  --run-label hop3
```

Accepted candidates exit with code `0`. Invalid or regressed comparisons still
write their evidence and manifest, then exit with code `3`.

## Mode profiles

Profiles are evaluation-only and never change production defaults:

```bash
PYTHONPATH=. venv/bin/python eval/ragas_evaluator.py --profile hybrid --run-label modes
PYTHONPATH=. venv/bin/python eval/ragas_evaluator.py --profile hybrid_lexical --run-label modes
PYTHONPATH=. venv/bin/python eval/ragas_evaluator.py --profile vectorless --run-label modes
```

`hybrid_lexical` is Hybrid with an opt-in BM25 fusion channel; it is not the
dedicated Vectorless path. Vectorless uses one deterministic attachment corpus
derived from all frozen reference contexts, rather than giving each query only
its own reference passage. Each profile has a separate report and manifest, and
no mode is preferred until accepted manifests establish the result.

## Bounded traversal-depth sweep

Issue #44 is implemented as an ablation, not a default change. After the
reference audit is accepted, execute:

```bash
PYTHONPATH=. venv/bin/python scripts/run_traversal_sweep.py \
  --baseline-hop 2 \
  --profile hybrid \
  --run-label bounded-hops
```

The runner executes `max_hops=1`, `2`, and `3` sequentially. Its summary keeps
hop 2 as the rollback setting and requires, for promotion: at least +0.05
context recall on `KGV1-009`, `KGV1-014`, `KGV1-017`, `KGV1-018`, `KGV1-019`,
and `KGV1-020`; overall context precision at least 0.90; no required metric
regression beyond 0.05; a passing weighted ratchet; graph-retrieval p95 within
25% of baseline; and complete evidence-bearing graph paths. Even an eligible
candidate requires human review and does not alter the runtime default.

Per-query provenance records seed IDs, traversal strategy/depth, full directed
relationship paths, weights and evidence, cycle-prevention counts, empty seeds,
failures, missing evidence, candidate survival, and graph-channel latency.

## Versioned reference audit

The evaluator loads `eval/kinegraph_benchmark_v1.audit.json` before creating
database clients. A missing, stale, draft, or unreviewed audit exits with code 2.
The effective dataset hash combines the source CSV, audit content, and dataset
version, so reference-only changes invalidate baseline comparisons.

Reference corrections are benchmark changes—not retrieval improvements. Review
and approval instructions and the current row-level findings are documented in
[Benchmark Reference Audit](BENCHMARK_REFERENCE_AUDIT.md).

Routing-policy experiments use the separate adaptive profile so fixed Hybrid
mode slices remain comparable. The conservative policy is disabled in
production until its candidate manifest passes the gate:

```bash
PYTHONPATH=. venv/bin/python eval/ragas_evaluator.py \
  --profile adaptive_hybrid --run-label routing-legacy
PYTHONPATH=. venv/bin/python eval/ragas_evaluator.py \
  --profile adaptive_hybrid --enable-conservative-routing \
  --baseline-manifest reports/ragas_routing-legacy-adaptive_hybrid_manifest.json \
  --run-label routing-conservative
```

## Manifest artifacts

Every accepted evaluation writes:

- `reports/ragas_<label>-<profile>_results.csv`
- `reports/ragas_<label>-<profile>_report.json`
- `reports/spider_graph_ragas_<label>-<profile>.png`
- `reports/ragas_<label>-<profile>_manifest.json`
- `reports/ragas_<label>-<profile>_provenance.jsonl`
- `reports/ragas_<label>-<profile>_diagnostics.json`

The manifest is the reproducible experiment record. Use it—not a README score
projection—as the source for baseline/candidate decisions.

The versioned JSONL contains redacted per-query evidence: requested/effective
mode, deterministic route rationale, candidate ranks and scores, graph paths,
recovery transformations, RRF/reranker decisions, final chunk IDs, citations,
critic removals, RAGAS provenance, and stage latency. It is written before the
acceptance gate so failed runs remain diagnosable without becoming benchmark
evidence.

## Scope

Only levers already connected to Kinegraph are recorded and compared. The
OpenResearch sample also proposes chunk strategies, embedding providers,
community resolution, and extraction limits; those are intentionally excluded
until Kinegraph exposes and tests them through stable runtime interfaces.
The current workflow does not expose reliable per-request token cost, so the
OpenResearch `$2` cycle budget is not enforced or claimed by this implementation.
