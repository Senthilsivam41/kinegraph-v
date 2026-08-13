# Evaluation Pipeline Guide

Kinegraph accepts a benchmark only when the reference audit is approved, the
declared retrieval profile is preserved for every query, and every requested
metric is produced by RAGAS. Heuristic fallback values are diagnostic-only.

## Reproducible environment

Install the pinned evaluation stack from the repository:

```bash
python -m pip install -r requirements.txt
```

For the default Qwen 3.6 judge through OpenRouter, configure:

```dotenv
OPENROUTER_API_KEY=sk-or-your-key
RAGAS_JUDGE_PROVIDER=openrouter
RAGAS_JUDGE_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=qwen/qwen3.6-27b
```

Judge output is bounded to 4,096 tokens by default to avoid reserving an
unaffordable provider context budget. Override with `RAGAS_JUDGE_MAX_TOKENS`
only when the judge account has sufficient credits.

To judge with NVIDIA Nemotron through OpenRouter, keep the provider as
`openrouter` and change only the judge model:

```bash
python scripts/run_ragas_evaluation.py \
  --model nvidia/nemotron-3-super-120b-a12b \
  --judge-provider openrouter \
  --judge-base-url https://openrouter.ai/api/v1 \
  --preflight-only
```

Fireworks is also supported as a separate OpenAI-compatible judge endpoint:

```bash
export FIREWORKS_API_KEY=fw_...
python scripts/run_ragas_evaluation.py \
  --model accounts/fireworks/models/nvidia-nemotron-3-super-120b-a12b-fp8 \
  --judge-provider fireworks \
  --judge-base-url https://api.fireworks.ai/inference/v1 \
  --preflight-only
```

`OPENAI_API_KEY` remains a backward-compatible fallback. Semantic RAGAS metrics
use the locally cached `sentence-transformers/all-MiniLM-L6-v2` model so the
judge does not try to call an OpenRouter embedding endpoint.

## Preflight without databases or paid judge calls

```bash
python scripts/run_ragas_evaluation.py \
  --model qwen/qwen3.6-27b \
  --judge-provider openrouter \
  --preflight-only
```

Preflight validates imports, credentials, the OpenRouter-compatible judge
client, local embedding availability, and metric registration. It does not call
the judge model or start Neo4j/ChromaDB.

To test the actual Qwen/OpenRouter/RAGAS request path with one small paid sample:

```bash
python scripts/run_ragas_evaluation.py \
  --model qwen/qwen3.6-27b \
  --judge-provider openrouter \
  --judge-smoke-test
```

The smoke test must produce all five finite RAGAS metrics or it exits with
status `2` and prints the underlying judge error.

## Synthetic testset generation (RAGAS TestsetGenerator)

Generate versioned **draft** benchmarks from ADR-002 chunks (never overwrites the
frozen accepted CSV by default):

```bash
PYTHONPATH=. python scripts/generate_testset.py \
  --docs-dir docs \
  --size 12 \
  --output-version v1.2.0-draft \
  --recursive-chunking
```

Use `--adaptive-chunking` for structural-first chunks, or
`scripts/generate_chunk_policy_ab.py` for paired recursive/adaptive drafts.
Graph-seeded drafts (Neo4j MENTIONS or offline entity fallback):

```bash
PYTHONPATH=. python scripts/generate_graph_testset.py --docs-dir docs --max-rows 20
```

Accept a reviewed audit:

```bash
PYTHONPATH=. python scripts/audit_benchmark_references.py \
  --dataset eval/kinegraph_benchmark_v1.csv \
  --audit eval/kinegraph_benchmark_v1.audit.json \
  --accept \
  --reviewer-name "Your Name"
```

Compare two manifests (Hybrid vs Graph/Vector/etc.):

```bash
PYTHONPATH=. python scripts/compare_benchmark_manifests.py \
  --baseline reports/…/manifest.json \
  --candidate reports/…/manifest.json \
  --skip-ratchet
```

Live reports now include IR metrics (`precision_at_5`, `recall_at_5`, `ndcg_at_5`),
shadow Kinetic Score calibration, and a cost placeholder until usage capture is complete.

## Accepted live benchmark

Approve `eval/kinegraph_benchmark_v1.audit.json` with a named human reviewer
(`scripts/audit_benchmark_references.py --accept`). Then start the services and
run one fixed profile (`hybrid`, `hybrid_lexical`, `vectorless`, `adaptive_hybrid`,
`vector`, or `graph`):

```bash
cd infra
docker compose up --build -d
cd ..

python scripts/run_ragas_evaluation.py \
  --profile hybrid \
  --max-hops 2 \
  --model qwen/qwen3.6-27b \
  --run-label v1-qwen36
```

Run other retrieval modes as separate benchmark slices:

```bash
python scripts/run_ragas_evaluation.py --profile hybrid_lexical --run-label v1-qwen36
python scripts/run_ragas_evaluation.py --profile vectorless --run-label v1-qwen36
```

The compatibility launcher delegates to `eval/ragas_evaluator.py`, which owns
the reference-approval, fixed-profile, provenance, and all-rows-success gates.
Offline answer-only evaluation is intentionally not exposed because it lacks
retrieved contexts and cannot produce comparable retrieval metrics.

## Acceptance and outputs

An accepted run requires:

- exactly 20 evaluated benchmark rows;
- no live workflow failures;
- no effective-mode escape from the declared profile;
- `ragas_failed == false` for every row;
- finite values for every requested metric;
- an approved, hash-matching reference audit;
- persisted Git revision, dataset hash, configuration, models, and provenance.

Accepted artifacts are written under `reports/` using the run label and profile.
If any gate fails, diagnostic provenance is retained but no accepted report,
manifest, or spider graph is updated.

### Thin DeepEval regression gate

An accepted run also writes `reports/run_output.json`. This is the input to the
zero-duplicate-judge ratchet gate:

```bash
deepeval test run regression_gate.py
```

`RagasCompositeGate` reads the persisted aggregate score and fails when it is
below either the last accepted `baseline_ref.json` score or the absolute
floor. DeepEval only judge-scores citation-to-context adherence, which is not
part of the RAGAS composite. The baseline is promoted only after both checks
pass. Set `KINEGRAPH_RUN_OUTPUT` and `KINEGRAPH_BASELINE_REF` to override the
default paths.

## Troubleshooting

- `RAGAS PREFLIGHT FAILED ... local cache`: cache the configured sentence
  transformer before running in local-only mode.
- `401` or `403`: verify `OPENROUTER_API_KEY` and the selected provider.
- `404` or model-not-found: verify the exact OpenRouter model slug.
- `BENCHMARK REJECTED ... audit`: complete the human reference review; do not
  bypass it to create a baseline.
- profile rejection: run Hybrid, Hybrid+BM25, Vectorless, or adaptive routing as
  separate slices rather than allowing silent route downgrades.
