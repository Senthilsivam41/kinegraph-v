# Evaluation Pipeline Guide

## Prerequisites

```bash
pip install ragas openai pandas datasets langchain-openai langchain-anthropic python-dotenv
```

Ensure `.env` file has:
- `OPENAI_API_KEY` or `OPENAI_AI_KEY` (for GPT-4o-mini)  
- `ANTHROPIC_API_KEY` (optional, for critic model)

---

## Running Evaluations

### 1. Generate Test Dataset (First Time Only)
```bash
python scripts/generate_testset.py --size 20 --output eval/kinegraph_benchmark_v1.csv
```

This creates 20 benchmark questions covering:
- 40% simple factual lookup
- 40% multi-hop reasoning  
- 20% abstract synthesis/comparison

### 2. Run Live Evaluation (Recommended)
```bash
cd /home/user/kinegraph-v
docker compose up --build -d    # Start services if not running
sleep 30                        # Wait for health checks

python scripts/run_ragas_evaluation.py \
    --dataset eval/kinegraph_benchmark_v1.csv \
    --max-hops 3 \
    --model gpt-4o-mini
```

### 3. Run Offline Evaluation (With Pre-computed Answers)
```bash
# First generate answers manually or from a previous run
python scripts/run_ragas_evaluation.py \
    --mode offline \
    --dataset eval/kinegraph_benchmark_v1.csv \
    -o reports/offline_eval

# This uses pre-computed answers file:
# eval/pre_computed_answers.json
```

### 4. Run with Custom Parameters
```bash
python scripts/run_ragas_evaluation.py \
    --max-hops 5 \
    --model gpt-4-turbo \
    -o reports/custom_run
```

---

## Output Files

Results are saved to `reports/` directory:

| File | Description |
|------|-------------|
| `v2_YYYYMMDD_HHMMSS_results.csv` | Detailed per-sample scores |
| `v2_YYYYMMDD_HHMMSS_report.json` | Aggregated metrics report |
| `spider_graph_ragas_v2_YYYYMMDD.png` | Visual spider chart of scores |

---

## Expected Results with v2 Improvements

| Metric | Current (v1) | Target (v2) | With v2 Improvements |
|--------|-------------|-------------|---------------------|
| Faithfulness | 0.33 | ≥0.75 | **0.80-0.90** ✅ |
| Answer Relevancy | 0.10 | ≥0.65 | **0.70-0.80** ✅ |
| Context Precision | 1.00 | Maintain | **0.95+** ⚠️ |
| Context Recall | 0.35 | ≥0.65 | **0.72-0.82** ✅ |
| Answer Correctness | 0.37 | ≥0.60 | **0.62-0.75** ⚠️ |

---

## Troubleshooting

### "RAGAS not installed" error
```bash
pip install ragas datasets openai langchain-openai
```

### "Workflow execution failed" errors  
- Check Neo4j and ChromaDB are running: `docker compose ps`
- Verify API keys in `.env` file
- Run validation script first to test core logic:
  ```bash
  python scripts/validate_improvements.py --demo
  ```

### "Benchmark CSV missing columns" error  
Ensure your dataset has both `user_input` and `reference` columns. The generator creates these automatically.

---

## Quick Start (All-in-One)
```bash
python scripts/generate_testset.py --size 20 --output eval/kinegraph_benchmark_v1.csv && \
docker compose up --build -d && sleep 30 && \
python scripts/run_ragas_evaluation.py --max-hops 3
```
