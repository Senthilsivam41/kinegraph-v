# Synthetic testset generation spike

- Generated at: 2026-07-31T03:36:55.044919+00:00
- Rows: 2
- Elapsed seconds: 0.000
- Chunk policy: legacy.recursive
- Adaptive enabled: False
- CSV: `/Users/sendils/work/repo/kinetic-v/kinegraph-v/eval/drafts/kinegraph_benchmark_spike-phase0-draft.csv`
- Draft audit: `/Users/sendils/work/repo/kinetic-v/kinegraph-v/eval/drafts/kinegraph_benchmark_spike-phase0-draft.audit.json`
- Manifest: `/Users/sendils/work/repo/kinetic-v/kinegraph-v/eval/drafts/kinegraph_benchmark_spike-phase0-draft.generation.json`
- Usage: `{"cost_complete": false, "note": "dry_run_chunks_only", "chunk_count": 239}`

## Synthesizer mix observed

- multi_hop_specific_query_synthesizer: 1
- single_hop_specific_query_synthesizer: 1

## Notes

- Generation embeddings use OpenRouter-compatible OpenAI embeddings.
- Live RAGAS scoring continues to use local MiniLM for eval identity.
- Output is a draft until `scripts/audit_benchmark_references.py --accept` is run.
