# RAGAS Evaluation Report

## Per-Metric Average Scores

- **faithfulness**: 0.3292
- **answer_relevancy**: 0.1016
- **context_precision**: 1.0000
- **context_recall**: 0.3476
- **answer_correctness**: 0.3745

## Actionable Recommendations

1. Low faithfulness: LLM hallucinating. Tighten system prompt with 'only use provided context'.
2. Low answer relevancy: Answers drift off-topic. Add query intent classification.
3. Low context recall: Missing relevant chunks. Increase chunk overlap or add BM25 fallback.
4. Low answer correctness: Verify knowledge-base freshness and consider fine-tuning.

## Quality Tier Distribution

- **fair**: 13 (65.0%)
- **poor**: 7 (35.0%)

**Overall Composite Score**: 0.4306
