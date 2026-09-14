# Neo4j hybrid search simplification experiment

- **Priority:** Medium
- **Status:** Proposed; no benchmark or migration performed
- **Reviewed:** 2026-09-14
- **Decision sought:** Does consolidating retrieval storage reduce operational cost enough to justify losing independent vector/graph scaling and increasing store coupling?

## Verified baseline and backlog correction

[PR #73](https://github.com/Senthilsivam41/kinegraph-v/pull/73) merged on September 7 at `16e1b90de96d3b97807d46ed677286f0e5a9fb03`, which was still GitHub `main` during this review. The local checkout is `codex/adr003-004-acceptance-calibration` at `b709ab9`. GitHub blob hashes for the workflow, configuration, API request model, reranking test, README, ADR-003/004, and deployment Compose file matched the local versions before this documentation change.

[Issue #48](https://github.com/Senthilsivam41/kinegraph-v/issues/48) remains valid: **cross-encoder scoring is disabled by default; the README was stale.** The README is corrected by this change. Keep the experiment open. Its request to expose the choice is already implemented; its remaining work is the accepted comparison, fallback evidence, and promotion decision. This review does not edit or close the GitHub issue.

| Evidence | Observed behavior |
| --- | --- |
| [Configuration](../../backend/core/config.py), [.env.example](../../.env.example) | `CROSS_ENCODER_RERANK_ENABLED=false` |
| [Workflow](../../backend/core/langgraph_workflow.py) | Constructor inherits the setting; the request-scoped ranker uses the original query and records requested/effective mode and fallback reason |
| [API request](../../backend/app/models.py) | `enable_cross_encoder_reranking` inherits the same setting |
| [Reranking test](../../tests/test_context_ranker.py) | `test_normal_workflow_keeps_cross_encoder_as_controlled_experiment` asserts the disabled default |
| [ADR-003](ADR-003-Retrieval-Orchestration.md), [ADR-004](ADR-004-Verification-Framework.md) | Retrieval promotion remains pending; verification and Kinetic Score remain experimental/shadow |
| [Deployment Compose](../../infra/docker-compose.yml) | Neo4j is pinned to `neo4j:5.16-community`; the deployed server version was not queried |

The September 7 gates reject invalid/non-finite benchmark metrics and incomplete provenance. Calibration requires usable positive and negative boolean labels and the sample floor; calibration does not promote the score. See the [acceptance report](../../reports/ADR003-004_ACCEPTANCE_REPORT.md).

## What Neo4j changes

Neo4j's [hybrid-search guide](https://neo4j.com/developer/genai-ecosystem/hybrid-search/) demonstrates vector and full-text indexes with WRRF implemented in Cypher. Each channel contributes its weight divided by the sum of the RRF constant and its rank. This is a query pattern, not evidence that consolidation outperforms this repository's existing weighted RRF.

The [SEARCH manual](https://neo4j.com/docs/cypher-manual/25/clauses/search/) identifies `SEARCH` as a Cypher 25 feature introduced in Neo4j 2026.01. It queries a vector index inside `MATCH` or `OPTIONAL MATCH`; index creation still uses `CREATE VECTOR INDEX`. The repository's 5.16 pin cannot execute that path. A trial must either use the version-compatible vector procedure or separately qualify an upgrade before testing `SEARCH`.

[Architecture principle 7](../ARCHITECTURE_PRINCIPLES.md) assigns embeddings to ChromaDB and topology to Neo4j. A proposed isolated experiment may copy frozen embeddings into a disposable Neo4j database to measure an alternative. Changing production ownership would require an explicit architecture decision supported by the experiment.

## Controlled experiment sequence

Use an isolated checkout and disposable stores. Preserve the original corpus, embeddings, graph, and accepted reports. Each step compares with its immediate predecessor so an engine upgrade, lexical change, and fusion change cannot masquerade as a storage benefit.

| Step | Change under test | Exit evidence |
| --- | --- | --- |
| 0. Freeze baseline | Record current dual-store Hybrid and Hybrid+BM25 behavior; retain Vectorless as the unchanged control | Accepted manifests and complete per-query traces |
| 1. Qualify version, if using SEARCH | Upgrade only Neo4j in the dual-store stack to an exact tested version/image digest supporting Cypher 25 SEARCH | Driver/APOC compatibility, graph/ingestion parity, backup/restore rehearsal; repeat baseline |
| 2. Move vector retrieval | Replace Chroma vector retrieval with Neo4j over identical frozen chunk embeddings; retain graph traversal, lexical retrieval, and application RRF | Candidate-level retrieval quality and provenance parity, then end-to-end results |
| 3. Move lexical retrieval | Replace existing BM25 with Neo4j full-text on the same chunks in the lexical-enabled slice | Exact-token and misspelling results; explicit analyzer/tokenization differences |
| 4. Move fusion, if justified | Execute equivalent WRRF in Cypher with unchanged weights, rank constant, stable-ID deduplication, and tie-breaking | Fusion parity on fixed channel lists, then live latency and resource results |

Hold dataset/query hashes, chunk IDs/content, embedding model/revision/dimensions, similarity function, graph snapshot, filters, traversal depth, candidate budgets, final context count, routing/recovery, reranker mode/model/revision, generator, judge, prompts, and verification policy fixed. Record necessary backend-specific ANN settings. Do not add structural embeddings or tune weights during this comparison.

Use the same aggregate CPU/memory budget and matched concurrency; record cold and warm runs separately and randomize paired query order. Repeat enough to report uncertainty. Include exact identifiers/URLs/punctuation, misspellings, paraphrases, single-hop, multi-hop, no-answer, and conflicting-evidence slices. Keep [#44](https://github.com/Senthilsivam41/kinegraph-v/issues/44), [#47](https://github.com/Senthilsivam41/kinegraph-v/issues/47), and [#48](https://github.com/Senthilsivam41/kinegraph-v/issues/48) as separate one-lever experiments.

## Acceptance and decision gates

1. **Valid evidence:** Reuse [retrieval acceptance](../../eval/retrieval_acceptance.py) for required profiles, finite/ranged metrics, non-negative latency, accepted RAGAS, and 100% candidate provenance. Preserve stable citation IDs, source ranks/scores, graph paths, candidate survival/drop reasons, fallback reasons, and stage timing. Reuse [experiment validation](../../eval/experiment_validation.py) for reproducibility and paired comparisons. Its current promotion rules target quality improvement; they do not automatically approve a cost-saving consolidation.
2. **Quality:** Report Precision@5, Recall@5, nDCG@5, context precision/recall, faithfulness, answer relevancy, citation validity, unsupported claims, and refusal behavior per slice. Register acceptable non-inferiority margins and confidence rules before collecting results. Missing samples, invalid metrics, or incomplete traces reject a run; no substitution, clamping, or diagnostic-to-baseline promotion.
3. **Operational value:** Measure end-to-end and retrieval p50/p95, throughput, aggregate CPU/RSS, disk/index size, ingestion/index-build time, backup/restore time, and store-link consistency work. Exercise vector-heavy traffic concurrently with graph-heavy traffic to quantify lost scaling independence. Record failure/restart behavior and the larger shared failure domain. Quantify infrastructure/licensing cost on a named environment; component count alone is insufficient.
4. **Reversibility and portability:** Demonstrate export of vectors/chunks and graph data, stable-ID reconciliation, and restoration of the original dual-store configuration without data loss. Verify post-ingestion evidence hydration has no mandatory Chroma dependency before claiming a Neo4j-only serving path. Keep original stores intact during the trial.
5. **Decision:** Adopt only if quality, reliability, and a predeclared operational improvement target all pass. Otherwise retain the dual-store design and record what benefit its complexity buys. Any production migration needs a subsequent ADR and explicit approval; ADR-003/004 and Kinetic Score status remain unchanged.

Before implementation, agree the target Neo4j version/edition, benchmark environment and spend cap, frozen accepted dataset/judge, quality margins, operational target, and run count. Storage backend/version changes need explicit manifest representation before an official comparison. The existing cross-encoder acceptance function is not a storage-consolidation gate.

Store trial reports, manifests, provenance, resource samples, and failures in a new experiment directory; do not overwrite `reports/run_output.json` or accepted baselines. Label every result as baseline, candidate, or diagnostic and record the commit plus configuration hashes.

## Verification of this review

```sh
PYTHONPATH=. venv311/bin/pytest -q tests/test_context_ranker.py tests/test_retrieval_orchestration.py tests/test_retrieval_acceptance.py tests/test_verification_framework.py
```

Result: **25 passed** on 2026-09-14. These tests validate local reranking and acceptance behavior; they provide no live Neo4j, cross-encoder model, RAGAS, performance, or migration evidence. No paid benchmark, service upgrade, database write, or ADR promotion was performed.
