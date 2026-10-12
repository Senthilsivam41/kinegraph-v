# Reflexion-guided query recovery — implementation and test plan

## Outcome and scope

Add an opt-in, request-scoped evidence-repair loop to Kinegraph-V's interactive query path. When the first answer has a specific, observable evidence gap, a bounded reflector selects a different retrieval action, then the workflow regenerates and verifies once. RAGAS and reference answers remain **offline evaluation tools**, never runtime feedback. This work is independent of the unfinished asynchronous ingestion plan in `plan.md` and `todo.md`.

The first release is a controlled pilot, not a default-on quality claim. It uses the existing LangGraph workflow and Chroma/Neo4j/lexical adapters; it adds no persistent memory table, shared cross-user reflection memory, arbitrary Cypher executor, or third-party Reflexion framework. Query scope and document filters cannot widen during a retry.

## Current seams and baseline

- `backend/core/langgraph_workflow.py` runs intent routing, retrieval, pre-generation query recovery, fusion/reranking, structured generation, grounding critique, then formatting. `_query_recovery` already decomposes weak queries; Reflexion must respond to **post-generation** failure rather than repeat that step.
- `WorkflowState` and `execute_with_answer` carry candidate lifecycle, citation validation, critique, answer relevancy, and optional trace. The critique exposes `missing_question_facets` and rejected claims. ADR-004 verification is experimental and disabled by default, so the new trigger must work without it.
- `backend/core/query_recovery.py` checks literal facet-token overlap. This is a diagnostic hint, not a proof that a facet is answered.
- `backend/app/models.py` and `backend/app/api/routes/query.py` expose request-scoped experimental flags and response diagnostics. Keep their disabled behavior backward compatible.
- The accepted v1 reference audit covers 20 rows, but v1 lacks direct refusal, conflict, citation-rejection, and provenance cases (`reports/BENCHMARK_V1_COVERAGE_AUDIT_20260829.md`). The four-metric Hybrid report is diagnostic for planning (composite 0.331, p95 about 135 seconds), not a clean same-judge Reflexion comparison baseline. Freeze a fresh clean revision and corpus before measuring lift.

## Execution contract

```text
intent → retrieval → existing query recovery → fusion/rerank → generate → critique
                                                               ├─ finish
                                                               └─ failure policy → reflect
                                                                                 → targeted retrieval
                                                                                 → merge/dedupe/rerank
                                                                                 → generate → critique → finish
```

1. First attempt remains byte-for-byte equivalent in behavior when Reflexion is disabled. A server-side `REFLEXION_ENABLED=false` flag gates an internal request opt-in; clients cannot raise the attempt, token, or deadline ceilings.
2. `FailureSignal` records a reason and evidence pointers, not a subjective score: missing question facet, invalid citation, unsupported claim, incomplete evidence path, or a retrieval-channel failure. A critic outage or malformed signal cannot authorize a retry. Existing one-shot structured-generation repair remains distinct from Reflexion.
3. `ReflectionAction` is validated structured data: `targeted_vector`, `targeted_lexical`, `targeted_graph`, or `stop`, with a bounded subquery, missing facet, reason, and observed evidence IDs. The existing generation node already handles one-shot structured-output/citation repair; the pilot does not add a second answer-only rewrite. No model-generated Cypher or new filters are executed. Actions respect the caller's explicit mode, authorization scope, and source filters.
4. An `AttemptTrace` in `WorkflowState` records up to two attempts, action, normalized query hash, candidate/evidence IDs, coverage and citation outcomes, latency, token usage, and stop reason. Reflection text is request-local and never reused across users or documents; logs and public responses contain only redacted summaries.
5. At most **one additional attempt**. Stop before rerunning generation if the action repeats a query, exceeds the request budget, fails to add genuinely new in-scope evidence, or would merely restate an unsupported claim. Preserve the safest supported first-attempt answer or existing partial/refusal policy on retry failure. Never promote an answer based on a self-awarded score.

## Delivery sequence

Tasks and checkpoints are tracked in [reflexion-todo.md](reflexion-todo.md): RFX-01 contract/baseline; RFX-02 failure policy; RFX-03 bounded reflector; RFX-04 action executor; RFX-05 graph loop; RFX-06 API/telemetry; RFX-07 paired evaluation; RFX-08 staged rollout. Each task is a small or medium slice with a focused test and explicit dependency. The first demoable slice is a missing-facet query that gains a new cited source with one retry while the disabled path stays unchanged.

## Test strategy and release gates

| Layer | Required evidence |
| --- | --- |
| Pure unit | Failure classification and action validation; malformed/unsafe reflection, repeated query, no new evidence, depleted budget, and critic failure all stop deterministically. |
| Workflow contract | Flag-off results remain unchanged; a flagged missing-facet case takes exactly one allowed retry; candidate IDs are deduped; final claims cite only returned evidence; failure keeps the first safe result. |
| Security and API | Source filters, requested mode, and authorized document scope are immutable; prompt-injected document text cannot add a tool/action; no cross-request memory or raw reflection leak; request/response compatibility holds. |
| Local integration | Seeded Chroma/Neo4j cases cover successful targeted recovery, absent graph evidence, conflicting evidence, retrieval outage, and deadline; capture the actual execution trace. |
| Paired evaluation | On the same clean revision, corpus, generator, judge, and fixed route, compare (A) no retry, (B) equal-budget ordinary retry, and (C) reflection-guided retry. Report per-query citations, context precision/recall, faithfulness, answer correctness/relevancy, p50/p95, tokens/cost, new-evidence rate, and failure/stop reasons. |

Keep accepted v1 frozen. Review and freeze additional held-out multi-hop, two-reference, insufficient-evidence, conflict, and citation cases before using them for promotion. The 20-row v1 set is a regression smoke, not enough to prove a small lift. Pre-register a practical quality-improvement threshold and absolute p95/token-cost ceilings **before** comparative runs; do not choose thresholds after seeing scores. No rollout if C fails to beat B on the targeted slice, if faithfulness/citation validity regresses, or if the operational ceiling fails. A feature flag and rollback path must remain available throughout canary.

## Risks and stop points

- **Missing or broken evidence:** Reflection cannot manufacture absent Chroma chunks or Neo4j provenance. Stop with partial/refusal and investigate ingestion or graph links separately.
- **Evaluator gaming:** The LLM critique is a diagnostic input, not the final authority. Deterministic citation IDs, scope checks, novel-evidence checks, and bounded attempts control the loop.
- **Latency and provider cost:** Existing p95 is high; stop promotion until a measured deadline and token budget are agreed and met. Do not default to three to five attempts from the paper.
- **Benchmark leakage:** Runtime prompts never receive ground truth, reference contexts, RAGAS results, or held-out labels.
- **Prompt injection and privacy:** Treat retrieved text as untrusted data. Store reflection only in request state; redact traces before API/telemetry export.

## Decisions required before promotion

1. Approve numeric interactive p95 deadline and incremental token/cost ceiling; implementation may use conservative configurable ceilings in the pilot but cannot be enabled broadly without these targets.
2. Approve a named reviewer and frozen, held-out Reflexion evaluation slice. Draft v2 rows are not accepted merely because they exist.
3. Confirm canary audience and whether an internal request opt-in is exposed to API clients or kept server-only for the pilot.
