# Reflexion-guided query recovery — task checklist

This checklist belongs to [reflexion-plan.md](reflexion-plan.md), not the unfinished ingestion checklist in `todo.md`. No task below is marked complete by planning. Use `PYTHONPATH=. venv311/bin/pytest -q` from the repository root for the full deterministic suite; focused test paths are proposed and should be added as implementation proceeds. Live Docker/provider results require saved evidence and are not implied by unit tests.

## RFX-01 — Freeze contract and comparison baseline

**Description:** Define the first-release behavior, typed attempt/action/failure schema, allowed actions, explicit-mode and source-scope invariants, stop reasons, and comparison protocol. Freeze a clean revision, corpus identity, model/judge/profile, and first-attempt latency/token baseline before quality claims.

**Acceptance criteria:**
- [ ] Contract limits execution to initial attempt plus one retry; benchmark answers/RAGAS are excluded from runtime inputs.
- [ ] Clean baseline manifest includes revision, accepted dataset audit, corpus/index versions, fixed route, models, and cost/latency provenance; invalid runs are labelled diagnostic.
- [ ] Existing flag-off query output and trace are captured as regression fixtures.

**Verification:** [ ] Review contract against `WorkflowState`, query API, and benchmark profile. [ ] Run reference audit and focused baseline smoke with clean-tree preflight. [ ] Reviewer signs off on comparison arms before implementation is promoted.

**Dependencies:** None. **Likely files:** `tasks/reflexion-plan.md`, new contract module under `backend/core/`, `tests/test_reflexion_contract.py`, benchmark harness configuration. **Estimated scope:** M (3–5 files).

## RFX-02 — Runtime failure policy

**Description:** Classify checkable first-attempt failures from citation validation, grounding critique, missing facets, graph path diagnostics, and retrieval failures without using a RAGAS score. Return `retry_eligible` only when there is a permitted recovery action and budget.

**Acceptance criteria:**
- [ ] Missing facet, invalid citation, unsupported claim, incomplete path, and channel failure map to stable reason codes; validation-only failures, critic failure, and unavailable verification stop safely rather than duplicate existing answer repair.
- [ ] Literal facet-token overlap is treated as a hint, never an answer-validity certificate; an already-supported complete answer does not retry.
- [ ] Policy preserves caller mode, filters, and authorized scope in every decision.

**Verification:** [ ] `PYTHONPATH=. venv311/bin/pytest -q tests/test_reflexion_policy.py`. [ ] Table-driven and adversarial unit cases cover every reason and stop code.

**Dependencies:** RFX-01. **Likely files:** new `backend/core/reflexion_policy.py`, `tests/test_reflexion_policy.py`. **Estimated scope:** S (2 files).

## Checkpoint 1 — Contract and decision quality

- [ ] RFX-01–02 focused tests and full deterministic suite pass.
- [ ] Human review confirms triggers are evidence-driven, not a self-judged quality score.
- [ ] No runtime ground-truth or benchmark-reference dependency exists.

## RFX-03 — Bounded reflector and request-local memory

**Description:** Convert an eligible `FailureSignal` and compact attempt trace into a validated `ReflectionAction`. Keep only the current/previous attempt in request state; treat retrieved text as untrusted and prevent model output from creating new permissions or arbitrary tool calls.

**Acceptance criteria:**
- [ ] Reflection output is parsed through a closed action schema; malformed, repeated, out-of-scope, or overlong output becomes `stop`.
- [ ] Reflections, normalized query hashes, and evidence IDs cannot leak into another request or authorization scope.
- [ ] Fixed server-side attempt/deadline/token ceilings cannot be increased through request content.

**Verification:** [ ] `PYTHONPATH=. venv311/bin/pytest -q tests/test_reflexion_planner.py`. [ ] Prompt-injection, malformed-JSON, duplicate-query, and cross-request isolation tests pass.

**Dependencies:** RFX-02. **Likely files:** new `backend/core/reflexion_planner.py`, `backend/core/langgraph_workflow.py` state contract, `tests/test_reflexion_planner.py`. **Estimated scope:** M (3 files).

## RFX-04 — Targeted retrieval action executor

**Description:** Execute only permitted vector, lexical, or graph retrieval actions through existing adapters. Merge new in-scope candidates with stable IDs and provenance; never run model-generated Cypher, perform an answer-only retry, or widen filters.

**Acceptance criteria:**
- [ ] Each allowed action uses the caller's immutable mode/scope and bounded fetch settings; unsupported channels stop with a reason.
- [ ] Duplicate and previously seen chunks are not counted as new evidence; zero novel evidence skips regeneration.
- [ ] Channel timeout/error preserves the original safe response and records an attributed failure.

**Verification:** [ ] `PYTHONPATH=. venv311/bin/pytest -q tests/test_reflexion_actions.py`. [ ] Adapter tests cover vector, lexical, graph, duplicate IDs, empty result, filter escape, and timeout.

**Dependencies:** RFX-03. **Likely files:** new `backend/core/reflexion_actions.py`, `backend/core/langgraph_workflow.py`, `tests/test_reflexion_actions.py`. **Estimated scope:** M (3 files).

## RFX-05 — LangGraph one-retry vertical slice

**Description:** Add a conditional edge after grounding critique to finish or run reflect/action/re-rank/generate/critique once. Keep pre-generation query recovery unchanged, keep cumulative usage/latency, reset attempt-local claim/citation state correctly, and return the safest verified/partial/refusal outcome.

**Acceptance criteria:**
- [ ] Disabled graph follows the existing node path without a reflection call; enabled missing-facet fixture gains one new cited source with at most two total attempts.
- [ ] Retry cannot loop, reuse stale citation IDs, or convert unsupported claims into a verified answer without new support.
- [ ] Retry failure/deadline leaves a truthful final state and original safe answer available.

**Verification:** [ ] `PYTHONPATH=. venv311/bin/pytest -q tests/test_reflexion_workflow.py tests/test_grounding.py tests/test_query_recovery.py`. [ ] Graph-edge/fake-store tests cover success, no-new-evidence, conflict, and failure. [ ] Full deterministic suite passes.

**Dependencies:** RFX-03–04. **Likely files:** `backend/core/langgraph_workflow.py`, `tests/test_reflexion_workflow.py`, `tests/test_grounding.py`. **Estimated scope:** M (3 files).

## Checkpoint 2 — Demoable query slice

- [ ] A seeded missing-facet query shows first and second attempts, new evidence, valid final citations, and a bounded stop.
- [ ] Flag-off behavior and existing query/grounding/recovery tests pass unchanged.
- [ ] Reviewer inspects prompt-injection, scope preservation, and refusal behavior before API exposure.

## RFX-06 — Server gate, API contract, and redacted observability

**Description:** Add default-off server authorization for Reflexion and an internal request opt-in; expose a compact response/trace summary for attempts and stop reasons. Client input cannot set budgets or override security scope. Instrument counts, extra latency, tokens, and failed actions without raw document/reflection text.

**Acceptance criteria:**
- [ ] Server flag off forces legacy behavior even when a client requests Reflexion; disabled responses remain backward compatible.
- [ ] Public diagnostics contain no raw reflection prompt, document excerpt, credential, or cross-request state.
- [ ] A kill switch disables new retries without rollback or data migration.

**Verification:** [ ] `PYTHONPATH=. venv311/bin/pytest -q tests/test_reflexion_api.py tests/test_query_telemetry.py tests/test_multi_hop.py`. [ ] Contract/auth/telemetry tests and OpenAPI diff review pass.

**Dependencies:** RFX-05. **Likely files:** `backend/core/config.py`, `backend/app/models.py`, `backend/app/api/routes/query.py`, `tests/test_reflexion_api.py`. **Estimated scope:** M (4 files).

## RFX-07 — Paired benchmark and fault evidence

**Description:** Build a reproducible three-arm comparison: no retry, equal-budget ordinary retry, reflection-guided retry. Hold revision, corpus, query profile, generator, and judge constant; keep reference answers out of execution. Review and freeze additional held-out cases before using them for promotion.

**Acceptance criteria:**
- [ ] Accepted v1 runs as frozen regression smoke; held-out multi-hop/two-reference/refusal/conflict/citation slice has reviewed sources and a versioned audit.
- [ ] Report contains paired per-query quality and citations plus p50/p95, tokens/cost, retry frequency, new-evidence rate, stop reasons, and failure count for all three arms.
- [ ] Promotion verdict is invalid if provenance is dirty/incomplete, any judge fails, profiles differ, or quality/cost thresholds were not pre-registered.

**Verification:** [ ] Evaluator/manifest contract tests pass. [ ] Live Docker Chroma/Neo4j/provider run is saved with clean-tree preflight, exact commands, manifests, and no secrets. [ ] Fault injection covers store outage, critic failure, timeout, and missing source.

**Dependencies:** RFX-06 and reviewer-approved held-out cases. **Likely files:** `eval/ragas_evaluator.py` or a separate comparison runner, benchmark profile/manifest code, `tests/test_reflexion_eval.py`, `reports/` artifacts. **Estimated scope:** M per slice; split harness and live evidence into separate commits if more than 5 files.

## Checkpoint 3 — Promotion decision

- [ ] Reflection-guided retry beats the equal-budget ordinary retry on the predeclared targeted slice; no citation/faithfulness regression is observed.
- [ ] Absolute p95 and incremental token/cost ceilings were approved before the run and are met.
- [ ] Security, provider failure, and absent-evidence cases fail closed; a reviewer signs the comparison manifest.

## RFX-08 — Canary and rollback

**Description:** Keep the flag off by default, enable a small authorized canary only after Checkpoint 3, compare traces and operational metrics against baseline, and document immediate disable/rollback steps.

**Acceptance criteria:**
- [ ] Canary metrics and alerts cover trigger rate, successful novel evidence, invalid actions, refusals, p95, tokens/cost, and provider errors.
- [ ] Disabling the server flag restores the original path with no state migration or stale shared memory.
- [ ] Human owner records a continue/rollback decision; no broad enablement without signed evidence.

**Verification:** [ ] Staging/Docker smoke with flag off/on and kill-switch drill. [ ] Saved canary report and runbook review.

**Dependencies:** RFX-07 and Checkpoint 3. **Likely files:** operations/runbook documentation, deployment configuration, telemetry tests. **Estimated scope:** S–M (2–4 files).
