"""Thin DeepEval ratchet gate over an accepted Kinegraph RAGAS run.

Run with::

    deepeval test run regression_gate.py

RAGAS metrics are not recomputed here.  The only judge-backed metric is the
additional citation-constraint check; the composite gate is pure threshold
logic over the score already produced by ``eval.ragas_evaluator``.
"""
from __future__ import annotations

from eval.regression_gate import (
    RatchetCase,
    RatchetRun,
    build_ragas_composite_metric,
    default_gate_paths,
    load_baseline_score,
    load_current_run,
    promote_baseline_if_passed,
)


def _deepeval():
    try:
        from deepeval import assert_test
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError(
            "DeepEval is required; install the evaluation dependencies before running "
            "`deepeval test run regression_gate.py`."
        ) from exc
    return assert_test, GEval, LLMTestCase, LLMTestCaseParams


def _citation_metric(GEval, LLMTestCaseParams):
    return GEval(
        name="CitationConstraintAdherence",
        criteria=(
            "Determine whether every factual claim in actual_output is backed by an "
            "explicit citation to a chunk in retrieval_context, and that no citation "
            "references a chunk that is not present in retrieval_context."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.8,
    )


def _case_test_case(case: RatchetCase, LLMTestCase):
    return LLMTestCase(
        input=case.query,
        actual_output=case.answer,
        retrieval_context=case.retrieved_chunks,
        additional_metadata={"ragas_composite_score": case.ragas_composite_score},
    )


def test_ratchet_regression_gate():
    """Fail CI on a ratchet regression or citation-constraint violation."""
    assert_test, GEval, LLMTestCase, LLMTestCaseParams = _deepeval()
    run_path, baseline_path = default_gate_paths()
    run = load_current_run(run_path)
    baseline = load_baseline_score(baseline_path)

    # The aggregate score is the only score used for promotion.  It is kept
    # separate from per-query citation checks so a low individual score cannot
    # silently rewrite the accepted aggregate baseline.
    aggregate_case = _case_test_case(
        RatchetCase(
            query=run.query,
            answer=run.answer,
            retrieved_chunks=run.retrieved_chunks,
            ragas_composite_score=run.ragas_composite_score,
        ),
        LLMTestCase,
    )
    composite_gate = build_ragas_composite_metric(baseline)
    assert_test(aggregate_case, [composite_gate])

    citation_adherence = _citation_metric(GEval, LLMTestCaseParams)
    for case in run.citation_cases():
        assert_test(_case_test_case(case, LLMTestCase), [citation_adherence])

    # This line is intentionally last: DeepEval raises on any failed metric,
    # so a failed gate never promotes the ratchet reference.
    promote_baseline_if_passed(run, baseline_path)
