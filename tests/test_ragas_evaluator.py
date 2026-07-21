import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from eval.ragas_evaluator import (
    RAGASEvaluator,
    RAGASValidationError,
    require_successful_ragas,
)


def _results(ragas_failed=False):
    return pd.DataFrame([{
        "question": "What is RRF?",
        "faithfulness": 0.8,
        "answer_relevancy": 0.7,
        "context_precision": 0.9,
        "context_recall": 0.7,
        "answer_correctness": 0.7,
        "ragas_failed": ragas_failed,
        "ragas_error": "judge unavailable" if ragas_failed else None,
    }])


def test_successful_ragas_rows_are_accepted():
    results = _results(ragas_failed=False)

    require_successful_ragas(results)
    report = RAGASEvaluator.__new__(RAGASEvaluator).generate_report(results)

    assert report["summary"]["eval_mode"] == "ragas"
    assert report["summary"]["accepted_as_ragas"] is True


def test_heuristic_fallback_is_rejected_as_benchmark_evidence():
    results = _results(ragas_failed=True)

    with pytest.raises(RAGASValidationError, match="heuristic fallback"):
        require_successful_ragas(results)

    report = RAGASEvaluator.__new__(RAGASEvaluator).generate_report(results)
    assert report["summary"]["eval_mode"] == "heuristic_or_mixed"
    assert report["summary"]["accepted_as_ragas"] is False


def test_missing_ragas_provenance_is_rejected():
    with pytest.raises(RAGASValidationError, match="Missing ragas_failed provenance"):
        require_successful_ragas(_results().drop(columns=["ragas_failed"]))


def test_incomplete_or_failed_live_workflow_is_rejected():
    with pytest.raises(RAGASValidationError, match="Expected 20 evaluation rows"):
        require_successful_ragas(_results(), expected_rows=20)

    results = _results()
    results["workflow_error"] = "Neo4j unavailable"
    with pytest.raises(RAGASValidationError, match="live workflow failed"):
        require_successful_ragas(results, expected_rows=1)
    report = RAGASEvaluator.__new__(RAGASEvaluator).generate_report(results)
    assert report["summary"]["accepted_as_ragas"] is False


def test_non_finite_or_missing_metrics_are_rejected_even_when_ragas_says_success():
    non_finite = _results()
    non_finite.loc[0, "faithfulness"] = float("nan")
    with pytest.raises(RAGASValidationError, match="invalid metric output"):
        require_successful_ragas(non_finite)

    missing = _results().drop(columns=["context_recall"])
    with pytest.raises(RAGASValidationError, match="missing required metric"):
        require_successful_ragas(missing)


def test_report_uses_weighted_composite_and_confidence_interval():
    report = RAGASEvaluator.__new__(RAGASEvaluator).generate_report(_results())

    assert report["summary"]["overall_composite_score"] == pytest.approx(0.795)
    assert report["summary"]["composite_metric_weights"] == {
        "faithfulness": 0.35,
        "context_precision": 0.30,
        "context_recall": 0.20,
        "answer_relevancy": 0.15,
    }
    assert report["summary"]["composite_confidence_interval_95"] == [0.795, 0.795]


def test_live_evaluation_forwards_precision_controls_to_workflow():
    workflow = AsyncMock()
    workflow.execute_with_answer.return_value = {
        "answer": "RRF combines ranked lists.",
        "chunks": [SimpleNamespace(content="RRF combines ranked lists.")],
    }
    evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
    evaluator.evaluate_single = lambda **kwargs: {"faithfulness": 1.0, "ragas_failed": False}

    result = asyncio.run(evaluator.evaluate_live_single(
        workflow,
        question="What is RRF?",
        max_results=6,
        max_hops=1,
        candidate_pool_size=25,
    ))

    call = workflow.execute_with_answer.await_args.kwargs
    assert call["query"] == "What is RRF?"
    assert call["max_results"] == 6
    assert call["max_hops"] == 1
    assert call["candidate_pool_size"] == 25
    assert result["max_hops"] == 1
    assert result["max_results"] == 6
    assert result["candidate_pool_size"] == 25
