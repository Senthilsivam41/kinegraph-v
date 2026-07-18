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
