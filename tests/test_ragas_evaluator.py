import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from backend.app.models import QueryMode
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


def test_report_keeps_profile_and_category_metrics_separate():
    results = pd.concat([_results(), _results()], ignore_index=True)
    results["categories"] = [["single_hop"], ["multi_hop", "exact_token"]]
    results["profile_name"] = "hybrid_lexical"
    results["workflow_latency_ms"] = [10.0, 20.0]
    results["sample_id"] = ["KGV1-001", "KGV1-009"]
    results["provenance"] = [
        {
            "latency_ms": {"stages": {"graph_retrieval_ms": 10.0}},
            "retrieval": {
                "candidate_lifecycle": [{"sent_to_generation": True, "dropped_at": None}],
                "graph_path_audit": {
                    "traversal_candidate_count": 1,
                    "complete_path_count": 1,
                    "retriever_diagnostics": {"cycle_prevention_count": 1},
                },
            },
        },
        {
            "latency_ms": {"stages": {"graph_retrieval_ms": 20.0}},
            "retrieval": {
                "candidate_lifecycle": [{"sent_to_generation": False, "dropped_at": "reranking"}],
                "graph_path_audit": {
                    "traversal_candidate_count": 1,
                    "complete_path_count": 1,
                    "retriever_diagnostics": {"missing_evidence_edge_count": 0},
                },
            },
        },
    ]

    report = RAGASEvaluator.__new__(RAGASEvaluator).generate_report(results)

    assert report["summary"]["profile"] == "hybrid_lexical"
    assert report["summary"]["profile_preference"].startswith("not established")
    assert report["per_category"]["exact_token"]["samples"] == 1
    assert report["per_category"]["multi_hop"]["mean_workflow_latency_ms"] == 20.0
    assert report["per_category"]["multi_hop"]["sample_ids"] == ["KGV1-009"]
    assert report["retrieval_diagnostics"]["graph_stage_latency_ms"]["p95"] == 19.5
    assert report["retrieval_diagnostics"]["graph_paths"]["all_complete"] is True
    assert report["retrieval_diagnostics"]["candidate_survival"]["dropped_reranking"] == 1


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
    assert call["include_trace"] is True
    assert result["max_hops"] == 1
    assert result["max_results"] == 6
    assert result["candidate_pool_size"] == 25


def test_mode_profile_escape_is_rejected():
    results = _results()
    results["mode_profile_error"] = "effective mode 'vector' is not declared"

    with pytest.raises(RAGASValidationError, match="declared benchmark profile"):
        require_successful_ragas(results)


def test_vectorless_profile_forwards_attachment_and_persists_effective_mode():
    workflow = AsyncMock()
    workflow.execute_with_answer.return_value = {
        "answer": "Use the query endpoint.",
        "chunks": [SimpleNamespace(content="Use the query endpoint.")],
        "effective_mode": "vectorless",
        "trace": {
            "requested_mode": "vectorless",
            "effective_mode": "vectorless",
            "routing": {"facets": ["How do I query the document?"]},
            "initial_candidates": {"vector": [], "graph": [], "lexical": []},
            "channel_candidates": {"vector": [], "graph": [], "lexical": [], "vectorless": []},
            "retrieval_failures": {},
            "fusion": {"candidates": []},
            "reranking": {"candidates": []},
            "final_contexts": [],
        },
    }
    evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
    evaluator.model = "judge"
    evaluator.embedding_model = "embedding"
    evaluator.evaluate_single = lambda **kwargs: {"faithfulness": 1.0, "ragas_failed": False}
    profile = {
        "name": "vectorless",
        "requested_mode": "vectorless",
        "declared_effective_modes": ["vectorless"],
    }

    result = asyncio.run(evaluator.evaluate_live_single(
        workflow,
        question="How do I query the document?",
        mode=QueryMode.VECTORLESS,
        attachment_content="source document",
        attachment_name="source.txt",
        allow_mode_downgrade=False,
        allow_vectorless_auto_route=False,
        profile=profile,
    ))

    call = workflow.execute_with_answer.await_args.kwargs
    assert call["attachment_content"] == "source document"
    assert call["allow_mode_downgrade"] is False
    assert result["effective_mode"] == "vectorless"
    assert result["mode_profile_error"] is None
    assert result["provenance"]["profile"]["name"] == "vectorless"
