import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from backend.app.models import QueryMode
import eval.ragas_evaluator as ragas_module
from eval.ragas_evaluator import (
    ALL_METRICS,
    DEFAULT_METRICS,
    RAGASConfigurationError,
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


def test_openrouter_qwen_preflight_uses_explicit_endpoint_and_local_embeddings(
    monkeypatch,
):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured["judge"] = kwargs

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured["embeddings"] = kwargs

    monkeypatch.setattr(ragas_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(ragas_module, "HuggingFaceEmbeddings", FakeEmbeddings)

    evaluator = RAGASEvaluator(
        openai_api_key="sk-or-test",
        model="qwen/qwen3.6-27b",
        metrics=ALL_METRICS,
        provider="openrouter",
    )

    assert evaluator.readiness() == {
        "ready": True,
        "provider": "openrouter",
        "judge_model": "qwen/qwen3.6-27b",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "local_embeddings_only": True,
        "metrics": ALL_METRICS,
        "setup_error": None,
    }
    assert captured["judge"]["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["judge"]["temperature"] == 0
    assert captured["judge"]["max_tokens"] == 4096
    assert captured["embeddings"]["model_kwargs"] == {"local_files_only": True}


def test_judge_exception_details_include_type_status_and_request_id():
    response = SimpleNamespace(status_code=503, headers={"x-request-id": "req-123"})
    error = RuntimeError()
    error.response = response

    details = RAGASEvaluator._judge_exception_details(error)

    assert "RuntimeError" in details
    assert "status_code=503" in details
    assert "request_id=req-123" in details
    assert RAGASEvaluator._is_transient_judge_error(error) is True


def test_judge_validation_errors_are_not_retryable():
    error = ValueError("malformed judge output")

    assert RAGASEvaluator._is_transient_judge_error(error) is False


def test_preflight_fails_closed_and_preserves_embedding_bootstrap_error(monkeypatch):
    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            pass

    class BrokenEmbeddings:
        def __init__(self, **kwargs):
            raise OSError("model is not present in the local cache")

    monkeypatch.setattr(ragas_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(ragas_module, "HuggingFaceEmbeddings", BrokenEmbeddings)

    with pytest.raises(
        RAGASConfigurationError,
        match="model is not present in the local cache",
    ):
        RAGASEvaluator(
            openai_api_key="sk-or-test",
            model="qwen/qwen3.6-27b",
            provider="openrouter",
        )


def test_explicit_diagnostic_fallback_reports_the_real_setup_error(monkeypatch):
    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            pass

    class BrokenEmbeddings:
        def __init__(self, **kwargs):
            raise OSError("local embedding unavailable")

    monkeypatch.setattr(ragas_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(ragas_module, "HuggingFaceEmbeddings", BrokenEmbeddings)
    evaluator = RAGASEvaluator(
        openai_api_key="sk-or-test",
        provider="openrouter",
        allow_heuristic_fallback=True,
    )

    result = evaluator.evaluate_single(
        question="What is RRF?",
        answer="RRF combines ranked lists.",
        contexts=["RRF combines ranked lists."],
    )

    assert result["ragas_failed"] is True
    assert "OSError: local embedding unavailable" in result["ragas_error"]


def test_openrouter_environment_key_is_supported(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-from-environment")
    monkeypatch.setattr(ragas_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(ragas_module, "HuggingFaceEmbeddings", FakeEmbeddings)

    evaluator = RAGASEvaluator(model="qwen/qwen3.6-27b")

    assert evaluator.readiness()["provider"] == "openrouter"
    assert captured["openai_api_key"] == "sk-or-from-environment"


def test_nvidia_nemotron_provider_uses_nvidia_endpoint_and_key(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(ragas_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(ragas_module, "HuggingFaceEmbeddings", FakeEmbeddings)

    evaluator = RAGASEvaluator(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        provider="nvidia",
    )

    assert evaluator.readiness()["provider"] == "nvidia"
    assert captured["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert captured["openai_api_key"] == "nvapi-test"


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


def test_diagnostic_answer_correctness_failure_does_not_reject_acceptance_metrics():
    results = _results()
    results.loc[0, "answer_correctness"] = float("nan")
    results["diagnostic_ragas_failed"] = True
    results["diagnostic_ragas_error"] = "TimeoutError"

    require_successful_ragas(results, required_metrics=DEFAULT_METRICS)
    report = RAGASEvaluator.__new__(RAGASEvaluator).generate_report(results)

    assert report["summary"]["accepted_as_ragas"] is True
    assert report["summary"]["acceptance_metrics"] == DEFAULT_METRICS
    assert report["summary"]["diagnostic_metrics"] == ["answer_correctness"]
    assert report["summary"]["answer_correctness_acceptance_role"] == "diagnostic_only"
    assert report["summary"]["diagnostic_failure_count"] == 1


def test_answer_correctness_is_evaluated_separately_and_cannot_poison_acceptance(monkeypatch):
    calls = []

    def fake_ragas_evaluate(**kwargs):
        metric_names = [metric.name for metric in kwargs["metrics"]]
        calls.append(metric_names)
        if metric_names == DEFAULT_METRICS:
            return {metric: [0.8] for metric in DEFAULT_METRICS}
        raise TimeoutError("diagnostic judge timed out")

    monkeypatch.setattr(ragas_module, "ragas_evaluate", fake_ragas_evaluate)
    evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
    evaluator._ragas_metrics = [SimpleNamespace(name=name) for name in ALL_METRICS]
    evaluator._setup_error = None
    evaluator._llm = None
    evaluator._critic_llm = None
    evaluator._embeddings = None
    evaluator.judge_max_retries = 2
    evaluator.judge_retry_backoff_seconds = 0

    result = evaluator.evaluate_single(
        question="What is RRF?",
        answer="RRF combines ranked lists.",
        contexts=["RRF combines ranked lists."],
        ground_truth="RRF combines ranked lists.",
    )

    assert calls == [DEFAULT_METRICS, ["answer_correctness"]]
    assert result["ragas_failed"] is False
    assert result["diagnostic_ragas_failed"] is True
    assert "TimeoutError" in result["diagnostic_ragas_error"]
    assert math.isnan(result["answer_correctness"])


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
        enable_adaptive_routing=True,
    ))

    call = workflow.execute_with_answer.await_args.kwargs
    assert call["query"] == "What is RRF?"
    assert call["max_results"] == 6
    assert call["max_hops"] == 1
    assert call["candidate_pool_size"] == 25
    assert call["enable_adaptive_routing"] is True
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


# ---------------------------------------------------------------------------
# Helper fixtures for retrieval diagnostic tests
# ---------------------------------------------------------------------------


def _results_with_provenance(provenance_list, ragas_failed=False):
    """Build a results DataFrame with realistic provenance for diagnostic tests."""
    rows = []
    for idx, prov in enumerate(provenance_list):
        rows.append({
            "question": f"Q{idx}",
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "context_precision": 0.9,
            "context_recall": 0.7,
            "ragas_failed": ragas_failed,
            "ragas_error": None,
            "provenance": prov,
            "sample_id": f"KGV1-{idx:03d}",
            "categories": [],
            "profile_name": "hybrid",
            "workflow_latency_ms": float(10 * (idx + 1)),
        })
    return pd.DataFrame(rows)


def _make_provenance(
    *,
    graph_latency=50.0,
    empty_seed=False,
    traversal_failure=False,
    cycle_prevention_count=0,
    missing_evidence_edge_count=0,
    traversal_candidate_count=1,
    complete_path_count=1,
    sent_to_generation=True,
    dropped_at=None,
):
    return {
        "latency_ms": {"stages": {"graph_retrieval_ms": graph_latency}},
        "retrieval": {
            "candidate_lifecycle": [
                {"sent_to_generation": sent_to_generation, "dropped_at": dropped_at}
            ],
            "graph_path_audit": {
                "traversal_candidate_count": traversal_candidate_count,
                "complete_path_count": complete_path_count,
                "retriever_diagnostics": {
                    "empty_seed": empty_seed,
                    "traversal_failure": traversal_failure,
                    "cycle_prevention_count": cycle_prevention_count,
                    "missing_evidence_edge_count": missing_evidence_edge_count,
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Retrieval diagnostic aggregation (commit 9562a93)
# ---------------------------------------------------------------------------


def test_report_aggregates_graph_diagnostic_counters():
    """empty_seed, traversal_failure, cycle_prevention, missing_evidence are summed correctly."""
    provenance = [
        _make_provenance(empty_seed=True, cycle_prevention_count=2),
        _make_provenance(traversal_failure=True, missing_evidence_edge_count=3),
    ]
    results = _results_with_provenance(provenance)

    report = RAGASEvaluator.__new__(RAGASEvaluator).generate_report(results)
    diag = report["retrieval_diagnostics"]["graph_paths"]

    assert diag["empty_seed_count"] == 1
    assert diag["traversal_failure_count"] == 1
    assert diag["cycle_prevention_count"] == 2
    assert diag["missing_evidence_edge_count"] == 3


def test_report_graph_latency_p50_and_p95_are_computed():
    """p50 and p95 percentiles are both present and ordered correctly."""
    provenance = [
        _make_provenance(graph_latency=10.0),
        _make_provenance(graph_latency=20.0),
        _make_provenance(graph_latency=100.0),
    ]
    results = _results_with_provenance(provenance)

    report = RAGASEvaluator.__new__(RAGASEvaluator).generate_report(results)
    latency = report["retrieval_diagnostics"]["graph_stage_latency_ms"]

    assert "p50" in latency
    assert "p95" in latency
    assert latency["p50"] <= latency["p95"]
    assert latency["samples"] == 3


def test_report_candidate_survival_counts_all_drop_stages():
    """pre_fusion, reranking, and final_truncation drops are counted independently."""
    provenance = [
        _make_provenance(sent_to_generation=True, dropped_at=None),
        _make_provenance(sent_to_generation=False, dropped_at="pre_fusion"),
        _make_provenance(sent_to_generation=False, dropped_at="reranking"),
        _make_provenance(sent_to_generation=False, dropped_at="final_truncation"),
    ]
    results = _results_with_provenance(provenance)

    report = RAGASEvaluator.__new__(RAGASEvaluator).generate_report(results)
    survival = report["retrieval_diagnostics"]["candidate_survival"]

    assert survival["sent_to_generation"] == 1
    assert survival["dropped_pre_fusion"] == 1
    assert survival["dropped_reranking"] == 1
    assert survival["dropped_final_truncation"] == 1


def test_report_graph_paths_all_complete_false_when_incomplete():
    """all_complete is False when complete_path_count < traversal_candidate_count."""
    provenance = [
        _make_provenance(traversal_candidate_count=2, complete_path_count=1),
    ]
    results = _results_with_provenance(provenance)

    report = RAGASEvaluator.__new__(RAGASEvaluator).generate_report(results)
    paths = report["retrieval_diagnostics"]["graph_paths"]

    assert paths["all_complete"] is False
    assert paths["traversal_candidate_count"] == 2
    assert paths["complete_path_count"] == 1


def test_report_returns_none_latency_when_no_provenance_records():
    """p50 and p95 are None when no provenance rows carry graph latency data."""
    results = _results()  # no provenance column

    report = RAGASEvaluator.__new__(RAGASEvaluator).generate_report(results)
    latency = report["retrieval_diagnostics"]["graph_stage_latency_ms"]

    assert latency["p50"] is None
    assert latency["p95"] is None
    assert latency["samples"] == 0


# ---------------------------------------------------------------------------
# per_category with missing sample_id column (commit 8b2bd43)
# ---------------------------------------------------------------------------


def test_per_category_sample_ids_empty_when_no_sample_id_column():
    """sample_ids gracefully falls back to [] when sample_id column is absent."""
    results = pd.concat([_results(), _results()], ignore_index=True)
    results["categories"] = [["single_hop"], ["single_hop"]]
    results["profile_name"] = "hybrid"
    results["workflow_latency_ms"] = [10.0, 20.0]
    # Intentionally omit 'sample_id' column

    report = RAGASEvaluator.__new__(RAGASEvaluator).generate_report(results)

    assert report["per_category"]["single_hop"]["sample_ids"] == []
    assert report["per_category"]["single_hop"]["samples"] == 2
