from copy import deepcopy

from eval.experiment_validation import ValidationPolicy
from eval.traversal_sweep import build_sweep_report, evaluate_traversal_candidate


def _manifest(hops, recall=0.6, precision=0.92, p95=100, complete=True):
    metrics = {
        "faithfulness": 0.8,
        "context_precision": precision,
        "context_recall": 0.7,
        "answer_relevancy": 0.75,
    }
    return {
        "provenance": {
            "dataset_sha256": "frozen",
            "git_revision": "same-revision",
            "working_tree_clean": True,
        },
        "pipeline_config": {"retrieval": {"max_hops": hops, "max_results": 6}},
        "models": {
            "generation": "generator",
            "grounding_critic": "critic",
            "judge": "judge",
            "embedding": "embedding",
        },
        "validation_policy": ValidationPolicy().__dict__,
        "report": {
            "summary": {"accepted_as_ragas": True},
            "per_metric": {name: {"mean": value} for name, value in metrics.items()},
            "per_category": {
                "two_reference_facets": {"metrics": {"context_recall": recall}}
            },
            "retrieval_diagnostics": {
                "graph_stage_latency_ms": {"p95": p95},
                "graph_paths": {
                    "all_complete": complete,
                    "traversal_failure_count": 0,
                },
            },
        },
    }


def test_traversal_candidate_requires_category_gain_precision_latency_and_paths():
    baseline = _manifest(2, recall=0.55, p95=100)
    candidate = _manifest(3, recall=0.62, p95=120)

    result = evaluate_traversal_candidate(baseline, candidate)

    assert result["promotion_eligible"] is True
    assert result["two_reference_facet_context_recall"]["delta"] == 0.07
    assert result["graph_p95_latency_ms"]["ratio"] == 1.2


def test_traversal_candidate_fails_closed_on_incomplete_paths():
    baseline = _manifest(2, recall=0.55)
    candidate = _manifest(3, recall=0.62, complete=False)

    result = evaluate_traversal_candidate(baseline, candidate)

    assert result["promotion_eligible"] is False
    assert any("paths" in reason for reason in result["reasons"])


def test_sweep_preserves_baseline_as_rollback_default():
    manifests = [_manifest(1, recall=0.62), _manifest(2, recall=0.55), _manifest(3, recall=0.63)]

    report = build_sweep_report(manifests[1], manifests)

    assert report["tested_hops"] == [1, 2, 3]
    assert report["rollback_hops"] == 2
    assert report["default_changed"] is False
    assert report["candidates"]["2"]["role"] == "accepted_baseline"


def test_sweep_rejects_missing_hop_manifest():
    manifests = [_manifest(1), _manifest(2)]
    try:
        build_sweep_report(manifests[1], manifests)
    except ValueError as exc:
        assert "1, 2, and 3" in str(exc)
    else:
        raise AssertionError("missing hop manifest was accepted")
