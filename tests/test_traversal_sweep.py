from copy import deepcopy

import pytest

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


def test_traversal_candidate_fails_on_insufficient_recall_delta():
    """Category recall improvement below 0.05 must block promotion."""
    baseline = _manifest(2, recall=0.55, p95=100)
    # delta = 0.04 — under the 0.05 threshold
    candidate = _manifest(3, recall=0.59, p95=110)

    result = evaluate_traversal_candidate(baseline, candidate)

    assert result["promotion_eligible"] is False
    assert any("context recall" in r for r in result["reasons"])
    assert result["two_reference_facet_context_recall"]["delta"] == pytest.approx(0.04)


def test_traversal_candidate_fails_on_precision_below_threshold():
    """Overall context_precision below 0.90 must block promotion even if recall improves."""
    baseline = _manifest(2, recall=0.55, precision=0.92, p95=100)
    # Recall improves but precision drops to 0.88
    candidate = _manifest(3, recall=0.62, precision=0.88, p95=110)

    result = evaluate_traversal_candidate(baseline, candidate)

    assert result["promotion_eligible"] is False
    assert any("context precision" in r for r in result["reasons"])


def test_traversal_candidate_fails_on_excessive_latency_ratio():
    """p95 latency > 125% of baseline must block promotion."""
    baseline = _manifest(2, recall=0.55, p95=100)
    # ratio = 1.30 — above the 1.25 ceiling
    candidate = _manifest(3, recall=0.62, p95=130)

    result = evaluate_traversal_candidate(baseline, candidate)

    assert result["promotion_eligible"] is False
    assert any("latency" in r for r in result["reasons"])
    assert result["graph_p95_latency_ms"]["ratio"] == pytest.approx(1.30)


def test_traversal_candidate_fails_on_traversal_failures():
    """Non-zero traversal_failure_count must block promotion."""
    baseline = _manifest(2, recall=0.55, p95=100)
    candidate = deepcopy(_manifest(3, recall=0.62, p95=110))
    candidate["report"]["retrieval_diagnostics"]["graph_paths"]["traversal_failure_count"] = 1

    result = evaluate_traversal_candidate(baseline, candidate)

    assert result["promotion_eligible"] is False
    assert any("traversal failure" in r for r in result["reasons"])


def test_traversal_candidate_fails_on_extra_changed_lever():
    """Changing more than max_hops must block promotion (one-lever rule)."""
    baseline = _manifest(2, recall=0.55, p95=100)
    # Change both max_hops and max_results — two levers
    candidate = deepcopy(_manifest(3, recall=0.62, p95=110))
    candidate["pipeline_config"]["retrieval"]["max_results"] = 10

    result = evaluate_traversal_candidate(baseline, candidate)

    assert result["promotion_eligible"] is False
    assert any("only changed experiment lever" in r for r in result["reasons"])


def test_sweep_rejects_missing_hop_manifest():
    manifests = [_manifest(1), _manifest(2)]
    try:
        build_sweep_report(manifests[1], manifests)
    except ValueError as exc:
        assert "1, 2, and 3" in str(exc)
    else:
        raise AssertionError("missing hop manifest was accepted")
