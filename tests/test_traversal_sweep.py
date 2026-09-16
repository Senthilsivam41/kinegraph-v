from copy import deepcopy
import json
import subprocess
from types import SimpleNamespace

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
        "schema_version": 1,
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
            "summary": {"accepted_as_ragas": True, "total_samples": 20, "metric_values_valid": True, "profile": "hybrid"},
            "per_metric": {name: {"mean": value} for name, value in metrics.items()},
            "per_category": {
                "two_reference_facets": {
                    "samples": 6,
                    "sample_ids": [f"KGV1-{i:03}" for i in (9, 14, 17, 18, 19, 20)],
                    "metrics": {"context_recall": recall},
                }
            },
            "retrieval_diagnostics": {
                "candidate_provenance_completeness": 1.0,
                "candidate_provenance_samples": 20,
                "graph_stage_latency_ms": {"samples": 20, "p95": p95},
                "graph_paths": {
                    "traversal_candidate_count": 20,
                    "complete_path_count": 20 if complete else 19,
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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, "invalid", None])
@pytest.mark.parametrize("role", ["baseline", "candidate"])
def test_sweep_rejects_invalid_slice_recall(role, value):
    baseline, candidate = _manifest(2, recall=0.55), _manifest(3, recall=0.65)
    target = baseline if role == "baseline" else candidate
    target["report"]["per_category"]["two_reference_facets"]["metrics"]["context_recall"] = value
    result = evaluate_traversal_candidate(baseline, candidate)
    assert result["promotion_eligible"] is False


@pytest.mark.parametrize("p95", [float("nan"), float("inf"), -1, True])
def test_sweep_rejects_invalid_graph_latency(p95):
    result = evaluate_traversal_candidate(_manifest(2, recall=0.55), _manifest(3, recall=0.65, p95=p95))
    assert result["promotion_eligible"] is False


@pytest.mark.parametrize("change", ["category", "samples", "provenance", "paths", "failed", "dirty", "metric"])
def test_sweep_validates_baseline_evidence(change):
    baseline = _manifest(2, recall=0.55)
    report = baseline["report"]
    if change == "category":
        report["per_category"]["two_reference_facets"]["sample_ids"][-1] = "KGV1-001"
    elif change == "samples":
        report["retrieval_diagnostics"]["graph_stage_latency_ms"]["samples"] = 19
    elif change == "provenance":
        report["retrieval_diagnostics"]["candidate_provenance_completeness"] = 0.99
    elif change == "paths":
        report["retrieval_diagnostics"]["graph_paths"]["complete_path_count"] = 19
    elif change == "failed":
        report["summary"]["accepted_as_ragas"] = False
    elif change == "dirty":
        baseline["provenance"]["working_tree_clean"] = False
    else:
        report["per_metric"]["faithfulness"]["mean"] = "invalid"
    result = evaluate_traversal_candidate(baseline, _manifest(3, recall=0.65))
    assert result["promotion_eligible"] is False
    assert result["ratchet"]["decision"] == "invalid"


def test_sweep_rejects_duplicate_hops_and_substituted_baseline():
    manifests = [_manifest(1), _manifest(2), _manifest(3)]
    with pytest.raises(ValueError, match="exactly one"):
        build_sweep_report(manifests[1], [*manifests, manifests[0]])
    changed_baseline = deepcopy(manifests[1])
    changed_baseline["run_label"] = "different-run"
    with pytest.raises(ValueError, match="baseline"):
        build_sweep_report(changed_baseline, manifests)


def test_sweep_does_not_claim_rejected_baseline_is_accepted():
    manifests = [_manifest(1), _manifest(2), _manifest(3)]
    manifests[1]["report"]["summary"]["accepted_as_ragas"] = False
    report = build_sweep_report(manifests[1], manifests)
    assert report["decision"] == "invalid"
    assert report["promotion_candidates"] == []
    assert report["candidates"]["2"]["role"] == "invalid_baseline"


def test_issue_thresholds_are_checked_before_display_rounding():
    baseline = _manifest(2, recall=0.55)
    assert not evaluate_traversal_candidate(baseline, _manifest(3, recall=0.59999))["promotion_eligible"]
    assert not evaluate_traversal_candidate(baseline, _manifest(3, recall=0.65, p95=125.001))["promotion_eligible"]
    assert evaluate_traversal_candidate(baseline, _manifest(3, recall=0.60, p95=125))["promotion_eligible"]


def test_runner_freezes_baseline_first_and_isolates_all_outputs(tmp_path, monkeypatch):
    from scripts import run_traversal_sweep as runner

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "working_tree_is_clean", lambda root: True, raising=False)
    commands = []
    output = tmp_path / "isolated"

    def evaluate(command, **kwargs):
        commands.append(command)
        option = lambda flag: command[command.index(flag) + 1]
        hops = int(option("--max-hops"))
        manifest = _manifest(hops, recall=0.55 if hops == 2 else 0.65)
        label = option("--run-label")
        path = output / f"ragas_{label}-hybrid_manifest.json"
        assert option("--output-dir") == str(output)
        assert option("--regression-run-output") == str(output / f"h{hops}_run_output.json")
        assert option("--concurrency") == "1"
        if hops != 2:
            assert option("--baseline-manifest") == str(output / "ragas_test-h2-hybrid_manifest.json")
        path.write_text(json.dumps(manifest))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", evaluate)
    assert runner.main(["--run-label", "test", "--output-dir", str(output), "--judge-model", "frozen-judge", "--judge-provider", "openrouter"]) == 0
    assert [int(cmd[cmd.index("--max-hops") + 1]) for cmd in commands] == [2, 1, 3]
    assert all(cmd[cmd.index("--judge-model") + 1] == "frozen-judge" for cmd in commands)
    report = json.loads((output / "sweep.json").read_text())
    assert report["default_changed"] is False
    assert report["promotion_candidates"] == [1, 3]
    with pytest.raises(SystemExit):
        runner.main(["--output-dir", str(output)])


@pytest.mark.parametrize("exit_code", [2, "timeout"])
def test_runner_persists_failure_and_stops_before_candidates(tmp_path, monkeypatch, exit_code):
    from scripts import run_traversal_sweep as runner

    monkeypatch.setattr(runner, "working_tree_is_clean", lambda root: True, raising=False)
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        if exit_code == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    output = tmp_path / "failed"
    assert runner.main(["--output-dir", str(output)]) == 2
    report = json.loads((output / "sweep.json").read_text())
    assert report["decision"] == "invalid"
    assert report["promotion_candidates"] == []
    assert len(calls) == 1
    assert report["runs"][0]["status"] == "failed"


def test_runner_rejects_dirty_baseline_before_spending_on_candidates(tmp_path, monkeypatch):
    from scripts import run_traversal_sweep as runner

    baseline = _manifest(2)
    baseline["provenance"]["working_tree_clean"] = False
    source = tmp_path / "baseline.json"
    source.write_text(json.dumps(baseline))
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("invalid baseline started a paid run"))
    output = tmp_path / "rejected"
    assert runner.main(["--baseline-manifest", str(source), "--output-dir", str(output)]) == 2
    assert json.loads((output / "sweep.json").read_text())["decision"] == "invalid"


def test_runner_can_validate_saved_manifests_without_rerunning_models(tmp_path, monkeypatch):
    from scripts import run_traversal_sweep as runner

    sources = []
    for hops in (1, 2, 3):
        path = tmp_path / f"hop{hops}.json"
        path.write_text(json.dumps(_manifest(hops, recall=0.55)))
        sources.append(str(path))
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("validation called a model"))
    output = tmp_path / "review"
    assert runner.main(["--manifests", *sources, "--output-dir", str(output)]) == 3
    report = json.loads((output / "sweep.json").read_text())
    assert report["decision"] == "retain_baseline"
    assert report["promotion_candidates"] == []
