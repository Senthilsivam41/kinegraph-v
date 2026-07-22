from pathlib import Path

import pytest

from eval.experiment_validation import (
    ValidationPolicy,
    bootstrap_mean_interval,
    build_manifest,
    changed_levers,
    compare_manifests,
    sha256_file,
    validate_metric_values,
    weighted_composite,
)


def _report(faith=0.8, precision=0.8, recall=0.8, relevancy=0.8):
    return {
        "summary": {"accepted_as_ragas": True},
        "per_metric": {
            "faithfulness": {"mean": faith},
            "context_precision": {"mean": precision},
            "context_recall": {"mean": recall},
            "answer_relevancy": {"mean": relevancy},
        },
    }


def _manifest(config, report=None, judge="judge", dataset_hash="frozen"):
    return {
        "provenance": {
            "dataset_sha256": dataset_hash,
            "git_revision": "same-revision",
            "working_tree_clean": True,
        },
        "pipeline_config": config,
        "models": {
            "generation": "generator",
            "grounding_critic": "grounding-critic",
            "judge": judge,
            "embedding": "embedding",
        },
        "report": report or _report(),
        "validation_policy": ValidationPolicy().__dict__,
    }


def test_weighted_composite_matches_faithfulness_first_policy():
    score = weighted_composite({
        "faithfulness": 0.8,
        "context_precision": 0.9,
        "context_recall": 0.7,
        "answer_relevancy": 0.7,
        "answer_correctness": 0.0,
    })

    assert score == pytest.approx(0.795)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, "bad"])
def test_metric_validation_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="finite values"):
        validate_metric_values({"faithfulness": value}, ["faithfulness"])


def test_bootstrap_interval_is_deterministic_and_contains_mean():
    first = bootstrap_mean_interval([0.4, 0.6, 0.8, 1.0])
    second = bootstrap_mean_interval([0.4, 0.6, 0.8, 1.0])

    assert first == second
    assert first[0] <= 0.7 <= first[1]


def test_changed_levers_flattens_nested_configuration():
    assert changed_levers(
        {"retrieval": {"max_hops": 2, "weight": 1.0}},
        {"retrieval": {"max_hops": 3, "weight": 1.0}},
    ) == ["retrieval.max_hops"]


def test_ratchet_keeps_one_lever_improvement():
    baseline = _manifest({"max_hops": 2})
    candidate = _manifest(
        {"max_hops": 3},
        _report(faith=0.82, precision=0.81, recall=0.83, relevancy=0.8),
    )

    comparison = compare_manifests(baseline, candidate)

    assert comparison["decision"] == "keep"
    assert comparison["changed_levers"] == ["pipeline.max_hops"]
    assert comparison["composite_delta"] > 0


def test_ratchet_reverts_large_metric_drop_even_when_composite_improves():
    baseline = _manifest({"max_hops": 2})
    candidate = _manifest(
        {"max_hops": 3},
        _report(faith=0.7, precision=1.0, recall=0.8, relevancy=0.8),
    )

    comparison = compare_manifests(baseline, candidate)

    assert comparison["candidate_composite"] > comparison["baseline_composite"]
    assert comparison["decision"] == "revert"
    assert comparison["large_metric_regressions"] == {"faithfulness": -0.1}


def test_ratchet_rejects_confounded_or_multi_lever_experiments():
    baseline = _manifest({"max_hops": 2, "max_results": 6})
    candidate = _manifest(
        {"max_hops": 3, "max_results": 8}, judge="different-judge"
    )

    comparison = compare_manifests(baseline, candidate)

    assert comparison["decision"] == "invalid"
    assert len(comparison["changed_levers"]) == 2
    assert any("judge model changed" in reason for reason in comparison["reasons"])
    assert any("exactly one changed lever" in reason for reason in comparison["reasons"])


def test_manifest_records_dataset_hash_revision_config_and_models(tmp_path):
    dataset = tmp_path / "golden.csv"
    dataset.write_text("question,reference\nq,a\n", encoding="utf-8")
    manifest = build_manifest(
        run_label="candidate",
        repo_root=Path(__file__).parents[1],
        dataset_path=dataset,
        pipeline_config={"max_hops": 2},
        models={"generation": "gen", "judge": "judge", "embedding": "embed"},
        report=_report(),
        artifacts={"results_csv": "results.csv"},
    )

    assert manifest["provenance"]["dataset_sha256"] == sha256_file(dataset)
    assert manifest["provenance"]["git_revision"] != ""
    assert isinstance(manifest["provenance"]["working_tree_clean"], bool)
    assert manifest["pipeline_config"] == {"max_hops": 2}
    assert manifest["models"]["judge"] == "judge"
    assert manifest["evaluation_risk_flags"] == []
    assert manifest["validation_policy"] == ValidationPolicy().__dict__
