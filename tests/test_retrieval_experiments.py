import json
from copy import deepcopy

import pytest

from eval.benchmark_profiles import get_profile
from eval.experiment_validation import sha256_file
from eval.provenance import SCHEMA_VERSION
from eval.retrieval_experiments import (
    EXACT_TOKEN_IDS,
    METRICS,
    compare_retrieval_experiment,
    load_experiment_provenance,
)


def _arm(profile, *, cross_encoder=False, gain=0.0, lexical_ms=50, rerank_ms=10):
    metrics = {metric: 0.90 for metric in METRICS}
    metrics["context_recall"] = 0.80 + gain
    records = []
    for sample_id in sorted(EXACT_TOKEN_IDS | {"KGV1-001"}):
        categories = ["exact_token", "multi_hop"] if sample_id in EXACT_TOKEN_IDS else ["single_hop"]
        if sample_id in {"KGV1-010", "KGV1-015"}:
            categories.append("misspelled")
        records.append({
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "categories": categories,
            "profile": {"name": profile, "effective_mode": "hybrid", "valid": True},
            "query": {"original": sample_id},
            "retrieval": {
                "candidate_provenance_completeness": 1.0,
                "failures": {},
                "fusion": {"candidates": [{"candidate_id": "shared"}]},
                "reranking": {
                    "mode": "cross_encoder" if cross_encoder else "keyword",
                    "fallback": False,
                    "model_revision": "c" * 40 if cross_encoder else None,
                },
            },
            "evaluation": {
                "metrics": metrics.copy(), "ragas_failed": False,
                "judge_model": "judge", "embedding_model": "embed",
            },
            "latency_ms": {"stages": {
                "rerank_ms": rerank_ms,
                **({"lexical_retrieval_ms": lexical_ms} if profile == "hybrid_lexical" else {}),
            }},
            "generation": {}, "routing": {}, "failure": {},
        })
    manifest = {
        "models": {"generation": "generator", "judge": "judge", "embedding": "embed"},
        "provenance": {
            "git_revision": "frozen-code", "dataset_sha256": "frozen-dataset",
            "working_tree_clean": True,
        },
        "report": {
            "summary": {"accepted_as_ragas": True, "total_samples": len(records)},
            "per_metric": {metric: {"mean": score} for metric, score in metrics.items()},
        },
        "pipeline_config": {
            "retrieval": {
                "profile": get_profile(profile).to_dict(),
                "mode": "hybrid",
                "enable_lexical_fusion": profile == "hybrid_lexical",
                "lexical_fusion_weight": 0.6,
            },
            "reranking": {"enable_cross_encoder_reranking": cross_encoder},
            "benchmark": {"corpus_sha256": "c" * 64},
        },
    }
    return manifest, records


def test_lexical_gate_counts_the_profile_toggle_as_one_lever_and_pairs_query_ids():
    baseline, before = _arm("hybrid")
    candidate, after = _arm("hybrid_lexical", gain=0.05)
    result = compare_retrieval_experiment("lexical-fusion", baseline, candidate, before, after[::-1])

    assert result["decision"] == "keep"
    assert result["changed_levers"] == ["pipeline.retrieval.enable_lexical_fusion"]
    assert set(result["per_category"]["exact_token"]["sample_ids"]) == EXACT_TOKEN_IDS
    assert result["per_query"]["KGV1-010"]["metric_deltas"]["context_recall"] == 0.05
    assert result["stage_latency_ms"]["p95_overhead"] == 50


@pytest.mark.parametrize(
    ("gain", "latency", "reason"),
    [(0.049, 50, "context recall"), (0.05, 51, "overhead")],
)
def test_lexical_gate_reverts_insufficient_gain_or_excess_stage_latency(gain, latency, reason):
    baseline, before = _arm("hybrid")
    candidate, after = _arm("hybrid_lexical", gain=gain, lexical_ms=latency)
    result = compare_retrieval_experiment("lexical-fusion", baseline, candidate, before, after)

    assert result["decision"] == "revert"
    assert any(reason in message for message in result["reasons"])


def test_cross_encoder_gate_reports_slices_and_rejects_fallback_or_revision_drift():
    baseline, before = _arm("hybrid")
    candidate, after = _arm("hybrid", cross_encoder=True, gain=0.06, rerank_ms=260)
    result = compare_retrieval_experiment("cross-encoder", baseline, candidate, before, after)
    assert result["decision"] == "keep"
    assert set(result["per_category"]) == {"exact_token", "misspelled", "single_hop", "multi_hop"}
    assert result["stage_latency_ms"]["p95_overhead"] == 250

    fallback = deepcopy(after)
    fallback[0]["retrieval"]["reranking"].update(
        mode="keyword", fallback=True, fallback_reason="cross-encoder load failed: OSError",
    )
    result = compare_retrieval_experiment("cross-encoder", baseline, candidate, before, fallback)
    assert result["decision"] == "revert"
    assert result["fallback_samples"] == {"KGV1-001": "cross-encoder load failed: OSError"}

    candidate["provenance"]["git_revision"] = "different-code"
    result = compare_retrieval_experiment("cross-encoder", baseline, candidate, before, after)
    assert result["decision"] == "invalid"


def test_cross_encoder_gate_rejects_composite_gain_at_noise_tolerance():
    baseline, before = _arm("hybrid")
    candidate, after = _arm("hybrid", cross_encoder=True, gain=0.05)
    result = compare_retrieval_experiment("cross-encoder", baseline, candidate, before, after)
    assert result["decision"] == "revert"
    assert any("0.01" in reason for reason in result["reasons"])


def test_provenance_loader_rejects_replaced_artifact(tmp_path):
    manifest, records = _arm("hybrid")
    path = tmp_path / "provenance.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    manifest["artifacts"] = {"provenance_jsonl": str(path)}
    manifest["artifact_sha256"] = {"provenance_jsonl": sha256_file(path)}
    assert load_experiment_provenance(manifest) == records

    path.write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        load_experiment_provenance(manifest)


def test_comparison_cli_persists_revert_evidence(tmp_path, monkeypatch):
    from scripts.compare_benchmark_manifests import main

    baseline, before = _arm("hybrid")
    candidate, after = _arm("hybrid_lexical", gain=0.01)
    paths = []
    for name, manifest, records in (("baseline", baseline, before), ("candidate", candidate, after)):
        provenance = tmp_path / f"{name}.jsonl"
        provenance.write_text("".join(json.dumps(record) + "\n" for record in records))
        manifest["artifacts"] = {"provenance_jsonl": str(provenance)}
        manifest["artifact_sha256"] = {"provenance_jsonl": sha256_file(provenance)}
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(manifest))
        paths.append(path)
    output = tmp_path / "comparison.json"
    monkeypatch.setattr("sys.argv", [
        "compare_benchmark_manifests.py", "--experiment", "lexical-fusion",
        "--baseline", str(paths[0]), "--candidate", str(paths[1]), "--output", str(output),
    ])

    assert main() == 2
    comparison = json.loads(output.read_text())
    assert comparison["decision"] == "revert"
    assert comparison["per_query"]["KGV1-015"]["metric_deltas"]["context_recall"] == 0.01
