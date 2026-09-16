"""Promotion gates for the bounded max-hops ablation in GitHub issue #44."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from eval.experiment_validation import ValidationPolicy, compare_manifests


SWEEP_HOPS = (1, 2, 3)
TWO_REFERENCE_CATEGORY = "two_reference_facets"
TWO_REFERENCE_IDS = tuple(f"KGV1-{i:03}" for i in (9, 14, 17, 18, 19, 20))


def _get(value: Any, *keys: str) -> Any:
    for key in keys:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


def _number(value: Any, maximum: float = math.inf) -> float | None:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
        return None
    return float(value)


def _category_recall(manifest: Mapping[str, Any]) -> float | None:
    return _number(_get(manifest, "report", "per_category", TWO_REFERENCE_CATEGORY, "metrics", "context_recall"), 1)


def _graph_p95(manifest: Mapping[str, Any]) -> float | None:
    return _number(_get(manifest, "report", "retrieval_diagnostics", "graph_stage_latency_ms", "p95"))


def _max_hops(manifest: Mapping[str, Any]) -> int | None:
    value = _get(manifest, "pipeline_config", "retrieval", "max_hops")
    return value if type(value) is int else None


def validate_traversal_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Reject incomplete evidence before comparisons can label a run accepted."""
    reasons = []
    if _max_hops(manifest) not in SWEEP_HOPS:
        reasons.append("max_hops must be one of 1, 2, or 3")
    for field in ("accepted_as_ragas", "metric_values_valid"):
        if _get(manifest, "report", "summary", field) is not True:
            reasons.append(f"{field} must be true")
    if _get(manifest, "report", "summary", "total_samples") != 20:
        reasons.append("all 20 benchmark samples are required")
    if _get(manifest, "report", "summary", "profile") not in ("hybrid", "hybrid_lexical"):
        reasons.append("an explicit Hybrid profile is required")
    if _get(manifest, "provenance", "working_tree_clean") is not True:
        reasons.append("run was produced from a dirty or unknown working tree")
    for field in ("git_revision", "dataset_sha256"):
        if not isinstance(_get(manifest, "provenance", field), str) or not _get(manifest, "provenance", field).strip():
            reasons.append(f"missing {field}")
    for field in ("generation", "grounding_critic", "judge", "embedding"):
        if not isinstance(_get(manifest, "models", field), str) or not _get(manifest, "models", field).strip():
            reasons.append(f"missing {field} model")
    for metric in ValidationPolicy().metric_weights:
        if _number(_get(manifest, "report", "per_metric", metric, "mean"), 1) is None:
            reasons.append(f"{metric} must be numeric, finite, and in [0, 1]")
    category = _get(manifest, "report", "per_category", TWO_REFERENCE_CATEGORY)
    ids = _get(category, "sample_ids")
    if not isinstance(ids, list) or sorted(map(str, ids)) != list(TWO_REFERENCE_IDS) or _get(category, "samples") != 6:
        reasons.append("two-reference-facet category must contain exactly queries 9, 14, 17, 18, 19, and 20")
    if _category_recall(manifest) is None:
        reasons.append("two-reference-facet context recall must be numeric, finite, and in [0, 1]")
    diagnostics = _get(manifest, "report", "retrieval_diagnostics")
    if _graph_p95(manifest) is None or _get(diagnostics, "graph_stage_latency_ms", "samples") != 20:
        reasons.append("graph latency requires 20 samples and a finite non-negative p95")
    if _number(_get(diagnostics, "candidate_provenance_completeness"), 1) != 1 or _get(diagnostics, "candidate_provenance_samples") != 20:
        reasons.append("candidate provenance must be complete for all 20 samples")
    paths = _get(diagnostics, "graph_paths")
    count = _get(paths, "traversal_candidate_count")
    complete = _get(paths, "complete_path_count")
    if type(count) is not int or count <= 0 or type(complete) is not int or complete != count or _get(paths, "all_complete") is not True:
        reasons.append("all returned traversal paths must be complete and evidence-bearing")
    failures = _get(paths, "traversal_failure_count")
    if type(failures) is not int or failures != 0:
        reasons.append("run contains graph traversal failures or missing failure diagnostics")
    return reasons


def evaluate_traversal_candidate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    policy: ValidationPolicy | None = None,
) -> dict[str, Any]:
    """Apply the issue-specific gates in addition to the generic ratchet."""
    policy = policy or ValidationPolicy()
    reasons = [
        f"{role}: {reason}"
        for role, manifest in (("baseline", baseline), ("candidate", candidate))
        for reason in validate_traversal_manifest(manifest)
    ]
    comparison = (
        {"decision": "invalid", "reasons": list(reasons), "changed_levers": []}
        if reasons else compare_manifests(baseline, candidate, policy)
    )

    expected_lever = "pipeline.retrieval.max_hops"
    if comparison.get("changed_levers") != [expected_lever]:
        reasons.append("max_hops must be the only changed experiment lever")

    baseline_recall = _category_recall(baseline)
    candidate_recall = _category_recall(candidate)
    category_delta = (
        candidate_recall - baseline_recall
        if baseline_recall is not None and candidate_recall is not None else None
    )
    if category_delta is None or category_delta + 1e-12 < 0.05:
        reasons.append("two-reference-facet context recall must improve by at least 0.05")

    precision = _number(_get(candidate, "report", "per_metric", "context_precision", "mean"), 1)
    if precision is None or precision < 0.90:
        reasons.append("overall context precision must remain at least 0.90")

    baseline_p95 = _graph_p95(baseline)
    candidate_p95 = _graph_p95(candidate)
    latency_ratio = (
        candidate_p95 / baseline_p95
        if baseline_p95 and candidate_p95 is not None else None
    )
    if latency_ratio is None or latency_ratio > 1.25:
        reasons.append("graph retrieval p95 latency must remain within 25% of baseline")

    graph_paths = _get(candidate, "report", "retrieval_diagnostics", "graph_paths")
    for metric in policy.metric_weights:
        before = _number(_get(baseline, "report", "per_metric", metric, "mean"), 1)
        after = _number(_get(candidate, "report", "per_metric", metric, "mean"), 1)
        if before is not None and after is not None and before - after > 0.05 + 1e-12:
            reasons.append(f"{metric} regressed by more than 0.05")

    if comparison.get("decision") != "keep":
        reasons.append("candidate did not pass the controlled-experiment ratchet")

    return {
        "baseline_hops": _max_hops(baseline),
        "candidate_hops": _max_hops(candidate),
        "promotion_eligible": not reasons,
        "reasons": reasons,
        "two_reference_facet_context_recall": {
            "baseline": baseline_recall,
            "candidate": candidate_recall,
            "delta": round(category_delta, 6) if category_delta is not None else None,
        },
        "context_precision": precision,
        "graph_p95_latency_ms": {
            "baseline": baseline_p95,
            "candidate": candidate_p95,
            "ratio": round(latency_ratio, 6) if latency_ratio is not None else None,
        },
        "graph_paths": graph_paths,
        "ratchet": comparison,
    }


def build_sweep_report(
    baseline: Mapping[str, Any],
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_hop = {_max_hops(manifest): manifest for manifest in manifests}
    if len(manifests) != len(SWEEP_HOPS) or set(by_hop) != set(SWEEP_HOPS):
        raise ValueError("traversal sweep requires exactly one manifest for max_hops 1, 2, and 3")
    baseline_hops = _max_hops(baseline)
    if baseline_hops not in SWEEP_HOPS:
        raise ValueError("baseline max_hops must be one of 1, 2, or 3")
    if by_hop[baseline_hops] != baseline:
        raise ValueError("baseline must match the sweep's baseline-hop manifest")
    baseline_errors = validate_traversal_manifest(baseline)
    candidates = {}
    for hops in SWEEP_HOPS:
        if hops == baseline_hops:
            candidates[str(hops)] = {
                "role": "invalid_baseline" if baseline_errors else "accepted_baseline",
                "promotion_eligible": False,
                "rollback_default": True,
                "reasons": baseline_errors,
            }
        else:
            candidates[str(hops)] = {
                "role": "candidate",
                "rollback_default": False,
                **evaluate_traversal_candidate(baseline, by_hop[hops]),
            }
    eligible = [
        int(hops) for hops, result in candidates.items()
        if result.get("role") == "candidate" and result.get("promotion_eligible")
    ]
    invalid = bool(baseline_errors) or any(
        result.get("ratchet", {}).get("decision") == "invalid" for result in candidates.values()
    )
    return {
        "schema_version": "kinegraph.eval.traversal-sweep.v2",
        "baseline_hops": baseline_hops,
        "rollback_hops": baseline_hops,
        "tested_hops": list(SWEEP_HOPS),
        "candidates": candidates,
        "promotion_candidates": [] if invalid else eligible,
        "default_changed": False,
        "decision": "invalid" if invalid else "human_review_required" if eligible else "retain_baseline",
    }


def write_sweep_report(path: str | Path, report: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)
