"""Promotion gates for the bounded max-hops ablation in GitHub issue #44."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from eval.experiment_validation import ValidationPolicy, compare_manifests


SWEEP_HOPS = (1, 2, 3)
TWO_REFERENCE_CATEGORY = "two_reference_facets"


def _category_recall(manifest: Mapping[str, Any]) -> float | None:
    value = (
        manifest.get("report", {})
        .get("per_category", {})
        .get(TWO_REFERENCE_CATEGORY, {})
        .get("metrics", {})
        .get("context_recall")
    )
    return float(value) if isinstance(value, (int, float)) else None


def _graph_p95(manifest: Mapping[str, Any]) -> float | None:
    value = (
        manifest.get("report", {})
        .get("retrieval_diagnostics", {})
        .get("graph_stage_latency_ms", {})
        .get("p95")
    )
    return float(value) if isinstance(value, (int, float)) else None


def _max_hops(manifest: Mapping[str, Any]) -> int | None:
    value = manifest.get("pipeline_config", {}).get("retrieval", {}).get("max_hops")
    return int(value) if isinstance(value, int) else None


def evaluate_traversal_candidate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    policy: ValidationPolicy | None = None,
) -> dict[str, Any]:
    """Apply the issue-specific gates in addition to the generic ratchet."""
    policy = policy or ValidationPolicy()
    comparison = compare_manifests(baseline, candidate, policy)
    reasons: list[str] = []

    expected_lever = "pipeline.retrieval.max_hops"
    if comparison.get("changed_levers") != [expected_lever]:
        reasons.append("max_hops must be the only changed experiment lever")

    baseline_recall = _category_recall(baseline)
    candidate_recall = _category_recall(candidate)
    category_delta = (
        round(candidate_recall - baseline_recall, 4)
        if baseline_recall is not None and candidate_recall is not None else None
    )
    if category_delta is None or category_delta < 0.05:
        reasons.append("two-reference-facet context recall must improve by at least 0.05")

    precision = (
        candidate.get("report", {}).get("per_metric", {})
        .get("context_precision", {}).get("mean")
    )
    if not isinstance(precision, (int, float)) or precision < 0.90:
        reasons.append("overall context precision must remain at least 0.90")

    baseline_p95 = _graph_p95(baseline)
    candidate_p95 = _graph_p95(candidate)
    latency_ratio = (
        round(candidate_p95 / baseline_p95, 4)
        if baseline_p95 and candidate_p95 is not None else None
    )
    if latency_ratio is None or latency_ratio > 1.25:
        reasons.append("graph retrieval p95 latency must remain within 25% of baseline")

    graph_paths = (
        candidate.get("report", {}).get("retrieval_diagnostics", {}).get("graph_paths", {})
    )
    if not graph_paths.get("all_complete", False):
        reasons.append("all returned traversal paths must be complete and evidence-bearing")
    if graph_paths.get("traversal_failure_count", 0):
        reasons.append("candidate contains graph traversal failures")

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
            "delta": category_delta,
        },
        "context_precision": precision,
        "graph_p95_latency_ms": {
            "baseline": baseline_p95,
            "candidate": candidate_p95,
            "ratio": latency_ratio,
        },
        "graph_paths": graph_paths,
        "ratchet": comparison,
    }


def build_sweep_report(
    baseline: Mapping[str, Any],
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_hop = {_max_hops(manifest): manifest for manifest in manifests}
    if set(by_hop) != set(SWEEP_HOPS):
        raise ValueError("traversal sweep requires exactly one manifest for max_hops 1, 2, and 3")
    baseline_hops = _max_hops(baseline)
    if baseline_hops not in SWEEP_HOPS:
        raise ValueError("baseline max_hops must be one of 1, 2, or 3")
    candidates = {}
    for hops in SWEEP_HOPS:
        if hops == baseline_hops:
            candidates[str(hops)] = {
                "role": "accepted_baseline",
                "promotion_eligible": True,
                "rollback_default": True,
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
    return {
        "schema_version": "kinegraph.eval.traversal-sweep.v1",
        "baseline_hops": baseline_hops,
        "rollback_hops": baseline_hops,
        "tested_hops": list(SWEEP_HOPS),
        "candidates": candidates,
        "promotion_candidates": eligible,
        "default_changed": False,
        "decision": "human_review_required" if eligible else "retain_baseline",
    }


def write_sweep_report(path: str | Path, report: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
