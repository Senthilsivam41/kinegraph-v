"""Benchmark acceptance gates for ADR-003 and controlled reranker experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


ACCEPTANCE_POLICY_VERSION = "kinegraph.retrieval-acceptance.v1"


@dataclass(frozen=True)
class RetrievalAcceptancePolicy:
    required_profiles: tuple[str, ...] = ("hybrid", "hybrid_lexical", "vectorless")
    required_metrics: tuple[str, ...] = (
        "precision_at_5",
        "recall_at_5",
        "ndcg_at_5",
        "context_precision",
        "context_recall",
        "p95_latency_ms",
        "candidate_provenance_completeness",
    )
    minimum_provenance_completeness: float = 1.0
    maximum_precision_regression: float = 0.02
    maximum_latency_regression_fraction: float = 0.50


def _lookup(report: Mapping[str, Any], metric: str) -> Any:
    if metric in report:
        return report[metric]
    for section in ("summary", "ir_metrics", "per_metric", "retrieval_diagnostics"):
        nested = report.get(section)
        if isinstance(nested, Mapping) and metric in nested:
            value = nested[metric]
            if isinstance(value, Mapping):
                return value.get("mean", value.get("score"))
            return value
    return None


def assess_retrieval_benchmark(
    profile_reports: Mapping[str, Mapping[str, Any]],
    policy: RetrievalAcceptancePolicy | None = None,
) -> dict[str, Any]:
    """Reject incomplete, heuristic, or provenance-poor ADR-003 benchmarks."""
    policy = policy or RetrievalAcceptancePolicy()
    failures = []
    profile_results = {}
    for profile in policy.required_profiles:
        report = profile_reports.get(profile)
        if not isinstance(report, Mapping):
            failures.append(f"missing required profile: {profile}")
            continue
        ragas_accepted = bool(
            report.get("ragas_accepted")
            or (report.get("summary") or {}).get("ragas_accepted")
            or (report.get("summary") or {}).get("accepted_as_ragas")
        )
        profile_failures = []
        if not ragas_accepted:
            profile_failures.append("RAGAS result is not accepted")
        measured = {}
        for metric in policy.required_metrics:
            value = _lookup(report, metric)
            measured[metric] = value
            if value is None:
                profile_failures.append(f"missing metric: {metric}")
        provenance = measured.get("candidate_provenance_completeness")
        if provenance is not None and float(provenance) < policy.minimum_provenance_completeness:
            profile_failures.append("candidate provenance is incomplete")
        if profile_failures:
            failures.extend(f"{profile}: {failure}" for failure in profile_failures)
        profile_results[profile] = {
            "accepted": not profile_failures,
            "metrics": measured,
            "failures": profile_failures,
        }
    return {
        "policy_version": ACCEPTANCE_POLICY_VERSION,
        "accepted": not failures,
        "profiles": profile_results,
        "failures": failures,
    }


def assess_cross_encoder_experiment(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    policy: RetrievalAcceptancePolicy | None = None,
) -> dict[str, Any]:
    """Require a one-lever, slice-level gain before cross-encoder promotion."""
    policy = policy or RetrievalAcceptancePolicy()
    def flatten(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
        flat = {}
        for key, value in mapping.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                flat.update(flatten(value, name))
            else:
                flat[name] = value
        return flat

    def config(payload: Mapping[str, Any]) -> dict[str, Any]:
        raw = payload.get("configuration") or payload.get("pipeline_config") or {}
        return flatten(raw) if isinstance(raw, Mapping) else {}

    def slices(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        raw = payload.get("slices")
        if not isinstance(raw, Mapping):
            raw = ((payload.get("report") or {}).get("per_category") or {})
        normalized = {}
        for name, value in raw.items():
            if not isinstance(value, Mapping):
                continue
            normalized[str(name)] = {
                **(value.get("metrics") or {}),
                **(value.get("retrieval_metrics") or {}),
                **{
                    key: item
                    for key, item in value.items()
                    if isinstance(item, (int, float))
                },
            }
        return normalized

    def p95(payload: Mapping[str, Any]) -> float:
        direct = payload.get("p95_latency_ms")
        if direct is None:
            direct = (
                ((payload.get("report") or {}).get("retrieval_diagnostics") or {})
                .get("p95_latency_ms")
            )
        return float(direct or 0.0)

    baseline_config = config(baseline)
    candidate_config = config(candidate)
    changed = sorted(
        key
        for key in set(baseline_config) | set(candidate_config)
        if baseline_config.get(key) != candidate_config.get(key)
    )
    failures = []
    allowed_key = next((
        key for key in changed if key.endswith("enable_cross_encoder_reranking")
    ), "enable_cross_encoder_reranking")
    if changed != [allowed_key]:
        failures.append(
            "controlled experiment must change only enable_cross_encoder_reranking"
        )
    if candidate_config.get(allowed_key) is not True:
        failures.append("candidate did not enable the cross-encoder")
    baseline_slices = slices(baseline)
    candidate_slices = slices(candidate)
    common_slices = sorted(set(baseline_slices) & set(candidate_slices))
    if not common_slices:
        failures.append("no comparable query-category slices")

    slice_results = {}
    gained_slice = False
    for slice_name in common_slices:
        before = baseline_slices[slice_name]
        after = candidate_slices[slice_name]
        before_precision = float(before.get("context_precision") or 0.0)
        after_precision = float(after.get("context_precision") or 0.0)
        before_ndcg = float(before.get("ndcg_at_5") or 0.0)
        after_ndcg = float(after.get("ndcg_at_5") or 0.0)
        precision_delta = after_precision - before_precision
        ndcg_delta = after_ndcg - before_ndcg
        gained_slice = gained_slice or precision_delta > 0 or ndcg_delta > 0
        if precision_delta < -policy.maximum_precision_regression:
            failures.append(f"{slice_name}: context precision regressed")
        slice_results[slice_name] = {
            "context_precision_delta": round(precision_delta, 6),
            "ndcg_at_5_delta": round(ndcg_delta, 6),
        }
    if common_slices and not gained_slice:
        failures.append("cross-encoder showed no slice-level precision or nDCG gain")

    baseline_p95 = p95(baseline)
    candidate_p95 = p95(candidate)
    latency_regression = (
        (candidate_p95 - baseline_p95) / baseline_p95 if baseline_p95 else None
    )
    if (
        latency_regression is not None
        and latency_regression > policy.maximum_latency_regression_fraction
    ):
        failures.append("p95 latency regression exceeds policy")

    return {
        "policy_version": ACCEPTANCE_POLICY_VERSION,
        "accepted": not failures,
        "changed_levers": changed,
        "slice_results": slice_results,
        "p95_latency_regression_fraction": latency_regression,
        "failures": failures,
    }
