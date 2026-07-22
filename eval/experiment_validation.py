"""Reproducible, fail-closed validation for Kinegraph benchmark experiments.

The policy is inspired by OpenResearch's one-lever ratchet loop, but only
contains controls that are wired into Kinegraph's live workflow.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_METRIC_WEIGHTS = {
    "faithfulness": 0.35,
    "context_precision": 0.30,
    "context_recall": 0.20,
    "answer_relevancy": 0.15,
}


@dataclass(frozen=True)
class ValidationPolicy:
    """Acceptance thresholds for one controlled experiment cycle."""

    metric_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_METRIC_WEIGHTS)
    )
    tie_tolerance: float = 0.01
    max_metric_regression: float = 0.05
    require_single_lever: bool = True
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 42

    def validate(self) -> None:
        if not self.metric_weights:
            raise ValueError("metric_weights cannot be empty")
        if any(weight < 0 for weight in self.metric_weights.values()):
            raise ValueError("metric weights cannot be negative")
        if not math.isclose(sum(self.metric_weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("metric weights must sum to 1.0")
        if not 0 <= self.tie_tolerance <= 1:
            raise ValueError("tie_tolerance must be between 0 and 1")
        if not 0 <= self.max_metric_regression <= 1:
            raise ValueError("max_metric_regression must be between 0 and 1")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def validate_metric_values(
    values: Mapping[str, Any],
    required_metrics: Sequence[str],
) -> None:
    """Require every benchmark metric to be numeric, finite, and in [0, 1]."""
    missing = [metric for metric in required_metrics if metric not in values]
    if missing:
        raise ValueError(f"missing required metric(s): {', '.join(missing)}")
    invalid = []
    for metric in required_metrics:
        try:
            value = float(values[metric])
        except (TypeError, ValueError):
            invalid.append(metric)
            continue
        if not math.isfinite(value) or not 0 <= value <= 1:
            invalid.append(metric)
    if invalid:
        raise ValueError(
            "metrics must be finite values between 0 and 1: " + ", ".join(invalid)
        )


def weighted_composite(
    values: Mapping[str, Any],
    policy: ValidationPolicy | None = None,
) -> float:
    policy = policy or ValidationPolicy()
    policy.validate()
    validate_metric_values(values, tuple(policy.metric_weights))
    return sum(float(values[metric]) * weight for metric, weight in policy.metric_weights.items())


def bootstrap_mean_interval(
    values: Sequence[float],
    policy: ValidationPolicy | None = None,
) -> tuple[float, float]:
    """Return a deterministic 95% bootstrap interval for a sample mean."""
    policy = policy or ValidationPolicy()
    policy.validate()
    sample = np.asarray(values, dtype=float)
    if sample.size == 0 or not np.isfinite(sample).all():
        raise ValueError("bootstrap values must be non-empty and finite")
    if sample.size == 1:
        value = float(sample[0])
        return value, value
    rng = np.random.default_rng(policy.bootstrap_seed)
    draws = rng.choice(sample, size=(policy.bootstrap_samples, sample.size), replace=True)
    means = draws.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_git_revision(repo_root: str | Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def working_tree_is_clean(repo_root: str | Path) -> bool:
    try:
        output = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return not output.strip()
    except (OSError, subprocess.CalledProcessError):
        return False


def _flatten(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten(value, path))
        else:
            flattened[path] = value
    return flattened


def changed_levers(
    baseline_config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
) -> list[str]:
    baseline = _flatten(baseline_config)
    candidate = _flatten(candidate_config)
    return sorted(
        key for key in set(baseline) | set(candidate)
        if baseline.get(key) != candidate.get(key)
    )


def build_manifest(
    *,
    run_label: str,
    repo_root: str | Path,
    dataset_path: str | Path,
    pipeline_config: Mapping[str, Any],
    models: Mapping[str, str],
    report: Mapping[str, Any],
    artifacts: Mapping[str, str],
    policy: ValidationPolicy | None = None,
    git_revision: str | None = None,
    working_tree_clean: bool | None = None,
) -> dict[str, Any]:
    policy = policy or ValidationPolicy()
    policy.validate()
    dataset = Path(dataset_path).resolve()
    model_values = dict(models)
    risk_flags = []
    if model_values.get("generation") == model_values.get("judge"):
        risk_flags.append("judge_model_matches_generation_model")
    return {
        "schema_version": 1,
        "run_label": run_label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_revision": git_revision or current_git_revision(repo_root),
            "working_tree_clean": (
                working_tree_is_clean(repo_root)
                if working_tree_clean is None
                else working_tree_clean
            ),
            "dataset_path": str(dataset),
            "dataset_sha256": sha256_file(dataset),
        },
        "pipeline_config": dict(pipeline_config),
        "models": model_values,
        "evaluation_risk_flags": risk_flags,
        "validation_policy": asdict(policy),
        "report": dict(report),
        "artifacts": dict(artifacts),
    }


def compare_manifests(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    policy: ValidationPolicy | None = None,
) -> dict[str, Any]:
    """Apply dataset/model invariants, one-lever attribution, and ratchet gates."""
    policy = policy or ValidationPolicy()
    policy.validate()
    reasons: list[str] = []

    for name, manifest in (("baseline", baseline), ("candidate", candidate)):
        summary = manifest.get("report", {}).get("summary", {})
        if not summary.get("accepted_as_ragas", False):
            reasons.append(f"{name} run is not accepted as RAGAS")
    if baseline.get("validation_policy") != candidate.get("validation_policy"):
        reasons.append("validation policy changed and would confound the comparison")

    baseline_hash = baseline.get("provenance", {}).get("dataset_sha256")
    candidate_hash = candidate.get("provenance", {}).get("dataset_sha256")
    if not baseline_hash or baseline_hash != candidate_hash:
        reasons.append("baseline and candidate must use the same frozen dataset hash")
    for name, manifest in (("baseline", baseline), ("candidate", candidate)):
        if not manifest.get("provenance", {}).get("working_tree_clean", False):
            reasons.append(f"{name} run was produced from a dirty or unknown working tree")

    baseline_models = baseline.get("models", {})
    candidate_models = candidate.get("models", {})
    for invariant in ("judge", "embedding"):
        if baseline_models.get(invariant) != candidate_models.get(invariant):
            reasons.append(f"{invariant} model changed and would confound the comparison")

    baseline_experiment = {
        "pipeline": baseline.get("pipeline_config", {}),
        "generation_model": baseline_models.get("generation"),
        "grounding_critic_model": baseline_models.get("grounding_critic"),
        "code_revision": baseline.get("provenance", {}).get("git_revision"),
    }
    candidate_experiment = {
        "pipeline": candidate.get("pipeline_config", {}),
        "generation_model": candidate_models.get("generation"),
        "grounding_critic_model": candidate_models.get("grounding_critic"),
        "code_revision": candidate.get("provenance", {}).get("git_revision"),
    }
    levers = changed_levers(baseline_experiment, candidate_experiment)
    if policy.require_single_lever and len(levers) != 1:
        reasons.append(f"expected exactly one changed lever, found {len(levers)}")

    baseline_metrics = {
        metric: baseline.get("report", {}).get("per_metric", {}).get(metric, {}).get("mean")
        for metric in policy.metric_weights
    }
    candidate_metrics = {
        metric: candidate.get("report", {}).get("per_metric", {}).get(metric, {}).get("mean")
        for metric in policy.metric_weights
    }
    try:
        baseline_score = weighted_composite(baseline_metrics, policy)
        candidate_score = weighted_composite(candidate_metrics, policy)
    except ValueError as exc:
        reasons.append(str(exc))
        baseline_score = candidate_score = float("nan")

    metric_deltas = {
        metric: round(float(candidate_metrics[metric]) - float(baseline_metrics[metric]), 4)
        for metric in policy.metric_weights
        if baseline_metrics.get(metric) is not None and candidate_metrics.get(metric) is not None
    }
    large_regressions = {
        metric: delta for metric, delta in metric_deltas.items()
        if delta < -policy.max_metric_regression
    }

    if reasons:
        decision = "invalid"
    elif large_regressions:
        decision = "revert"
        reasons.append("one or more metrics exceeded the maximum allowed regression")
    elif candidate_score >= baseline_score - policy.tie_tolerance:
        decision = "keep"
        reasons.append("weighted composite improved or tied within tolerance")
    else:
        decision = "revert"
        reasons.append("weighted composite regressed beyond tie tolerance")

    return {
        "decision": decision,
        "reasons": reasons,
        "changed_levers": levers,
        "baseline_composite": round(baseline_score, 4) if math.isfinite(baseline_score) else None,
        "candidate_composite": round(candidate_score, 4) if math.isfinite(candidate_score) else None,
        "composite_delta": (
            round(candidate_score - baseline_score, 4)
            if math.isfinite(baseline_score) and math.isfinite(candidate_score)
            else None
        ),
        "metric_deltas": metric_deltas,
        "large_metric_regressions": large_regressions,
        "policy": asdict(policy),
    }


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError("manifest must contain a JSON object")
    return payload


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(destination)
