#!/usr/bin/env python3
"""Compare two accepted benchmark manifests and print a metric delta table."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.experiment_validation import compare_manifests
from eval.retrieval_experiments import (
    EXPERIMENTS,
    compare_retrieval_experiment,
    load_experiment_provenance,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return payload


def _metric_means(manifest: Mapping[str, Any]) -> dict[str, float]:
    per_metric = (manifest.get("report") or {}).get("per_metric") or {}
    means = {
        name: float(details["mean"])
        for name, details in per_metric.items()
        if isinstance(details, Mapping) and details.get("mean") is not None
    }
    ir = (manifest.get("report") or {}).get("ir_metrics") or {}
    for name, details in ir.items():
        if isinstance(details, Mapping) and details.get("mean") is not None:
            means[name] = float(details["mean"])
    summary = (manifest.get("report") or {}).get("summary") or {}
    if summary.get("overall_composite_score") is not None:
        means["overall_composite_score"] = float(summary["overall_composite_score"])
    cost = (manifest.get("report") or {}).get("cost") or {}
    if cost.get("estimated_cost_usd") is not None:
        means["estimated_cost_usd"] = float(cost["estimated_cost_usd"])
    return means


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--output", type=Path, help="Write the paired experiment decision and deltas as JSON")
    parser.add_argument(
        "--skip-ratchet",
        action="store_true",
        help="Print deltas even when the one-lever ratchet rejects the pair",
    )
    args = parser.parse_args()
    if args.experiment:
        if args.output is None or args.skip_ratchet:
            parser.error("--experiment requires --output and does not support --skip-ratchet")
        paths = (Path(args.baseline), Path(args.candidate))
        try:
            baseline, candidate = (_load(path) for path in paths)
            comparison = compare_retrieval_experiment(
                args.experiment, baseline, candidate,
                load_experiment_provenance(baseline), load_experiment_provenance(candidate),
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            comparison = {
                "experiment": args.experiment, "decision": "invalid",
                "accepted": False, "reasons": [str(exc)],
            }
        comparison["manifests"] = {
            name: str(path.resolve()) for name, path in zip(("baseline", "candidate"), paths)
        }
        # Exclusive creation protects the input manifests and previous experiment reports.
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(comparison, output, indent=2, allow_nan=False)
            output.write("\n")
        print(f"{comparison['decision'].upper()} | {args.output}")
        return 0 if comparison["accepted"] else 2
    if args.output is not None:
        parser.error("--output requires --experiment")
    baseline = _load(Path(args.baseline))
    candidate = _load(Path(args.candidate))
    comparison = compare_manifests(baseline, candidate)
    accepted = comparison.get("accepted")
    if accepted is None:
        accepted = comparison.get("decision") == "keep"
    print(f"accepted={accepted}")
    print(f"levers={comparison.get('changed_levers')}")
    for reason in comparison.get("reasons") or []:
        print(f"REASON: {reason}")
    if not accepted and not args.skip_ratchet:
        return 2

    baseline_means = _metric_means(baseline)
    candidate_means = _metric_means(candidate)
    names = sorted(set(baseline_means) | set(candidate_means))
    print("metric\tbaseline\tcandidate\tdelta")
    for name in names:
        left = baseline_means.get(name)
        right = candidate_means.get(name)
        if left is None or right is None:
            delta = ""
        else:
            delta = f"{right - left:+.4f}"
        print(
            f"{name}\t"
            f"{'' if left is None else f'{left:.4f}'}\t"
            f"{'' if right is None else f'{right:.4f}'}\t"
            f"{delta}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
