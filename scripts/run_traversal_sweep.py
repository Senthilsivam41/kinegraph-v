#!/usr/bin/env python3
"""Run max_hops=1/2/3 sequentially with every other evaluator lever frozen."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.experiment_validation import load_manifest
from eval.traversal_sweep import SWEEP_HOPS, build_sweep_report, write_sweep_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", default="traversal-sweep")
    parser.add_argument("--baseline-hop", type=int, choices=SWEEP_HOPS, default=2)
    parser.add_argument("--profile", default="hybrid", choices=("hybrid", "hybrid_lexical"))
    parser.add_argument("--max-results", type=int, default=6)
    parser.add_argument("--candidate-pool-size", type=int, default=25)
    parser.add_argument("--benchmark-audit", default="eval/kinegraph_benchmark_v1.audit.json")
    args = parser.parse_args()

    manifests = []
    for hops in SWEEP_HOPS:
        label = f"{args.run_label}-h{hops}"
        command = [
            sys.executable,
            str(REPO_ROOT / "eval" / "ragas_evaluator.py"),
            "--profile", args.profile,
            "--max-hops", str(hops),
            "--max-results", str(args.max_results),
            "--candidate-pool-size", str(args.candidate_pool_size),
            "--benchmark-audit", args.benchmark_audit,
            "--run-label", label,
        ]
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            print(f"Traversal sweep stopped: max_hops={hops} exited {completed.returncode}.", file=sys.stderr)
            return completed.returncode
        manifest_path = REPO_ROOT / "reports" / f"ragas_{label}-{args.profile}_manifest.json"
        manifests.append(load_manifest(manifest_path))

    baseline = next(
        manifest for manifest in manifests
        if manifest["pipeline_config"]["retrieval"]["max_hops"] == args.baseline_hop
    )
    report = build_sweep_report(baseline, manifests)
    output = REPO_ROOT / "reports" / f"traversal_sweep_{args.run_label}.json"
    write_sweep_report(output, report)
    print(f"Traversal sweep report saved to {output}")
    print("No production default was changed; promotion requires human review of an eligible candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
