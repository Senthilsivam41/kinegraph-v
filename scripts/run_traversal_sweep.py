#!/usr/bin/env python3
"""Run max_hops=1/2/3 sequentially with every other evaluator lever frozen."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.experiment_validation import current_git_revision, load_manifest, sha256_file, working_tree_is_clean
from eval.traversal_sweep import SWEEP_HOPS, build_sweep_report, validate_traversal_manifest, write_sweep_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", default="traversal-sweep")
    parser.add_argument("--baseline-hop", type=int, choices=SWEEP_HOPS, default=2)
    parser.add_argument("--profile", default="hybrid", choices=("hybrid", "hybrid_lexical"))
    parser.add_argument("--max-results", type=int, default=6)
    parser.add_argument("--candidate-pool-size", type=int, default=25)
    parser.add_argument("--benchmark-audit", default="eval/kinegraph_benchmark_v1.audit.json")
    parser.add_argument("--baseline-manifest", type=Path, help="Reuse an accepted baseline from this code revision")
    parser.add_argument("--manifests", nargs=3, type=Path, help="Validate saved 1/2/3-hop manifests without running models")
    parser.add_argument("--output-dir", type=Path, help="New artifact directory; existing directories are never overwritten")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=3600, help="Maximum time for each hop run")
    parser.add_argument("--generation-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-provider", choices=("openrouter", "openai", "nvidia", "fireworks"))
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-embedding-model")
    args = parser.parse_args(argv)
    if not 1 <= args.max_results <= 100 or not 5 <= args.candidate_pool_size <= 100:
        parser.error("max-results must be 1..100 and candidate-pool-size must be 5..100")
    if args.concurrency < 1 or args.timeout_seconds < 1:
        parser.error("concurrency and timeout-seconds must be positive")
    label = re.sub(r"[^a-zA-Z0-9_-]+", "-", args.run_label).strip("-") or "traversal-sweep"
    output = (args.output_dir or REPO_ROOT / "reports" / "traversal_sweeps" / label).resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"output directory already exists: {output}")
    report_path = output / "sweep.json"
    progress = {
        "schema_version": "kinegraph.eval.traversal-sweep.v2",
        "baseline_hops": args.baseline_hop,
        "rollback_hops": args.baseline_hop,
        "default_changed": False,
        "promotion_candidates": [],
        "decision": "invalid",
        "runs": [],
    }

    def reject(reason: str) -> int:
        progress["reasons"] = [reason]
        write_sweep_report(report_path, progress)
        print(f"Traversal sweep rejected: {reason}\nEvidence: {report_path}", file=sys.stderr)
        return 2

    try:
        manifests = []
        baseline_path = args.baseline_manifest.resolve() if args.baseline_manifest else None
        baseline = load_manifest(baseline_path) if baseline_path else None
        if args.manifests:
            manifests = [load_manifest(path) for path in args.manifests]
            if baseline is None:
                baseline = next((item for item in manifests if item.get("pipeline_config", {}).get("retrieval", {}).get("max_hops") == args.baseline_hop), None)
            if baseline is None:
                return reject("no baseline-hop manifest supplied")
        else:
            if baseline is not None:
                errors = validate_traversal_manifest(baseline)
                if errors:
                    return reject("baseline: " + "; ".join(errors))
                if baseline["pipeline_config"]["retrieval"]["max_hops"] != args.baseline_hop:
                    return reject("baseline manifest does not match baseline-hop")
                if baseline["provenance"]["git_revision"] != current_git_revision(REPO_ROOT):
                    return reject("baseline code revision differs; create a fresh baseline")
                manifests.append(baseline)
            order = [args.baseline_hop, *(hop for hop in SWEEP_HOPS if hop != args.baseline_hop)]
            for hops in order:
                if hops == args.baseline_hop and baseline is not None:
                    continue
                if not working_tree_is_clean(REPO_ROOT):
                    return reject("working tree is dirty; commit code and keep generated artifacts ignored or outside the repository")
                run_label = f"{label}-h{hops}"
                manifest_path = output / f"ragas_{run_label}-{args.profile}_manifest.json"
                command = [
                    sys.executable, str(REPO_ROOT / "eval" / "ragas_evaluator.py"),
                    "--profile", args.profile, "--max-hops", str(hops),
                    "--max-results", str(args.max_results),
                    "--candidate-pool-size", str(args.candidate_pool_size),
                    "--benchmark-audit", args.benchmark_audit,
                    "--run-label", run_label, "--concurrency", str(args.concurrency),
                    "--output-dir", str(output),
                    "--regression-run-output", str(output / f"h{hops}_run_output.json"),
                ]
                for option in ("generation_model", "judge_model", "judge_provider", "judge_base_url", "judge_embedding_model"):
                    if getattr(args, option):
                        command.extend(["--" + option.replace("_", "-"), getattr(args, option)])
                if hops != args.baseline_hop:
                    command.extend(["--baseline-manifest", str(baseline_path)])
                log_path = output / f"h{hops}.log"
                run = {"max_hops": hops, "status": "running", "log": str(log_path)}
                progress["runs"].append(run)
                write_sweep_report(report_path, progress)
                print(f"Running max_hops={hops}; log: {log_path}", flush=True)
                try:
                    with log_path.open("w", encoding="utf-8") as log:
                        completed = subprocess.run(command, cwd=REPO_ROOT, check=False, timeout=args.timeout_seconds, stdout=log, stderr=subprocess.STDOUT)
                except subprocess.TimeoutExpired:
                    run["status"] = "failed"
                    return reject(f"max_hops={hops} exceeded {args.timeout_seconds} seconds")
                run["exit_code"] = completed.returncode
                # Exit 3 retains a valid candidate manifest whose generic ratchet rejected it.
                if completed.returncode not in (0, 3) or not manifest_path.is_file():
                    run["status"] = "failed"
                    return reject(f"max_hops={hops} exited {completed.returncode} or produced no manifest")
                manifest = load_manifest(manifest_path)
                errors = validate_traversal_manifest(manifest)
                if errors:
                    run["status"] = "failed"
                    return reject(f"max_hops={hops}: " + "; ".join(errors))
                run.update(status="validated", manifest=str(manifest_path), sha256=sha256_file(manifest_path))
                manifests.append(manifest)
                if hops == args.baseline_hop:
                    baseline, baseline_path = manifest, manifest_path
                write_sweep_report(report_path, progress)
        report = build_sweep_report(baseline, manifests)
    except (OSError, ValueError) as exc:
        return reject(f"{type(exc).__name__}: {exc}")
    report["runs"] = progress["runs"]
    write_sweep_report(report_path, report)
    print(f"Traversal sweep: {report['decision']}. Evidence: {report_path}")
    print("Production defaults unchanged; eligible candidates require human review.")
    return {"human_review_required": 0, "retain_baseline": 3, "invalid": 2}[report["decision"]]


if __name__ == "__main__":
    raise SystemExit(main())
