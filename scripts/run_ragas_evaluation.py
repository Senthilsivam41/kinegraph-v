#!/usr/bin/env python3
"""Compatibility launcher for the canonical, fail-closed RAGAS benchmark.

The accepted benchmark implementation lives in ``eval.ragas_evaluator``.  This
launcher intentionally exposes only live, fixed-profile runs so a convenience
script cannot bypass reference approval, route validation, provenance capture,
or the all-rows RAGAS success gate.
"""
from __future__ import annotations

import argparse
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except ImportError:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the canonical Kinegraph live RAGAS benchmark",
    )
    parser.add_argument(
        "--model",
        default="qwen/qwen3.6-27b",
        help="Compatibility alias for --judge-model",
    )
    parser.add_argument("--generation-model", default="qwen/qwen3.6-27b")
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--max-results", type=int, default=6)
    parser.add_argument("--candidate-pool-size", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--profile",
        choices=["hybrid", "hybrid_lexical", "vectorless", "adaptive_hybrid"],
        default="hybrid",
    )
    parser.add_argument("--run-label", default="latest")
    parser.add_argument(
        "--judge-provider",
        choices=["openrouter", "openai", "nvidia", "fireworks"],
        default=os.getenv("RAGAS_JUDGE_PROVIDER", "openrouter"),
    )
    parser.add_argument("--judge-base-url", default=os.getenv("RAGAS_JUDGE_BASE_URL"))
    parser.add_argument(
        "--judge-embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--judge-smoke-test", action="store_true")
    parser.add_argument(
        "--regression-run-output",
        default=os.getenv("KINEGRAPH_RUN_OUTPUT", "reports/run_output.json"),
        help="Accepted RAGAS output path consumed by regression_gate.py",
    )
    parser.add_argument("--enable-adaptive-routing", action="store_true")
    args = parser.parse_args()

    forwarded = [
        os.path.join(REPO_ROOT, "eval", "ragas_evaluator.py"),
        "--judge-model",
        args.model,
        "--generation-model",
        args.generation_model,
        "--judge-provider",
        args.judge_provider,
        "--judge-embedding-model",
        args.judge_embedding_model,
        "--max-hops",
        str(args.max_hops),
        "--max-results",
        str(args.max_results),
        "--candidate-pool-size",
        str(args.candidate_pool_size),
        "--concurrency",
        str(max(1, args.concurrency)),
        "--profile",
        args.profile,
        "--run-label",
        args.run_label,
        "--regression-run-output",
        args.regression_run_output,
    ]
    if args.judge_base_url:
        forwarded.extend(["--judge-base-url", args.judge_base_url])
    if args.preflight_only:
        forwarded.append("--preflight-only")
    if args.judge_smoke_test:
        forwarded.append("--judge-smoke-test")
    if args.enable_adaptive_routing:
        forwarded.append("--enable-adaptive-routing")

    os.execv(sys.executable, [sys.executable, *forwarded])


if __name__ == "__main__":
    main()
