#!/usr/bin/env python3
"""Validate ADR-003 profile reports and an optional cross-encoder experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.retrieval_acceptance import (
    assess_cross_encoder_experiment,
    assess_retrieval_benchmark,
)


def _load(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-report", required=True)
    parser.add_argument("--hybrid-lexical-report", required=True)
    parser.add_argument("--vectorless-report", required=True)
    parser.add_argument("--cross-encoder-baseline")
    parser.add_argument("--cross-encoder-candidate")
    args = parser.parse_args()
    if bool(args.cross_encoder_baseline) != bool(args.cross_encoder_candidate):
        parser.error("cross-encoder baseline and candidate must be supplied together")

    result = {
        "retrieval_benchmark": assess_retrieval_benchmark({
            "hybrid": _load(args.hybrid_report),
            "hybrid_lexical": _load(args.hybrid_lexical_report),
            "vectorless": _load(args.vectorless_report),
        })
    }
    if args.cross_encoder_baseline:
        result["cross_encoder_experiment"] = assess_cross_encoder_experiment(
            baseline=_load(args.cross_encoder_baseline),
            candidate=_load(args.cross_encoder_candidate),
        )
    accepted = all(section.get("accepted", False) for section in result.values())
    result["accepted"] = accepted
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    sys.exit(main())
