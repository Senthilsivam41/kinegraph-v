#!/usr/bin/env python3
"""Generate paired recursive vs adaptive draft corpora for chunk-policy A/B."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.testset_generation import generate_testset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--dry-run-chunks-only", action="store_true")
    args = parser.parse_args()

    recursive = generate_testset(
        repo_root=REPO_ROOT,
        docs_dir=args.docs_dir,
        size=args.size,
        output_version="chunk-ab-recursive-draft",
        adaptive_enabled=False,
        write_spike_note_file=False,
        dry_run_chunks_only=args.dry_run_chunks_only,
    )
    adaptive = generate_testset(
        repo_root=REPO_ROOT,
        docs_dir=args.docs_dir,
        size=args.size,
        output_version="chunk-ab-adaptive-draft",
        adaptive_enabled=True,
        write_spike_note_file=False,
        dry_run_chunks_only=args.dry_run_chunks_only,
    )
    print(f"recursive={recursive.csv_path} policy={recursive.chunk_policy_version}")
    print(f"adaptive={adaptive.csv_path} policy={adaptive.chunk_policy_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
