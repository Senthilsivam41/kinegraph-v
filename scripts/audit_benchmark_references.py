#!/usr/bin/env python3
"""Generate or validate the versioned Kinegraph benchmark reference audit."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.benchmark_reference_audit import (
    accept_reference_audit,
    build_draft_audit,
    load_reference_audit,
    refresh_audit_content_hash,
    validate_reference_audit,
    write_reference_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="eval/kinegraph_benchmark_v1.csv")
    parser.add_argument("--audit", default="eval/kinegraph_benchmark_v1.audit.json")
    parser.add_argument("--write-draft", action="store_true")
    parser.add_argument("--rehash", action="store_true", help="Refresh the hash after deliberate review edits")
    parser.add_argument(
        "--accept",
        action="store_true",
        help="Apply fail-closed human acceptance using verified checked-in source excerpts",
    )
    parser.add_argument("--reviewer-name", default="Benchmark Reviewer")
    parser.add_argument("--dataset-version", default="1.1.0")
    args = parser.parse_args()
    dataset_path = (REPO_ROOT / args.dataset).resolve()
    audit_path = (REPO_ROOT / args.audit).resolve()
    if args.write_draft:
        write_reference_audit(
            audit_path,
            build_draft_audit(
                dataset_path,
                REPO_ROOT,
                dataset_version=f"{args.dataset_version}-draft",
            ),
        )
        print(f"Draft audit written to {audit_path}")
    elif args.accept:
        accepted = accept_reference_audit(
            load_reference_audit(audit_path),
            repo_root=REPO_ROOT,
            reviewer_name=args.reviewer_name,
            dataset_version=args.dataset_version,
        )
        write_reference_audit(audit_path, accepted)
        print(f"Accepted audit written to {audit_path}")
    elif args.rehash:
        write_reference_audit(
            audit_path,
            refresh_audit_content_hash(load_reference_audit(audit_path)),
        )
        print(f"Audit content hash refreshed in {audit_path}")
    with dataset_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    validation = validate_reference_audit(
        rows, load_reference_audit(audit_path), REPO_ROOT, dataset_path
    )
    print(f"dataset_version={validation.dataset_version}")
    print(f"accepted={validation.accepted}")
    print(f"effective_dataset_sha256={validation.effective_dataset_sha256}")
    for warning in validation.warnings:
        print(f"WARNING: {warning}")
    for error in validation.errors:
        print(f"ERROR: {error}")
    return 0 if validation.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
