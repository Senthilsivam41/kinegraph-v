#!/usr/bin/env python3
"""Generate graph-seeded draft benchmark rows (Neo4j MENTIONS or offline fallback)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.benchmark_reference_audit import build_draft_audit, write_reference_audit
from eval.graph_testset_synthesis import (
    extract_offline_seeds,
    load_neo4j_seeds,
    synthesize_graph_rows,
)
from eval.testset_generation import REQUIRED_COLUMNS, load_markdown_documents, chunk_documents_for_generation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--output-version", default="graph-seed-v1-draft")
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--use-neo4j", action="store_true")
    args = parser.parse_args()

    drafts = REPO_ROOT / "eval" / "drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    csv_path = drafts / f"kinegraph_benchmark_{args.output_version}.csv"
    audit_path = drafts / f"kinegraph_benchmark_{args.output_version}.audit.json"

    seeds = []
    if args.use_neo4j:
        from backend.core.config import settings
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        try:
            seeds = load_neo4j_seeds(driver, limit=args.max_rows)
        finally:
            driver.close()
    if not seeds:
        docs = load_markdown_documents(REPO_ROOT / args.docs_dir)
        records = chunk_documents_for_generation(docs, adaptive_enabled=False)
        seeds = extract_offline_seeds([record.text for record in records], limit=args.max_rows)

    rows = synthesize_graph_rows(seeds, max_rows=args.max_rows)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in REQUIRED_COLUMNS})

    write_reference_audit(
        audit_path,
        build_draft_audit(
            csv_path,
            REPO_ROOT,
            dataset_version=args.output_version,
            id_prefix="KGGRAPH",
        ),
    )
    print(f"Wrote {len(rows)} graph-seeded rows to {csv_path}")
    print(f"Wrote draft audit to {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
