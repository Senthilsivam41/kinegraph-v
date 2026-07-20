"""Audit Neo4j ontology coverage against the checked-in golden benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from neo4j import GraphDatabase

from backend.core.config import settings
from backend.graph_ingestion.schema import OntologySchema
from backend.graph_ingestion.schema_coverage import (
    audit_schema_coverage,
    load_benchmark_rows,
    load_graph_snapshot,
    render_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="eval/kinegraph_benchmark_v1.csv")
    parser.add_argument("--schema", default="config/ontology_schema.yaml")
    parser.add_argument("--output-prefix", default="reports/schema_coverage")
    args = parser.parse_args()

    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    try:
        entities, relationships = load_graph_snapshot(driver)
    finally:
        driver.close()

    report = audit_schema_coverage(
        OntologySchema(args.schema),
        load_benchmark_rows(args.benchmark),
        entities,
        relationships,
    )
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    markdown_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Schema coverage JSON: {json_path}")
    print(f"Schema coverage Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
