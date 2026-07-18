"""One-time v2 -> v3 graph enrichment migration.

Run with ``python scripts/enrich_kinetic_v_nodes.py --dry-run`` first.  It only
updates entity properties and never deletes or re-embeds existing data.
"""
import argparse

from backend.graph_ingestion.enrichment import NodeEnricher
from backend.services.neo4j_service import Neo4jService


def main() -> None:
    parser = argparse.ArgumentParser(description="Add v3 context metadata to graph entities")
    parser.add_argument("--dry-run", action="store_true", help="report candidates without writing")
    args = parser.parse_args()
    service = Neo4jService()
    try:
        print(NodeEnricher(service.driver).enrich(dry_run=args.dry_run))
    finally:
        service.close()


if __name__ == "__main__":
    main()
