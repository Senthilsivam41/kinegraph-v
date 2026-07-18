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
    parser.add_argument("--batch-size", type=int, default=200, help="Neo4j/Chroma read and write batch size")
    args = parser.parse_args()
    service = Neo4jService()
    try:
        from backend.services.chroma_service import ChromaService

        chroma = ChromaService()
        print(NodeEnricher(service.driver, chroma.client).enrich(
            dry_run=args.dry_run,
            batch_size=args.batch_size,
        ))
    finally:
        service.close()


if __name__ == "__main__":
    main()
