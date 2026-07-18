# Kinetic-V v2 → v3 schema audit

## Baseline findings

1. The legacy `Entity` graph path creates `Document → MENTIONS → Entity` but stores no source chunk identifiers on entities.
2. The `PropertyGraphIndex` path stores `Chunk`/`__Entity__` nodes, while the idempotency check looks for `__Chunk__`; this label mismatch can make duplicate checks unreliable.
3. Chroma document chunks (`kinetic_vectors`) and PropertyGraphIndex vectors (`kg_nodes`) are separate collections with no common retrieval contract.
4. Graph retrieval returns LlamaIndex node content only; linked `MENTIONS` chunk evidence is not consistently included in the generation context.
5. Entity relationships retain a type but not evidence text, weight, direction metadata, or provenance.
6. Nodes have no stored summary, community, centrality, or traversal-depth metadata.

## v3 contract

`KineticVNode` adds `description`, citable `parent_context_chunk_ids`, a JSON context snapshot, relationship evidence, and graph-positioning fields. Chroma remains the authoritative home for embeddings and full chunk bodies; Neo4j holds stable IDs and a small evidence snapshot. This is intentional: Neo4j properties cannot safely store nested maps and duplicating vectors would introduce drift.

## Migration and validation

Run `python scripts/enrich_kinetic_v_nodes.py --dry-run`, inspect the counts, then rerun without `--dry-run`. The migration is additive and idempotent. It migrates only entities with existing `MENTIONS` links; nodes without source evidence are reported rather than inventing citations.

For the requested six-query baseline, use `eval/kinegraph_benchmark_v1.csv` as the source and record vector, graph, hybrid, and vectorless outputs after services are running. The repository cannot establish the live DB link from source inspection alone; run the migration dry-run against the target environment to verify it.
