# Kinetic-V v2 → v3 schema audit

## Baseline findings

1. The legacy `Entity` graph path creates `Document → MENTIONS → Entity` but stores no source chunk identifiers on entities.
2. The `PropertyGraphIndex` path stores `Chunk`/`__Entity__` nodes, while the idempotency check looks for `__Chunk__`; this label mismatch can make duplicate checks unreliable.
3. Chroma document chunks (`kinetic_vectors`) and PropertyGraphIndex vectors (`kg_nodes`) are separate collections with no common retrieval contract.
4. Graph retrieval returns LlamaIndex node content only; linked `MENTIONS` chunk evidence is not consistently included in the generation context.
5. Entity relationships retain a type but not evidence text, weight, direction metadata, or provenance.
6. Nodes have no stored summary, community, centrality, or traversal-depth metadata.

## v3 contract

`KineticVNode` adds `description`, citable `parent_context_chunk_ids`, a JSON context snapshot, relationship evidence, and graph-positioning fields. Chroma remains the authoritative home for embeddings and full chunk bodies; Neo4j holds stable IDs and a small evidence snapshot. A link is persisted only after its Chroma record and embedding are found in `kg_nodes` or `kinetic_vectors`. This is intentional: Neo4j properties cannot safely store nested maps and duplicating vectors would introduce drift.

## Migration and validation

Run `python scripts/enrich_kinetic_v_nodes.py --dry-run --batch-size 200`, inspect `verified_vector_links` and `missing_vector_links`, then rerun without `--dry-run`. The migration is additive, idempotent, and paginated. It computes connected-component communities, normalized degree centrality, and distance from chunk roots without requiring Neo4j GDS. Missing relationship evidence receives a context-grounded fallback and a derived confidence weight.

Both PropertyGraphIndex ingestion and the legacy worker ingestion path invoke enrichment automatically after graph and vector writes. Their result includes an `enrichment` summary (or `last_enrichment_result` for `Neo4jService`) so a vector-link failure cannot remain invisible.

For the requested six-query baseline, use `eval/kinegraph_benchmark_v1.csv` as the source and record vector, graph, hybrid, and vectorless outputs after services are running. The repository cannot establish the live DB link from source inspection alone; run the migration dry-run against the target environment to verify it.
