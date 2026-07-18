"""Context-first graph-node enrichment.

Neo4j properties cannot contain nested maps, so ``parent_context_chunks`` is
represented by stable chunk ids plus a JSON snapshot.  The authoritative chunk
text and embedding remain in Chroma; this avoids duplicating vectors in Neo4j
and keeps graph-to-vector links durable across retrieval paths.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Optional


@dataclass
class ContextChunkLink:
    chunk_id: str
    text: str
    relevance_score: float = 1.0
    source_type: str = "V"


@dataclass
class RelationshipEvidence:
    target_node_id: str
    relationship_type: str
    weight: float = 1.0
    evidence_text: str = ""


@dataclass
class KineticVNode:
    """Serializable v3 representation of an enriched graph node."""

    node_id: str
    entity_type: str
    description: str
    parent_context_chunks: list[ContextChunkLink] = field(default_factory=list)
    relationships: list[RelationshipEvidence] = field(default_factory=list)
    graph_positioning: dict[str, Any] = field(default_factory=dict)

    def neo4j_properties(self) -> dict[str, Any]:
        """Return Neo4j-safe scalar/list properties for this node."""
        return {
            "description": self.description,
            "parent_context_chunk_ids": [c.chunk_id for c in self.parent_context_chunks],
            "parent_context_chunks_json": json.dumps(
                [asdict(c) for c in self.parent_context_chunks], ensure_ascii=False
            ),
            "relationships_json": json.dumps(
                [asdict(r) for r in self.relationships], ensure_ascii=False
            ),
            "community_id": self.graph_positioning.get("community_id"),
            "centrality_score": self.graph_positioning.get("centrality_score", 0.0),
            "depth_from_root": self.graph_positioning.get("depth_from_root", 0),
            "schema_version": "3",
        }


class NodeEnricher:
    """Enrich existing Neo4j entities using their already-linked chunk nodes."""

    def __init__(self, driver: Any, description_generator: Optional[Callable[[dict, list[ContextChunkLink]], str]] = None):
        self.driver = driver
        self.description_generator = description_generator or self._default_description

    @staticmethod
    def _default_description(node: dict, chunks: list[ContextChunkLink]) -> str:
        name = node.get("name") or node.get("id") or "Unnamed entity"
        entity_type = node.get("type") or node.get("label") or "entity"
        evidence = " ".join(c.text.strip() for c in chunks[:2]).replace("\n", " ")
        if evidence:
            return f"{name} is a {entity_type}. Evidence: {evidence[:420]}"
        return f"{name} is a {entity_type} represented in the Kinetic-V knowledge graph."

    def enrich(self, batch_size: int = 200, dry_run: bool = False) -> dict[str, int]:
        """Migrate legacy ``Entity`` and PropertyGraphIndex ``__Entity__`` nodes."""
        query = """
        MATCH (e) WHERE e:Entity OR e:`__Entity__`
        OPTIONAL MATCH (e)<-[m:MENTIONS]-(c)
        WHERE c:Chunk OR c:`__Chunk__`
        WITH e, [chunk IN collect(DISTINCT {id: c.id, text: c.text, evidence: m.evidence_text})
                 WHERE chunk.id IS NOT NULL][..10] AS chunks
        OPTIONAL MATCH (e)-[r]->(target)
        WHERE NOT type(r) = 'MENTIONS'
        RETURN elementId(e) AS element_id, properties(e) AS node, chunks,
               collect(DISTINCT {target: coalesce(target.id, target.name), type: type(r), evidence: r.evidence_text})[..20] AS relationships
        """
        enriched = skipped = 0
        with self.driver.session() as session:
            records = list(session.run(query))
            for record in records:
                node = dict(record["node"])
                raw_chunks = [c for c in record["chunks"] if c.get("id") and c.get("text")]
                chunks = [ContextChunkLink(chunk_id=c["id"], text=c["text"]) for c in raw_chunks]
                if not chunks:
                    skipped += 1
                    continue
                relationships = [
                    RelationshipEvidence(
                        target_node_id=str(r.get("target") or ""),
                        relationship_type=r.get("type") or "RELATED_TO",
                        evidence_text=r.get("evidence") or "",
                    )
                    for r in record["relationships"]
                    if r.get("target")
                ]
                vnode = KineticVNode(
                    node_id=str(node.get("id") or node.get("name") or record["element_id"]),
                    entity_type=str(node.get("type") or "Entity"),
                    description=node.get("description") or self.description_generator(node, chunks),
                    parent_context_chunks=chunks,
                    relationships=relationships,
                )
                if not dry_run:
                    session.run(
                        "MATCH (e) WHERE elementId(e) = $element_id SET e += $properties",
                        element_id=record["element_id"], properties=vnode.neo4j_properties(),
                    )
                enriched += 1
        return {"enriched": enriched, "skipped_without_context": skipped, "scanned": len(records)}
