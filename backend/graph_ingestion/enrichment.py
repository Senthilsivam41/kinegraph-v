"""Context-first graph-node enrichment with verified vector provenance."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Sequence

from backend.core.config import settings


@dataclass
class ContextChunkLink:
    chunk_id: str
    text: str
    relevance_score: float = 1.0
    source_type: str = "V"
    embedding_verified: bool = False
    vector_collection: str = ""
    vector_record_id: str = ""
    verification_method: str = ""


@dataclass
class RelationshipEvidence:
    target_node_id: str
    relationship_type: str
    weight: float
    evidence_text: str
    direction: str = "OUTGOING"


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
        return {
            "description": self.description,
            "parent_context_chunk_ids": [c.chunk_id for c in self.parent_context_chunks],
            "parent_context_chunks_json": json.dumps([asdict(c) for c in self.parent_context_chunks], ensure_ascii=False),
            "relationships_json": json.dumps([asdict(r) for r in self.relationships], ensure_ascii=False),
            "community_id": self.graph_positioning["community_id"],
            "centrality_score": self.graph_positioning["centrality_score"],
            "depth_from_root": self.graph_positioning["depth_from_root"],
            "schema_version": "3",
            "vector_links_verified": bool(self.parent_context_chunks) and all(
                c.embedding_verified for c in self.parent_context_chunks
            ),
        }


class ChromaChunkValidator:
    """Resolve graph chunk IDs to Chroma records that contain embeddings."""

    def __init__(self, client: Any, collection_names: Optional[Sequence[str]] = None):
        self.client = client
        self.collection_names = tuple(dict.fromkeys(collection_names or ("kg_nodes", settings.CHROMA_COLLECTION_NAME)))

    @staticmethod
    def _has_embedding(value: Any) -> bool:
        if value is None:
            return False
        size = getattr(value, "size", None)
        if size is not None:
            return bool(size)
        try:
            return len(value) > 0
        except TypeError:
            return False

    def resolve(self, raw_chunks: list[dict[str, Any]], batch_size: int) -> tuple[list[ContextChunkLink], set[str]]:
        by_id = {str(c["id"]): c for c in raw_chunks if c.get("id")}
        unresolved = set(by_id)
        verified: dict[str, ContextChunkLink] = {}
        for collection_name in self.collection_names:
            if not unresolved:
                break
            try:
                collection = self.client.get_collection(name=collection_name)
            except Exception:
                continue
            ids = sorted(unresolved)
            for offset in range(0, len(ids), batch_size):
                batch_ids = ids[offset : offset + batch_size]
                payload = collection.get(
                    ids=batch_ids,
                    include=["documents", "metadatas", "embeddings"],
                )
                result_ids = payload.get("ids") or []
                documents = payload.get("documents") or []
                embeddings = payload.get("embeddings")
                for index, chunk_id in enumerate(result_ids):
                    embedding = embeddings[index] if embeddings is not None and index < len(embeddings) else None
                    if not self._has_embedding(embedding):
                        continue
                    raw = by_id[str(chunk_id)]
                    text = documents[index] if index < len(documents) and documents[index] else raw.get("text", "")
                    verified[str(chunk_id)] = ContextChunkLink(
                        chunk_id=str(chunk_id),
                        text=text,
                        relevance_score=float(raw.get("relevance_score") or 1.0),
                        embedding_verified=True,
                        vector_collection=collection_name,
                        vector_record_id=str(chunk_id),
                        verification_method="exact_id",
                    )
                    unresolved.discard(str(chunk_id))
                provenance_ids = [chunk_id for chunk_id in batch_ids if chunk_id in unresolved]
                if not provenance_ids:
                    continue
                payload = collection.get(
                    where={"triplet_source_id": {"$in": provenance_ids}},
                    include=["documents", "metadatas", "embeddings"],
                )
                result_ids = payload.get("ids") or []
                metadatas = payload.get("metadatas") or []
                embeddings = payload.get("embeddings")
                for index, vector_record_id in enumerate(result_ids):
                    metadata = metadatas[index] if index < len(metadatas) else {}
                    chunk_id = str(metadata.get("triplet_source_id") or "")
                    embedding = embeddings[index] if embeddings is not None and index < len(embeddings) else None
                    if chunk_id not in unresolved or not self._has_embedding(embedding):
                        continue
                    raw = by_id[chunk_id]
                    verified[chunk_id] = ContextChunkLink(
                        chunk_id=chunk_id,
                        text=raw.get("text", ""),
                        relevance_score=float(raw.get("relevance_score") or 1.0),
                        embedding_verified=True,
                        vector_collection=collection_name,
                        vector_record_id=str(vector_record_id),
                        verification_method="triplet_source_id",
                    )
                    unresolved.discard(chunk_id)
        return [verified[cid] for cid in by_id if cid in verified], unresolved


class NodeEnricher:
    """Enrich Neo4j entities using graph evidence verified against Chroma."""

    def __init__(
        self,
        driver: Any,
        chroma_client: Any,
        description_generator: Optional[Callable[[dict, list[ContextChunkLink]], str]] = None,
        collection_names: Optional[Sequence[str]] = None,
    ):
        self.driver = driver
        self.validator = ChromaChunkValidator(chroma_client, collection_names)
        self.description_generator = description_generator or self._default_description

    @staticmethod
    def _default_description(node: dict, chunks: list[ContextChunkLink]) -> str:
        name = node.get("name") or node.get("id") or "Unnamed entity"
        entity_type = node.get("type") or node.get("label") or "entity"
        evidence = " ".join(c.text.strip() for c in chunks[:2]).replace("\n", " ")
        return f"{name} is a {entity_type}. Evidence: {evidence[:420]}"

    @staticmethod
    def _graph_positioning(topology: dict[str, set[str]], rooted: set[str]) -> dict[str, dict[str, Any]]:
        max_degree = max((len(edges) for edges in topology.values()), default=1) or 1
        depths = {node_id: 1 for node_id in rooted}
        frontier = list(rooted)
        while frontier:
            node_id = frontier.pop(0)
            for neighbor in topology.get(node_id, set()):
                candidate = depths[node_id] + 1
                if neighbor not in depths or candidate < depths[neighbor]:
                    depths[neighbor] = candidate
                    frontier.append(neighbor)
        positions: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        for seed in sorted(topology):
            if seed in seen:
                continue
            stack, component = [seed], []
            while stack:
                node_id = stack.pop()
                if node_id in seen:
                    continue
                seen.add(node_id)
                component.append(node_id)
                stack.extend(topology.get(node_id, set()) - seen)
            community_id = min(component)
            for node_id in component:
                positions[node_id] = {
                    "community_id": community_id,
                    "centrality_score": round(len(topology.get(node_id, set())) / max_degree, 6),
                    "depth_from_root": depths.get(node_id, -1),
                }
        return positions

    @staticmethod
    def _relationship(raw: dict[str, Any], node: dict, chunks: list[ContextChunkLink]) -> RelationshipEvidence:
        explicit_evidence = (raw.get("evidence") or "").strip()
        target = str(raw.get("target") or "")
        rel_type = raw.get("type") or "RELATED_TO"
        if explicit_evidence:
            evidence = explicit_evidence
            weight = float(raw.get("weight") or 1.0)
        else:
            source = node.get("name") or node.get("id") or "This entity"
            excerpt = chunks[0].text.replace("\n", " ")[:240]
            evidence = f"{source} {rel_type} {target}. Supporting context: {excerpt}"
            weight = min(0.95, 0.55 + (0.05 * len(chunks)))
        return RelationshipEvidence(target, rel_type, round(max(0.0, min(weight, 1.0)), 4), evidence, raw.get("direction") or "OUTGOING")

    def enrich(
        self,
        batch_size: int = 200,
        dry_run: bool = False,
        chunk_ids: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        selection = [str(chunk_id) for chunk_id in chunk_ids or []]
        query = """
        MATCH (e) WHERE (e:Entity OR e:`__Entity__`)
          AND (NOT $restricted OR EXISTS { MATCH (e)<-[:MENTIONS]-(selected) WHERE selected.id IN $chunk_ids })
        CALL {
          WITH e OPTIONAL MATCH (e)<-[m:MENTIONS]-(c)
          WHERE c:Chunk OR c:`__Chunk__`
          RETURN [chunk IN collect(DISTINCT {id: c.id, text: c.text}) WHERE chunk.id IS NOT NULL][..10] AS chunks
        }
        CALL {
          WITH e OPTIONAL MATCH (e)-[r]-(target)
          WHERE (target:Entity OR target:`__Entity__`) AND type(r) <> 'MENTIONS'
          RETURN [rel IN collect(DISTINCT {
            target: coalesce(target.id, target.name), target_element_id: elementId(target),
            type: coalesce(r.type, type(r)), evidence: r.evidence_text, weight: r.weight,
            direction: CASE WHEN elementId(startNode(r)) = elementId(e) THEN 'OUTGOING' ELSE 'INCOMING' END
          }) WHERE rel.target IS NOT NULL][..20] AS relationships
        }
        RETURN elementId(e) AS element_id, properties(e) AS node, chunks, relationships
        ORDER BY element_id SKIP $offset LIMIT $batch_size
        """
        topology_query = """
        MATCH (e) WHERE e:Entity OR e:`__Entity__`
        OPTIONAL MATCH (e)-[r]-(other) WHERE (other:Entity OR other:`__Entity__`) AND type(r) <> 'MENTIONS'
        OPTIONAL MATCH (e)<-[:MENTIONS]-(root) WHERE root:Chunk OR root:`__Chunk__`
        RETURN elementId(e) AS id, collect(DISTINCT elementId(other)) AS neighbors, count(DISTINCT root) > 0 AS rooted
        """
        enriched = skipped = verified_count = missing_count = scanned = 0
        with self.driver.session() as session:
            topology_records = list(session.run(topology_query))
            topology = {r["id"]: {n for n in r["neighbors"] if n} for r in topology_records}
            rooted = {r["id"] for r in topology_records if r["rooted"]}
            positions = self._graph_positioning(topology, rooted)
            offset = 0
            while True:
                records = list(session.run(
                    query, restricted=bool(selection), chunk_ids=selection,
                    offset=offset, batch_size=batch_size,
                ))
                if not records:
                    break
                updates = []
                for record in records:
                    scanned += 1
                    raw_chunks = [c for c in record["chunks"] if c.get("id") and c.get("text")]
                    chunks, missing = self.validator.resolve(raw_chunks, batch_size)
                    missing_count += len(missing)
                    verified_count += len(chunks)
                    if not chunks:
                        skipped += 1
                        continue
                    node = dict(record["node"])
                    relationships = [self._relationship(r, node, chunks) for r in record["relationships"]]
                    vnode = KineticVNode(
                        node_id=str(node.get("id") or node.get("name") or record["element_id"]),
                        entity_type=str(node.get("type") or "Entity"),
                        description=node.get("description") or self.description_generator(node, chunks),
                        parent_context_chunks=chunks,
                        relationships=relationships,
                        graph_positioning=positions[record["element_id"]],
                    )
                    updates.append({"element_id": record["element_id"], "properties": vnode.neo4j_properties()})
                    enriched += 1
                if updates and not dry_run:
                    session.run(
                        "UNWIND $updates AS row MATCH (e) WHERE elementId(e) = row.element_id SET e += row.properties",
                        updates=updates,
                    )
                offset += batch_size
        return {
            "enriched": enriched,
            "skipped_without_verified_context": skipped,
            "scanned": scanned,
            "verified_vector_links": verified_count,
            "missing_vector_links": missing_count,
            "complete": missing_count == 0 and skipped == 0,
        }
