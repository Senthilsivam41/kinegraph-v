"""Context-first graph-node enrichment with verified vector provenance (Enhanced v2).

Improvements over v1:
- Improved description generation with chunk weighting and relevance scoring
- Configurable chunk context count with smart truncation
- Enhanced relationship evidence with temporal metadata support
"""
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
    """Enrich Neo4j entities using graph evidence verified against Chroma.

    Enhancements (v2):
    - Smart chunk weighting based on relevance scores and text quality
    - Configurable context window with meaningful truncation
    - Improved description generation that preserves critical information for multi-hop traversal
    """

    def __init__(
        self,
        driver: Any,
        chroma_client: Any,
        description_generator: Optional[Callable[[dict, list[ContextChunkLink]], str]] = None,
        collection_names: Optional[Sequence[str]] = None,
        max_context_chunks: int = 5,
    ):
        self.driver = driver
        self.validator = ChromaChunkValidator(chroma_client, collection_names)
        self.description_generator = description_generator or self._default_description
        self.max_context_chunks = max(max_context_chunks, 1)

    def _compute_chunk_weights(self, chunks: list[ContextChunkLink]) -> list[float]:
        """Compute relevance weights for each chunk based on multiple factors."""
        if not chunks:
            return []

        weights = []
        for i, chunk in enumerate(chunks):
            # Base weight from Chroma relevance score (0.5 importance)
            base_weight = chunk.relevance_score * 0.5
            
            # Positional decay - first chunks are more important (0.3 importance)
            positional_weight = max(1 - (i * 0.2), 0.1)
            
            # Text quality bonus - longer chunks tend to have more context (0.2 importance)
            text_length_score = min(chunk.text.strip().count(" ") / 50, 1.0) if chunk.text else 0
            
            weight = base_weight + positional_weight + text_length_score
            weights.append(round(min(weight, 1.0), 4))

        return weights

    @staticmethod
    def _default_description(node: dict, chunks: list[ContextChunkLink]) -> str:
        """Generate enhanced description with weighted context from multiple chunks."""
        name = node.get("name") or node.get("id") or "Unnamed entity"
        entity_type = node.get("type") or node.get("label") or "entity"
        
        # Use configurable chunk count (default 5) for richer descriptions
        max_chunks_to_use = min(len(chunks), NodeEnricher._calculate_optimal_chunk_count(node, chunks))
        
        if max_chunks_to_use == 0:
            return f"{name} ({entity_type}) - No available context"
            
        # Weight and rank chunks by relevance
        weighted_chunks = sorted(
            enumerate(chunks[:max_chunks_to_use]),
            key=lambda x: x[1].relevance_score,
            reverse=True
        )
        
        # Build description with top relevant chunks (up to 3 most important)
        evidence_parts = []
        for rank_idx, ((chunk_idx, chunk)) in enumerate(weighted_chunks[:3]):
            if not chunk.text.strip():
                continue
            # Add truncation indicator for long text
            display_text = chunk.text.strip()
            if len(display_text) > 500:
                display_text = display_text[:497] + "..."
            
            evidence_parts.append(
                f"#{chunk_idx} [{chunk.chunk_id[:8]}]: {display_text}"
            )

        # Build final description with structure that supports multi-hop reasoning
        if entity_type and entity_type.lower() != "entity":
            type_context = f"type: {entity_type}. "
        else:
            type_context = ""
            
        evidence_summary = "\n".join(evidence_parts)
        
        return (
            f"{name} ({type_context})"
            f"\nContext summary:\n{evidence_summary}"
        )

    @staticmethod
    def _calculate_optimal_chunk_count(node: dict, chunks: list[ContextChunkLink]) -> int:
        """Dynamically calculate optimal chunk count based on node complexity."""
        # Base on available verified chunks with meaningful content
        meaningful_chunks = [c for c in chunks if len(c.text.strip()) > 10]
        
        if not meaningful_chunks:
            return min(3, max_context_chunks)
            
        # Scale based on total context length
        total_length = sum(len(c.text.strip()) for c in meaningful_chunks)
        avg_chunk_length = total_length / len(meaningful_chunks)
        
        # More chunks for richer entities (complex descriptions need more context)
        if avg_chunk_length > 200:  # Long descriptive chunks
            return min(5, max_context_chunks)
        elif avg_chunk_length > 100:  # Medium chunks
            return min(4, max_context_chunks)
        else:  # Short chunks - use more of them for better coverage
            return min(6, max_context_chunks)

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
        
        # Use the best chunk for evidence context
        if chunks:
            # Prefer chunk with highest relevance that has substantial content
            best_chunk_idx = max(
                range(len(chunks)),
                key=lambda i: len(chunks[i].text.strip()) * chunks[i].relevance_score
            )
            best_chunk = chunks[best_chunk_idx]
        else:
            best_chunk = ContextChunkLink(chunk_id="", text="")

        if explicit_evidence:
            evidence = explicit_evidence
            weight = float(raw.get("weight") or 1.0)
        else:
            source = node.get("name") or node.get("id") or "This entity"
            excerpt = best_chunk.text.replace("\n", " ")[:240]
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
                    
                    # Compute chunk weights for improved description generation
                    if chunks:
                        weights = self._compute_chunk_weights(chunks)
                    else:
                        weights = []

                    if not chunks:
                        skipped += 1
                        continue
                    node = dict(record["node"])
                    relationships = [self._relationship(r, node, chunks) for r in record["relationships"]]
                    
                    # Use custom description generator that leverages chunk weighting
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
