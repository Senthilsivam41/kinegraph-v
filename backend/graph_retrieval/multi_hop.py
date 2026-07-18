"""Bounded, evidence-preserving multi-hop traversal for Neo4j entities."""
from __future__ import annotations

import json
import re
from collections import deque
from enum import Enum
from typing import Any, Optional, Sequence


class TraversalStrategy(str, Enum):
    BFS = "bfs"
    DFS = "dfs"
    COMMUNITY = "community"


class MultiHopGraphRetriever:
    """Traverse entity relationships and return explicit paths up to ``max_hops``."""

    _STOPWORDS = {
        "about", "does", "from", "have", "into", "that", "their", "then",
        "this", "what", "when", "where", "which", "with", "would", "your",
    }

    def __init__(self, driver: Any, max_hops: int = 3, strategy: TraversalStrategy | str = TraversalStrategy.BFS):
        self.driver = driver
        self.max_hops = self._validate_max_hops(max_hops)
        self.strategy = TraversalStrategy(strategy)

    @staticmethod
    def _validate_max_hops(max_hops: int) -> int:
        if not 1 <= max_hops <= 5:
            raise ValueError("max_hops must be between 1 and 5")
        return max_hops

    @classmethod
    def _terms(cls, query: str) -> list[str]:
        return sorted({
            term for term in re.findall(r"[a-z0-9_-]{3,}", query.lower())
            if term not in cls._STOPWORDS
        })

    def _find_seeds(
        self,
        session: Any,
        query: str,
        seed_node_ids: Sequence[str],
        limit: int,
        community_id: Optional[str],
    ) -> list[dict[str, Any]]:
        cypher = """
        MATCH (e) WHERE e:Entity OR e:`__Entity__`
        WITH e, [term IN $terms WHERE
          toLower(coalesce(e.name, '') + ' ' + coalesce(e.description, '')) CONTAINS term
        ] AS matches
        WHERE ((size($seed_ids) > 0 AND (
          elementId(e) IN $seed_ids OR e.id IN $seed_ids OR e.name IN $seed_ids
        )) OR size(matches) > 0)
        AND ($community_id IS NULL OR e.community_id = $community_id)
        RETURN elementId(e) AS element_id, coalesce(e.id, e.name) AS node_id,
               properties(e) AS properties, size(matches) AS relevance
        ORDER BY relevance DESC, coalesce(e.centrality_score, 0) DESC, node_id
        LIMIT $limit
        """
        return [dict(record) for record in session.run(
            cypher,
            terms=self._terms(query),
            seed_ids=list(seed_node_ids),
            community_id=community_id,
            limit=limit,
        )]

    def _neighbors(self, session: Any, element_id: str, community_id: Optional[str]) -> list[dict[str, Any]]:
        cypher = """
        MATCH (current) WHERE elementId(current) = $element_id
        MATCH (current)-[r]-(neighbor)
        WHERE (neighbor:Entity OR neighbor:`__Entity__`) AND type(r) <> 'MENTIONS'
          AND ($community_id IS NULL OR neighbor.community_id = $community_id)
        RETURN elementId(neighbor) AS element_id,
               coalesce(neighbor.id, neighbor.name) AS node_id,
               properties(neighbor) AS properties,
               coalesce(r.type, type(r)) AS relationship_type,
               coalesce(r.weight, 0.5) AS weight,
               coalesce(r.evidence_text, '') AS evidence_text,
               CASE WHEN elementId(startNode(r)) = $element_id THEN 'OUTGOING' ELSE 'INCOMING' END AS direction
        ORDER BY weight DESC, coalesce(neighbor.centrality_score, 0) DESC, node_id
        """
        return [dict(record) for record in session.run(
            cypher, element_id=element_id, community_id=community_id
        )]

    @staticmethod
    def _node_content(properties: dict[str, Any]) -> str:
        chunks_raw = properties.get("parent_context_chunks_json") or "[]"
        try:
            chunks = json.loads(chunks_raw) if isinstance(chunks_raw, str) else chunks_raw
        except (TypeError, json.JSONDecodeError):
            chunks = []
        evidence = "\n".join(
            f"[{chunk.get('chunk_id', 'graph-context')}] {chunk.get('text', '')}"
            for chunk in chunks[:3] if chunk.get("text")
        )
        return "\n\n".join(filter(None, [
            properties.get("description"), properties.get("name"), evidence,
        ]))

    @staticmethod
    def _path_text(path: list[dict[str, Any]]) -> str:
        if not path:
            return ""
        parts = [str(path[0]["from_node_id"])]
        for step in path:
            arrow = "->" if step["direction"] == "OUTGOING" else "<-"
            parts.append(f"-[{step['relationship_type']}]{arrow} {step['to_node_id']}")
        return " ".join(parts)

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        max_hops: Optional[int] = None,
        strategy: TraversalStrategy | str | None = None,
        seed_node_ids: Optional[Sequence[str]] = None,
        community_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        hops = self._validate_max_hops(max_hops or self.max_hops)
        selected_strategy = TraversalStrategy(strategy or self.strategy)
        with self.driver.session() as session:
            seeds = self._find_seeds(
                session, query, seed_node_ids or (), max(1, min(n_results, 5)), community_id
            )
            results: list[dict[str, Any]] = []
            discovered: set[str] = {seed["element_id"] for seed in seeds}
            work = deque()
            for seed in seeds:
                seed_community = community_id
                if selected_strategy == TraversalStrategy.COMMUNITY:
                    seed_community = seed["properties"].get("community_id")
                    if not seed_community:
                        continue
                work.append((seed, seed, [], 0, seed_community))

            while work and len(results) < n_results:
                seed, current, path, depth, active_community = (
                    work.popleft() if selected_strategy in (TraversalStrategy.BFS, TraversalStrategy.COMMUNITY)
                    else work.pop()
                )
                if depth > 0:
                    path_weight = sum(item["weight"] for item in path) / len(path)
                    centrality = float(current["properties"].get("centrality_score") or 0.0)
                    score = min(1.0, (path_weight / (1 + 0.15 * depth)) + 0.05 * centrality)
                    path_text = self._path_text(path)
                    content = self._node_content(current["properties"])
                    results.append({
                        "content": f"Traversal path (depth {depth}): {path_text}\n\n{content}".strip(),
                        "metadata": {
                            **current["properties"],
                            "relationship_path": path,
                            "traversal_depth": depth,
                            "traversal_strategy": selected_strategy.value,
                            "max_hops": hops,
                            "seed_node_id": seed["node_id"],
                            "community_id": active_community or current["properties"].get("community_id"),
                        },
                        "score": round(score, 6),
                        "source": "graph_traversal",
                    })
                    if len(results) >= n_results:
                        break
                if depth >= hops:
                    continue
                neighbors = self._neighbors(session, current["element_id"], active_community)
                iterable = neighbors if selected_strategy != TraversalStrategy.DFS else reversed(neighbors)
                for neighbor in iterable:
                    if neighbor["element_id"] in discovered:
                        continue
                    discovered.add(neighbor["element_id"])
                    step = {
                        "from_node_id": current["node_id"],
                        "relationship_type": neighbor["relationship_type"],
                        "direction": neighbor["direction"],
                        "to_node_id": neighbor["node_id"],
                        "weight": round(float(neighbor["weight"]), 4),
                        "evidence_text": neighbor["evidence_text"],
                    }
                    next_path = [*path, step]
                    traversal_depth = depth + 1
                    work.append((seed, neighbor, next_path, traversal_depth, active_community))
            return results
