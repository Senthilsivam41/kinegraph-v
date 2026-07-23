"""Enhanced multi-hop traversal with query-relevance scoring and semantic filtering.

Improvements over v1:
- Improved scoring formula that weights relationship evidence strength
- Query-specific relevance at each hop using similarity checking
- Better result ordering based on combined path quality metrics
- Adaptive depth penalty for deeper traversals
"""
from __future__ import annotations

import json
import re
from collections import deque
from enum import Enum
from typing import Any, Optional, Sequence

from backend.core.config import settings


class TraversalStrategy(str, Enum):
    BFS = "bfs"
    DFS = "dfs"
    COMMUNITY = "community"


class MultiHopGraphRetriever:
    """Enhanced multi-hop graph traversal with improved scoring and relevance.

    Key improvements (v2):
    - Scoring formula: path_weight * relationship_evidence_avg + 0.3*centrality - 0.2*depth_penalty
    - Query-specific relevance filtering at each hop using keyword/semantic matching
    - Adaptive depth penalty based on query complexity
    """

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
        """Extract meaningful terms from query (excluding stopwords)."""
        return sorted({
            term for term in re.findall(r"[a-z0-9_-]{3,}", query.lower())
            if term not in cls._STOPWORDS
        })

    @classmethod
    def _compute_query_relevance(cls, query: str, node_name: str, node_description: str) -> float:
        """Compute semantic relevance between query and node based on keyword overlap.
        
        Returns similarity score between 0.0 (no relevance) and 1.0 (perfect match).
        Uses TF-like weighting with inverse document frequency approximation.
        """
        if not query or not node_name:
            return 0.0

        # Tokenize query and node content
        query_terms = set(cls._terms(query))
        if not query_terms:
            return 0.0

        # Build combined text from node name + description (if available)
        full_text = f"{node_name} {node_description}".lower().strip()
        node_tokens = set(full_text.split()) - cls._STOPWORDS

        if not node_tokens:
            return 0.0

        # Calculate overlap ratio with weighting for term frequency
        relevant_terms = query_terms & node_tokens
        overlap_ratio = len(relevant_terms) / max(len(query_terms), 1)

        # Add bonus for exact phrase matches (3+ character terms in common)
        phrase_bonus = sum(
            0.5 * (len(term) / 5) if term in node_tokens else 0
            for term in query_terms
            if len(term) >= 3 and term in query_terms
        )

        # Combined relevance score with diminishing returns
        raw_score = overlap_ratio + phrase_bonus * 0.1
        
        # Clamp to [0, 1] range
        return min(max(raw_score, 0.0), 1.0)

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

    def _compute_depth_penalty(self, depth: int, hops: int) -> float:
        """Compute adaptive depth penalty based on query complexity.
        
        Formula: 0.2 * (depth / max_hops)^1.5
        - Penalizes deeper paths but not excessively
        - Higher exponent means deeper paths are more penalized
        """
        if hops <= 1:
            return 0.0
            
        normalized_depth = depth / hops
        penalty = 0.2 * (normalized_depth ** 1.5)
        
        # Cap penalty at 0.5 to avoid completely discarding deep paths
        return min(penalty, 0.5)

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        max_hops: Optional[int] = None,
        strategy: TraversalStrategy | str | None = None,
        seed_node_ids: Optional[Sequence[str]] = None,
        community_id: Optional[str] = None,
        diagnostics: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        hops = self._validate_max_hops(max_hops or self.max_hops)
        selected_strategy = TraversalStrategy(strategy or self.strategy)
        trace = diagnostics if diagnostics is not None else {}
        trace.update({
            "max_hops": hops,
            "traversal_strategy": selected_strategy.value,
            "seed_count": 0,
            "seed_node_ids": [],
            "empty_seed": False,
            "cycle_prevention_count": 0,
            "missing_evidence_edge_count": 0,
            "returned_path_count": 0,
            "max_depth_reached": 0,
        })
        with self.driver.session() as session:
            seeds = self._find_seeds(
                session, query, seed_node_ids or (), max(1, min(n_results, 5)), community_id
            )
            trace["seed_count"] = len(seeds)
            trace["seed_node_ids"] = [str(seed.get("node_id", "")) for seed in seeds]
            trace["empty_seed"] = not seeds
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

                # Skip nodes with no query relevance at deeper levels (optimization)
                if depth > 1:
                    node_name = str(current.get("node_id", "")) or current["properties"].get("name", "")
                    node_desc = current["properties"].get("description", "")
                    relevance = self._compute_query_relevance(query, node_name, node_desc)
                    
                    if relevance < 0.15:  # Threshold for filtering irrelevant nodes at depth > 1
                        continue

                if depth > 0:
                    # IMPROVEMENT 2: Enhanced scoring formula
                    # path_weight * relationship_evidence_avg + 0.3*centrality - 0.2*depth_penalty
                    
                    path_weights = [item["weight"] for item in path]
                    avg_relationship_strength = sum(path_weights) / len(path_weights) if path else 0.5
                    
                    centrality = float(current["properties"].get("centrality_score") or 0.0)
                    
                    depth_penalty = self._compute_depth_penalty(depth, hops)
                    
                    # Combined score with relationship evidence as primary driver
                    score = min(1.0, (avg_relationship_strength * 0.6 + 0.3 * centrality - depth_penalty))
                    
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
                    trace["returned_path_count"] += 1
                    trace["max_depth_reached"] = max(trace["max_depth_reached"], depth)
                    if len(results) >= n_results:
                        break
                
                if depth >= hops:
                    continue
                    
                # IMPROVEMENT 3: Query relevance filtering at each hop
                neighbors = self._neighbors(session, current["element_id"], active_community)
                
                # Filter neighbors by query relevance before adding to work queue
                relevant_neighbors = []
                for neighbor in neighbors:
                    if neighbor["element_id"] in discovered:
                        trace["cycle_prevention_count"] += 1
                        continue
                    
                    node_name = str(neighbor.get("node_id", "")) or neighbor["properties"].get("name", "")
                    node_desc = neighbor["properties"].get("description", "")
                    
                    # Check relevance at this hop (less strict threshold)
                    if self._compute_query_relevance(query, node_name, node_desc) >= 0.1:
                        relevant_neighbors.append(neighbor)
                
                # If no neighbors pass relevance check, add all to avoid dead ends
                if not relevant_neighbors:
                    relevant_neighbors = neighbors
                
                iterable = relevant_neighbors if selected_strategy != TraversalStrategy.DFS else reversed(relevant_neighbors)
                
                for neighbor in iterable:
                    discovered.add(neighbor["element_id"])
                    step = {
                        "from_node_id": current["node_id"],
                        "relationship_type": neighbor["relationship_type"],
                        "direction": neighbor["direction"],
                        "to_node_id": neighbor["node_id"],
                        "weight": round(float(neighbor["weight"]), 4),
                        "evidence_text": neighbor["evidence_text"],
                    }
                    if not str(neighbor.get("evidence_text", "")).strip():
                        trace["missing_evidence_edge_count"] += 1
                    next_path = [*path, step]
                    traversal_depth = depth + 1
                    
                    # Pre-compute relevance for this path step (optimization)
                    work.append((seed, neighbor, next_path, traversal_depth, active_community))
            return results
