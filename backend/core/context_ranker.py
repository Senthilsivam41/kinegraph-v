"""
Context Ranker — KineticGraph-Vectra
Semantic-first, graph-aware reranker applied after RRF fusion.
Improves context_precision by pruning low-relevance chunks before generation.
Works in two modes:
  • keyword mode (no extra deps) — fast, always available
  • cross-encoder mode         — accurate, requires sentence-transformers
"""
from __future__ import annotations

import logging
import math
import json
from collections import Counter
from typing import Any, Dict, List, Tuple

from backend.core.retrieval_orchestration import candidate_identity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional cross-encoder import
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import CrossEncoder
    _CE_AVAILABLE = True
except ImportError:
    _CE_AVAILABLE = False
    logger.info(
        "sentence-transformers not installed — using keyword reranker. "
        "pip install sentence-transformers for cross-encoder reranking."
    )

_DEFAULT_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Keyword-based relevance scorer (always available)
# ---------------------------------------------------------------------------

def _keyword_score(query: str, content: str) -> float:
    """Normalised keyword overlap score (TF-inspired, no external deps)."""
    q_terms = set(query.lower().split())
    c_terms = content.lower().split()
    if not q_terms or not c_terms:
        return 0.0
    hits = sum(1 for t in c_terms if t in q_terms)
    # Normalise by log of content length to prevent length bias
    return hits / (1 + math.log(len(c_terms) + 1))


# ---------------------------------------------------------------------------
# ContextRanker
# ---------------------------------------------------------------------------

class ContextRanker:
    """
    Post-fusion reranker that improves context_precision without mutating RRF scores.

    1. Scores every chunk against the query (keyword or cross-encoder).
    2. Filters out semantically weak chunks before graph signals are considered.
    3. Adds bounded centrality, community, edge-strength, and distance signals.
    4. Re-ranks survivors while preserving source, original, and RRF scores.

    Usage::

        ranker = ContextRanker(use_cross_encoder=True)
        ranked = ranker.rerank(query="What is RRF?", chunks=[...], top_k=5)
    """

    def __init__(
        self,
        use_cross_encoder: bool = False,
        model_name: str = _DEFAULT_CE_MODEL,
        min_relevance_threshold: float = 0.05,
        semantic_weight: float = 0.70,
        centrality_weight: float = 0.10,
        community_weight: float = 0.05,
        edge_weight: float = 0.10,
        distance_weight: float = 0.05,
    ) -> None:
        weights = {
            "semantic": semantic_weight,
            "centrality": centrality_weight,
            "community": community_weight,
            "edge": edge_weight,
            "distance": distance_weight,
        }
        if any(weight < 0 for weight in weights.values()) or not math.isclose(sum(weights.values()), 1.0):
            raise ValueError("reranker weights must be non-negative and sum to 1.0")
        if semantic_weight < 0.5:
            raise ValueError("semantic_weight must remain at least 0.5")
        self.requested_cross_encoder = use_cross_encoder
        self.fallback_reason = None
        self.use_cross_encoder = use_cross_encoder and _CE_AVAILABLE
        if use_cross_encoder and not _CE_AVAILABLE:
            self.fallback_reason = "sentence-transformers unavailable"
        if not 0.0 <= min_relevance_threshold <= 1.0:
            raise ValueError("min_relevance_threshold must be between 0 and 1")
        self.min_relevance_threshold = min_relevance_threshold
        self.model_name = model_name
        self.weights = weights
        self._encoder = None

        if self.use_cross_encoder:
            try:
                self._encoder = CrossEncoder(model_name)
                logger.info("ContextRanker: cross-encoder loaded (%s)", model_name)
            except Exception as exc:
                logger.warning("CrossEncoder load failed (%s) — falling back to keyword.", exc)
                self.fallback_reason = f"cross-encoder load failed: {type(exc).__name__}"
                self._encoder = None
                self.use_cross_encoder = False

    def _score_chunks(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Return (chunk, score) pairs sorted descending by relevance."""
        if self.use_cross_encoder and self._encoder:
            pairs = [(query, c.get("content", "")) for c in chunks]
            scores = self._encoder.predict(pairs)
            # Preserve calibrated probability scores. Convert raw logits with a
            # sigmoid rather than batch-relative min/max scaling so relevance
            # thresholds mean the same thing across queries.
            raw_scores = [float(score) for score in scores]
            if all(0.0 <= score <= 1.0 for score in raw_scores):
                norm = raw_scores
            else:
                norm = [1.0 / (1.0 + math.exp(-max(-60.0, min(score, 60.0)))) for score in raw_scores]
            scored = list(zip(chunks, norm))
        else:
            scored = [
                (c, max(0.0, min(_keyword_score(query, c.get("content", "")), 1.0)))
                for c in chunks
            ]

        return sorted(scored, key=lambda x: x[1], reverse=True)

    @staticmethod
    def _metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
        return chunk.get("metadata") or {}

    @classmethod
    def _number(cls, chunk: Dict[str, Any], key: str) -> float | None:
        value = cls._metadata(chunk).get(key, chunk.get(key))
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _community(cls, chunk: Dict[str, Any]) -> str | None:
        value = cls._metadata(chunk).get("community_id", chunk.get("community_id"))
        return str(value) if value not in (None, "") else None

    @classmethod
    def _edge_strength(cls, chunk: Dict[str, Any]) -> float | None:
        metadata = cls._metadata(chunk)
        path = metadata.get("relationship_path") or chunk.get("relationship_path") or []
        weights = []
        for edge in path:
            try:
                weights.append(float(edge["weight"]))
            except (KeyError, TypeError, ValueError):
                continue
        if not weights:
            raw_relationships = metadata.get("relationships_json") or chunk.get("relationships_json")
            if raw_relationships:
                try:
                    relationships = json.loads(raw_relationships) if isinstance(raw_relationships, str) else raw_relationships
                    weights = [float(rel["weight"]) for rel in relationships if rel.get("weight") is not None]
                except (TypeError, ValueError, json.JSONDecodeError):
                    weights = []
        if not weights:
            return None
        return max(0.0, min(sum(weights) / len(weights), 1.0))

    def _graph_components(
        self,
        chunk: Dict[str, Any],
        community_counts: Counter,
        max_community_count: int,
        preferred_community_id: str | None,
    ) -> Dict[str, float | None]:
        centrality = self._number(chunk, "centrality_score")
        if centrality is not None:
            centrality = max(0.0, min(centrality, 1.0))

        community = self._community(chunk)
        community_score = None
        if community:
            if preferred_community_id:
                community_score = 1.0 if community == preferred_community_id else 0.0
            else:
                community_score = community_counts[community] / max_community_count

        traversal_depth = self._number(chunk, "traversal_depth")
        distance_score = None
        if traversal_depth is not None:
            distance_score = 1.0 / max(1.0, traversal_depth)

        return {
            "centrality": centrality,
            "community": community_score,
            "edge": self._edge_strength(chunk),
            "distance": distance_score,
        }

    def _combined_score(self, semantic_score: float, graph_components: Dict[str, float | None]) -> tuple[float, float | None]:
        weighted_score = self.weights["semantic"] * semantic_score
        active_weight = self.weights["semantic"]
        graph_weighted = 0.0
        graph_active_weight = 0.0
        for name, value in graph_components.items():
            if value is None:
                continue
            weight = self.weights[name]
            weighted_score += weight * value
            active_weight += weight
            graph_weighted += weight * value
            graph_active_weight += weight
        combined = weighted_score / active_weight
        graph_score = graph_weighted / graph_active_weight if graph_active_weight else None
        return combined, graph_score

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
        preferred_community_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank *chunks* by relevance to *query* and return the top-k.

        Args:
            query:   The user query (or rewritten query).
            chunks:  List of chunk dicts (must have 'content' key).
            top_k:   Maximum number of chunks to return.
            preferred_community_id: Optional community to favor explicitly.

        Returns:
            Filtered and reranked list of chunk dicts, each augmented with
            ``rerank_score``, ``semantic_score``, and auditable graph components.
        """
        result, _ = self.rerank_with_report(
            query=query,
            chunks=chunks,
            top_k=top_k,
            preferred_community_id=preferred_community_id,
        )
        return result

    def rerank_with_report(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
        preferred_community_id: str | None = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Rerank once and emit deterministic survival reasons for ADR-003."""
        if not chunks:
            return [], {
                "input_count": 0,
                "output_count": 0,
                "decisions": [],
                "forced_top_candidate": False,
            }

        scored = self._score_chunks(query, chunks)
        semantic_by_id = {
            candidate_identity(chunk): float(score) for chunk, score in scored
        }
        survivors = [
            (chunk, score)
            for chunk, score in scored
            if score >= self.min_relevance_threshold
        ]
        forced_top_candidate = False
        if not survivors:
            survivors = scored[:1]
            forced_top_candidate = bool(survivors)

        communities = [self._community(chunk) for chunk, _ in survivors]
        community_counts = Counter(community for community in communities if community)
        max_community_count = max(community_counts.values(), default=1)

        reranked = []
        for chunk, semantic_score in survivors:
            components = self._graph_components(
                chunk, community_counts, max_community_count, preferred_community_id
            )
            score, graph_score = self._combined_score(semantic_score, components)
            enriched = dict(chunk)
            enriched["rerank_score"] = round(float(score), 4)
            enriched["semantic_score"] = round(float(semantic_score), 4)
            enriched["graph_signal_score"] = (
                round(graph_score, 4) if graph_score is not None else None
            )
            enriched["graph_signals_applied"] = graph_score is not None
            enriched["rerank_mode"] = (
                "graph_aware_cross_encoder"
                if self.use_cross_encoder
                else "graph_aware_keyword"
            )
            enriched["rerank_components"] = {
                key: round(value, 4) if value is not None else None
                for key, value in components.items()
            }
            reranked.append(enriched)

        reranked.sort(
            key=lambda chunk: (
                chunk["rerank_score"],
                float(chunk.get("rrf_score", chunk.get("score", 0.0))),
            ),
            reverse=True,
        )
        result = reranked[:top_k]
        retained_ids = {candidate_identity(chunk) for chunk in result}
        forced_id = candidate_identity(survivors[0][0]) if forced_top_candidate else None
        decisions = []
        for chunk in chunks:
            candidate_id = candidate_identity(chunk)
            semantic_score = semantic_by_id.get(candidate_id, 0.0)
            if candidate_id in retained_ids:
                decision = "survived"
                reason = (
                    "forced_top_candidate_after_threshold"
                    if candidate_id == forced_id
                    else "selected_by_reranker"
                )
            elif semantic_score < self.min_relevance_threshold:
                decision, reason = "dropped", "below_semantic_relevance_threshold"
            else:
                decision, reason = "dropped", "reranker_top_k_exceeded"
            decisions.append({
                "candidate_id": candidate_id,
                "stage": "semantic_reranking",
                "decision": decision,
                "reason": reason,
                "semantic_score": round(semantic_score, 6),
            })

        logger.debug(
            "ContextRanker: %d → %d chunks (threshold=%.3f, mode=%s)",
            len(chunks),
            len(result),
            self.min_relevance_threshold,
            "cross-encoder" if self.use_cross_encoder else "keyword",
        )
        return result, {
            "input_count": len(chunks),
            "output_count": len(result),
            "minimum_relevance": self.min_relevance_threshold,
            "forced_top_candidate": forced_top_candidate,
            "decisions": decisions,
        }
