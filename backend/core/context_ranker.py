"""
Context Ranker — KineticGraph-Vectra
Cross-encoder reranker + relevance filter applied after RRF fusion.
Improves context_precision by pruning low-relevance chunks before generation.
Works in two modes:
  • keyword mode (no extra deps) — fast, always available
  • cross-encoder mode         — accurate, requires sentence-transformers
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple

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
    Post-fusion reranker that improves context_precision.

    1. Scores every chunk against the query (keyword or cross-encoder).
    2. Filters out chunks below a relevance threshold.
    3. Re-ranks the survivors by relevance score.

    Usage::

        ranker = ContextRanker(use_cross_encoder=True)
        ranked = ranker.rerank(query="What is RRF?", chunks=[...], top_k=5)
    """

    def __init__(
        self,
        use_cross_encoder: bool = False,
        model_name: str = _DEFAULT_CE_MODEL,
        min_relevance_threshold: float = 0.05,
    ) -> None:
        self.use_cross_encoder = use_cross_encoder and _CE_AVAILABLE
        self.min_relevance_threshold = min_relevance_threshold
        self._encoder = None

        if self.use_cross_encoder:
            try:
                self._encoder = CrossEncoder(model_name)
                logger.info("ContextRanker: cross-encoder loaded (%s)", model_name)
            except Exception as exc:
                logger.warning("CrossEncoder load failed (%s) — falling back to keyword.", exc)
                self._encoder = None
                self.use_cross_encoder = False

    def _score_chunks(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Return (chunk, score) pairs sorted descending by relevance."""
        if self.use_cross_encoder and self._encoder:
            pairs = [(query, c.get("content", "")) for c in chunks]
            scores = self._encoder.predict(pairs)
            # Normalise cross-encoder logit scores to 0-1
            min_s, max_s = min(scores), max(scores)
            span = (max_s - min_s) or 1.0
            norm = [(s - min_s) / span for s in scores]
            scored = list(zip(chunks, norm))
        else:
            scored = [
                (c, _keyword_score(query, c.get("content", "")))
                for c in chunks
            ]

        return sorted(scored, key=lambda x: x[1], reverse=True)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank *chunks* by relevance to *query* and return the top-k.

        Args:
            query:   The user query (or rewritten query).
            chunks:  List of chunk dicts (must have 'content' key).
            top_k:   Maximum number of chunks to return.

        Returns:
            Filtered and reranked list of chunk dicts, each augmented with
            a ``rerank_score`` field.
        """
        if not chunks:
            return []

        scored = self._score_chunks(query, chunks)

        # Apply relevance floor
        survivors = [
            (c, s) for c, s in scored if s >= self.min_relevance_threshold
        ]
        # If everything is filtered out, keep top-1 to avoid empty context
        if not survivors:
            survivors = scored[:1]

        result = []
        for chunk, score in survivors[:top_k]:
            enriched = dict(chunk)
            enriched["rerank_score"] = round(float(score), 4)
            result.append(enriched)

        logger.debug(
            "ContextRanker: %d → %d chunks (threshold=%.3f, mode=%s)",
            len(chunks),
            len(result),
            self.min_relevance_threshold,
            "cross-encoder" if self.use_cross_encoder else "keyword",
        )
        return result
