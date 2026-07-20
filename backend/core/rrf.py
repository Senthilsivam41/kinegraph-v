"""RRF fusion and retrieval-time near-duplicate filtering."""
import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional
from backend.core.config import settings


def reciprocal_rank_fusion(
    results_list: List[List[Dict[str, Any]]],
    k: Optional[int] = None,
    weights: Optional[List[float]] = None,
    source_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Perform Reciprocal Rank Fusion on multiple result lists
    
    RRF Formula: RRF(d) = Σ 1/(k + rank(d))
    
    Args:
        results_list: List of result lists from different sources
        k: Constant for RRF formula (default from settings)
        weights: Optional non-negative multiplier for each result list.
        source_names: Optional names used to expose per-channel contributions.
        
    Returns:
        Fused and ranked results
    """
    if k is None:
        k = settings.RRF_K
    if weights is None:
        weights = [1.0] * len(results_list)
    if len(weights) != len(results_list):
        raise ValueError("weights must match results_list length")
    if any(weight < 0 for weight in weights) or (weights and not any(weights)):
        raise ValueError("weights must be non-negative with at least one positive value")
    if source_names is None:
        source_names = [f"source_{index}" for index in range(len(results_list))]
    if len(source_names) != len(results_list):
        raise ValueError("source_names must match results_list length")

    # Dictionary to store RRF scores
    rrf_scores: Dict[str, float] = {}
    document_data: Dict[str, Dict[str, Any]] = {}
    contributions: Dict[str, Dict[str, float]] = {}
    
    # Calculate RRF scores for each result list
    for results, weight, source_name in zip(results_list, weights, source_names):
        if weight == 0:
            continue
        for rank, result in enumerate(results, start=1):
            # Use content as unique identifier
            doc_id = result.get('content', '')[:100]  # Use first 100 chars as ID
            
            # Calculate RRF score
            score = weight / (k + rank)
            
            # Accumulate scores
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += score
            else:
                rrf_scores[doc_id] = score
                document_data[doc_id] = result
            source_contributions = contributions.setdefault(doc_id, {})
            source_contributions[source_name] = round(
                source_contributions.get(source_name, 0.0) + score,
                8,
            )
    
    # Sort by RRF score (descending)
    sorted_docs = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Format final results
    fused_results = []
    for doc_id, rrf_score in sorted_docs:
        result = document_data[doc_id].copy()
        result['rrf_score'] = rrf_score
        # Keep original score as well
        result['original_score'] = result.get('score', 0)
        result['score'] = rrf_score  # Use RRF score as primary score
        result['rrf_contributions'] = contributions.get(doc_id, {})
        fused_results.append(result)
    
    return fused_results


def deduplicate_results(
    results: List[Dict[str, Any]],
    similarity_threshold: float = 0.95,
) -> List[Dict[str, Any]]:
    """Keep the highest-ranked representative of near-identical chunk text.

    Stored dense embeddings are used when both candidates provide them; graph-only
    candidates fall back to normalized token-frequency cosine. This adds no
    embedding API call and preserves the incoming (normally RRF) ordering.
    """
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")

    deduplicated: List[Dict[str, Any]] = []
    accepted_vectors: List[tuple[Counter[str], Any]] = []
    for result in results:
        vector = Counter(re.findall(r"[a-z0-9]+", result.get("content", "").lower()))
        embedding = result.get("embedding")
        if any(
            _result_similarity(vector, embedding, existing_vector, existing_embedding)
            >= similarity_threshold
            for existing_vector, existing_embedding in accepted_vectors
        ):
            continue
        deduplicated.append(result)
        accepted_vectors.append((vector, embedding))

    return deduplicated


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    """Cosine similarity for sparse token-frequency vectors."""
    if not left or not right:
        return 1.0 if left == right else 0.0
    dot = sum(value * right.get(token, 0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _result_similarity(
    left_tokens: Counter[str],
    left_embedding: Any,
    right_tokens: Counter[str],
    right_embedding: Any,
) -> float:
    """Prefer stored dense embeddings, falling back to lexical cosine."""
    try:
        left = [float(value) for value in left_embedding]
        right = [float(value) for value in right_embedding]
        if left and len(left) == len(right):
            dot = sum(a * b for a, b in zip(left, right))
            left_norm = math.sqrt(sum(value * value for value in left))
            right_norm = math.sqrt(sum(value * value for value in right))
            if left_norm and right_norm:
                return dot / (left_norm * right_norm)
    except (TypeError, ValueError):
        pass
    return _cosine_similarity(left_tokens, right_tokens)
