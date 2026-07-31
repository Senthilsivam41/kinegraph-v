"""RRF fusion and retrieval-time near-duplicate filtering."""
import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional
from backend.core.config import settings
from backend.core.retrieval_orchestration import (
    annotate_channel_candidates,
    candidate_identity,
    merge_candidate_provenance,
)


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
    
    # Calculate RRF scores for each result list while retaining every channel's
    # rank, score, and graph-path representation.
    for results, weight, source_name in zip(results_list, weights, source_names):
        if weight == 0:
            continue
        annotated_results = annotate_channel_candidates(results, source_name)
        for rank, result in enumerate(annotated_results, start=1):
            doc_id = candidate_identity(result)
            
            # Calculate RRF score
            score = weight / (k + rank)
            
            # Accumulate scores
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += score
                document_data[doc_id] = merge_candidate_provenance(
                    document_data[doc_id], result
                )
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
    for rrf_rank, (doc_id, rrf_score) in enumerate(sorted_docs, start=1):
        result = document_data[doc_id].copy()
        result['rrf_score'] = rrf_score
        result['rrf_rank'] = rrf_rank
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
    deduplicated, _ = deduplicate_results_with_report(
        results,
        similarity_threshold=similarity_threshold,
    )
    return deduplicated


def deduplicate_results_with_report(
    results: List[Dict[str, Any]],
    similarity_threshold: float = 0.95,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Deduplicate and report the exact survival decision for every candidate."""
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")

    deduplicated: List[Dict[str, Any]] = []
    accepted: List[tuple[str, Counter[str], Any]] = []
    decisions: List[Dict[str, Any]] = []
    for result in results:
        candidate_id = candidate_identity(result)
        vector = Counter(re.findall(r"[a-z0-9]+", result.get("content", "").lower()))
        embedding = result.get("embedding")
        duplicate_of = None
        similarity = None
        reason = "unique_candidate"
        for existing_id, existing_vector, existing_embedding in accepted:
            if candidate_id == existing_id:
                duplicate_of = existing_id
                similarity = 1.0
                reason = "duplicate_identity"
                break
            current_similarity = _result_similarity(
                vector,
                embedding,
                existing_vector,
                existing_embedding,
            )
            if current_similarity >= similarity_threshold:
                duplicate_of = existing_id
                similarity = current_similarity
                reason = "near_duplicate_content"
                break

        if duplicate_of is None:
            deduplicated.append(result)
            accepted.append((candidate_id, vector, embedding))
            decision = "survived"
        else:
            decision = "dropped"
        decisions.append({
            "candidate_id": candidate_id,
            "stage": "identity_deduplication",
            "decision": decision,
            "reason": reason,
            "duplicate_of": duplicate_of,
            "similarity": round(float(similarity), 6) if similarity is not None else None,
        })

    return deduplicated, {
        "input_count": len(results),
        "output_count": len(deduplicated),
        "similarity_threshold": similarity_threshold,
        "decisions": decisions,
    }


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
