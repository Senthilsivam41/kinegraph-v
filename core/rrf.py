"""
Reciprocal Rank Fusion (RRF) Implementation
Combines results from multiple retrieval systems
"""
from typing import List, Dict, Any
from core.config import settings


def reciprocal_rank_fusion(
    results_list: List[List[Dict[str, Any]]],
    k: int = None
) -> List[Dict[str, Any]]:
    """
    Perform Reciprocal Rank Fusion on multiple result lists
    
    RRF Formula: RRF(d) = Σ 1/(k + rank(d))
    
    Args:
        results_list: List of result lists from different sources
        k: Constant for RRF formula (default from settings)
        
    Returns:
        Fused and ranked results
    """
    if k is None:
        k = settings.RRF_K
    
    # Dictionary to store RRF scores
    rrf_scores: Dict[str, float] = {}
    document_data: Dict[str, Dict[str, Any]] = {}
    
    # Calculate RRF scores for each result list
    for results in results_list:
        for rank, result in enumerate(results, start=1):
            # Use content as unique identifier
            doc_id = result.get('content', '')[:100]  # Use first 100 chars as ID
            
            # Calculate RRF score
            score = 1 / (k + rank)
            
            # Accumulate scores
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += score
            else:
                rrf_scores[doc_id] = score
                document_data[doc_id] = result
    
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
        fused_results.append(result)
    
    return fused_results


def deduplicate_results(
    results: List[Dict[str, Any]],
    similarity_threshold: float = 0.9
) -> List[Dict[str, Any]]:
    """
    Remove duplicate results based on content similarity
    
    Args:
        results: List of results to deduplicate
        similarity_threshold: Threshold for considering documents as duplicates
        
    Returns:
        Deduplicated results
    """
    # Simple deduplication based on exact content match
    # For production, consider using fuzzy matching or embedding similarity
    
    seen_content = set()
    deduplicated = []
    
    for result in results:
        content = result.get('content', '')
        
        # Use first 200 characters for deduplication
        content_signature = content[:200].strip().lower()
        
        if content_signature not in seen_content:
            seen_content.add(content_signature)
            deduplicated.append(result)
    
    return deduplicated
