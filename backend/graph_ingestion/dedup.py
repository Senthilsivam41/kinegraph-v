import numpy as np
from typing import List, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(v1)
    b = np.array(v2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

class EntityResolver:
    """
    Deduplicates entity names using exact-match and semantic embedding similarity clustering.
    """
    def __init__(self, embed_model: Any, threshold: float = 0.85):
        self.embed_model = embed_model
        self.threshold = threshold
        # lowercase_name -> (canonical_name, embedding)
        self.resolved_cache: Dict[str, Tuple[str, List[float]]] = {}

    def resolve_entities(self, entity_names: List[str]) -> Dict[str, str]:
        """
        Deduplicates a list of entity names against each other and against cached entities.
        
        Returns a mapping from the original name to the resolved canonical name.
        """
        resolution_map = {}
        for name in entity_names:
            clean_name = name.strip()
            lower_name = clean_name.lower()
            
            # 1. Exact match check against cache
            if lower_name in self.resolved_cache:
                resolved_name = self.resolved_cache[lower_name][0]
                resolution_map[name] = resolved_name
                continue

            # 2. Compute embedding for semantic matching
            emb = self.embed_model.get_text_embedding(clean_name)
            
            # 3. Embedding similarity check
            match_found = False
            for cached_lower, (cached_canonical, cached_emb) in self.resolved_cache.items():
                sim = cosine_similarity(emb, cached_emb)
                if sim >= self.threshold:
                    logger.warning(
                        "Dedup collision: Resolved alias '%s' to canonical '%s' (similarity: %.3f)",
                        clean_name, cached_canonical, sim
                    )
                    resolution_map[name] = cached_canonical
                    # Update cache to resolve this variant immediately next time
                    self.resolved_cache[lower_name] = (cached_canonical, cached_emb)
                    match_found = True
                    break

            if not match_found:
                # Store new canonical entity in cache
                self.resolved_cache[lower_name] = (clean_name, emb)
                resolution_map[name] = clean_name

        return resolution_map
