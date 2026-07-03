from backend.graph_ingestion.dedup import EntityResolver, cosine_similarity
import pytest

class FakeEmbedModel:
    def get_text_embedding(self, text: str):
        # Return mock embeddings to force similarity collisions
        if "Sendil" in text:
            # High similarity between Sendil K. and Sendil Kumar
            return [0.99, 0.01, 0.0]
        elif "Kumar" in text:
            return [0.95, 0.05, 0.0]
        # completely different entity
        return [0.0, 0.0, 1.0]

def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([0, 0], [1, 1]) == 0.0

def test_entity_resolution():
    resolver = EntityResolver(FakeEmbedModel(), threshold=0.85)
    
    # Run resolution
    mapping = resolver.resolve_entities(["Sendil K.", "Sendil Kumar", "Python"])
    
    # Verify exact match / high similarity resolves to first variant
    assert mapping["Sendil K."] == "Sendil K."
    assert mapping["Sendil Kumar"] == "Sendil K." # Similarity collision resolved!
    assert mapping["Python"] == "Python"
