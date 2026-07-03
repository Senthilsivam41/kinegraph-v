from backend.graph_ingestion.extractors import get_extractor_stack, TaggedSimpleLLMPathExtractor
from backend.graph_ingestion.schema import OntologySchema
from llama_index.core.schema import TextNode
from llama_index.core.graph_stores.types import Relation
from llama_index.core.llms import MockLLM
from unittest.mock import MagicMock, patch
import pytest

def test_extractor_stack_creation():
    schema = OntologySchema("config/ontology_schema.yaml")
    mock_llm = MockLLM()
    stack = get_extractor_stack(schema, mock_llm)
    assert len(stack) == 2
    assert stack[0].__class__.__name__ == "SchemaLLMPathExtractor"
    assert isinstance(stack[1], TaggedSimpleLLMPathExtractor)

@pytest.mark.asyncio
async def test_tagged_simple_extractor_acall():
    # Setup mock super class behavior for acall
    mock_node = TextNode(text="dummy text")
    
    # Create relationship that should be tagged
    mock_relation = Relation(label="DEVELOPED", source_id="Alice", target_id="Concept", properties={})
    
    # We must patch the base class acall method
    with patch("llama_index.core.indices.property_graph.SimpleLLMPathExtractor.acall") as mock_super_acall:
        # Prepare returned nodes
        returned_node = TextNode(text="dummy text")
        returned_node.metadata["relations"] = [mock_relation]
        mock_super_acall.return_value = [returned_node]
        
        extractor = TaggedSimpleLLMPathExtractor(llm=MockLLM())
        result_nodes = await extractor.acall([mock_node])
        
        # Verify the relationship is tagged
        assert "relations" in result_nodes[0].metadata
        relations = result_nodes[0].metadata["relations"]
        assert len(relations) == 1
        assert relations[0].properties["source"] == "simple_fallback"
