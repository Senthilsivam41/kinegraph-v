from backend.graph_retrieval.retrievers import ComposedGraphRetriever
from backend.graph_retrieval.multi_hop import TraversalStrategy
from llama_index.core.schema import TextNode, NodeWithScore
from unittest.mock import ANY, MagicMock, patch
import pytest

@patch("backend.graph_retrieval.retrievers.PropertyGraphIndex")
@patch("backend.graph_retrieval.retrievers.VectorContextRetriever")
@patch("backend.graph_retrieval.retrievers.LLMSynonymRetriever")
@patch("backend.graph_retrieval.retrievers.TextToCypherRetriever")
@patch("backend.graph_retrieval.retrievers.ChromaService")
@patch("backend.graph_retrieval.retrievers.get_neo4j_graph_store")
@patch("backend.graph_retrieval.retrievers.get_chroma_vector_store")
@patch("backend.graph_retrieval.retrievers.get_llm")
def test_composed_retriever(
    mock_get_llm, 
    mock_get_chroma_store, 
    mock_get_neo4j_store, 
    mock_chroma_service,
    mock_cypher_retriever,
    mock_synonym_retriever,
    mock_vector_retriever,
    mock_property_graph_index
):
    # Setup mock index
    mock_index = MagicMock()
    mock_property_graph_index.from_existing.return_value = mock_index
    
    # Setup mock LlamaIndex retriever and its return value
    mock_retriever = MagicMock()
    mock_index.as_retriever.return_value = mock_retriever
    
    # Prepare dummy NodeWithScore list
    node1 = TextNode(text="first chunk content", metadata={"key": "val1"})
    node2 = TextNode(text="second chunk content", metadata={"key": "val2"})
    mock_retriever.retrieve.return_value = [
        NodeWithScore(node=node1, score=0.9),
        NodeWithScore(node=node2, score=0.7)
    ]
    
    # Instantiate ComposedGraphRetriever
    retriever = ComposedGraphRetriever(use_cypher=True, neo4j_driver=MagicMock())
    retriever.multi_hop_retriever.retrieve = MagicMock(return_value=[{
        "content": "Traversal path (depth 1): A -[USES]-> B",
        "metadata": {"traversal_depth": 1, "relationship_path": [{"relationship_type": "USES"}]},
        "score": 0.6,
        "source": "graph_traversal",
    }])
    results = retriever.retrieve("query text", n_results=5)
    
    # Verify sub-retrievers were constructed and index was initialized
    mock_property_graph_index.from_existing.assert_called_once()
    mock_vector_retriever.assert_called_once()
    mock_synonym_retriever.assert_called_once()
    mock_cypher_retriever.assert_called_once()
    
    # Verify outputs are formatted properly
    assert len(results) == 3
    assert results[0]["content"] == "first chunk content"
    assert results[0]["metadata"]["key"] == "val1"
    assert results[0]["score"] == 0.9
    assert results[0]["source"] == "graph"
    assert results[1]["content"] == "second chunk content"
    assert results[1]["metadata"]["key"] == "val2"
    assert results[1]["score"] == 0.7
    assert results[1]["source"] == "graph"
    assert results[2]["source"] == "graph_traversal"
    assert results[2]["metadata"]["traversal_depth"] == 1
    retriever.multi_hop_retriever.retrieve.assert_called_once_with(
        query="query text",
        n_results=5,
        max_hops=3,
        strategy=TraversalStrategy.BFS,
        seed_node_ids=[],
        community_id=None,
        diagnostics=ANY,
    )
    diagnostics = retriever.multi_hop_retriever.retrieve.call_args.kwargs["diagnostics"]
    assert diagnostics["base_candidate_count"] == 2
    assert diagnostics["traversal_candidate_count"] == 1
