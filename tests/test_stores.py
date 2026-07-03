from backend.graph_ingestion.stores import get_neo4j_graph_store, get_chroma_vector_store
from unittest.mock import MagicMock, patch

@patch("backend.graph_ingestion.stores.Neo4jPropertyGraphStore")
def test_neo4j_store_initialization(mock_neo4j_store):
    mock_neo4j_store.return_value = MagicMock()
    store = get_neo4j_graph_store()
    assert store is not None
    mock_neo4j_store.assert_called_once()

@patch("backend.graph_ingestion.stores.chromadb.HttpClient")
@patch("backend.graph_ingestion.stores.ChromaVectorStore")
def test_chroma_store_initialization(mock_chroma_vector_store, mock_chroma_client):
    mock_client_instance = MagicMock()
    mock_chroma_client.return_value = mock_client_instance
    mock_chroma_vector_store.return_value = MagicMock()
    
    store = get_chroma_vector_store()
    assert store is not None
    mock_chroma_client.assert_called_once()
    mock_client_instance.get_or_create_collection.assert_called_once_with("kg_nodes")
    mock_chroma_vector_store.assert_called_once()
