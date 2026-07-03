from backend.graph_ingestion.ingest import IdempotentGraphIngester
from llama_index.core.schema import TextNode
from unittest.mock import MagicMock, patch
import pytest

@patch("backend.graph_ingestion.ingest.GraphDatabase.driver")
@patch("backend.graph_ingestion.ingest.ChromaService")
@patch("backend.graph_ingestion.ingest.get_neo4j_graph_store")
@patch("backend.graph_ingestion.ingest.get_chroma_vector_store")
@patch("backend.graph_ingestion.ingest.get_llm")
def test_ingest_idempotency_skip(mock_get_llm, mock_get_chroma_store, mock_get_neo4j_store, mock_chroma_service, mock_driver_cls):
    # Setup mock driver to return count = 1 (meaning already ingested)
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.single.return_value = {"count": 1}
    mock_session.run.return_value = mock_result
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_driver_cls.return_value = mock_driver

    # Instantiate ingester
    ingester = IdempotentGraphIngester()
    
    with patch.object(IdempotentGraphIngester, "chunk_text", return_value=["test chunk content"]):
        # Run ingestion
        res = ingester.ingest_file("tests/test_schema.py")
        
        # Check results
        assert res["status"] == "skipped"
        assert res["total_chunks"] == 1
        assert res["skipped_chunks"] == 1
        assert res["ingested_chunks"] == 0
        
        # Check that query was called
        mock_session.run.assert_called_once()
            
    ingester.close()

@patch("backend.graph_ingestion.ingest.GraphDatabase.driver")
@patch("backend.graph_ingestion.ingest.ChromaService")
@patch("backend.graph_ingestion.ingest.get_neo4j_graph_store")
@patch("backend.graph_ingestion.ingest.get_chroma_vector_store")
@patch("backend.graph_ingestion.ingest.get_llm")
@patch("backend.graph_ingestion.ingest.PropertyGraphIndex")
@patch("backend.graph_ingestion.ingest.get_extractor_stack")
def test_ingest_idempotency_ingest(mock_get_extractors, mock_property_graph_index, mock_get_llm, mock_get_chroma_store, mock_get_neo4j_store, mock_chroma_service, mock_driver_cls):
    # Setup mock driver to return count = 0 (meaning NOT ingested)
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.single.return_value = {"count": 0}
    mock_session.run.return_value = mock_result
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_driver_cls.return_value = mock_driver
    
    # Setup extractors mock
    mock_extractor = MagicMock()
    mock_extractor.side_effect = lambda nodes: nodes
    mock_get_extractors.return_value = [mock_extractor]

    # Instantiate ingester
    ingester = IdempotentGraphIngester()
    
    with patch.object(IdempotentGraphIngester, "chunk_text", return_value=["test chunk content"]):
        # Run ingestion
        res = ingester.ingest_file("tests/test_schema.py")
        
        # Check results
        assert res["status"] == "success"
        assert res["total_chunks"] == 1
        assert res["skipped_chunks"] == 0
        assert res["ingested_chunks"] == 1
        
        # Verify PropertyGraphIndex was built
        mock_property_graph_index.assert_called_once()
            
    ingester.close()
