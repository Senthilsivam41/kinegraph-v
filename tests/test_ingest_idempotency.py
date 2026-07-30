from backend.graph_ingestion.ingest import IdempotentGraphIngester
from backend.graph_ingestion.adaptive_chunking import (
    CHUNK_POLICY_VERSION,
    ChunkRecord,
    content_hash,
    stable_chunk_id,
)
from unittest.mock import MagicMock, patch


def _record(text: str = "test chunk content", document_id: str = "doc_test") -> ChunkRecord:
    digest = content_hash(text)
    return ChunkRecord(
        chunk_id=stable_chunk_id(text, 0, document_id),
        text=text,
        chunk_type="recursive",
        document_id=document_id,
        ordinal=0,
        policy_version=CHUNK_POLICY_VERSION,
        boundary_reason="unstructured_text",
        tokenizer_version="recursive-character-v1",
        overlap=200,
    )


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

    with patch.object(IdempotentGraphIngester, "build_chunks", return_value=[_record()]):
        # Run ingestion
        res = ingester.ingest_file("tests/test_schema.py")

        # Check results
        assert res["status"] == "skipped"
        assert res["total_chunks"] == 1
        assert res["skipped_chunks"] == 1
        assert res["ingested_chunks"] == 0
        assert res["document_id"].startswith("doc_")

        # Check that query was called
        mock_session.run.assert_called_once()
        existence_call = mock_session.run.call_args
        assert "n.document_id = $document_id" in existence_call.args[0]
        assert existence_call.kwargs["document_id"] == "doc_test"

    ingester.close()


def test_document_scoped_dedup_keeps_identical_text_from_another_document():
    ingester = IdempotentGraphIngester.__new__(IdempotentGraphIngester)
    ingester.is_chunk_ingested = MagicMock(
        side_effect=lambda chunk_hash, document_id: document_id == "doc_a"
    )
    records = [
        _record("shared boilerplate", document_id="doc_a"),
        _record("shared boilerplate", document_id="doc_b"),
    ]

    nodes, skipped, accepted = ingester._nodes_from_records(
        records,
        file_name="shared.md",
        metadata={},
    )

    assert skipped == 1
    assert [record.document_id for record in accepted] == ["doc_b"]
    assert [node.metadata["document_id"] for node in nodes] == ["doc_b"]


@patch("backend.graph_ingestion.ingest.GraphDatabase.driver")
@patch("backend.graph_ingestion.ingest.ChromaService")
@patch("backend.graph_ingestion.ingest.get_neo4j_graph_store")
@patch("backend.graph_ingestion.ingest.get_chroma_vector_store")
@patch("backend.graph_ingestion.ingest.get_llm")
@patch("backend.graph_ingestion.ingest.PropertyGraphIndex")
@patch("backend.graph_ingestion.ingest.get_extractor_stack")
@patch("backend.graph_ingestion.ingest.NodeEnricher")
def test_ingest_idempotency_ingest(mock_node_enricher, mock_get_extractors, mock_property_graph_index, mock_get_llm, mock_get_chroma_store, mock_get_neo4j_store, mock_chroma_service, mock_driver_cls):
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
    mock_node_enricher.return_value.enrich.return_value = {
        "enriched": 1, "verified_vector_links": 1, "missing_vector_links": 0,
        "complete": True, "skipped_without_verified_context": 0,
    }

    # Instantiate ingester
    ingester = IdempotentGraphIngester()

    with patch.object(IdempotentGraphIngester, "build_chunks", return_value=[_record()]):
        # Run ingestion
        res = ingester.ingest_file("tests/test_schema.py")

        # Check results
        assert res["status"] == "success"
        assert res["total_chunks"] == 1
        assert res["skipped_chunks"] == 0
        assert res["ingested_chunks"] == 1
        assert res["enrichment"]["verified_vector_links"] == 1
        assert res["enrichment"]["status"] == "success"
        assert "ingestion_validation" in res
        assert res["ingestion_validation"]["complete"] is True
        assert res["document_id"].startswith("doc_")

        # Verify PropertyGraphIndex was built with a stable chunk id
        mock_property_graph_index.assert_called_once()
        nodes = mock_property_graph_index.call_args.kwargs["nodes"]
        assert len(nodes) == 1
        assert nodes[0].node_id.startswith("chunk_")
        mock_node_enricher.return_value.enrich.assert_called_once()

    ingester.close()
