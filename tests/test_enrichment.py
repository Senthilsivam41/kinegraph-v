import json
import asyncio
from unittest.mock import MagicMock
from unittest.mock import patch

from backend.graph_ingestion.enrichment import (
    ChromaChunkValidator,
    ContextChunkLink,
    KineticVNode,
    NodeEnricher,
)
from backend.graph_retrieval.retrievers import contextualize_graph_result
from backend.services.neo4j_service import Neo4jService


def test_v3_node_properties_are_neo4j_safe_and_preserve_chunk_ids():
    node = KineticVNode(
        node_id="neo4j",
        entity_type="Technology",
        description="A graph database.",
        parent_context_chunks=[ContextChunkLink(
            "vec_0452", "Neo4j stores relationships as edges.",
            embedding_verified=True, vector_collection="kg_nodes",
        )],
        graph_positioning={"community_id": "storage", "centrality_score": 0.92, "depth_from_root": 1},
    )

    properties = node.neo4j_properties()
    assert properties["parent_context_chunk_ids"] == ["vec_0452"]
    assert json.loads(properties["parent_context_chunks_json"])[0]["text"].startswith("Neo4j")
    assert properties["schema_version"] == "3"
    assert properties["vector_links_verified"] is True


def test_contextualize_graph_result_includes_citable_chunk_text():
    content = contextualize_graph_result(
        "Neo4j",
        {"description": "A graph database.", "parent_context_chunks_json": json.dumps([{"chunk_id": "c1", "text": "It stores data as nodes and edges."}])},
    )
    assert "A graph database." in content
    assert "[c1] It stores data as nodes and edges." in content


def test_chroma_validator_rejects_missing_or_embeddingless_links():
    collection = MagicMock()
    def collection_get(ids=None, where=None, include=None):
        if where:
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        selected = [chunk_id for chunk_id in (ids or []) if chunk_id in {"verified", "no_embedding"}]
        return {
            "ids": selected,
            "documents": ["authoritative text" if chunk_id == "verified" else "not valid" for chunk_id in selected],
            "metadatas": [{} for _ in selected],
            "embeddings": [[0.1, 0.2] if chunk_id == "verified" else None for chunk_id in selected],
        }
    collection.get.side_effect = collection_get
    client = MagicMock()
    client.get_collection.return_value = collection

    links, missing = ChromaChunkValidator(client, ["kg_nodes"]).resolve([
        {"id": "verified", "text": "graph text"},
        {"id": "no_embedding", "text": "graph text"},
        {"id": "absent", "text": "graph text"},
    ], batch_size=2)

    assert [link.chunk_id for link in links] == ["verified"]
    assert links[0].text == "authoritative text"
    assert links[0].embedding_verified is True
    assert missing == {"no_embedding", "absent"}
    assert collection.get.call_count >= 2


def test_chroma_validator_accepts_triplet_source_provenance():
    collection = MagicMock()
    collection.get.side_effect = [
        {"ids": [], "documents": [], "metadatas": [], "embeddings": []},
        {"ids": ["entity-vector-1"], "documents": ["entity vector"],
         "metadatas": [{"triplet_source_id": "chunk-1"}], "embeddings": [[0.2, 0.3]]},
    ]
    client = MagicMock()
    client.get_collection.return_value = collection

    links, missing = ChromaChunkValidator(client, ["kg_nodes"]).resolve(
        [{"id": "chunk-1", "text": "source chunk text"}], batch_size=10
    )

    assert missing == set()
    assert links[0].vector_record_id == "entity-vector-1"
    assert links[0].verification_method == "triplet_source_id"
    assert links[0].text == "source chunk text"


def test_graph_positioning_computes_communities_centrality_and_root_depth():
    positions = NodeEnricher._graph_positioning(
        {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}, "isolated": set()},
        rooted={"a"},
    )

    assert positions["a"]["community_id"] == positions["c"]["community_id"]
    assert positions["isolated"]["community_id"] == "isolated"
    assert positions["b"]["centrality_score"] == 1.0
    assert positions["c"]["depth_from_root"] == 3
    assert positions["isolated"]["depth_from_root"] == -1


def test_relationship_fallback_generates_evidence_and_non_default_weight():
    relationship = NodeEnricher._relationship(
        {"target": "Neo4j", "type": "USES", "direction": "OUTGOING"},
        {"name": "Kinegraph"},
        [ContextChunkLink("c1", "Kinegraph uses Neo4j for graph storage.", embedding_verified=True)],
    )

    assert relationship.evidence_text.startswith("Kinegraph USES Neo4j")
    assert relationship.weight == 0.6


def test_node_enricher_batches_writes_and_reports_verified_links():
    topology = [
        {"id": "e1", "neighbors": ["e2"], "rooted": True},
        {"id": "e2", "neighbors": ["e1"], "rooted": True},
    ]
    records = [
        {"element_id": "e1", "node": {"name": "One", "type": "Concept"},
         "chunks": [{"id": "c1", "text": "one evidence"}], "relationships": []},
        {"element_id": "e2", "node": {"name": "Two", "type": "Concept"},
         "chunks": [{"id": "c2", "text": "two evidence"}], "relationships": []},
    ]
    session = MagicMock()
    session.run.side_effect = [topology, [records[0]], MagicMock(), [records[1]], MagicMock(), []]
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session

    collection = MagicMock()
    collection.get.side_effect = lambda ids=None, include=None, where=None: {
        "ids": ids,
        "documents": [f"verified {chunk_id}" for chunk_id in ids],
        "metadatas": [{} for _ in ids],
        "embeddings": [[0.1] for _ in ids],
    }
    client = MagicMock()
    client.get_collection.return_value = collection

    result = NodeEnricher(driver, client, collection_names=["kg_nodes"]).enrich(batch_size=1)

    assert result == {
        "enriched": 2,
        "skipped_without_verified_context": 0,
        "scanned": 2,
        "verified_vector_links": 2,
        "missing_vector_links": 0,
        "complete": True,
    }
    write_calls = [call for call in session.run.call_args_list if "updates" in call.kwargs]
    assert len(write_calls) == 2
    assert all(len(call.kwargs["updates"]) == 1 for call in write_calls)


def test_legacy_document_ingestion_creates_chunks_and_runs_enrichment():
    service = Neo4jService.__new__(Neo4jService)
    session = MagicMock()
    service.driver = MagicMock()
    service.driver.session.return_value.__enter__.return_value = session
    enrichment = {"enriched": 1, "complete": True}

    with patch("backend.services.chroma_service.ChromaService") as chroma_cls, patch(
        "backend.graph_ingestion.enrichment.NodeEnricher"
    ) as enricher_cls:
        enricher_cls.return_value.enrich.return_value = enrichment
        result = asyncio.run(service.add_document_graph(
            doc_id="doc-1",
            content="Kinegraph uses Neo4j.",
            metadata={"file_name": "test.md"},
            entities=[{"name": "Neo4j", "type": "Technology"}],
            relationships=[],
            chunks=["Kinegraph uses Neo4j."],
            chunk_ids=["chunk-1"],
        ))

    assert result.success is True
    assert result.enrichment == enrichment
    assert any("MERGE (c:__Node__:Chunk" in call.args[0] for call in session.run.call_args_list)
    enricher_cls.return_value.enrich.assert_called_once_with(chunk_ids=["chunk-1"])
    assert chroma_cls.called
    chroma_cls.return_value.close.assert_called_once_with()


def test_legacy_relationship_upsert_uses_valid_merge_clause_order():
    service = Neo4jService.__new__(Neo4jService)
    session = MagicMock()
    service.driver = MagicMock()
    service.driver.session.return_value.__enter__.return_value = session

    result = asyncio.run(service.add_document_graph(
        doc_id="doc-relationship",
        content="Kinegraph uses Neo4j.",
        metadata={"file_name": "test.md"},
        entities=[
            {"name": "Kinegraph", "type": "System"},
            {"name": "Neo4j", "type": "Technology"},
        ],
        relationships=[{
            "source": "Kinegraph",
            "target": "Neo4j",
            "type": "USES",
            "evidence_text": "Kinegraph uses Neo4j.",
            "weight": 0.9,
        }],
    ))

    assert result.success is True
    relationship_query = next(
        call.args[0] for call in session.run.call_args_list
        if "MERGE (e1)-[r:RELATES_TO" in call.args[0]
    )
    assert relationship_query.index("ON CREATE SET") < relationship_query.index(
        "SET r.evidence_text"
    )
