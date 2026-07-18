import json

from backend.graph_ingestion.enrichment import ContextChunkLink, KineticVNode
from backend.graph_retrieval.retrievers import contextualize_graph_result


def test_v3_node_properties_are_neo4j_safe_and_preserve_chunk_ids():
    node = KineticVNode(
        node_id="neo4j",
        entity_type="Technology",
        description="A graph database.",
        parent_context_chunks=[ContextChunkLink("vec_0452", "Neo4j stores relationships as edges.")],
        graph_positioning={"community_id": "storage", "centrality_score": 0.92, "depth_from_root": 1},
    )

    properties = node.neo4j_properties()
    assert properties["parent_context_chunk_ids"] == ["vec_0452"]
    assert json.loads(properties["parent_context_chunks_json"])[0]["text"].startswith("Neo4j")
    assert properties["schema_version"] == "3"


def test_contextualize_graph_result_includes_citable_chunk_text():
    content = contextualize_graph_result(
        "Neo4j",
        {"description": "A graph database.", "parent_context_chunks_json": json.dumps([{"chunk_id": "c1", "text": "It stores data as nodes and edges."}])},
    )
    assert "A graph database." in content
    assert "[c1] It stores data as nodes and edges." in content
