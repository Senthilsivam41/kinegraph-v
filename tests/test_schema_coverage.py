from backend.graph_ingestion.schema import OntologySchema
from backend.graph_ingestion.schema_coverage import audit_schema_coverage, render_markdown


def test_schema_audit_surfaces_fallback_candidates_and_golden_misses():
    schema = OntologySchema("config/ontology_schema.yaml")
    benchmark = [
        {"question": "How is Neo4j used?", "reference_text": "Kinegraph integrates Neo4j."},
        {"question": "What does the API expose?", "reference_text": "The endpoint accepts JSON."},
    ]
    entities = [
        {"name": "Kinegraph", "entity_type": "Component"},
        {"name": "Neo4j", "entity_type": "Technology"},
        {"name": "REST endpoint", "entity_type": "Interface"},
    ]
    relationships = [
        {
            "source_type": "Component",
            "relation_type": "INTEGRATES",
            "target_type": "Technology",
            "extraction_source": "",
        },
        {
            "source_type": "Component",
            "relation_type": "DEPENDS_ON",
            "target_type": "Interface",
            "extraction_source": "simple_fallback",
        },
        {
            "source_type": "Interface",
            "relation_type": "DEPENDS_ON",
            "target_type": "Technology",
            "extraction_source": "simple_fallback",
        },
    ]

    report = audit_schema_coverage(schema, benchmark, entities, relationships)

    assert report["golden_entity_mention_coverage"] == 0.5
    assert report["strict_triple_coverage"] == 0.3333
    assert report["out_of_schema_entity_types"] == {"Interface": 1}
    assert report["recurring_fallback_relation_candidates"] == {"DEPENDS_ON": 2}
    assert report["uncovered_questions"] == ["What does the API expose?"]
    assert "diagnostic proxy" in render_markdown(report)
