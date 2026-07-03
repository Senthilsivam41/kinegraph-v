from backend.graph_ingestion.schema import OntologySchema

def test_schema_load():
    schema = OntologySchema("config/ontology_schema.yaml")
    assert "Person" in schema.entity_types
    assert "DEVELOPED" in schema.relation_types
    assert schema.validate_triple("Person", "DEVELOPED", "Concept") is True
    assert schema.validate_triple("Person", "USES", "Person") is False
