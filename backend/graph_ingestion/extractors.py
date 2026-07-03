from typing import List, Any, Sequence
from enum import Enum as PyEnum
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor, SimpleLLMPathExtractor
from llama_index.core.schema import TransformComponent, BaseNode
from backend.graph_ingestion.schema import OntologySchema

class TaggedSimpleLLMPathExtractor(SimpleLLMPathExtractor):
    """
    Subclass of SimpleLLMPathExtractor that tags all extracted relationships
    with metadata marking them as fallback extractions.
    """
    async def acall(self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any) -> Sequence[BaseNode]:
        res_nodes = await super().acall(nodes, show_progress=show_progress, **kwargs)
        for node in res_nodes:
            if "relations" in node.metadata:
                for rel in node.metadata["relations"]:
                    if not rel.properties:
                        rel.properties = {}
                    rel.properties["source"] = "simple_fallback"
        return res_nodes

    def __call__(self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any) -> Sequence[BaseNode]:
        res_nodes = super().__call__(nodes, show_progress=show_progress, **kwargs)
        for node in res_nodes:
            if "relations" in node.metadata:
                for rel in node.metadata["relations"]:
                    if not rel.properties:
                        rel.properties = {}
                    rel.properties["source"] = "simple_fallback"
        return res_nodes

def get_extractor_stack(schema: OntologySchema, llm: Any) -> List[TransformComponent]:
    """
    Construct the stacked extractor pipeline:
    1. SchemaLLMPathExtractor: Strict, schema-guided ontology extraction.
    2. TaggedSimpleLLMPathExtractor: Free-form fallback tagged with 'source: simple_fallback'.
    """
    # Create Python Enums dynamically from ontology types as required by LlamaIndex SchemaLLMPathExtractor
    EntityTypesEnum = PyEnum(
        "EntityTypesEnum", 
        {et.replace(" ", "_").upper(): et for et in schema.entity_types}, 
        type=str
    )
    RelationTypesEnum = PyEnum(
        "RelationTypesEnum", 
        {rt.replace(" ", "_").upper(): rt for rt in schema.relation_types}, 
        type=str
    )
    
    # Map valid triples to Enums format if needed, or pass as list of tuples of Enum members/strings
    # LlamaIndex accepts List[Tuple[str, str, str]] as kg_validation_schema or possible_triples
    schema_extractor = SchemaLLMPathExtractor(
        llm=llm,
        possible_entities=EntityTypesEnum,
        possible_relations=RelationTypesEnum,
        kg_validation_schema=schema.valid_triples,
        strict=True
    )
    
    fallback_extractor = TaggedSimpleLLMPathExtractor(
        llm=llm,
        num_workers=1
    )
    
    return [schema_extractor, fallback_extractor]
