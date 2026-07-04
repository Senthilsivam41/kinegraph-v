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

from llama_index.core.indices.property_graph import ImplicitPathExtractor

class EntityResolutionExtractor(TransformComponent):
    """
    TransformComponent that resolves and deduplicates entities
    across all nodes in-place in their metadata.
    """
    resolver: Any

    def __init__(self, resolver: Any, **kwargs: Any) -> None:
        super().__init__(resolver=resolver, **kwargs)

    async def acall(self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any) -> Sequence[BaseNode]:
        return self(nodes, show_progress=show_progress, **kwargs)

    def __call__(self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any) -> Sequence[BaseNode]:
        entity_names = set()
        for node in nodes:
            if "nodes" in node.metadata:
                for ent in node.metadata["nodes"]:
                    entity_names.add(ent.name)
        if not entity_names:
            return nodes

        resolution_map = self.resolver.resolve_entities(list(entity_names))
        for node in nodes:
            if "nodes" in node.metadata:
                for ent in node.metadata["nodes"]:
                    ent.name = resolution_map.get(ent.name, ent.name)
            if "relations" in node.metadata:
                for rel in node.metadata["relations"]:
                    rel.source_id = resolution_map.get(rel.source_id, rel.source_id)
                    rel.target_id = resolution_map.get(rel.target_id, rel.target_id)
        return nodes

def get_extractor_stack(schema: OntologySchema, llm: Any, resolver: Any) -> List[TransformComponent]:
    """
    Construct the stacked extractor pipeline:
    1. SchemaLLMPathExtractor: Strict, schema-guided ontology extraction.
    2. TaggedSimpleLLMPathExtractor: Free-form fallback tagged with 'source: simple_fallback'.
    3. EntityResolutionExtractor: Global entity resolution/disambiguation.
    4. ImplicitPathExtractor: Mention and link extraction.
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
    
    resolver_extractor = EntityResolutionExtractor(resolver=resolver)
    
    implicit_extractor = ImplicitPathExtractor()
    
    return [schema_extractor, fallback_extractor, resolver_extractor, implicit_extractor]
