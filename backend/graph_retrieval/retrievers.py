from typing import List, Dict, Any
import json
import logging

from llama_index.core.indices.property_graph import (
    VectorContextRetriever,
    LLMSynonymRetriever,
    TextToCypherRetriever,
    PropertyGraphIndex
)

from backend.graph_ingestion.stores import get_neo4j_graph_store, get_chroma_vector_store, get_llm
from backend.graph_ingestion.embedding_wrapper import LangChainEmbeddingWrapper
from backend.services.chroma_service import ChromaService
from backend.graph_retrieval.multi_hop import MultiHopGraphRetriever, TraversalStrategy
from backend.core.config import settings
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


def contextualize_graph_result(content: str, metadata: Dict[str, Any]) -> str:
    """Attach persisted evidence chunks to an entity result for grounded generation."""
    description = metadata.get("description", "")
    raw_chunks = metadata.get("parent_context_chunks_json", "")
    try:
        chunks = json.loads(raw_chunks) if isinstance(raw_chunks, str) else raw_chunks
    except (TypeError, json.JSONDecodeError):
        chunks = []
    evidence = [f"[{c.get('chunk_id', 'graph-context')}] {c.get('text', '')}" for c in chunks[:3] if c.get("text")]
    sections = [part for part in (description, content, "\n".join(evidence)) if part]
    return "\n\n".join(sections)

class ComposedGraphRetriever:
    """
    Composed retriever for the hybrid RAG system using PropertyGraphIndex.
    Combines vector search, synonym-expansion graph traversal, and optional Text-to-Cypher.
    """
    def __init__(
        self,
        use_cypher: bool = False,
        neo4j_driver: Any = None,
        max_hops: int = 3,
        traversal_strategy: TraversalStrategy | str = TraversalStrategy.BFS,
    ):
        self.chroma_service = ChromaService()
        self.embed_model = LangChainEmbeddingWrapper(self.chroma_service.embeddings)
        self.graph_store = get_neo4j_graph_store()
        self.vector_store = get_chroma_vector_store()
        self.llm = get_llm()
        self.use_cypher = use_cypher
        self._owns_driver = neo4j_driver is None
        self.neo4j_driver = neo4j_driver or GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        self.multi_hop_retriever = MultiHopGraphRetriever(
            self.neo4j_driver, max_hops=max_hops, strategy=traversal_strategy
        )

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        max_hops: int = 3,
        traversal_strategy: TraversalStrategy | str = TraversalStrategy.BFS,
        community_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relevant document chunks and context using composed retrievers.
        """
        try:
            # Initialize index from existing store configuration
            index = PropertyGraphIndex.from_existing(
                property_graph_store=self.graph_store,
                vector_store=self.vector_store,
                embed_model=self.embed_model,
                llm=self.llm
            )
            
            # Setup sub retrievers
            vector_retriever = VectorContextRetriever(
                graph_store=self.graph_store,
                vector_store=self.vector_store,
                embed_model=self.embed_model,
                similarity_top_k=n_results
            )
            
            synonym_retriever = LLMSynonymRetriever(
                graph_store=self.graph_store,
                llm=self.llm,
                limit=n_results * 2
            )
            
            sub_retrievers = [vector_retriever, synonym_retriever]
            
            # TextToCypher is run behind a feature flag and is sandboxed (using Neo4j read-only credentials if possible)
            if self.use_cypher:
                cypher_retriever = TextToCypherRetriever(
                    graph_store=self.graph_store,
                    llm=self.llm
                )
                sub_retrievers.append(cypher_retriever)
                
            # Compose retrievers
            retriever = index.as_retriever(sub_retrievers=sub_retrievers)
            
            # Run retrieval
            results = retriever.retrieve(query)
            
            # Format and deduplicate results
            formatted_results = []
            seen_contents = set()
            for r in results:
                metadata = r.node.metadata or {}
                content = contextualize_graph_result(r.node.get_content(), metadata)
                if not content.strip():
                    continue
                if content not in seen_contents:
                    seen_contents.add(content)
                    
                    # Convert LlamaIndex NodeWithScore to our project standard dictionary format
                    formatted_results.append({
                        "id": r.node.node_id,
                        "content": content,
                        "metadata": metadata,
                        "score": r.score if r.score is not None else 1.0,
                        "source": "graph"
                    })
                    
            seed_ids = []
            for result in formatted_results:
                metadata = result["metadata"]
                for key in ("id", "name", "entity_id", "vector_source_id"):
                    if metadata.get(key):
                        seed_ids.append(str(metadata[key]))
            try:
                traversal_results = self.multi_hop_retriever.retrieve(
                    query=query,
                    n_results=n_results,
                    max_hops=max_hops,
                    strategy=traversal_strategy,
                    seed_node_ids=seed_ids,
                    community_id=community_id,
                )
            except Exception as exc:
                logger.warning("Multi-hop traversal failed; returning composed base results: %s", exc)
                traversal_results = []
            traversal_limit = min(len(traversal_results), max(1, n_results // 2))
            base_limit = max(0, n_results - traversal_limit)
            return [*formatted_results[:base_limit], *traversal_results[:traversal_limit]]
            
        except Exception as e:
            logger.error("Error during composed graph retrieval: %s", e)
            return []

    def close(self) -> None:
        if self._owns_driver:
            self.neo4j_driver.close()
