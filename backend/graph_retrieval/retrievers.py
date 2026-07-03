from typing import List, Dict, Any
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

logger = logging.getLogger(__name__)

class ComposedGraphRetriever:
    """
    Composed retriever for the hybrid RAG system using PropertyGraphIndex.
    Combines vector search, synonym-expansion graph traversal, and optional Text-to-Cypher.
    """
    def __init__(self, use_cypher: bool = False):
        self.chroma_service = ChromaService()
        self.embed_model = LangChainEmbeddingWrapper(self.chroma_service.embeddings)
        self.graph_store = get_neo4j_graph_store()
        self.vector_store = get_chroma_vector_store()
        self.llm = get_llm()
        self.use_cypher = use_cypher

    def retrieve(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
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
                content = r.node.get_content()
                if not content.strip():
                    continue
                if content not in seen_contents:
                    seen_contents.add(content)
                    
                    # Convert LlamaIndex NodeWithScore to our project standard dictionary format
                    formatted_results.append({
                        "content": content,
                        "metadata": r.node.metadata,
                        "score": r.score if r.score is not None else 1.0,
                        "source": "graph"
                    })
                    
            return formatted_results[:n_results]
            
        except Exception as e:
            logger.error("Error during composed graph retrieval: %s", e)
            return []
