from typing import List, Dict, Any
import json
import logging

from llama_index.core.indices.property_graph import (
    VectorContextRetriever,
    LLMSynonymRetriever,
    PropertyGraphIndex
)

from backend.graph_ingestion.stores import get_neo4j_graph_store, get_chroma_vector_store, get_llm
from backend.graph_ingestion.embedding_wrapper import LangChainEmbeddingWrapper
from backend.services.chroma_service import ChromaService
from backend.graph_retrieval.multi_hop import MultiHopGraphRetriever, TraversalStrategy
from backend.core.config import settings
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


def metadata_matches(metadata: Dict[str, Any] | None, filters: Dict[str, Any] | None) -> bool:
    """Return whether graph metadata satisfies the same equality filters as Chroma.

    Missing fields and unsupported operators fail closed so out-of-scope
    evidence cannot enter graph candidate selection.
    """
    if not filters:
        return True
    return _match_where(metadata or {}, filters)


def _match_where(metadata: Dict[str, Any], clause: Any) -> bool:
    if not isinstance(clause, dict) or not clause:
        return False
    if "$and" in clause:
        parts = clause["$and"]
        return isinstance(parts, list) and all(_match_where(metadata, part) for part in parts)
    if "$or" in clause:
        parts = clause["$or"]
        return isinstance(parts, list) and bool(parts) and any(_match_where(metadata, part) for part in parts)
    for key, expected in clause.items():
        if not isinstance(key, str) or key.startswith("$"):
            return False
        actual = metadata.get(key)
        if isinstance(expected, dict):
            if "$eq" in expected and actual != expected["$eq"]:
                return False
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            if "$in" in expected and actual not in list(expected["$in"]):
                return False
            if not any(operator in expected for operator in ("$eq", "$ne", "$in")):
                return False
        elif actual != expected:
            return False
    return True


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
    Combines vector search and synonym-expansion graph traversal.
    """
    def __init__(
        self,
        use_cypher: bool = False,
        neo4j_driver: Any = None,
        max_hops: int = 3,
        traversal_strategy: TraversalStrategy | str = TraversalStrategy.BFS,
    ):
        if use_cypher:
            raise ValueError(
                "Text-to-Cypher is disabled until validated, bounded read-only execution is available"
            )
        self.chroma_service = ChromaService()
        self.embed_model = LangChainEmbeddingWrapper(self.chroma_service.embeddings)
        self.graph_store = get_neo4j_graph_store()
        self.vector_store = get_chroma_vector_store()
        self.llm = get_llm()
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
        filters: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        results, _ = self.retrieve_with_diagnostics(
            query=query,
            n_results=n_results,
            max_hops=max_hops,
            traversal_strategy=traversal_strategy,
            community_id=community_id,
            filters=filters,
        )
        return results

    def retrieve_with_diagnostics(
        self,
        query: str,
        n_results: int = 10,
        max_hops: int = 3,
        traversal_strategy: TraversalStrategy | str = TraversalStrategy.BFS,
        community_id: str | None = None,
        filters: Dict[str, Any] | None = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieve graph context together with bounded traversal diagnostics.
        """
        diagnostics: Dict[str, Any] = {
            "traversal_failure": None,
            "base_candidate_count": 0,
        }
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
            unfiltered_count = len(formatted_results)
            formatted_results = [
                result for result in formatted_results
                if metadata_matches(result.get("metadata"), filters)
            ]
            diagnostics["base_candidate_count"] = len(formatted_results)
            diagnostics["metadata_filter_removed"] = unfiltered_count - len(formatted_results)
                    
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
                    diagnostics=diagnostics,
                )
            except Exception as exc:
                logger.warning("Multi-hop traversal failed; returning composed base results: %s", exc)
                diagnostics["traversal_failure"] = f"{type(exc).__name__}: {exc}"[:500]
                traversal_results = []
            traversal_count = len(traversal_results)
            traversal_results = [
                result for result in traversal_results
                if metadata_matches(result.get("metadata"), filters)
            ]
            diagnostics["traversal_candidate_count"] = len(traversal_results)
            diagnostics["metadata_filter_removed"] = (
                diagnostics.get("metadata_filter_removed", 0)
                + traversal_count
                - len(traversal_results)
            )
            traversal_limit = min(len(traversal_results), max(1, n_results // 2))
            base_limit = max(0, n_results - traversal_limit)
            results = [*formatted_results[:base_limit], *traversal_results[:traversal_limit]]
            diagnostics["returned_candidate_count"] = len(results)
            return results, diagnostics
            
        except Exception as e:
            logger.error("Error during composed graph retrieval: %s", e)
            diagnostics["retrieval_failure"] = f"{type(e).__name__}: {e}"[:500]
            return [], diagnostics

    def close(self) -> None:
        if self._owns_driver:
            self.neo4j_driver.close()
