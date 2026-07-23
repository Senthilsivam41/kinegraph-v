from typing import Dict, Any, List
from backend.graph_retrieval.retrievers import ComposedGraphRetriever
from backend.graph_retrieval.multi_hop import TraversalStrategy

class LangGraphGraphRetrieverNode:
    """
    Wraps the ComposedGraphRetriever to match the interface contract expected by
    the existing LangGraph orchestration.
    """
    def __init__(self, use_cypher: bool = False, neo4j_driver: Any = None):
        self.retriever = ComposedGraphRetriever(use_cypher=use_cypher, neo4j_driver=neo4j_driver)

    async def retrieve_chunks(
        self,
        query: str,
        n_results: int = 10,
        max_hops: int = 3,
        traversal_strategy: TraversalStrategy | str = TraversalStrategy.BFS,
        community_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves matching chunks and returns standard formatted dicts for LangGraph workflow.
        """
        # Call the synchronous retriever inside this async wrapper (matching project conventions)
        import asyncio
        results = await asyncio.to_thread(
            self.retriever.retrieve,
            query=query,
            n_results=n_results,
            max_hops=max_hops,
            traversal_strategy=traversal_strategy,
            community_id=community_id,
        )
        return results

    async def retrieve_chunks_with_diagnostics(
        self,
        query: str,
        n_results: int = 10,
        max_hops: int = 3,
        traversal_strategy: TraversalStrategy | str = TraversalStrategy.BFS,
        community_id: str | None = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Return graph candidates and traversal diagnostics without shared mutable state."""
        import asyncio
        return await asyncio.to_thread(
            self.retriever.retrieve_with_diagnostics,
            query=query,
            n_results=n_results,
            max_hops=max_hops,
            traversal_strategy=traversal_strategy,
            community_id=community_id,
        )
