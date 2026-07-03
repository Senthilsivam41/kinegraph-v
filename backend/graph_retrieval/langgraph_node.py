from typing import Dict, Any, List
from backend.graph_retrieval.retrievers import ComposedGraphRetriever

class LangGraphGraphRetrieverNode:
    """
    Wraps the ComposedGraphRetriever to match the interface contract expected by
    the existing LangGraph orchestration.
    """
    def __init__(self, use_cypher: bool = False):
        self.retriever = ComposedGraphRetriever(use_cypher=use_cypher)

    async def retrieve_chunks(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves matching chunks and returns standard formatted dicts for LangGraph workflow.
        """
        # Call the synchronous retriever inside this async wrapper (matching project conventions)
        import asyncio
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, 
            self.retriever.retrieve, 
            query, 
            n_results
        )
        return results
