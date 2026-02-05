"""
LangGraph Workflow for Hybrid RAG System
Orchestrates Vector and Graph searches with RRF fusion
"""
from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
from app.models import QueryMode, DocumentChunk
from services.chroma_service import ChromaService
from services.neo4j_service import Neo4jService
from core.rrf import reciprocal_rank_fusion, deduplicate_results
from core.config import settings


class WorkflowState(TypedDict):
    """State for the LangGraph workflow"""
    query: str
    mode: QueryMode
    max_results: int
    filters: Optional[Dict[str, Any]]
    vector_results: List[Dict[str, Any]]
    graph_results: List[Dict[str, Any]]
    fused_results: List[Dict[str, Any]]
    final_results: List[DocumentChunk]


class HybridRAGWorkflow:
    """
    LangGraph workflow for orchestrating hybrid RAG queries
    """
    
    def __init__(
        self,
        chroma_service: ChromaService,
        neo4j_service: Neo4jService
    ):
        """
        Initialize the workflow
        
        Args:
            chroma_service: ChromaDB service instance
            neo4j_service: Neo4j service instance
        """
        self.chroma = chroma_service
        self.neo4j = neo4j_service
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow
        
        Flow:
        START → Route → [Vector Agent | Graph Agent | Both] → Fusion → Format → END
        """
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("vector_agent", self._vector_agent)
        workflow.add_node("graph_agent", self._graph_agent)
        workflow.add_node("fusion_node", self._fusion_node)
        workflow.add_node("format_results", self._format_results)
        
        # Add conditional routing
        workflow.set_conditional_entry_point(
            self._route_query,
            {
                "vector": "vector_agent",
                "graph": "graph_agent",
                "hybrid": "vector_agent"  # Hybrid starts with vector, then graph
            }
        )
        
        # Add edges
        workflow.add_edge("vector_agent", "fusion_node")
        workflow.add_edge("graph_agent", "fusion_node")
        workflow.add_edge("fusion_node", "format_results")
        workflow.add_edge("format_results", END)
        
        return workflow.compile()
    
    def _route_query(self, state: WorkflowState) -> str:
        """Determine which path to take based on query mode"""
        return state["mode"].value
    
    async def _vector_agent(self, state: WorkflowState) -> WorkflowState:
        """
        Vector Agent: Performs similarity search in ChromaDB
        """
        print(f"[Vector Agent] Searching for: {state['query']}")
        
        results = await self.chroma.similarity_search(
            query=state["query"],
            n_results=state["max_results"],
            filters=state.get("filters")
        )
        
        state["vector_results"] = results
        
        # If hybrid mode, also call graph agent
        if state["mode"] == QueryMode.HYBRID:
            state = await self._graph_agent(state)
        
        return state
    
    async def _graph_agent(self, state: WorkflowState) -> WorkflowState:
        """
        Graph Agent: Performs Cypher query in Neo4j
        """
        print(f"[Graph Agent] Searching for: {state['query']}")
        
        results = await self.neo4j.graph_search(
            query=state["query"],
            n_results=state["max_results"]
        )
        
        state["graph_results"] = results
        return state
    
    async def _fusion_node(self, state: WorkflowState) -> WorkflowState:
        """
        Fusion Node: Combines results using Reciprocal Rank Fusion (RRF)
        """
        print("[Fusion Node] Merging results with RRF")
        
        mode = state["mode"]
        
        if mode == QueryMode.VECTOR:
            # Only vector results
            fused = state["vector_results"]
        elif mode == QueryMode.GRAPH:
            # Only graph results
            fused = state["graph_results"]
        else:  # HYBRID
            # Combine both with RRF
            results_to_fuse = []
            if state.get("vector_results"):
                results_to_fuse.append(state["vector_results"])
            if state.get("graph_results"):
                results_to_fuse.append(state["graph_results"])
            
            if results_to_fuse:
                fused = reciprocal_rank_fusion(results_to_fuse)
            else:
                fused = []
        
        # Deduplicate
        fused = deduplicate_results(fused)
        
        # Limit to max_results
        fused = fused[:state["max_results"]]
        
        state["fused_results"] = fused
        return state
    
    async def _format_results(self, state: WorkflowState) -> WorkflowState:
        """
        Format results into DocumentChunk objects
        """
        print("[Format] Converting to DocumentChunk objects")
        
        formatted = []
        for result in state["fused_results"]:
            chunk = DocumentChunk(
                content=result.get("content", ""),
                metadata=result.get("metadata", {}),
                score=result.get("score", 0.0),
                source=result.get("source", "unknown")
            )
            formatted.append(chunk)
        
        state["final_results"] = formatted
        return state
    
    async def execute(
        self,
        query: str,
        mode: QueryMode = QueryMode.HYBRID,
        max_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Execute the workflow
        
        Args:
            query: User's query
            mode: Query mode (vector, graph, or hybrid)
            max_results: Maximum number of results to return
            filters: Optional filters for the query
            
        Returns:
            List of DocumentChunk results
        """
        # Initialize state
        initial_state = WorkflowState(
            query=query,
            mode=mode,
            max_results=max_results,
            filters=filters,
            vector_results=[],
            graph_results=[],
            fused_results=[],
            final_results=[]
        )
        
        # Run workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        return final_state["final_results"]
