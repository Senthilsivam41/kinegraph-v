"""
Query Endpoints for Hybrid RAG
"""
from fastapi import APIRouter, Request, HTTPException
from app.models import QueryRequest, QueryResponse, QueryMode
from core.langgraph_workflow import HybridRAGWorkflow
import time

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query_system(query_request: QueryRequest, request: Request):
    """
    Query the hybrid RAG system
    
    Modes:
    - vector: Search only in ChromaDB using semantic similarity
    - graph: Search only in Neo4j using Cypher queries
    - hybrid: Search both and fuse results with RRF
    """
    start_time = time.time()
    
    try:
        # Initialize workflow
        workflow = HybridRAGWorkflow(
            chroma_service=request.app.state.chroma,
            neo4j_service=request.app.state.neo4j
        )
        
        # Execute query based on mode
        results = await workflow.execute(
            query=query_request.query,
            mode=query_request.mode,
            max_results=query_request.max_results,
            filters=query_request.filters
        )
        
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return QueryResponse(
            query=query_request.query,
            mode=query_request.mode,
            results=results,
            total_results=len(results),
            execution_time_ms=round(execution_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query execution failed: {str(e)}"
        )


@router.get("/test")
async def test_query(request: Request):
    """
    Test endpoint to verify query system is operational
    """
    return {
        "status": "operational",
        "chroma_connected": hasattr(request.app.state, 'chroma'),
        "neo4j_connected": hasattr(request.app.state, 'neo4j')
    }
