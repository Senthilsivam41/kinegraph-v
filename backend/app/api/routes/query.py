"""
Query Endpoints for Hybrid RAG — v2
Uses execute_with_answer() to surface generated answer + latency breakdown.
"""
import time
import uuid

from fastapi import APIRouter, Request, HTTPException

from backend.app.models import QueryRequest, QueryResponse, QueryMode
from backend.core.langgraph_workflow import HybridRAGWorkflow

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query_system(query_request: QueryRequest, request: Request):
    """
    Query the hybrid RAG system.

    v2 improvements:
    - Intent classification routes queries to the optimal retrieval mode.
    - Parallel vector + graph retrieval (hybrid mode) reduces latency.
    - Post-fusion reranker removes irrelevant chunks.
    - Grounded LLM generation node produces a faithfulness-focused answer.
    """
    start_time = time.perf_counter()
    query_id = str(uuid.uuid4())

    try:
        workflow = HybridRAGWorkflow(
            chroma_service=request.app.state.chroma,
            neo4j_service=request.app.state.neo4j,
        )

        result = await workflow.execute_with_answer(
            query=query_request.query,
            mode=query_request.mode,
            max_results=query_request.max_results,
            candidate_pool_size=query_request.candidate_pool_size,
            max_hops=query_request.max_hops,
            traversal_strategy=query_request.traversal_strategy,
            community_id=query_request.community_id,
            enable_conditional_recovery=query_request.enable_conditional_recovery,
            enable_hyde_fallback=query_request.enable_hyde_fallback,
            enable_grounding_critique=query_request.enable_grounding_critique,
            enable_lexical_fusion=query_request.enable_lexical_fusion,
            vector_fusion_weight=query_request.vector_fusion_weight,
            graph_fusion_weight=query_request.graph_fusion_weight,
            lexical_fusion_weight=query_request.lexical_fusion_weight,
            filters=query_request.filters,
            attachment_content=query_request.attachment_content,
            attachment_name=query_request.attachment_name,
        )
        execution_time = round((time.perf_counter() - start_time) * 1000, 2)
        chunks = result["chunks"]

        return QueryResponse(
            query=query_request.query,
            mode=query_request.mode,
            results=chunks,
            total_results=len(chunks),
            execution_time_ms=execution_time,
            generated_answer=result["answer"],
            answer_confidence=result["confidence"],
            intent=result["intent"],
            latency_breakdown=result["latency"],
            recovery_triggered=result["recovery_triggered"],
            recovery_details=result["recovery"],
            fusion_details=result["fusion"],
            grounded_claims=result["grounded_claims"],
            citation_validation=result["citation_validation"],
            grounding_critique=result["grounding_critique"],
            answer_relevancy=result["answer_relevancy"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query execution failed: {str(e)}",
        )


@router.get("/test")
async def test_query(request: Request):
    """Test endpoint to verify the query system is operational."""
    return {
        "status": "operational",
        "chroma_connected": hasattr(request.app.state, "chroma"),
        "neo4j_connected":  hasattr(request.app.state, "neo4j"),
        "workflow_version": "v2 (intent + parallel + rerank + generate)",
    }
