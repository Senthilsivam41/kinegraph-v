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

    tracer    = getattr(request.app.state, "tracer", None)
    collector = getattr(request.app.state, "metrics_collector", None)

    try:
        workflow = HybridRAGWorkflow(
            chroma_service=request.app.state.chroma,
            neo4j_service=request.app.state.neo4j,
        )

        # ── Trace + execute ──────────────────────────────────────────
        if tracer is not None:
            with tracer.trace_query(
                query=query_request.query,
                mode=query_request.mode.value,
                session_id=query_id,
            ) as trace_ctx:
                result = await workflow.execute_with_answer(
                    query=query_request.query,
                    mode=query_request.mode,
                    max_results=query_request.max_results,
                    filters=query_request.filters,
                    attachment_content=query_request.attachment_content,
                    attachment_name=query_request.attachment_name,
                )
                execution_time = round((time.perf_counter() - start_time) * 1000, 2)

                # Log per-step latency to tracer
                for step_name, lat_ms in result["latency"].items():
                    tracer.log_step(
                        run_id=trace_ctx["run_id"],
                        step_name=step_name,
                        latency_ms=lat_ms,
                    )
        else:
            result = await workflow.execute_with_answer(
                query=query_request.query,
                mode=query_request.mode,
                max_results=query_request.max_results,
                filters=query_request.filters,
                attachment_content=query_request.attachment_content,
                attachment_name=query_request.attachment_name,
            )
            execution_time = round((time.perf_counter() - start_time) * 1000, 2)

        chunks = result["chunks"]

        # ── Persist metrics ──────────────────────────────────────────
        if collector is not None:
            collector.record_query(
                query=query_request.query,
                mode=query_request.mode.value,
                latency_ms=execution_time,
                chunk_count=len(chunks),
                query_id=query_id,
            )
            collector.record_generation(
                query_id=query_id,
                answer=result["answer"],
                confidence=result["confidence"],
                tokens_used=0,  # token count wired separately via LangSmith callback
            )

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
        "tracer_active": getattr(
            getattr(request.app.state, "tracer", None), "is_active", False
        ),
        "metrics_db": "postgres"
        if getattr(
            getattr(request.app.state, "metrics_collector", None),
            "_use_postgres", False,
        )
        else "sqlite",
        "workflow_version": "v2 (intent + parallel + rerank + generate)",
    }
