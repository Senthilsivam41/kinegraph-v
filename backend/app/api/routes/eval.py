"""
FastAPI routes for evaluation observability:
  POST /api/v1/eval/feedback     — thumbs up/down on a run
  GET  /api/v1/eval/metrics      — dashboard stats
  GET  /api/v1/eval/slow-queries — slowest queries
  GET  /api/v1/eval/mode-perf    — mode comparison
"""
from fastapi import APIRouter, Request, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class FeedbackRequest(BaseModel):
    run_id: str
    score: float          # 1.0 = thumbs up, 0.0 = thumbs down
    comment: Optional[str] = None


@router.post("/feedback")
async def submit_feedback(body: FeedbackRequest, request: Request):
    """Submit user feedback (thumbs up/down) for a completed query run."""
    tracer = getattr(request.app.state, "tracer", None)
    if tracer is None:
        raise HTTPException(status_code=503, detail="Tracer not initialised")
    tracer.log_feedback(
        run_id=body.run_id,
        score=body.score,
        comment=body.comment,
    )
    return {"status": "ok", "run_id": body.run_id, "score": body.score}


@router.get("/metrics")
async def get_metrics(
    request: Request,
    hours: int = Query(default=24, ge=1, le=720, description="Look-back window in hours"),
):
    """Return aggregated RAG pipeline metrics for the dashboard."""
    collector = getattr(request.app.state, "metrics_collector", None)
    if collector is None:
        raise HTTPException(status_code=503, detail="Metrics collector not initialised")
    return collector.get_dashboard_stats(time_window_hours=hours)


@router.get("/slow-queries")
async def get_slow_queries(
    request: Request,
    threshold_ms: float = Query(default=2000.0, description="Latency threshold in ms"),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Return queries exceeding the latency threshold."""
    collector = getattr(request.app.state, "metrics_collector", None)
    if collector is None:
        raise HTTPException(status_code=503, detail="Metrics collector not initialised")
    return {"slow_queries": collector.get_slow_queries(threshold_ms=threshold_ms, limit=limit)}


@router.get("/mode-perf")
async def get_mode_performance(request: Request):
    """Compare hybrid vs vector vs graph query performance."""
    collector = getattr(request.app.state, "metrics_collector", None)
    if collector is None:
        raise HTTPException(status_code=503, detail="Metrics collector not initialised")
    return collector.get_mode_performance()


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request):
    """Return detailed metrics for a specific run ID."""
    tracer = getattr(request.app.state, "tracer", None)
    if tracer is None:
        raise HTTPException(status_code=503, detail="Tracer not initialised")
    data = tracer.get_run_metrics(run_id)
    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])
    return data
