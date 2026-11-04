"""
Health Check Endpoints
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from backend.app.models import HealthResponse
from backend.core.config import settings
import httpx

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Health check endpoint that verifies all services are operational
    """
    services = {
        "api": True,
        "chroma": False,
        "neo4j": False,
        "redis": False
    }
    
    # Check ChromaDB
    try:
        chroma_service = request.app.state.chroma
        chroma_service.get_or_create_collection()
        services["chroma"] = True
    except Exception:
        pass
    
    # Check Neo4j
    try:
        neo4j_service = request.app.state.neo4j
        neo4j_service.verify_connectivity()
        services["neo4j"] = True
    except Exception:
        pass
    
    # Check Redis
    try:
        import redis
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB
        )
        r.ping()
        services["redis"] = True
    except Exception:
        pass
    
    overall_status = "healthy" if all(services.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        services=services,
        version=settings.APP_VERSION
    )


@router.get("/liveness")
async def liveness():
    """Kubernetes liveness probe"""
    return {"status": "alive"}


@router.get("/readiness")
async def readiness(request: Request):
    """Kubernetes readiness probe. Dependency failures are HTTP 503."""
    try:
        request.app.state.chroma.get_or_create_collection()
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": "chroma_unavailable"},
        )
    try:
        neo4j_ready = request.app.state.neo4j.verify_connectivity()
    except Exception:
        neo4j_ready = False
    if not neo4j_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": "neo4j_unavailable"},
        )
    return {"status": "ready"}
