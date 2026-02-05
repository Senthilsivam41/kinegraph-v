"""
Health Check Endpoints
"""
from fastapi import APIRouter, Request
from app.models import HealthResponse
from core.config import settings
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
    """Kubernetes readiness probe"""
    try:
        # Check critical services
        request.app.state.chroma.get_or_create_collection()
        request.app.state.neo4j.verify_connectivity()
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}, 503
