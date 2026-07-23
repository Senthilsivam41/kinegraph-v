"""
FastAPI Application Entry Point — KineticGraph-Vectra
Includes observability layer: LangSmithTracer + MetricsCollector
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.app.api.routes import query, ingest, health
from backend.core.config import settings
from backend.services.chroma_service import ChromaService
from backend.services.neo4j_service import Neo4jService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Core services
    app.state.chroma = ChromaService()
    app.state.neo4j = Neo4jService()

    yield

    # Shutdown
    try:
        app.state.neo4j.close()
    finally:
        app.state.chroma.close()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Hybrid RAG System with Vector + Graph Databases and full observability",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Routers
app.include_router(health.router,          prefix="/health",              tags=["Health"])
app.include_router(ingest.router,          prefix="/api/v1/ingest",       tags=["Ingestion"])
app.include_router(query.router,           prefix="/api/v1/query",        tags=["Query"])


@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }
