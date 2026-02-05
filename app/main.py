"""
FastAPI Application Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import query, ingest, health
from core.config import settings
from services.chroma_service import ChromaService
from services.neo4j_service import Neo4jService

# Initialize services at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    app.state.chroma = ChromaService()
    app.state.neo4j = Neo4jService()
    
    yield
    
    # Shutdown
    app.state.neo4j.close()


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Hybrid RAG System with Vector and Graph Databases",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingestion"])
app.include_router(query.router, prefix="/api/v1/query", tags=["Query"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }
