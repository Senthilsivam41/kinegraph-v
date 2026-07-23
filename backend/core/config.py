"""
Application Configuration
"""
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "KineticGraph-Vectra"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ALLOWED_ORIGINS: str = "http://localhost:8080,http://127.0.0.1:8080"
    CORS_ALLOW_CREDENTIALS: bool = False
    UPLOAD_DIR: Path = Path("data/uploads")
    
    # OpenAI
    OPENAI_API_KEY: str
    # OpenAI-compatible model routing. The default is served through OpenRouter
    # when OPENAI_API_KEY is an OpenRouter key (sk-or-...).
    LLM_MODEL: str = "qwen/qwen3.6-27b"
    
    # ChromaDB
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION_NAME: str = "kinetic_vectors"
    
    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = Field(min_length=16)
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # LiteParse (local layout-aware document parsing service)
    PARSER_URL: str = "http://localhost:5707"
    PARSER_TIMEOUT_SECONDS: int = 120

    # RRF Configuration
    RRF_K: int = 60
    MAX_RESULTS: int = 10

    # Context precision controls
    CONTEXT_TOP_K: int = 6
    RETRIEVAL_CANDIDATE_LIMIT: int = 25
    RETRIEVAL_DEDUP_THRESHOLD: float = 0.95
    GRAPH_MAX_HOPS: int = 2
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_MIN_RELEVANCE: float = 0.20
    FUSION_VECTOR_WEIGHT: float = 1.0
    FUSION_GRAPH_WEIGHT: float = 1.0
    FUSION_LEXICAL_WEIGHT: float = 0.7
    CONSERVATIVE_ROUTING_ENABLED: bool = False

    # Faithfulness controls
    GENERATION_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=0.2)
    FAITHFULNESS_CRITIC_MODEL: str = "qwen/qwen3.6-27b"
    FAITHFULNESS_CRITIC_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=0.2)

    # LangSmith (optional — leave blank to disable remote tracing)
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "kinegraph-vectra"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    # Metrics DB (optional — defaults to SQLite eval/metrics.db)
    DATABASE_URL: Optional[str] = None

    @property
    def cors_allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]

    @model_validator(mode="after")
    def validate_security_configuration(self):
        if not self.cors_allowed_origins:
            raise ValueError("CORS_ALLOWED_ORIGINS must contain at least one origin")
        if "*" in self.cors_allowed_origins and self.CORS_ALLOW_CREDENTIALS:
            raise ValueError("credentialed CORS cannot use a wildcard origin")
        return self

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build settings on first use so imports remain side-effect free."""
    return Settings()


class _LazySettings:
    """Backward-compatible lazy proxy for modules that import ``settings``."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_settings(), name)

    def __repr__(self) -> str:
        return repr(get_settings())


settings = _LazySettings()
