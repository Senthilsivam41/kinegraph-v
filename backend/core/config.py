"""
Application Configuration
"""
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional


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
    
    # OpenAI
    OPENAI_API_KEY: str
    
    # ChromaDB
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION_NAME: str = "kinetic_vectors"
    
    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str
    
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
    FAITHFULNESS_CRITIC_MODEL: str = "gpt-4o-mini"
    FAITHFULNESS_CRITIC_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=0.2)

    # LangSmith (optional — leave blank to disable remote tracing)
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "kinegraph-vectra"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    # Metrics DB (optional — defaults to SQLite eval/metrics.db)
    DATABASE_URL: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
