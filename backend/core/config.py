"""
Application Configuration
"""
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
    
    # RRF Configuration
    RRF_K: int = 60
    MAX_RESULTS: int = 10

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
