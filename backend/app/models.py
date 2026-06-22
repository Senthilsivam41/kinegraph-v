"""
Pydantic Models for API Requests and Responses
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class QueryMode(str, Enum):
    """Query mode selection"""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"
    VECTORLESS = "vectorless"


class QueryRequest(BaseModel):
    """Query request model"""
    query: str = Field(..., description="Natural language query")
    mode: QueryMode = Field(default=QueryMode.HYBRID, description="Query mode")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional filters")
    attachment_content: Optional[str] = Field(default=None, description="Raw text of the document attachment")
    attachment_name: Optional[str] = Field(default=None, description="Name of the document attachment")


class DocumentChunk(BaseModel):
    """Document chunk result"""
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str  # 'vector' or 'graph'


class QueryResponse(BaseModel):
    """Query response model"""
    query: str
    mode: QueryMode
    results: List[DocumentChunk]
    total_results: int
    execution_time_ms: float
    # New observability fields (v2)
    generated_answer: Optional[str] = None
    answer_confidence: Optional[float] = None
    intent: Optional[str] = None
    latency_breakdown: Optional[Dict[str, float]] = None


class IngestRequest(BaseModel):
    """Document ingestion request"""
    file_name: str = Field(..., description="Name of the file")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")


class IngestResponse(BaseModel):
    """Ingestion response"""
    task_id: str
    status: str
    message: str


class TaskStatus(BaseModel):
    """Celery task status"""
    task_id: str
    status: str  # PENDING, STARTED, SUCCESS, FAILURE
    result: Optional[Any] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    services: Dict[str, bool]
    version: str
