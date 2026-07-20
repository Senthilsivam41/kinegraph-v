"""
Pydantic Models for API Requests and Responses
"""
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any
from enum import Enum
from backend.core.config import settings
from backend.graph_retrieval.multi_hop import TraversalStrategy


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
    max_results: int = Field(
        default=settings.CONTEXT_TOP_K,
        ge=1,
        le=100,
        description="Maximum reranked contexts sent to the generator",
    )
    candidate_pool_size: int = Field(
        default=settings.RETRIEVAL_CANDIDATE_LIMIT,
        ge=5,
        le=100,
        description="Per-channel retrieval candidates before fusion and reranking",
    )
    max_hops: int = Field(
        default=settings.GRAPH_MAX_HOPS,
        ge=1,
        le=5,
        description="Maximum graph traversal depth",
    )
    traversal_strategy: TraversalStrategy = Field(default=TraversalStrategy.BFS, description="Graph traversal strategy")
    community_id: Optional[str] = Field(default=None, description="Optional graph community restriction")
    enable_conditional_recovery: bool = Field(default=True, description="Recover weak retrieval before RRF")
    enable_hyde_fallback: bool = Field(default=False, description="Opt-in constrained HyDE vector fallback")
    enable_grounding_critique: bool = Field(
        default=True,
        description="Verify structured citations and filter unsupported claims before returning the answer",
    )
    enable_lexical_fusion: bool = Field(default=False, description="Opt-in BM25 channel for hybrid fusion")
    vector_fusion_weight: float = Field(default=settings.FUSION_VECTOR_WEIGHT, ge=0, le=5)
    graph_fusion_weight: float = Field(default=settings.FUSION_GRAPH_WEIGHT, ge=0, le=5)
    lexical_fusion_weight: float = Field(default=settings.FUSION_LEXICAL_WEIGHT, ge=0, le=5)
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional filters")
    attachment_content: Optional[str] = Field(default=None, description="Raw text of the document attachment")
    attachment_name: Optional[str] = Field(default=None, description="Name of the document attachment")

    @model_validator(mode="after")
    def validate_active_fusion_weights(self):
        active_total = self.vector_fusion_weight + self.graph_fusion_weight
        if self.enable_lexical_fusion:
            active_total += self.lexical_fusion_weight
        if active_total <= 0:
            raise ValueError("at least one active fusion weight must be positive")
        return self


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
    recovery_triggered: bool = False
    recovery_details: Optional[Dict[str, Any]] = None
    fusion_details: Optional[Dict[str, Any]] = None
    grounded_claims: Optional[List[Dict[str, Any]]] = None
    citation_validation: Optional[Dict[str, Any]] = None
    grounding_critique: Optional[Dict[str, Any]] = None
    answer_relevancy: Optional[Dict[str, Any]] = None


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
