"""
LangGraph Workflow for Hybrid RAG System — v2
Improvements over v1:
  ① Query intent classification  → better routing, higher answer_relevancy
  ② Query rewriting              → broader recall, higher context_recall
  ③ Parallel vector + graph      → lower response time (async gather)
  ④ Post-fusion reranker         → higher context_precision
  ⑤ Grounded generation node     → faithfulness-first system prompt
  ⑥ Deeper retrieval (2× fetch)  → higher context_recall before filtering
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, TypedDict

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from backend.app.models import DocumentChunk, QueryMode
from backend.graph_retrieval.multi_hop import TraversalStrategy
from backend.core.config import settings
from backend.core.context_ranker import ContextRanker
from backend.core.intent_classifier import classify_intent, rewrite_query_for_retrieval
from backend.core.rrf import deduplicate_results, reciprocal_rank_fusion
from backend.services.chroma_service import ChromaService
from backend.services.neo4j_service import Neo4jService
from backend.graph_retrieval.langgraph_node import LangGraphGraphRetrieverNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Faithfulness-first generation prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an AI research agent with the sole mandate to provide factually grounded, maximally relevant information. You must only use the provided context to answer the user's question; do not extrapolate, assume, or bring in outside knowledge. Your task is threefold: first, to extract all supporting evidence; second, to synthesize a coherent answer based strictly and solely on the provided context; and third, to critique your own synthesis against the provided context.

---
### 🎯 TASK 1: EVIDENCE EXTRACTION (CRITICAL STEP FOR FAITHFULNESS)
Review the provided Contextual Data. Do not attempt to answer yet. Instead, identify and list every claim from the context that directly addresses or contributes information towards the User Question. You must only use the provided context; do not extrapolate or assume anything.
*   Format each finding as a concise bullet point claim, followed immediately by the specific source chunk ID that supports it.
*   If a relationship (Graph-derived) is critical, note the path/nodes involved.
*   **Goal:** Prove existence and location of all answer components.

### 🎯 TASK 2: SYNTHESIS AND ANSWER GENERATION (GOAL: RELEVANCY)
Using *only* the claims identified in Task 1 (and thus solely supported by the provided context), synthesize a single, unified answer to the User Question. Do not bring in any outside or parametric knowledge, assumptions, or external facts.
*   The tone must be precise and objective.
*   Structure the answer logically (e.g., using headings or sequential steps) to match the complexity of the question.
*   If there are conflicting claims across the sources (V vs G), you MUST flag them in your answer, stating the conflict found in the context.

### 🎯 TASK 3: CRITICAL REVIEW (SELF-CORRECTION)
Critique your synthesized answer against the original context and question. Answer the following:
1.  **Faithfulness Check:** Are all claims in your answer directly supported by the extracted context? Only answer Yes if there is 100% direct context support without any extrapolation or outside knowledge.
2.  **Completeness Check:** Does the answer fully address all components of the original question? (Yes/No, and identify any missing facets).

---
### FINAL OUTPUT FORMAT
Provide only the synthesized answer from Task 2, followed by a concise confidence score based on your assessment in Task 3.
"""


_HUMAN_PROMPT = """## [INPUT CONTEXTUAL DATA]
{context}

## [USER QUESTION]
{question}
"""

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human",  _HUMAN_PROMPT),
])

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class WorkflowState(TypedDict):
    """Full state passed between LangGraph nodes."""
    query: str
    rewritten_query: str
    intent: str
    suggested_mode: str
    mode: QueryMode
    max_results: int
    max_hops: int
    traversal_strategy: TraversalStrategy
    community_id: Optional[str]
    filters: Optional[Dict[str, Any]]
    vector_results: List[Dict[str, Any]]
    graph_results: List[Dict[str, Any]]
    fused_results: List[Dict[str, Any]]
    reranked_results: List[Dict[str, Any]]
    generated_answer: str
    answer_confidence: float
    latency_breakdown: Dict[str, float]
    final_results: List[DocumentChunk]
    attachment_content: Optional[str]
    attachment_name: Optional[str]


# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------

def parse_research_synthesis_output(text: str) -> tuple[str, float]:
    parts = re.split(r"###?\s*FINAL\s*OUTPUT\s*FORMAT", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        final_section = parts[1].strip()
    else:
        parts = re.split(r"###?\s*TASK\s*2(?::|and|synthesis|answer|generation|\s)*", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            task2_section = parts[1].strip()
            task2_parts = re.split(r"###?\s*TASK\s*3", task2_section, flags=re.IGNORECASE)
            final_section = task2_parts[0].strip()
        else:
            final_section = text.strip()

    score_match = re.search(r"(?:confidence|score)\s*(?:score)?\s*[:\-\s]\s*(\d+(?:\.\d+)?)(?:\s*/\s*(\d+))?", final_section, re.IGNORECASE)
    
    confidence = 0.85
    clean_answer = final_section
    
    if score_match:
        val_str = score_match.group(1)
        max_str = score_match.group(2)
        try:
            val = float(val_str)
            if max_str:
                max_val = float(max_str)
                if max_val > 0:
                    confidence = round(val / max_val, 2)
            else:
                if 0 <= val <= 1:
                    confidence = val
                elif 0 <= val <= 10:
                    confidence = round(val / 10.0, 2)
                elif 0 <= val <= 5:
                    confidence = round(val / 5.0, 2)
        except ValueError:
            pass
        
        # Remove the confidence score line
        lines = clean_answer.split("\n")
        clean_lines = []
        for line in lines:
            if re.search(r"(?:confidence|score)\s*(?:score)?\s*[:\-\s]\s*\d+", line, re.IGNORECASE):
                continue
            clean_lines.append(line)
        clean_answer = "\n".join(clean_lines).strip()

    # Clean leading colons or headers
    clean_answer = re.sub(r"^(?::|and|synthesis|answer|generation|\s)*", "", clean_answer, flags=re.IGNORECASE).strip()
    clean_answer = re.sub(r"^(?:Provide only the synthesized answer from Task 2, followed by a concise confidence score based on your assessment in Task 3\.?|Synthesized Answer:)\s*", "", clean_answer, flags=re.IGNORECASE).strip()
    
    return clean_answer, confidence


# ---------------------------------------------------------------------------
# HybridRAGWorkflow
# ---------------------------------------------------------------------------

class HybridRAGWorkflow:
    """
    Improved LangGraph workflow with intent routing, parallel retrieval,
    reranking, and grounded LLM generation.
    """

    def __init__(
        self,
        chroma_service: ChromaService,
        neo4j_service: Neo4jService,
        use_cross_encoder: bool = False,
        generation_model: str = "gpt-4o-mini",
    ) -> None:
        self.chroma = chroma_service
        self.neo4j = neo4j_service
        self.graph_retriever_node = LangGraphGraphRetrieverNode(
            use_cypher=False, neo4j_driver=neo4j_service.driver
        )
        self.ranker = ContextRanker(
            use_cross_encoder=use_cross_encoder,
            min_relevance_threshold=0.03,
        )

        openai_key = settings.OPENAI_API_KEY
        kw = {
            "model": generation_model,
            "openai_api_key": openai_key,
            "temperature": 0,
        }
        if openai_key and (openai_key.startswith("sk-or-") or "openrouter" in openai_key):
            kw["base_url"] = "https://openrouter.ai/api/v1"
        self.llm = ChatOpenAI(**kw)

        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        """
        Build the improved LangGraph workflow.

        Flow (v2):
        START
          → intent_router          (classify + rewrite query)
          → [vector_agent ‖ graph_agent]   (parallel via asyncio.gather)
          → fusion_node            (RRF merge)
          → rerank_node            (cross-encoder / keyword reranker)
          → generate_node          (grounded LLM answer)
          → format_results
        END
        """
        workflow = StateGraph(WorkflowState)

        workflow.add_node("intent_router",   self._intent_router)
        workflow.add_node("vector_agent",    self._vector_agent)
        workflow.add_node("graph_agent",     self._graph_agent)
        workflow.add_node("parallel_fetch",  self._parallel_fetch)
        workflow.add_node("vectorless_agent", self._vectorless_agent)
        workflow.add_node("fusion_node",     self._fusion_node)
        workflow.add_node("rerank_node",     self._rerank_node)
        workflow.add_node("generate_node",   self._generate_node)
        workflow.add_node("format_results",  self._format_results)

        workflow.set_entry_point("intent_router")

        # Conditional routing after intent classification
        workflow.add_conditional_edges(
            "intent_router",
            self._route_decision,
            {
                "vector":  "vector_agent",
                "graph":   "graph_agent",
                "hybrid":  "parallel_fetch",   # ← parallel branch
                "vectorless": "vectorless_agent",
            },
        )

        # Single-mode paths
        workflow.add_edge("vector_agent", "fusion_node")
        workflow.add_edge("graph_agent",  "fusion_node")
        workflow.add_edge("vectorless_agent", "fusion_node")

        # Hybrid parallel path
        workflow.add_edge("parallel_fetch", "fusion_node")

        # Common tail
        workflow.add_edge("fusion_node",    "rerank_node")
        workflow.add_edge("rerank_node",    "generate_node")
        workflow.add_edge("generate_node",  "format_results")
        workflow.add_edge("format_results", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Node: intent_router
    # ------------------------------------------------------------------

    async def _intent_router(self, state: WorkflowState) -> WorkflowState:
        """Classify intent, rewrite query for retrieval, and set mode."""
        t0 = time.perf_counter()

        intent_result = classify_intent(state["query"])
        intent = intent_result["intent"]
        suggested_mode = intent_result["suggested_mode"]

        # Rewrite the query to expand recall
        rewritten = rewrite_query_for_retrieval(state["query"], intent)

        # Override mode only if caller chose HYBRID (let explicit vector/graph override stand)
        effective_mode = state["mode"]
        if effective_mode == QueryMode.HYBRID:
            if suggested_mode == "vector":
                effective_mode = QueryMode.VECTOR
            elif suggested_mode == "graph":
                effective_mode = QueryMode.GRAPH
            # else: stays hybrid

        # Check for auto-routing to Vectorless RAG
        attachment_content = state.get("attachment_content")
        filters = state.get("filters")
        query_lower = state["query"].lower()

        # Rule 1: Explicitly requested Vectorless
        # Rule 2: Query is a global/summarization query (vector chunking is bad and expensive)
        # Rule 3: Direct attachment content is small (<40k chars)
        is_summary_query = any(kw in query_lower for kw in [
            "summarize", "summary", "tldr", "overall theme", "recap", "synopsis", "outline", "explain the main"
        ])

        should_use_vectorless = False

        if effective_mode == QueryMode.VECTORLESS:
            should_use_vectorless = True
        elif attachment_content:
            if len(attachment_content) < 40000 or is_summary_query:
                should_use_vectorless = True
                logger.info("[IntentRouter] Auto-routing to VECTORLESS: Attachment content detected (len=%d, is_summary=%s)", len(attachment_content), is_summary_query)
        elif filters and "file_name" in filters:
            file_name = filters["file_name"]
            from backend.services.vectorless_service import VectorlessService
            vectorless = VectorlessService()
            doc_text = vectorless.get_local_document_text(file_name)
            if doc_text and (len(doc_text) < 40000 or is_summary_query):
                should_use_vectorless = True
                logger.info("[IntentRouter] Auto-routing to VECTORLESS: Local file '%s' is small/queried for summary", file_name)

        if should_use_vectorless:
            effective_mode = QueryMode.VECTORLESS

        logger.info(
            "[IntentRouter] query=%r intent=%s mode=%s→%s rewritten=%r",
            state["query"][:60], intent, state["mode"].value,
            effective_mode.value, rewritten[:60],
        )

        state["intent"] = intent
        state["suggested_mode"] = suggested_mode
        state["rewritten_query"] = rewritten
        state["mode"] = effective_mode
        state["latency_breakdown"] = {
            "intent_router_ms": round((time.perf_counter() - t0) * 1000, 2)
        }
        return state

    def _route_decision(self, state: WorkflowState) -> str:
        return state["mode"].value

    # ------------------------------------------------------------------
    # Node: parallel_fetch (hybrid mode — runs both agents concurrently)
    # ------------------------------------------------------------------

    async def _parallel_fetch(self, state: WorkflowState) -> WorkflowState:
        """Run vector and graph retrieval in parallel using asyncio.gather."""
        t0 = time.perf_counter()

        # Fetch 2× more chunks than needed — reranker will prune
        fetch_n = min(state["max_results"] * 2, 20)
        rq = state["rewritten_query"]

        vector_task = self.chroma.similarity_search(
            query=rq, n_results=fetch_n, filters=state.get("filters")
        )
        graph_task = self.graph_retriever_node.retrieve_chunks(
            query=rq,
            n_results=fetch_n,
            max_hops=state["max_hops"],
            traversal_strategy=state["traversal_strategy"],
            community_id=state.get("community_id"),
        )

        vector_results, graph_results = await asyncio.gather(
            vector_task, graph_task, return_exceptions=True
        )

        state["vector_results"] = vector_results if isinstance(vector_results, list) else []
        state["graph_results"]  = graph_results  if isinstance(graph_results,  list) else []
        state["latency_breakdown"]["parallel_fetch_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        logger.info(
            "[ParallelFetch] vector=%d graph=%d (%.0fms)",
            len(state["vector_results"]), len(state["graph_results"]),
            state["latency_breakdown"]["parallel_fetch_ms"],
        )
        return state

    # ------------------------------------------------------------------
    # Nodes: single-mode retrieval agents
    # ------------------------------------------------------------------

    async def _vector_agent(self, state: WorkflowState) -> WorkflowState:
        """Vector-only retrieval with expanded fetch for better recall."""
        t0 = time.perf_counter()
        fetch_n = min(state["max_results"] * 2, 20)
        results = await self.chroma.similarity_search(
            query=state["rewritten_query"],
            n_results=fetch_n,
            filters=state.get("filters"),
        )
        state["vector_results"] = results
        state["graph_results"] = []
        state["latency_breakdown"]["vector_agent_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        logger.info("[VectorAgent] %d results (%.0fms)", len(results),
                    state["latency_breakdown"]["vector_agent_ms"])
        return state

    async def _graph_agent(self, state: WorkflowState) -> WorkflowState:
        """Graph-only retrieval with expanded fetch for better recall."""
        t0 = time.perf_counter()
        fetch_n = min(state["max_results"] * 2, 20)
        results = await self.graph_retriever_node.retrieve_chunks(
            query=state["rewritten_query"],
            n_results=fetch_n,
            max_hops=state["max_hops"],
            traversal_strategy=state["traversal_strategy"],
            community_id=state.get("community_id"),
        )
        state["graph_results"] = results
        state["vector_results"] = []
        state["latency_breakdown"]["graph_agent_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        logger.info("[GraphAgent] %d results (%.0fms)", len(results),
                    state["latency_breakdown"]["graph_agent_ms"])
        return state

    async def _vectorless_agent(self, state: WorkflowState) -> WorkflowState:
        """Vectorless retrieval agent using local BM25 and attachment filtering."""
        t0 = time.perf_counter()
        
        query = state["rewritten_query"]
        attachment_content = state.get("attachment_content")
        attachment_name = state.get("attachment_name")
        filters = state.get("filters")
        max_results = state["max_results"]

        results = []
        
        try:
            from backend.services.vectorless_service import VectorlessService
            vectorless = VectorlessService()

            if attachment_content:
                # Retrieve from direct request attachment
                results = vectorless.search_attachment(
                    query=query,
                    attachment_content=attachment_content,
                    attachment_name=attachment_name,
                    max_results=max_results
                )
                logger.info("[VectorlessAgent] Extracted %d chunks from attachment", len(results))
            else:
                # Retrieve from local document chunk cache
                results = vectorless.search_chunks(
                    query=query,
                    top_k=max_results,
                    filters=filters
                )
                logger.info("[VectorlessAgent] Retrieved %d chunks from local chunks", len(results))
        except Exception as e:
            logger.error("[VectorlessAgent] Failed vectorless retrieval: %s", e)

        state["vector_results"] = results
        state["graph_results"] = []
        state["latency_breakdown"]["vectorless_agent_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        return state

    # ------------------------------------------------------------------
    # Node: fusion_node
    # ------------------------------------------------------------------

    async def _fusion_node(self, state: WorkflowState) -> WorkflowState:
        """Merge results using RRF; deduplicate."""
        t0 = time.perf_counter()
        mode = state["mode"]

        if mode in (QueryMode.VECTOR, QueryMode.VECTORLESS):
            fused = state["vector_results"]
        elif mode == QueryMode.GRAPH:
            fused = state["graph_results"]
        else:
            lists = [r for r in [state["vector_results"], state["graph_results"]] if r]
            fused = reciprocal_rank_fusion(lists) if lists else []

        fused = deduplicate_results(fused)
        state["fused_results"] = fused
        state["latency_breakdown"]["fusion_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        logger.info("[Fusion] %d unique chunks after RRF", len(fused))
        return state

    # ------------------------------------------------------------------
    # Node: rerank_node  ← NEW (fixes context_precision)
    # ------------------------------------------------------------------

    async def _rerank_node(self, state: WorkflowState) -> WorkflowState:
        """
        Rerank and filter fused chunks.
        Returns top-k most relevant chunks (capped at max_results).
        This directly improves context_precision.
        """
        t0 = time.perf_counter()
        reranked = self.ranker.rerank(
            query=state["query"],           # use original query for relevance check
            chunks=state["fused_results"],
            top_k=state["max_results"],
        )
        state["reranked_results"] = reranked
        state["latency_breakdown"]["rerank_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        logger.info(
            "[Rerank] %d → %d chunks after filtering",
            len(state["fused_results"]), len(reranked),
        )
        return state

    # ------------------------------------------------------------------
    # Node: generate_node  ← NEW (fixes faithfulness + answer_relevancy)
    # ------------------------------------------------------------------

    async def _generate_node(self, state: WorkflowState) -> WorkflowState:
        """
        Generate a grounded answer from the reranked context.
        The faithfulness-first system prompt forces the LLM to stay in-context.
        """
        t0 = time.perf_counter()
        chunks = state["reranked_results"]

        if not chunks:
            state["generated_answer"] = "No relevant context was retrieved for this query."
            state["answer_confidence"] = 0.0
            return state

        # Build context string — numbered for citation clarity
        context_parts = [
            f"[{i+1}] {c['content'].strip()}"
            for i, c in enumerate(chunks)
        ]
        context_str = "\n\n".join(context_parts)

        try:
            chain = GENERATION_PROMPT | self.llm
            response = await chain.ainvoke({
                "context":  context_str,
                "question": state["query"],
            })
            raw_answer = response.content.strip()
            answer, confidence = parse_research_synthesis_output(raw_answer)

            state["generated_answer"] = answer
            state["answer_confidence"] = confidence

        except Exception as exc:
            logger.error("[GenerateNode] LLM call failed: %s", exc)
            state["generated_answer"] = "Generation failed. Please retry."
            state["answer_confidence"] = 0.0

        state["latency_breakdown"]["generation_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        logger.info(
            "[Generate] answer=%d chars confidence=%.2f (%.0fms)",
            len(state["generated_answer"]),
            state["answer_confidence"],
            state["latency_breakdown"]["generation_ms"],
        )
        return state

    # ------------------------------------------------------------------
    # Node: format_results
    # ------------------------------------------------------------------

    async def _format_results(self, state: WorkflowState) -> WorkflowState:
        """Convert reranked chunks to DocumentChunk objects."""
        formatted = []
        for result in state["reranked_results"]:
            chunk = DocumentChunk(
                content=result.get("content", ""),
                metadata={
                    **result.get("metadata", {}),
                    "rerank_score": result.get("rerank_score", 0.0),
                    "intent": state.get("intent", ""),
                },
                score=result.get("score", 0.0),
                source=result.get("source", "unknown"),
            )
            formatted.append(chunk)
        state["final_results"] = formatted
        return state

    # ------------------------------------------------------------------
    # Public execute()
    # ------------------------------------------------------------------

    async def execute(
        self,
        query: str,
        mode: QueryMode = QueryMode.HYBRID,
        max_results: int = 10,
        max_hops: int = 3,
        traversal_strategy: TraversalStrategy = TraversalStrategy.BFS,
        community_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        attachment_content: Optional[str] = None,
        attachment_name: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Execute the full retrieval + generation workflow."""
        initial_state = WorkflowState(
            query=query,
            rewritten_query=query,
            intent="",
            suggested_mode="",
            mode=mode,
            max_results=max_results,
            max_hops=max_hops,
            traversal_strategy=traversal_strategy,
            community_id=community_id,
            filters=filters,
            vector_results=[],
            graph_results=[],
            fused_results=[],
            reranked_results=[],
            generated_answer="",
            answer_confidence=0.0,
            latency_breakdown={},
            final_results=[],
            attachment_content=attachment_content,
            attachment_name=attachment_name,
        )
        final_state = await self.graph.ainvoke(initial_state)
        return final_state["final_results"]

    async def execute_with_answer(
        self,
        query: str,
        mode: QueryMode = QueryMode.HYBRID,
        max_results: int = 10,
        max_hops: int = 3,
        traversal_strategy: TraversalStrategy = TraversalStrategy.BFS,
        community_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        attachment_content: Optional[str] = None,
        attachment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute and return both retrieved chunks AND the generated answer.

        Returns::

            {
                "answer":       str,
                "confidence":   float,
                "chunks":       List[DocumentChunk],
                "intent":       str,
                "latency":      Dict[str, float],
            }
        """
        initial_state = WorkflowState(
            query=query,
            rewritten_query=query,
            intent="",
            suggested_mode="",
            mode=mode,
            max_results=max_results,
            max_hops=max_hops,
            traversal_strategy=traversal_strategy,
            community_id=community_id,
            filters=filters,
            vector_results=[],
            graph_results=[],
            fused_results=[],
            reranked_results=[],
            generated_answer="",
            answer_confidence=0.0,
            latency_breakdown={},
            final_results=[],
            attachment_content=attachment_content,
            attachment_name=attachment_name,
        )
        final_state = await self.graph.ainvoke(initial_state)
        return {
            "answer":     final_state["generated_answer"],
            "confidence": final_state["answer_confidence"],
            "chunks":     final_state["final_results"],
            "intent":     final_state["intent"],
            "latency":    final_state["latency_breakdown"],
        }
