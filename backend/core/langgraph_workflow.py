"""
LangGraph Workflow for Hybrid RAG System — v2
Improvements over v1:
  ① Query intent classification  → better routing, higher answer_relevancy
  ② Query rewriting              → broader recall, higher context_recall
  ③ Parallel vector + graph      → lower response time (async gather)
  ④ Post-fusion reranker         → higher context_precision
  ⑤ Grounded generation node     → faithfulness-first system prompt
  ⑥ Deeper retrieval (2× fetch)  → higher context_recall before filtering
  ⑦ Conditional query recovery   → decomposition/vocabulary, opt-in HyDE before RRF
"""
from __future__ import annotations

import asyncio
import hashlib
import json
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
from backend.core.grounding import (
    apply_critic_response,
    assign_citation_ids,
    build_citation_context,
    format_grounded_claims,
    validate_grounded_response,
)
from backend.core.intent_classifier import classify_intent, rewrite_query_for_retrieval
from backend.core.rrf import deduplicate_results, reciprocal_rank_fusion
from backend.core.query_recovery import QueryRecoveryEngine
from backend.services.chroma_service import ChromaService
from backend.services.neo4j_service import Neo4jService
from backend.services.vectorless_service import VectorlessService
from backend.graph_retrieval.langgraph_node import LangGraphGraphRetrieverNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Faithfulness-first generation prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are Kinegraph's grounded synthesis node. Use only the
provided context. Write atomic claims that directly answer the question, and attach
one or more exact context chunk IDs to every claim. Never invent, shorten, translate,
or renumber a chunk ID. Omit claims that the context does not support.

Answer the literal question asked. Do not summarize the retrieved context, add
background, or include merely related facts. For a narrow question, return only the
narrow answer. For a compound question, include only claims that resolve an explicit
question component.

Return JSON only, with exactly this shape:
{{"claims":[{{"text":"one atomic supported claim","chunk_ids":["exact-chunk-id"]}}],"confidence":0.0}}

Confidence must be between 0 and 1 and reflect only support in the supplied context.
If no claim is supported, return an empty claims array and confidence 0.
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

_CRITIC_SYSTEM_PROMPT = """You are Kinegraph's grounding and answer-relevance critic.
Evaluate each existing claim against two independent objective criteria:
1. Grounding: every material statement is directly supported by its cited chunks.
2. Direct relevance: the claim answers the literal user question or one explicit
   component of a compound question. Related background and context summaries are
   not directly relevant.

You may filter existing claims, but must never rewrite a claim, add a claim, or add a
citation. Judge relevance independently of length, detail, tone, or writing quality.

Return JSON only:
{{"supported_claim_ids":["claim-1"],"directly_relevant_claim_ids":["claim-1"],"unsupported_reasons":{{"claim-2":"concise reason"}},"irrelevant_reasons":{{"claim-3":"concise reason"}},"question_coverage":"complete","missing_question_facets":[]}}
"""

_CRITIC_HUMAN_PROMPT = """## LITERAL USER QUESTION
{question}

Each claim below includes only the chunks it cited.
Do not use one claim's evidence to support another claim.

## CLAIMS WITH CITED CONTEXT
{claims_json}
"""

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _CRITIC_SYSTEM_PROMPT),
    ("human", _CRITIC_HUMAN_PROMPT),
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
    requested_mode: QueryMode
    mode: QueryMode
    allow_mode_downgrade: bool
    enable_conservative_routing: bool
    allow_vectorless_auto_route: bool
    routing_details: Dict[str, Any]
    max_results: int
    candidate_pool_size: int
    max_hops: int
    traversal_strategy: TraversalStrategy
    community_id: Optional[str]
    enable_conditional_recovery: bool
    enable_hyde_fallback: bool
    filters: Optional[Dict[str, Any]]
    vector_results: List[Dict[str, Any]]
    graph_results: List[Dict[str, Any]]
    lexical_results: List[Dict[str, Any]]
    initial_candidates: Dict[str, List[Dict[str, Any]]]
    retrieval_failures: Dict[str, str]
    graph_retrieval_diagnostics: Dict[str, Any]
    enable_lexical_fusion: bool
    vector_fusion_weight: float
    graph_fusion_weight: float
    lexical_fusion_weight: float
    recovery_triggered: bool
    recovery_details: Dict[str, Any]
    fused_results: List[Dict[str, Any]]
    deduplication_details: Dict[str, Any]
    reranked_results: List[Dict[str, Any]]
    reranker_details: Dict[str, Any]
    enable_grounding_critique: bool
    grounded_claims: List[Dict[str, Any]]
    citation_context: Dict[str, str]
    citation_validation: Dict[str, Any]
    grounding_critique: Dict[str, Any]
    answer_relevancy: Dict[str, Any]
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


def _candidate_identity(result: Dict[str, Any]) -> str:
    """Return a stable candidate identifier for trace comparisons."""
    metadata = result.get("metadata") or {}
    for key in ("chunk_id", "vector_record_id", "node_id", "id", "citation_id"):
        value = result.get(key, metadata.get(key))
        if value not in (None, ""):
            return str(value)
    digest = hashlib.sha256(result.get("content", "").encode("utf-8")).hexdigest()
    return f"content-sha256:{digest}"


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
        use_cross_encoder: bool = True,
        generation_model: Optional[str] = None,
        critic_model: Optional[str] = None,
    ) -> None:
        self.chroma = chroma_service
        self.neo4j = neo4j_service
        self.graph_retriever_node = LangGraphGraphRetrieverNode(
            use_cypher=False, neo4j_driver=neo4j_service.driver
        )
        self.ranker = ContextRanker(
            use_cross_encoder=use_cross_encoder,
            model_name=settings.RERANKER_MODEL,
            min_relevance_threshold=settings.RERANKER_MIN_RELEVANCE,
        )

        generation_model = generation_model or settings.LLM_MODEL
        critic_model = critic_model or settings.FAITHFULNESS_CRITIC_MODEL
        openai_key = settings.OPENAI_API_KEY
        kw = {
            "model": generation_model,
            "openai_api_key": openai_key,
            "temperature": settings.GENERATION_TEMPERATURE,
        }
        if openai_key and (openai_key.startswith("sk-or-") or "openrouter" in openai_key):
            kw["base_url"] = "https://openrouter.ai/api/v1"
        self.llm = ChatOpenAI(**kw)
        critic_kw = {
            **kw,
            "model": critic_model,
            "temperature": settings.FAITHFULNESS_CRITIC_TEMPERATURE,
        }
        self.critic_llm = ChatOpenAI(**critic_kw)
        self.recovery = QueryRecoveryEngine(self.llm)

        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        """
        Build the improved LangGraph workflow.

        Flow (v3):
        START
          → intent_router          (classify + rewrite query)
          → [vector_agent ‖ graph_agent]   (parallel via asyncio.gather)
          → query_recovery         (only when initial retrieval is weak)
          → fusion_node            (RRF merge)
          → rerank_node            (cross-encoder / keyword reranker)
          → generate_node          (grounded LLM answer)
          → grounding_critique     (remove unsupported claims; cannot rewrite)
          → format_results
        END
        """
        workflow = StateGraph(WorkflowState)

        workflow.add_node("intent_router",   self._intent_router)
        workflow.add_node("vector_agent",    self._vector_agent)
        workflow.add_node("graph_agent",     self._graph_agent)
        workflow.add_node("parallel_fetch",  self._parallel_fetch)
        workflow.add_node("vectorless_agent", self._vectorless_agent)
        workflow.add_node("query_recovery", self._query_recovery)
        workflow.add_node("fusion_node",     self._fusion_node)
        workflow.add_node("rerank_node",     self._rerank_node)
        workflow.add_node("generate_node",   self._generate_node)
        workflow.add_node("grounding_critique", self._grounding_critique)
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
        workflow.add_edge("vector_agent", "query_recovery")
        workflow.add_edge("graph_agent",  "query_recovery")
        workflow.add_edge("vectorless_agent", "query_recovery")

        # Hybrid parallel path
        workflow.add_edge("parallel_fetch", "query_recovery")
        workflow.add_edge("query_recovery", "fusion_node")

        # Common tail
        workflow.add_edge("fusion_node",    "rerank_node")
        workflow.add_edge("rerank_node",    "generate_node")
        workflow.add_edge("generate_node",  "grounding_critique")
        workflow.add_edge("grounding_critique", "format_results")
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

        requested_mode = state.get("requested_mode", state["mode"])
        effective_mode = requested_mode
        downgrade_reason = "explicit caller mode preserved"
        if requested_mode == QueryMode.HYBRID:
            conservative = state.get("enable_conservative_routing", False)
            can_downgrade = state.get("allow_mode_downgrade", True) and (
                not conservative
                or (
                    intent_result.get("confidence") == "high"
                    and not intent_result.get("coverage_sensitive", False)
                )
            )
            if can_downgrade and suggested_mode in {"vector", "graph"}:
                effective_mode = QueryMode(suggested_mode)
                downgrade_reason = (
                    "high-confidence single-facet classifier downgrade"
                    if conservative
                    else "legacy classifier downgrade"
                )
            elif not state.get("allow_mode_downgrade", True):
                downgrade_reason = "benchmark profile requires requested mode"
            else:
                downgrade_reason = "conservative routing retained hybrid for uncertain or coverage-sensitive query"

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

        if requested_mode == QueryMode.VECTORLESS:
            should_use_vectorless = True
        elif (
            requested_mode == QueryMode.HYBRID
            and state.get("allow_vectorless_auto_route", True)
            and attachment_content
        ):
            if len(attachment_content) < 40000 or is_summary_query:
                should_use_vectorless = True
                logger.info("[IntentRouter] Auto-routing to VECTORLESS: Attachment content detected (len=%d, is_summary=%s)", len(attachment_content), is_summary_query)
        elif (
            requested_mode == QueryMode.HYBRID
            and state.get("allow_vectorless_auto_route", True)
            and filters
            and "file_name" in filters
        ):
            file_name = filters["file_name"]
            from backend.services.vectorless_service import VectorlessService
            vectorless = VectorlessService()
            doc_text = vectorless.get_local_document_text(file_name)
            if doc_text and (len(doc_text) < 40000 or is_summary_query):
                should_use_vectorless = True
                logger.info("[IntentRouter] Auto-routing to VECTORLESS: Local file '%s' is small/queried for summary", file_name)

        if should_use_vectorless:
            effective_mode = QueryMode.VECTORLESS
            downgrade_reason = (
                "explicit vectorless mode"
                if requested_mode == QueryMode.VECTORLESS
                else "eligible attachment or local-file vectorless route"
            )

        logger.info(
            "[IntentRouter] query=%r intent=%s mode=%s→%s rewritten=%r",
            state["query"][:60], intent, state["mode"].value,
            effective_mode.value, rewritten[:60],
        )

        state["intent"] = intent
        state["suggested_mode"] = suggested_mode
        state["rewritten_query"] = rewritten
        state["mode"] = effective_mode
        state["routing_details"] = {
            **intent_result,
            "requested_mode": requested_mode.value,
            "effective_mode": effective_mode.value,
            "allow_mode_downgrade": state.get("allow_mode_downgrade", True),
            "enable_conservative_routing": state.get("enable_conservative_routing", False),
            "allow_vectorless_auto_route": state.get("allow_vectorless_auto_route", True),
            "decision": downgrade_reason,
        }
        state["latency_breakdown"] = {
            "intent_router_ms": round((time.perf_counter() - t0) * 1000, 2)
        }
        return state

    def _route_decision(self, state: WorkflowState) -> str:
        return state["mode"].value

    async def _retrieve_graph(
        self,
        *,
        query: str,
        n_results: int,
        max_hops: int,
        traversal_strategy: TraversalStrategy,
        community_id: Optional[str],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Use the diagnostic graph contract when supported, preserving test adapters."""
        diagnostic_method = getattr(
            type(self.graph_retriever_node), "retrieve_chunks_with_diagnostics", None
        )
        if diagnostic_method is not None:
            return await self.graph_retriever_node.retrieve_chunks_with_diagnostics(
                query=query,
                n_results=n_results,
                max_hops=max_hops,
                traversal_strategy=traversal_strategy,
                community_id=community_id,
            )
        results = await self.graph_retriever_node.retrieve_chunks(
            query=query,
            n_results=n_results,
            max_hops=max_hops,
            traversal_strategy=traversal_strategy,
            community_id=community_id,
        )
        return results, {}

    # ------------------------------------------------------------------
    # Node: parallel_fetch (hybrid mode — runs both agents concurrently)
    # ------------------------------------------------------------------

    async def _parallel_fetch(self, state: WorkflowState) -> WorkflowState:
        """Run vector and graph retrieval in parallel using asyncio.gather."""
        t0 = time.perf_counter()

        # Retrieve wide, then let the post-fusion reranker cut generator context.
        fetch_n = max(state["max_results"], state["candidate_pool_size"])
        rq = state["rewritten_query"]

        async def timed(awaitable):
            started = time.perf_counter()
            try:
                value = await awaitable
            except Exception as exc:  # surfaced below with channel attribution
                value = exc
            return value, round((time.perf_counter() - started) * 1000, 2)

        vector_task = timed(self.chroma.similarity_search(
            query=rq, n_results=fetch_n, filters=state.get("filters")
        ))
        graph_task = timed(self._retrieve_graph(
            query=rq,
            n_results=fetch_n,
            max_hops=state["max_hops"],
            traversal_strategy=state["traversal_strategy"],
            community_id=state.get("community_id"),
        ))

        tasks = [vector_task, graph_task]
        if state.get("enable_lexical_fusion", False):
            tasks.append(timed(asyncio.to_thread(
                VectorlessService().search_chunks,
                query=rq,
                top_k=fetch_n,
                filters=state.get("filters"),
            )))

        timed_results = await asyncio.gather(*tasks)
        retrieved = [value for value, _ in timed_results]
        vector_results = retrieved[0]
        graph_payload = retrieved[1]
        lexical_results = retrieved[2] if len(retrieved) > 2 else []
        graph_results, graph_diagnostics = (
            graph_payload if isinstance(graph_payload, tuple) else (graph_payload, {})
        )
        for channel, (_, latency_ms) in zip(("vector", "graph", "lexical"), timed_results):
            state["latency_breakdown"][f"{channel}_retrieval_ms"] = latency_ms

        failures = state.setdefault("retrieval_failures", {})
        for channel, value in zip(("vector", "graph", "lexical"), retrieved):
            if isinstance(value, Exception):
                failures[channel] = f"{type(value).__name__}: {value}"[:500]

        state["vector_results"] = vector_results if isinstance(vector_results, list) else []
        state["graph_results"]  = graph_results  if isinstance(graph_results,  list) else []
        state["lexical_results"] = lexical_results if isinstance(lexical_results, list) else []
        state["graph_retrieval_diagnostics"] = graph_diagnostics
        state["initial_candidates"] = {
            "vector": list(state["vector_results"]),
            "graph": list(state["graph_results"]),
            "lexical": list(state["lexical_results"]),
        }
        state["latency_breakdown"]["parallel_fetch_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        logger.info(
            "[ParallelFetch] vector=%d graph=%d lexical=%d (%.0fms)",
            len(state["vector_results"]), len(state["graph_results"]), len(state["lexical_results"]),
            state["latency_breakdown"]["parallel_fetch_ms"],
        )
        return state

    # ------------------------------------------------------------------
    # Nodes: single-mode retrieval agents
    # ------------------------------------------------------------------

    async def _vector_agent(self, state: WorkflowState) -> WorkflowState:
        """Vector-only retrieval with expanded fetch for better recall."""
        t0 = time.perf_counter()
        fetch_n = max(state["max_results"], state["candidate_pool_size"])
        try:
            results = await self.chroma.similarity_search(
                query=state["rewritten_query"],
                n_results=fetch_n,
                filters=state.get("filters"),
            )
        except Exception as exc:
            state.setdefault("retrieval_failures", {})["vector"] = f"{type(exc).__name__}: {exc}"[:500]
            results = []
        state["vector_results"] = results
        state["graph_results"] = []
        state["lexical_results"] = []
        state["initial_candidates"] = {"vector": list(results), "graph": [], "lexical": []}
        state["latency_breakdown"]["vector_agent_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        logger.info("[VectorAgent] %d results (%.0fms)", len(results),
                    state["latency_breakdown"]["vector_agent_ms"])
        return state

    async def _graph_agent(self, state: WorkflowState) -> WorkflowState:
        """Graph-only retrieval with expanded fetch for better recall."""
        t0 = time.perf_counter()
        fetch_n = max(state["max_results"], state["candidate_pool_size"])
        try:
            results, diagnostics = await self._retrieve_graph(
                query=state["rewritten_query"],
                n_results=fetch_n,
                max_hops=state["max_hops"],
                traversal_strategy=state["traversal_strategy"],
                community_id=state.get("community_id"),
            )
        except Exception as exc:
            state.setdefault("retrieval_failures", {})["graph"] = f"{type(exc).__name__}: {exc}"[:500]
            results = []
            diagnostics = {"retrieval_failure": f"{type(exc).__name__}: {exc}"[:500]}
        state["graph_results"] = results
        state["vector_results"] = []
        state["lexical_results"] = []
        state["initial_candidates"] = {"vector": [], "graph": list(results), "lexical": []}
        state["graph_retrieval_diagnostics"] = diagnostics
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
            state.setdefault("retrieval_failures", {})["vectorless"] = f"{type(e).__name__}: {e}"[:500]

        state["vector_results"] = results
        state["graph_results"] = []
        state["lexical_results"] = []
        state["initial_candidates"] = {"vector": [], "graph": [], "lexical": list(results)}
        state["latency_breakdown"]["vectorless_agent_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        return state

    # ------------------------------------------------------------------
    # Node: query_recovery (conditional, immediately before RRF)
    # ------------------------------------------------------------------

    async def _query_recovery(self, state: WorkflowState) -> WorkflowState:
        """Recover weak retrieval through decomposition, vocabulary, then optional HyDE."""
        t0 = time.perf_counter()
        mode = state["mode"]
        require_graph = mode in (QueryMode.GRAPH, QueryMode.HYBRID)
        initial = self.recovery.assess(
            state["vector_results"],
            state["graph_results"],
            require_graph=require_graph,
            require_source_diversity=mode == QueryMode.HYBRID,
        )
        routing = state.get("routing_details", {})
        facet_coverage = self.recovery.assess_facet_coverage(
            routing.get("facets", [state["query"]]),
            [*state["vector_results"], *state["graph_results"], *state.get("lexical_results", [])],
        )
        if routing.get("coverage_sensitive") and not facet_coverage.complete:
            initial.weak = True
            if "incomplete_question_facet_coverage" not in initial.reasons:
                initial.reasons.append("incomplete_question_facet_coverage")
        details: Dict[str, Any] = {
            "initial_assessment": initial.to_dict(),
            "initial_facet_coverage": facet_coverage.to_dict(),
            "structured_recovery_used": False,
            "hyde_used": False,
            "subqueries": [],
            "vocabulary": [],
            "generated_hypothesis": None,
        }
        state["recovery_triggered"] = False

        if (
            not state["enable_conditional_recovery"]
            or mode == QueryMode.VECTORLESS
            or not initial.weak
        ):
            details["final_assessment"] = initial.to_dict()
            details["final_facet_coverage"] = facet_coverage.to_dict()
            state["recovery_details"] = details
            state["latency_breakdown"]["query_recovery_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return state

        state["recovery_triggered"] = True
        plan = await self.recovery.create_plan(state["query"], state["intent"])
        details["subqueries"] = plan.subqueries
        details["vocabulary"] = plan.vocabulary
        details["structured_recovery_used"] = bool(plan.subqueries or plan.vocabulary)
        fetch_n = max(state["max_results"], state["candidate_pool_size"])
        vector_lists = [state["vector_results"]] if state["vector_results"] else []
        graph_lists = [state["graph_results"]] if state["graph_results"] else []

        async def execute_subquery(subquery: str):
            vector_task = None
            graph_task = None
            if mode in (QueryMode.VECTOR, QueryMode.HYBRID):
                vector_task = self.chroma.similarity_search(
                    query=subquery, n_results=fetch_n, filters=state.get("filters")
                )
            if mode in (QueryMode.GRAPH, QueryMode.HYBRID):
                graph_task = self.graph_retriever_node.retrieve_chunks(
                    query=subquery,
                    n_results=fetch_n,
                    max_hops=state["max_hops"],
                    traversal_strategy=state["traversal_strategy"],
                    community_id=state.get("community_id"),
                )
            tasks = [task for task in (vector_task, graph_task) if task is not None]
            values = await asyncio.gather(*tasks, return_exceptions=True)
            vector_value, graph_value = [], []
            index = 0
            if vector_task is not None:
                value = values[index]
                if isinstance(value, Exception):
                    state.setdefault("retrieval_failures", {})["recovery_vector"] = (
                        f"{type(value).__name__}: {value}"[:500]
                    )
                vector_value = value if isinstance(value, list) else []
                index += 1
            if graph_task is not None:
                value = values[index]
                if isinstance(value, Exception):
                    state.setdefault("retrieval_failures", {})["recovery_graph"] = (
                        f"{type(value).__name__}: {value}"[:500]
                    )
                graph_value = value if isinstance(value, list) else []
            return vector_value, graph_value

        if plan.subqueries:
            recovered = await asyncio.gather(*(execute_subquery(query) for query in plan.subqueries))
            for subquery, (vector_value, graph_value) in zip(plan.subqueries, recovered):
                if vector_value:
                    vector_lists.append(self.recovery.annotate_results(
                        vector_value, state["query"], subquery, "decomposition"
                    ))
                if graph_value:
                    graph_lists.append(self.recovery.annotate_results(
                        graph_value, state["query"], subquery, "decomposition"
                    ))

        if plan.vocabulary and mode in (QueryMode.VECTOR, QueryMode.HYBRID):
            vocabulary_query = f"{state['query']} {' '.join(plan.vocabulary)}"
            vocabulary_results = await self.chroma.similarity_search(
                query=vocabulary_query, n_results=fetch_n, filters=state.get("filters")
            )
            if vocabulary_results:
                vector_lists.append(self.recovery.annotate_results(
                    vocabulary_results, state["query"], vocabulary_query, "vocabulary"
                ))

        if vector_lists:
            state["vector_results"] = (
                reciprocal_rank_fusion(vector_lists) if len(vector_lists) > 1 else vector_lists[0]
            )
        if graph_lists:
            state["graph_results"] = (
                reciprocal_rank_fusion(graph_lists) if len(graph_lists) > 1 else graph_lists[0]
            )

        after_structured = self.recovery.assess(
            state["vector_results"],
            state["graph_results"],
            require_graph=require_graph,
            require_source_diversity=mode == QueryMode.HYBRID,
        )
        structured_facet_coverage = self.recovery.assess_facet_coverage(
            routing.get("facets", [state["query"]]),
            [*state["vector_results"], *state["graph_results"], *state.get("lexical_results", [])],
        )
        if routing.get("coverage_sensitive") and not structured_facet_coverage.complete:
            after_structured.weak = True
            if "incomplete_question_facet_coverage" not in after_structured.reasons:
                after_structured.reasons.append("incomplete_question_facet_coverage")
        details["structured_assessment"] = after_structured.to_dict()
        details["structured_facet_coverage"] = structured_facet_coverage.to_dict()

        if (
            after_structured.weak
            and state["enable_hyde_fallback"]
            and mode in (QueryMode.VECTOR, QueryMode.HYBRID)
        ):
            hypothesis = await self.recovery.generate_hypothesis(state["query"])
            if hypothesis:
                hyde_results = await self.chroma.similarity_search(
                    query=hypothesis, n_results=fetch_n, filters=state.get("filters")
                )
                if hyde_results:
                    annotated_hyde = self.recovery.annotate_results(
                        hyde_results, state["query"], hypothesis, "hyde"
                    )
                    state["vector_results"] = reciprocal_rank_fusion([
                        state["vector_results"], annotated_hyde
                    ]) if state["vector_results"] else annotated_hyde
                    details["hyde_used"] = True
                    details["generated_hypothesis"] = hypothesis

        final_assessment = self.recovery.assess(
            state["vector_results"],
            state["graph_results"],
            require_graph=require_graph,
            require_source_diversity=mode == QueryMode.HYBRID,
        )
        final_facet_coverage = self.recovery.assess_facet_coverage(
            routing.get("facets", [state["query"]]),
            [*state["vector_results"], *state["graph_results"], *state.get("lexical_results", [])],
        )
        if routing.get("coverage_sensitive") and not final_facet_coverage.complete:
            final_assessment.weak = True
            if "incomplete_question_facet_coverage" not in final_assessment.reasons:
                final_assessment.reasons.append("incomplete_question_facet_coverage")
        details["final_assessment"] = final_assessment.to_dict()
        details["final_facet_coverage"] = final_facet_coverage.to_dict()
        state["recovery_details"] = details
        state["latency_breakdown"]["query_recovery_ms"] = round((time.perf_counter() - t0) * 1000, 2)
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
            channels = [
                ("vector", state["vector_results"], state.get("vector_fusion_weight", 1.0)),
                ("graph", state["graph_results"], state.get("graph_fusion_weight", 1.0)),
                ("lexical", state.get("lexical_results", []), state.get("lexical_fusion_weight", 0.0)),
            ]
            active = [(name, results, weight) for name, results, weight in channels if results and weight > 0]
            fused = reciprocal_rank_fusion(
                [results for _, results, _ in active],
                weights=[weight for _, _, weight in active],
                source_names=[name for name, _, _ in active],
            ) if active else []

        before_deduplication = list(fused)
        fused = deduplicate_results(
            before_deduplication,
            similarity_threshold=settings.RETRIEVAL_DEDUP_THRESHOLD,
        )
        state["fused_results"] = fused
        retained_ids = {_candidate_identity(result) for result in fused}
        state["deduplication_details"] = {
            "input_count": len(before_deduplication),
            "output_count": len(fused),
            "removed_candidate_ids": [
                _candidate_identity(result)
                for result in before_deduplication
                if _candidate_identity(result) not in retained_ids
            ],
        }
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
            preferred_community_id=state.get("community_id"),
        )
        state["reranked_results"] = reranked
        retained_ids = {_candidate_identity(result) for result in reranked}
        state["reranker_details"] = {
            "requested_mode": "cross_encoder" if self.ranker.requested_cross_encoder else "keyword",
            "mode": "cross_encoder" if self.ranker.use_cross_encoder else "keyword",
            "model": self.ranker.model_name,
            "fallback": bool(self.ranker.fallback_reason),
            "fallback_reason": self.ranker.fallback_reason,
            "minimum_relevance": self.ranker.min_relevance_threshold,
            "input_count": len(state["fused_results"]),
            "output_count": len(reranked),
            "removed_candidate_ids": [
                _candidate_identity(result)
                for result in state["fused_results"]
                if _candidate_identity(result) not in retained_ids
            ],
        }
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
        """Generate structured claims and reject unverifiable citations."""
        t0 = time.perf_counter()
        chunks = state["reranked_results"]

        if not chunks:
            state["generated_answer"] = "No relevant context was retrieved for this query."
            state["answer_confidence"] = 0.0
            state["grounded_claims"] = []
            state["citation_context"] = {}
            state["citation_validation"] = {
                "structured_output_valid": False,
                "reason": "no_retrieved_context",
                "valid_chunk_ids": [],
            }
            return state

        cited_chunks = assign_citation_ids(chunks)
        state["reranked_results"] = cited_chunks
        context_str, context_map = build_citation_context(cited_chunks)
        state["citation_context"] = context_map

        try:
            response = await self._invoke_prompt(GENERATION_PROMPT, self.llm, {
                "context":  context_str,
                "question": state["query"],
            })
            claims, confidence, validation = validate_grounded_response(
                str(response.content), set(context_map)
            )
            state["grounded_claims"] = claims
            state["citation_validation"] = validation
            state["generated_answer"] = (
                format_grounded_claims(claims)
                if claims
                else "No fully supported claims could be generated from the retrieved context."
            )
            state["answer_confidence"] = confidence

        except Exception as exc:
            logger.error("[GenerateNode] LLM call failed: %s", exc)
            state["generated_answer"] = "Generation failed. Please retry."
            state["answer_confidence"] = 0.0
            state["grounded_claims"] = []
            state["citation_validation"] = {
                "structured_output_valid": False,
                "reason": "generation_failed",
                "error": str(exc),
                "valid_chunk_ids": sorted(context_map),
            }

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

    async def _invoke_prompt(self, prompt, llm, payload: Dict[str, Any]):
        """Small seam for testing structured generation and critique."""
        return await (prompt | llm).ainvoke(payload)

    async def _grounding_critique(self, state: WorkflowState) -> WorkflowState:
        """Remove semantically unsupported claims after citation-ID validation."""
        t0 = time.perf_counter()
        claims = state.get("grounded_claims", [])
        if not state.get("enable_grounding_critique", True) or not claims:
            state["grounding_critique"] = {
                "completed": False,
                "reason": "disabled" if claims else "no_validated_claims",
                "removed_claim_ids": [],
            }
            state["answer_relevancy"] = {
                "completed": False,
                "reason": "disabled" if claims else "no_validated_claims",
                "question_coverage": "none" if not claims else "unverified",
                "removed_irrelevant_claim_ids": [],
            }
            state["latency_breakdown"]["grounding_critique_ms"] = round(
                (time.perf_counter() - t0) * 1000, 2
            )
            return state

        original_count = len(claims)
        try:
            context_map = state.get("citation_context", {})
            critic_claims = [
                {
                    **claim,
                    "cited_context": {
                        chunk_id: context_map[chunk_id]
                        for chunk_id in claim["chunk_ids"]
                        if chunk_id in context_map
                    },
                }
                for claim in claims
            ]
            response = await self._invoke_prompt(CRITIC_PROMPT, self.critic_llm, {
                "question": state["query"],
                "claims_json": json.dumps(critic_claims, ensure_ascii=False),
            })
            retained, critique = apply_critic_response(claims, str(response.content))
            state["grounded_claims"] = retained
            state["grounding_critique"] = critique
            state["answer_relevancy"] = {
                "completed": critique.get("completed", False),
                "question_coverage": critique.get("question_coverage", "unverified"),
                "missing_question_facets": critique.get("missing_question_facets", []),
                "retained_relevant_claim_ids": critique.get("retained_claim_ids", []),
                "removed_irrelevant_claim_ids": critique.get(
                    "removed_irrelevant_claim_ids", []
                ),
                "irrelevant_reasons": critique.get("irrelevant_reasons", {}),
            }
            state["generated_answer"] = (
                format_grounded_claims(retained)
                if retained
                else "No fully supported claims remained after grounding verification."
            )
            if critique.get("completed"):
                state["answer_confidence"] = round(
                    state["answer_confidence"] * (len(retained) / original_count), 4
                )
        except Exception as exc:
            logger.error("[GroundingCritique] critic call failed: %s", exc)
            state["grounding_critique"] = {
                "completed": False,
                "reason": "critic_failed",
                "error": str(exc),
                "removed_claim_ids": [],
            }
            state["answer_relevancy"] = {
                "completed": False,
                "reason": "critic_failed",
                "question_coverage": "unverified",
                "removed_irrelevant_claim_ids": [],
            }

        state["latency_breakdown"]["grounding_critique_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
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
                    "semantic_score": result.get("semantic_score", 0.0),
                    "graph_signal_score": result.get("graph_signal_score"),
                    "graph_signals_applied": result.get("graph_signals_applied", False),
                    "rerank_mode": result.get("rerank_mode", "unknown"),
                    "rerank_components": result.get("rerank_components", {}),
                    "rrf_contributions": result.get("rrf_contributions", {}),
                    "citation_id": result.get("citation_id"),
                    "intent": state.get("intent", ""),
                },
                score=result.get("score", 0.0),
                source=result.get("source", "unknown"),
            )
            formatted.append(chunk)
        state["final_results"] = formatted
        return state

    @staticmethod
    def _build_trace(state: WorkflowState) -> Dict[str, Any]:
        """Expose raw internal evidence for evaluator-side redaction/persistence."""
        effective_mode = state["mode"]
        channel_candidates = {
            "vector": [] if effective_mode == QueryMode.VECTORLESS else list(state["vector_results"]),
            "graph": list(state["graph_results"]),
            "lexical": list(state.get("lexical_results", [])),
            "vectorless": list(state["vector_results"]) if effective_mode == QueryMode.VECTORLESS else [],
        }
        return {
            "requested_mode": state["requested_mode"].value,
            "effective_mode": effective_mode.value,
            "original_query": state["query"],
            "rewritten_query": state["rewritten_query"],
            "routing": state.get("routing_details", {}),
            "initial_candidates": state.get("initial_candidates", {}),
            "channel_candidates": channel_candidates,
            "retrieval_failures": state.get("retrieval_failures", {}),
            "graph_retrieval_diagnostics": state.get("graph_retrieval_diagnostics", {}),
            "recovery": state.get("recovery_details", {}),
            "fusion": {
                "weights": {
                    "vector": state["vector_fusion_weight"],
                    "graph": state["graph_fusion_weight"],
                    "lexical": state["lexical_fusion_weight"],
                },
                "lexical_enabled": state["enable_lexical_fusion"],
                "candidates": list(state["fused_results"]),
                "deduplication": state.get("deduplication_details", {}),
            },
            "reranking": {
                **state.get("reranker_details", {}),
                "candidates": list(state["reranked_results"]),
            },
            "final_contexts": list(state["reranked_results"]),
            "latency_ms": state["latency_breakdown"],
        }

    # ------------------------------------------------------------------
    # Public execute()
    # ------------------------------------------------------------------

    async def execute(
        self,
        query: str,
        mode: QueryMode = QueryMode.HYBRID,
        max_results: int = settings.CONTEXT_TOP_K,
        candidate_pool_size: int = settings.RETRIEVAL_CANDIDATE_LIMIT,
        max_hops: int = settings.GRAPH_MAX_HOPS,
        enable_lexical_fusion: bool = False,
        vector_fusion_weight: float = settings.FUSION_VECTOR_WEIGHT,
        graph_fusion_weight: float = settings.FUSION_GRAPH_WEIGHT,
        lexical_fusion_weight: float = settings.FUSION_LEXICAL_WEIGHT,
        traversal_strategy: TraversalStrategy = TraversalStrategy.BFS,
        community_id: Optional[str] = None,
        enable_conditional_recovery: bool = True,
        enable_hyde_fallback: bool = False,
        enable_grounding_critique: bool = True,
        enable_conservative_routing: bool = settings.CONSERVATIVE_ROUTING_ENABLED,
        allow_mode_downgrade: bool = True,
        allow_vectorless_auto_route: bool = True,
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
            requested_mode=mode,
            mode=mode,
            allow_mode_downgrade=allow_mode_downgrade,
            enable_conservative_routing=enable_conservative_routing,
            allow_vectorless_auto_route=allow_vectorless_auto_route,
            routing_details={},
            max_results=max_results,
            candidate_pool_size=min(max(max_results, candidate_pool_size), 100),
            max_hops=max_hops,
            traversal_strategy=traversal_strategy,
            community_id=community_id,
            enable_conditional_recovery=enable_conditional_recovery,
            enable_hyde_fallback=enable_hyde_fallback,
            filters=filters,
            vector_results=[],
            graph_results=[],
            lexical_results=[],
            initial_candidates={},
            retrieval_failures={},
            graph_retrieval_diagnostics={},
            enable_lexical_fusion=enable_lexical_fusion,
            vector_fusion_weight=vector_fusion_weight,
            graph_fusion_weight=graph_fusion_weight,
            lexical_fusion_weight=lexical_fusion_weight,
            recovery_triggered=False,
            recovery_details={},
            fused_results=[],
            deduplication_details={},
            reranked_results=[],
            reranker_details={},
            enable_grounding_critique=enable_grounding_critique,
            grounded_claims=[],
            citation_context={},
            citation_validation={},
            grounding_critique={},
            answer_relevancy={},
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
        max_results: int = settings.CONTEXT_TOP_K,
        candidate_pool_size: int = settings.RETRIEVAL_CANDIDATE_LIMIT,
        max_hops: int = settings.GRAPH_MAX_HOPS,
        enable_lexical_fusion: bool = False,
        vector_fusion_weight: float = settings.FUSION_VECTOR_WEIGHT,
        graph_fusion_weight: float = settings.FUSION_GRAPH_WEIGHT,
        lexical_fusion_weight: float = settings.FUSION_LEXICAL_WEIGHT,
        traversal_strategy: TraversalStrategy = TraversalStrategy.BFS,
        community_id: Optional[str] = None,
        enable_conditional_recovery: bool = True,
        enable_hyde_fallback: bool = False,
        enable_grounding_critique: bool = True,
        enable_conservative_routing: bool = settings.CONSERVATIVE_ROUTING_ENABLED,
        allow_mode_downgrade: bool = True,
        allow_vectorless_auto_route: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        attachment_content: Optional[str] = None,
        attachment_name: Optional[str] = None,
        include_trace: bool = False,
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
            requested_mode=mode,
            mode=mode,
            allow_mode_downgrade=allow_mode_downgrade,
            enable_conservative_routing=enable_conservative_routing,
            allow_vectorless_auto_route=allow_vectorless_auto_route,
            routing_details={},
            max_results=max_results,
            candidate_pool_size=min(max(max_results, candidate_pool_size), 100),
            max_hops=max_hops,
            traversal_strategy=traversal_strategy,
            community_id=community_id,
            enable_conditional_recovery=enable_conditional_recovery,
            enable_hyde_fallback=enable_hyde_fallback,
            filters=filters,
            vector_results=[],
            graph_results=[],
            lexical_results=[],
            initial_candidates={},
            retrieval_failures={},
            graph_retrieval_diagnostics={},
            enable_lexical_fusion=enable_lexical_fusion,
            vector_fusion_weight=vector_fusion_weight,
            graph_fusion_weight=graph_fusion_weight,
            lexical_fusion_weight=lexical_fusion_weight,
            recovery_triggered=False,
            recovery_details={},
            fused_results=[],
            deduplication_details={},
            reranked_results=[],
            reranker_details={},
            enable_grounding_critique=enable_grounding_critique,
            grounded_claims=[],
            citation_context={},
            citation_validation={},
            grounding_critique={},
            answer_relevancy={},
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
            "requested_mode": final_state["requested_mode"].value,
            "effective_mode": final_state["mode"].value,
            "routing": final_state["routing_details"],
            "latency":    final_state["latency_breakdown"],
            "recovery_triggered": final_state["recovery_triggered"],
            "recovery": final_state["recovery_details"],
            "grounded_claims": final_state["grounded_claims"],
            "citation_validation": final_state["citation_validation"],
            "grounding_critique": final_state["grounding_critique"],
            "answer_relevancy": final_state["answer_relevancy"],
            "fusion": {
                "lexical_enabled": final_state["enable_lexical_fusion"],
                "vector_weight": final_state["vector_fusion_weight"],
                "graph_weight": final_state["graph_fusion_weight"],
                "lexical_weight": final_state["lexical_fusion_weight"],
            },
            "trace": self._build_trace(final_state) if include_trace else {},
        }
