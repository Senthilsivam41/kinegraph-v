import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from backend.app.models import QueryMode
from backend.core.langgraph_workflow import HybridRAGWorkflow
from backend.core.query_recovery import (
    QueryRecoveryEngine,
    RecoveryPlan,
    WeaknessAssessment,
)
from backend.graph_retrieval.multi_hop import TraversalStrategy


def _assessment(weak, reasons=None):
    return WeaknessAssessment(
        weak=weak,
        reasons=reasons or [],
        result_count=1 if weak else 5,
        top_score=0.2 if weak else 0.8,
        graph_result_count=0 if weak else 2,
        source_count=1 if weak else 2,
    )


def _state(**overrides):
    state = {
        "query": "How do Neo4j and ChromaDB work together?",
        "rewritten_query": "Neo4j ChromaDB integration",
        "intent": "relationship",
        "mode": QueryMode.HYBRID,
        "max_results": 5,
        "candidate_pool_size": 20,
        "max_hops": 3,
        "traversal_strategy": TraversalStrategy.BFS,
        "community_id": None,
        "enable_conditional_recovery": True,
        "enable_hyde_fallback": False,
        "filters": None,
        "vector_results": [{"content": "initial", "score": 0.2, "source": "vector", "metadata": {}}],
        "graph_results": [],
        "recovery_triggered": False,
        "recovery_details": {},
        "latency_breakdown": {},
    }
    state.update(overrides)
    return state


def test_weakness_detection_uses_original_score_after_subquery_rrf():
    engine = QueryRecoveryEngine(MagicMock(), min_results=1, min_top_score=0.35)
    assessment = engine.assess(
        [{"content": "result", "score": 0.016, "rrf_score": 0.016, "original_score": 0.8, "source": "vector"}],
        [],
    )
    assert assessment.weak is False
    assert assessment.top_score == 0.8


def test_graph_only_results_do_not_require_cross_channel_diversity():
    engine = QueryRecoveryEngine(MagicMock(), min_results=2, min_top_score=0.35)
    graph_results = [
        {"content": "path one", "score": 0.8, "source": "graph"},
        {"content": "path two", "score": 0.7, "source": "graph"},
    ]

    assessment = engine.assess(graph_results=[], vector_results=[], require_graph=True)
    graph_assessment = engine.assess([], graph_results, require_graph=True)

    assert assessment.weak is True
    assert graph_assessment.weak is False
    assert "low_source_diversity" not in graph_assessment.reasons


def test_hyde_rejects_new_named_entities_numbers_and_citations():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content="Neo4j integrates with ChromaDB using Kubernetes in 2025 [1]."
    ))
    engine = QueryRecoveryEngine(llm)

    hypothesis = asyncio.run(engine.generate_hypothesis("How do Neo4j and ChromaDB integrate?"))

    assert hypothesis == ""


def test_plan_filters_offtrack_subqueries_and_limits_structured_output():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content=json.dumps({
        "subqueries": [
            "Compare Neo4j and ChromaDB storage responsibilities",
            "How is Kubernetes deployed to production?",
        ],
        "vocabulary": ["graph database", "vector store", "semantic retrieval"],
    })))
    engine = QueryRecoveryEngine(llm)

    plan = asyncio.run(engine.create_plan("Compare Neo4j and ChromaDB storage", "comparison"))

    assert plan.subqueries == ["Compare Neo4j and ChromaDB storage responsibilities"]
    assert plan.vocabulary == ["graph database", "vector store", "semantic retrieval"]


def test_subqueries_execute_and_fuse_before_cross_channel_rrf():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    workflow.chroma = MagicMock()
    workflow.chroma.similarity_search = AsyncMock(return_value=[
        {"content": "subquery evidence", "score": 0.75, "source": "vector", "metadata": {}}
    ])
    workflow.graph_retriever_node = MagicMock()
    workflow.graph_retriever_node.retrieve_chunks = AsyncMock(return_value=[])
    workflow.recovery = QueryRecoveryEngine(MagicMock())
    workflow.recovery.assess = MagicMock(side_effect=[_assessment(True, ["insufficient_results"]), _assessment(False), _assessment(False)])
    workflow.recovery.create_plan = AsyncMock(return_value=RecoveryPlan(
        subqueries=["Neo4j ChromaDB integration"], vocabulary=[]
    ))

    state = asyncio.run(workflow._query_recovery(_state()))

    assert state["recovery_triggered"] is True
    assert state["recovery_details"]["structured_recovery_used"] is True
    assert state["recovery_details"]["hyde_used"] is False
    assert all("rrf_score" in result for result in state["vector_results"])
    recovered = next(result for result in state["vector_results"] if result["content"] == "subquery evidence")
    assert recovered["metadata"]["recovery_stage"] == "decomposition"


def test_hyde_runs_only_after_structured_recovery_and_never_queries_graph():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    workflow.chroma = MagicMock()
    workflow.chroma.similarity_search = AsyncMock(side_effect=[
        [],  # decomposition vector retrieval
        [],  # vocabulary vector retrieval
        [{"content": "verified source chunk", "score": 0.8, "source": "vector", "metadata": {}}],
    ])
    workflow.graph_retriever_node = MagicMock()
    workflow.graph_retriever_node.retrieve_chunks = AsyncMock(return_value=[])
    workflow.recovery = QueryRecoveryEngine(MagicMock())
    workflow.recovery.assess = MagicMock(side_effect=[
        _assessment(True, ["insufficient_results"]),
        _assessment(True, ["insufficient_results"]),
        _assessment(False),
    ])
    workflow.recovery.create_plan = AsyncMock(return_value=RecoveryPlan(
        subqueries=["Neo4j ChromaDB integration"], vocabulary=["hybrid retrieval"]
    ))
    hypothesis = "Neo4j and ChromaDB participate in a hybrid retrieval workflow."
    workflow.recovery.generate_hypothesis = AsyncMock(return_value=hypothesis)

    state = asyncio.run(workflow._query_recovery(_state(enable_hyde_fallback=True)))

    assert state["recovery_details"]["hyde_used"] is True
    assert state["recovery_details"]["generated_hypothesis"] == hypothesis
    assert workflow.graph_retriever_node.retrieve_chunks.await_count == 1
    assert hypothesis not in str(workflow.graph_retriever_node.retrieve_chunks.await_args_list)
    hyde_result = next(result for result in state["vector_results"] if result["content"] == "verified source chunk")
    assert hyde_result["metadata"]["recovery_stage"] == "hyde"
    assert hyde_result["metadata"]["hypothesis_is_evidence"] is False


def test_strong_results_skip_all_recovery_generation():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    workflow.recovery = QueryRecoveryEngine(MagicMock())
    workflow.recovery.assess = MagicMock(return_value=_assessment(False))
    workflow.recovery.create_plan = AsyncMock()

    state = asyncio.run(workflow._query_recovery(_state(
        vector_results=[{"content": "strong", "score": 0.9, "source": "vector", "metadata": {}}],
        graph_results=[{"content": "path", "score": 0.8, "source": "graph", "metadata": {}}],
    )))

    assert state["recovery_triggered"] is False
    workflow.recovery.create_plan.assert_not_awaited()
