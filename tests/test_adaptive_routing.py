import asyncio
from unittest.mock import AsyncMock

from backend.app.models import QueryMode
from backend.core.adaptive_routing import ROUTING_POLICY_VERSION
from backend.core.langgraph_workflow import HybridRAGWorkflow
from backend.core.query_recovery import QueryRecoveryEngine
from backend.graph_retrieval.multi_hop import TraversalStrategy


def _routing_state(query: str, mode: QueryMode = QueryMode.HYBRID, **overrides):
    state = {
        "query": query,
        "rewritten_query": query,
        "intent": "",
        "suggested_mode": "",
        "requested_mode": mode,
        "mode": mode,
        "allow_mode_downgrade": True,
        "enable_adaptive_routing": True,
        "enable_conservative_routing": False,
        "allow_vectorless_auto_route": True,
        "enable_lexical_fusion": False,
        "routing_details": {},
        "attachment_content": None,
        "filters": None,
        "latency_breakdown": {},
    }
    state.update(overrides)
    return state


def test_high_confidence_single_facet_plan_is_versioned_and_reversible():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)

    state = asyncio.run(workflow._intent_router(_routing_state(
        "Define the definition of PageRank"
    )))

    plan = state["routing_details"]["execution_plan"]
    assert state["mode"] == QueryMode.VECTOR
    assert plan["policy_version"] == ROUTING_POLICY_VERSION
    assert plan["policy"] == "adaptive"
    assert plan["route_confidence"] >= 0.80
    assert plan["required_channels"] == ["vector"]
    assert plan["fallback_mode"] == "hybrid"
    assert plan["fallback_trigger"] == "measurable_initial_retrieval_weakness"
    assert {item["mode"] for item in plan["alternatives"]} == {
        "hybrid", "graph", "vectorless"
    }


def test_exact_token_query_retains_hybrid_and_recommends_lexical_channel():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)

    state = asyncio.run(workflow._intent_router(_routing_state(
        "What are the OPENAI_API_KEY rules and how should .env be secured?"
    )))

    plan = state["routing_details"]["execution_plan"]
    assert state["mode"] == QueryMode.HYBRID
    assert "OPENAI_API_KEY" in plan["signals"]["exact_tokens"]
    assert plan["required_channels"] == ["vector", "graph"]
    assert plan["recommended_channels"] == ["vector", "graph", "lexical"]
    assert "exact-token evidence" in plan["decision"]


def test_explicit_mode_is_authoritative_even_with_attachment():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)

    state = asyncio.run(workflow._intent_router(_routing_state(
        "Explain the attached source",
        mode=QueryMode.GRAPH,
        attachment_content="small attachment",
    )))

    plan = state["routing_details"]["execution_plan"]
    assert state["mode"] == QueryMode.GRAPH
    assert plan["pinned"] is True
    assert plan["decision"] == "explicit caller mode 'graph' preserved"


def test_hybrid_can_be_pinned_against_attachment_auto_route():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)

    state = asyncio.run(workflow._intent_router(_routing_state(
        "Summarize this attachment",
        attachment_content="small attachment",
        allow_mode_downgrade=False,
        allow_vectorless_auto_route=False,
    )))

    assert state["mode"] == QueryMode.HYBRID
    assert state["routing_details"]["execution_plan"]["pinned"] is True


def test_eligible_attachment_routes_to_vectorless_with_observable_signal():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)

    state = asyncio.run(workflow._intent_router(_routing_state(
        "Summarize this attachment",
        attachment_content="bounded source content",
    )))

    plan = state["routing_details"]["execution_plan"]
    assert state["mode"] == QueryMode.VECTORLESS
    assert plan["signals"]["attachment_eligible"] is True
    assert plan["required_channels"] == ["vectorless"]


def test_low_confidence_or_misspelled_query_cannot_downgrade():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)

    state = asyncio.run(workflow._intent_router(_routing_state(
        "How do I fix documnt.pdf?"
    )))

    plan = state["routing_details"]["execution_plan"]
    assert state["mode"] == QueryMode.HYBRID
    assert plan["route_confidence"] < 0.80
    assert "documnt.pdf" in plan["signals"]["exact_tokens"]


def test_compound_and_relationship_queries_keep_broad_evidence_coverage():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)

    compound = asyncio.run(workflow._intent_router(_routing_state(
        "Compare Neo4j and ChromaDB and explain how they work together"
    )))
    relationship = asyncio.run(workflow._intent_router(_routing_state(
        "How does Neo4j relate to ChromaDB?"
    )))

    compound_plan = compound["routing_details"]["execution_plan"]
    relationship_plan = relationship["routing_details"]["execution_plan"]
    assert compound["mode"] == QueryMode.HYBRID
    assert compound_plan["signals"]["coverage_sensitive"] is True
    assert compound_plan["signals"]["entity_candidates"] == ["Neo4j", "ChromaDB"]
    assert relationship["mode"] == QueryMode.HYBRID
    assert relationship_plan["signals"]["relationship_signal"] is True


def test_weak_single_channel_route_escalates_to_hybrid_before_fusion():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    workflow.recovery = QueryRecoveryEngine(llm=None)
    workflow._retrieve_graph = AsyncMock(return_value=(
        [
            {
                "content": f"PageRank definition graph evidence {index}",
                "score": 0.9 - (index * 0.05),
                "source": "graph",
                "metadata": {"id": f"graph-{index}"},
            }
            for index in range(3)
        ],
        {"seed_count": 1},
    ))

    state = asyncio.run(workflow._intent_router(_routing_state(
        "Define the definition of PageRank"
    )))
    assert state["mode"] == QueryMode.VECTOR
    state.update({
        "max_results": 3,
        "candidate_pool_size": 5,
        "max_hops": 2,
        "traversal_strategy": TraversalStrategy.BFS,
        "community_id": None,
        "enable_conditional_recovery": True,
        "enable_hyde_fallback": False,
        "vector_results": [{
            "content": "PageRank overview",
            "score": 0.2,
            "source": "vector",
            "metadata": {"id": "vector-1"},
        }],
        "graph_results": [],
        "lexical_results": [],
        "initial_candidates": {},
        "retrieval_failures": {},
        "graph_retrieval_diagnostics": {},
        "recovery_triggered": False,
        "recovery_details": {},
    })

    recovered = asyncio.run(workflow._query_recovery(state))

    escalation = recovered["recovery_details"]["route_escalation"]
    assert recovered["mode"] == QueryMode.HYBRID
    assert escalation["triggered"] is True
    assert escalation["from_mode"] == "vector"
    assert escalation["to_mode"] == "hybrid"
    assert escalation["added_channels"] == ["graph"]
    assert escalation["initial_channel_counts"] == {
        "vector": 1,
        "graph": 0,
        "lexical": 0,
    }
    assert escalation["added_candidate_counts"] == {"graph": 3}
    assert recovered["recovery_triggered"] is True
    assert recovered["routing_details"]["effective_mode"] == "hybrid"
    assert recovered["routing_details"]["execution_plan"]["effective_mode"] == "hybrid"
    assert {item["mode"] for item in recovered["routing_details"]["execution_plan"]["alternatives"]} == {
        "vector", "graph", "vectorless"
    }
    assert recovered["graph_results"][0]["metadata"]["recovery_stage"] == (
        "adaptive_route_escalation"
    )
