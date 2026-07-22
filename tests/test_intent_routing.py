import asyncio
import csv

from backend.app.models import QueryMode
from backend.core.intent_classifier import classify_intent
from backend.core.langgraph_workflow import HybridRAGWorkflow
from eval.benchmark_profiles import get_profile


def _state(query, mode=QueryMode.HYBRID, **overrides):
    state = {
        "query": query,
        "rewritten_query": query,
        "intent": "",
        "suggested_mode": "",
        "requested_mode": mode,
        "mode": mode,
        "allow_mode_downgrade": True,
        "enable_conservative_routing": False,
        "allow_vectorless_auto_route": True,
        "routing_details": {},
        "attachment_content": None,
        "filters": None,
        "latency_breakdown": {},
    }
    state.update(overrides)
    return state


def _benchmark_questions():
    with open("eval/kinegraph_benchmark_v1.csv", newline="") as source:
        return [row["user_input"] for row in csv.DictReader(source)]


def test_classifier_exposes_scores_triggers_facets_and_rationale():
    result = classify_intent(
        "What are the API key formatting rules, and what security practices apply?"
    )

    assert result["scores"]["definition"] > 0
    assert result["matched_triggers"]["definition"] == ["what are"]
    assert len(result["facets"]) == 2
    assert result["coverage_sensitive"] is True
    assert "deterministic trigger scores" in result["route_rationale"]


def test_trigger_matching_does_not_treat_overview_as_comparison():
    result = classify_intent("Provide an overview of graph retrieval")

    assert result["scores"]["comparison"] == 0


def test_lossy_benchmark_downgrades_are_retained_as_hybrid():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    questions = _benchmark_questions()

    for row_number in (1, 6, 11, 13, 14):
        state = asyncio.run(workflow._intent_router(_state(
            questions[row_number - 1], enable_conservative_routing=True
        )))
        assert state["mode"] == QueryMode.HYBRID, row_number
        assert "retained hybrid" in state["routing_details"]["decision"]


def test_query_12_remains_comparison_hybrid():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    query = _benchmark_questions()[11]

    state = asyncio.run(workflow._intent_router(_state(query)))

    assert state["intent"] == "comparison"
    assert state["mode"] == QueryMode.HYBRID


def test_high_confidence_single_facet_can_still_use_simpler_vector_path():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)

    state = asyncio.run(workflow._intent_router(_state(
        "Define the definition of PageRank", enable_conservative_routing=True
    )))

    assert state["routing_details"]["confidence"] == "high"
    assert state["routing_details"]["coverage_sensitive"] is False
    assert state["mode"] == QueryMode.VECTOR


def test_profile_can_pin_hybrid_and_explicit_modes_remain_authoritative():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    query = "Define the definition of PageRank"

    pinned = asyncio.run(workflow._intent_router(_state(
        query,
        allow_mode_downgrade=False,
        allow_vectorless_auto_route=False,
    )))
    explicit = asyncio.run(workflow._intent_router(_state(
        query,
        mode=QueryMode.VECTOR,
        attachment_content="small attachment",
    )))

    assert pinned["mode"] == QueryMode.HYBRID
    assert pinned["routing_details"]["decision"] == "benchmark profile requires requested mode"
    assert explicit["mode"] == QueryMode.VECTOR


def test_legacy_routing_remains_default_until_conservative_policy_is_accepted():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    query = _benchmark_questions()[5]

    state = asyncio.run(workflow._intent_router(_state(query)))

    assert state["routing_details"]["enable_conservative_routing"] is False
    assert state["mode"] == QueryMode.GRAPH


def test_fixed_benchmark_profiles_reach_their_declared_route():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    query = "Define the definition of PageRank"

    for profile_name, expected_mode in (
        ("hybrid", QueryMode.HYBRID),
        ("hybrid_lexical", QueryMode.HYBRID),
        ("vectorless", QueryMode.VECTORLESS),
    ):
        profile = get_profile(profile_name)
        state = asyncio.run(workflow._intent_router(_state(
            query,
            mode=profile.requested_mode,
            allow_mode_downgrade=profile.allow_mode_downgrade,
            allow_vectorless_auto_route=profile.allow_vectorless_auto_route,
        )))
        assert state["mode"] == expected_mode
        assert workflow._route_decision(state) in profile.declared_effective_modes
