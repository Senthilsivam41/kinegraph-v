"""Corrective RAG grading and the parental-leave retrieval comparison.

Exercises backend.core.corrective_rag.apply_corrective_rag, refine_knowledge,
and HybridRAGWorkflow._corrective_rag_node, plus the parental-leave cases in
eval.corrective_rag_validation. No earlier test file covers that module.
"""
import asyncio

from backend.app.models import QueryMode
from backend.core.corrective_rag import apply_corrective_rag, refine_knowledge, tokenize
from backend.core.langgraph_workflow import HybridRAGWorkflow
from eval.corrective_rag_validation import (
    PARENTAL_LEAVE_CORPUS,
    PARENTAL_LEAVE_POLICY,
    PARENTAL_LEAVE_QUERY,
    run_distractor_validation,
    run_missed_context_validation,
    run_parental_leave_validation,
    weak_term_frequency_retrieve,
)


def test_knowledge_refinement_drops_the_unrelated_sentence():
    text = (
        "Primary caregivers receive 16 weeks of paid parental leave. "
        "The cafeteria serves pasta on Friday."
    )
    refined = refine_knowledge(text, set(tokenize(PARENTAL_LEAVE_QUERY)), {"week"})

    assert "16 weeks" in refined
    assert "cafeteria" not in refined


def test_distractor_case_promotes_the_parental_leave_policy():
    report = run_distractor_validation()

    assert report["baseline_ids"][0] == "bereavement-leave"
    assert "parental-leave" in report["baseline_ids"]
    assert report["corrected_ids"] == ["parental-leave"]
    assert report["corrective_rag"]["action"] == "refine"
    assert report["verdict"] == "improved"
    assert report["corrected_metrics"]["precision_at_1"] > report["baseline_metrics"]["precision_at_1"]
    assert report["corrected_metrics"]["ndcg_at_3"] > report["baseline_metrics"]["ndcg_at_3"]
    assert PARENTAL_LEAVE_POLICY == report["reference"]


def test_missed_context_case_recovers_the_policy_on_the_follow_up_search():
    report = run_missed_context_validation()

    assert report["baseline_ids"] == ["bereavement-leave"]
    assert report["corrected_ids"] == ["parental-leave"]
    assert report["corrective_rag"]["action"] == "retrieve"
    assert "week week" in report["corrective_rag"]["corrective_query"]
    assert report["baseline_metrics"]["precision_at_1"] == 0.0
    assert report["corrected_metrics"]["precision_at_1"] == 1.0
    assert report["verdict"] == "improved"


def test_parental_leave_validation_accepts_only_when_both_cases_improve():
    report = run_parental_leave_validation()

    assert report["improved"] is True
    assert [case["use_case"] for case in report["cases"]] == [
        "parental_leave_distractor",
        "parental_leave_missed_context",
    ]


def test_ambiguous_overlap_requests_one_follow_up_and_keeps_the_supporting_strip():
    query = "How many weeks of paid parental leave does a primary caregiver receive?"
    mixed = {
        "id": "mixed-leave",
        "content": (
            "Primary caregivers receive 16 weeks of paid leave. "
            "The cafeteria serves pasta on Friday."
        ),
    }
    unrelated = [
        {"id": "parking-access", "content": "Primary badge holders receive paid parking access."},
        {"id": "remote-stipend", "content": "Remote employees receive a home office stipend."},
    ]
    captured = {}

    def retrieve(corrective_query: str):
        captured["query"] = corrective_query
        return [{
            "id": "parental-leave",
            "content": (
                "Primary caregivers receive 16 weeks of paid parental leave "
                "after the birth or adoption of a child."
            ),
        }]

    selected, report = apply_corrective_rag(
        query, [mixed, *unrelated], top_k=2, retrieve=retrieve
    )

    assert report["action"] == "refine_and_retrieve"
    assert "parental" in captured["query"]
    assert [item["id"] for item in selected] == ["parental-leave", "mixed-leave"]
    assert "cafeteria" not in selected[1]["content"]


def test_corrective_node_leaves_results_unchanged_when_disabled():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    state = {
        "query": PARENTAL_LEAVE_QUERY,
        "mode": QueryMode.VECTOR,
        "max_results": 3,
        "enable_corrective_rag": False,
        "reranked_results": [{"id": "bereavement-leave", "content": "paid leave"}],
        "latency_breakdown": {},
        "candidate_lifecycle": {"already": True},
    }

    updated = asyncio.run(workflow._corrective_rag_node(state))

    assert updated["reranked_results"][0]["id"] == "bereavement-leave"
    assert updated["corrective_rag"]["enabled"] is False
    assert updated["corrective_rag"]["action"] == "passthrough"
    assert updated["candidate_lifecycle"] == {"already": True}


def test_corrective_node_filters_a_distractor_without_calling_the_stores():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    workflow.chroma = None
    initial = weak_term_frequency_retrieve(PARENTAL_LEAVE_QUERY, PARENTAL_LEAVE_CORPUS, 3)
    state = {
        "query": PARENTAL_LEAVE_QUERY,
        "mode": QueryMode.VECTOR,
        "max_results": 3,
        "candidate_pool_size": 5,
        "enable_corrective_rag": True,
        "reranked_results": initial,
        "vector_results": initial,
        "graph_results": [],
        "lexical_results": [],
        "fusion_candidates_before_dedup": initial,
        "deduplication_details": {},
        "reranker_details": {},
        "context_optimization_details": {},
        "latency_breakdown": {},
    }

    updated = asyncio.run(workflow._corrective_rag_node(state))

    assert [item["id"] for item in updated["reranked_results"]] == ["parental-leave"]
    assert updated["corrective_rag"]["action"] == "refine"
    assert updated["candidate_lifecycle"]["candidate_count"] >= 1
