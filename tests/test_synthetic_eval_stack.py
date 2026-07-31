"""Tests for IR metrics, kinetic score shadow, and testset generation helpers."""
from pathlib import Path

from eval.graph_testset_synthesis import extract_offline_seeds, synthesize_graph_rows
from eval.ir_metrics import ndcg_at_k, precision_at_k, recall_at_k, score_retrieval
from eval.kinetic_score import compute_kinetic_score_shadow
from eval.testset_generation import (
    REQUIRED_COLUMNS,
    chunk_documents_for_generation,
    load_markdown_documents,
    records_to_langchain_docs,
)
from eval.usage_cost import UsageTracker
from langchain_core.documents import Document


REPO_ROOT = Path(__file__).parents[1]


def test_ir_metrics_reward_relevant_top_hits():
    retrieved = ["alpha evidence", "noise", "beta evidence"]
    relevant = ["alpha evidence", "beta evidence"]
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    assert recall_at_k(retrieved, relevant, k=3) == 1.0
    assert ndcg_at_k(retrieved, relevant, k=3) > 0.5
    scores = score_retrieval(
        retrieved_contexts=retrieved,
        reference_contexts=relevant,
        k=3,
    )
    assert scores["precision_at_3"] == round(2 / 3, 4)


def test_kinetic_score_shadow_is_bounded_and_versioned():
    payload = compute_kinetic_score_shadow(
        ragas_scores={
            "faithfulness": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
            "answer_relevancy": 0.9,
        },
        ir_metrics={"precision_at_5": 0.5, "recall_at_5": 0.4},
        path_completeness=1.0,
    )
    assert payload["mode"] == "shadow"
    assert 0 <= payload["kinetic_score"] <= 100
    assert payload["policy_version"].startswith("kinegraph.kinetic-score")


def test_adr002_chunks_convert_to_langchain_docs_with_provenance():
    docs = [Document(page_content="# Title\n\nBody paragraph about Kinetic-V.", metadata={"source": "docs/demo.md"})]
    records = chunk_documents_for_generation(docs, adaptive_enabled=True, chunk_size=200, chunk_overlap=20)
    assert records
    lc_docs = records_to_langchain_docs(records)
    assert lc_docs[0].metadata["chunk_id"]
    assert lc_docs[0].metadata["document_id"]
    assert "chunk_policy_version" in lc_docs[0].metadata


def test_load_markdown_documents_from_repo_docs():
    docs = load_markdown_documents(REPO_ROOT / "docs")
    assert docs
    assert all(doc.page_content.strip() for doc in docs)


def test_graph_seed_synthesis_is_grounded_in_chunk_text():
    seeds = extract_offline_seeds(
        ["KineticGraph uses Neo4j and ChromaDB for hybrid retrieval."],
        limit=5,
    )
    rows = synthesize_graph_rows(seeds, max_rows=3)
    assert rows
    assert set(REQUIRED_COLUMNS).issubset(rows[0])
    assert "Neo4j" in rows[0]["reference"] or "KineticGraph" in rows[0]["reference"]
    assert "graph_mentions_seed_v1" == rows[0]["synthesizer_name"]


def test_usage_tracker_marks_incomplete_cost_without_estimates():
    tracker = UsageTracker()
    tracker.record(label="llm", prompt_tokens=10, completion_tokens=5)
    snapshot = tracker.snapshot()
    assert snapshot["total_tokens"] == 15
    assert snapshot["cost_complete"] is False
