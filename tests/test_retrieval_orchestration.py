from backend.core.context_ranker import ContextRanker
from backend.core.retrieval_orchestration import (
    ORCHESTRATION_POLICY_VERSION,
    annotate_channel_candidates,
    build_candidate_lifecycle,
    optimize_context,
)
from backend.core.rrf import (
    deduplicate_results_with_report,
    reciprocal_rank_fusion,
)


def test_rrf_preserves_channel_scores_ranks_and_graph_path():
    vector = [{
        "id": "chunk-1",
        "content": "shared evidence",
        "score": 0.91,
        "source": "vector",
        "metadata": {},
    }]
    path = [{"source": "A", "target": "B", "weight": 0.8}]
    graph = [{
        "metadata": {"chunk_id": "chunk-1", "relationship_path": path},
        "content": "shared evidence",
        "score": 0.72,
        "source": "graph",
    }]

    fused = reciprocal_rank_fusion(
        [vector, graph], source_names=["vector", "graph"], k=60
    )

    assert len(fused) == 1
    assert fused[0]["candidate_id"] == "chunk-1"
    assert fused[0]["source_channels"] == ["vector", "graph"]
    assert fused[0]["original_scores"] == {"vector": 0.91, "graph": 0.72}
    assert fused[0]["channel_ranks"] == {"vector": 1, "graph": 1}
    assert fused[0]["graph_paths"]["graph"] == path


def test_deduplication_reports_exact_drop_reason_and_representative():
    first = {"id": "a", "content": "alpha beta gamma delta"}
    duplicate = {"id": "b", "content": "delta gamma beta alpha"}

    survivors, report = deduplicate_results_with_report(
        [first, duplicate], similarity_threshold=0.95
    )

    assert survivors == [first]
    assert report["decisions"][1] == {
        "candidate_id": "b",
        "stage": "identity_deduplication",
        "decision": "dropped",
        "reason": "near_duplicate_content",
        "duplicate_of": "a",
        "similarity": 1.0,
    }


def test_context_optimizer_preserves_rank_and_declares_diversity_drops():
    candidates = [
        {"id": "a", "content": "a", "metadata": {"document_id": "doc-1"}},
        {"id": "b", "content": "b", "metadata": {"document_id": "doc-1"}},
        {"id": "c", "content": "c", "metadata": {"document_id": "doc-2"}},
    ]

    selected, report = optimize_context(
        candidates, top_k=2, max_per_source=1, max_per_community=0
    )

    assert [candidate["id"] for candidate in selected] == ["a", "c"]
    assert report["decisions"][1]["reason"] == "source_diversity_cap"
    assert report["decisions"][1]["decision"] == "dropped"


def test_reranker_report_distinguishes_threshold_and_top_k_drops():
    ranker = ContextRanker(use_cross_encoder=False, min_relevance_threshold=0.2)
    chunks = [
        {"id": "strong", "content": "strong"},
        {"id": "medium", "content": "medium"},
        {"id": "weak", "content": "weak"},
    ]
    scores = {"strong": 0.9, "medium": 0.8, "weak": 0.1}
    ranker._score_chunks = lambda query, items: [
        (item, scores[item["id"]]) for item in items
    ]

    results, report = ranker.rerank_with_report("query", chunks, top_k=1)

    assert [result["id"] for result in results] == ["strong"]
    reasons = {item["candidate_id"]: item["reason"] for item in report["decisions"]}
    assert reasons == {
        "strong": "selected_by_reranker",
        "medium": "reranker_top_k_exceeded",
        "weak": "below_semantic_relevance_threshold",
    }


def test_candidate_lifecycle_reports_provenance_completeness():
    path = [{"source": "A", "target": "B", "weight": 1.0}]
    graph = annotate_channel_candidates([{
        "metadata": {"chunk_id": "g-1", "relationship_path": path},
        "content": "graph evidence",
        "score": 0.8,
    }], "graph")
    report = build_candidate_lifecycle(
        channel_candidates={"graph": graph},
        fused_candidates=graph,
        final_candidates=graph,
        stage_reports=(),
    )

    assert report["policy_version"] == ORCHESTRATION_POLICY_VERSION
    assert report["candidate_provenance_completeness"] == 1.0
    assert report["candidates"][0]["final_status"] == "selected_for_generation"

