import inspect
from unittest.mock import MagicMock

import pytest

from backend.core.context_ranker import ContextRanker
from backend.core.config import settings
from backend.core.langgraph_workflow import HybridRAGWorkflow
from backend.core.rrf import reciprocal_rank_fusion


def _graph_chunk(name, centrality, community, edge_weight, depth):
    return {
        "content": name,
        "metadata": {
            "centrality_score": centrality,
            "community_id": community,
            "traversal_depth": depth,
            "relationship_path": [{"weight": edge_weight}],
        },
        "score": 0.25,
        "source": "graph_traversal",
    }


def test_graph_signals_refine_equal_semantic_scores():
    strong = _graph_chunk("strong graph result", 0.9, "preferred", 0.9, 1)
    weak = _graph_chunk("weak graph result", 0.1, "other", 0.2, 3)
    ranker = ContextRanker(use_cross_encoder=False, min_relevance_threshold=0)
    ranker._score_chunks = lambda query, chunks: [(weak, 0.6), (strong, 0.6)]

    results = ranker.rerank(
        "graph result", [weak, strong], top_k=2, preferred_community_id="preferred"
    )

    assert [result["content"] for result in results] == ["strong graph result", "weak graph result"]
    assert results[0]["rerank_components"] == {
        "centrality": 0.9,
        "community": 1.0,
        "edge": 0.9,
        "distance": 1.0,
    }
    assert results[0]["graph_signals_applied"] is True


def test_semantic_relevance_remains_dominant_over_graph_importance():
    relevant = {"content": "directly relevant", "metadata": {}, "score": 0.4}
    central_but_weak = _graph_chunk("off topic central node", 1.0, "community", 1.0, 1)
    ranker = ContextRanker(use_cross_encoder=False, min_relevance_threshold=0)
    ranker._score_chunks = lambda query, chunks: [(relevant, 0.9), (central_but_weak, 0.2)]

    results = ranker.rerank("direct question", [relevant, central_but_weak], top_k=2)

    assert results[0]["content"] == "directly relevant"
    assert results[0]["rerank_score"] > results[1]["rerank_score"]


def test_rrf_and_original_scores_survive_graph_aware_reranking():
    vector = [
        {"content": "alpha", "score": 0.91, "source": "vector", "metadata": {}},
        {"content": "beta", "score": 0.72, "source": "vector", "metadata": {}},
    ]
    graph = [
        {"content": "alpha", "score": 0.83, "source": "graph", "metadata": {"centrality_score": 0.8}},
    ]
    fused = reciprocal_rank_fusion([vector, graph], k=60)
    before = {
        item["content"]: (item["score"], item["rrf_score"], item["original_score"], item["source"])
        for item in fused
    }
    ranker = ContextRanker(use_cross_encoder=False, min_relevance_threshold=0)
    semantic = {"alpha": 0.7, "beta": 0.8}
    ranker._score_chunks = lambda query, chunks: [(item, semantic[item["content"]]) for item in chunks]

    results = ranker.rerank("alpha beta", fused, top_k=2)

    for result in results:
        expected = before[result["content"]]
        assert (result["score"], result["rrf_score"], result["original_score"], result["source"]) == expected
        assert "rerank_score" in result
        assert "semantic_score" in result


def test_normal_workflow_keeps_cross_encoder_as_controlled_experiment():
    default = inspect.signature(HybridRAGWorkflow.__init__).parameters["use_cross_encoder"].default
    assert default is None
    assert settings.CROSS_ENCODER_RERANK_ENABLED is False


def test_cross_encoder_logits_use_query_independent_sigmoid_scores():
    ranker = ContextRanker(use_cross_encoder=False, min_relevance_threshold=0)
    ranker.use_cross_encoder = True
    ranker._encoder = MagicMock()
    ranker._encoder.predict.return_value = [-2.0, 2.0]
    chunks = [{"content": "weak"}, {"content": "strong"}]

    scored = ranker._score_chunks("query", chunks)

    assert [item[0]["content"] for item in scored] == ["strong", "weak"]
    assert scored[0][1] == pytest.approx(0.8808, abs=0.0001)
    assert scored[1][1] == pytest.approx(0.1192, abs=0.0001)


def test_weighted_rrf_changes_channel_priority_and_reports_contributions():
    vector = [{"content": "vector evidence", "score": 0.8, "source": "vector"}]
    graph = [{"content": "graph evidence", "score": 0.7, "source": "graph"}]

    fused = reciprocal_rank_fusion(
        [vector, graph],
        k=60,
        weights=[1.0, 2.0],
        source_names=["vector", "graph"],
    )

    assert [item["content"] for item in fused] == ["graph evidence", "vector evidence"]
    assert fused[0]["rrf_contributions"] == {"graph": pytest.approx(2 / 61)}


@pytest.mark.parametrize("failure", ["load", "predict", "nonfinite"])
def test_cross_encoder_failure_reports_effective_keyword_fallback(monkeypatch, failure):
    from backend.core import context_ranker

    encoder = MagicMock()
    factory = MagicMock(return_value=encoder)
    if failure == "load":
        factory.side_effect = OSError("model unavailable")
    elif failure == "predict":
        encoder.predict.side_effect = RuntimeError("inference unavailable")
    else:
        encoder.predict.return_value = [float("nan"), 0.9]
    monkeypatch.setattr(context_ranker, "_CE_AVAILABLE", True)
    monkeypatch.setattr(
        context_ranker, "hf_hub_download",
        lambda *args: f"/hub/snapshots/{'c' * 40}/config.json", raising=False,
    )
    monkeypatch.setattr(context_ranker, "CrossEncoder", factory, raising=False)
    ranker = ContextRanker(use_cross_encoder=True)
    chunks = [{"content": "needle"}, {"content": "unrelated"}]

    results, trace = ranker.rerank_with_report("needle", chunks, top_k=1)

    assert results[0]["content"] == "needle"
    assert results[0]["rerank_mode"] == "graph_aware_keyword"
    assert trace["mode"] == "keyword"
    assert trace["fallback"] is True
    expected = "load failed: OSError" if failure == "load" else "inference failed:"
    assert expected in trace["fallback_reason"]
    if failure != "load":
        assert trace["model_revision"] == "c" * 40
        encoder.predict.side_effect = None
        encoder.predict.return_value = [0.9, 0.3]
        _, retry_trace = ranker.rerank_with_report("needle", chunks)
        assert retry_trace["mode"] == "cross_encoder"
        assert retry_trace["fallback"] is False


def test_rerank_trace_includes_scores_for_candidate_dropped_at_top_k():
    ranker = ContextRanker(min_relevance_threshold=0)
    chunks = [
        _graph_chunk("needle strong", 0.9, "preferred", 0.9, 1),
        _graph_chunk("needle weak", 0.1, "other", 0.1, 2),
    ]

    _, trace = ranker.rerank_with_report("needle", chunks, top_k=1)

    dropped = trace["decisions"][1]
    assert dropped["reason"] == "reranker_top_k_exceeded"
    assert dropped["rerank_rank"] == 2
    assert dropped["rerank_score"] > 0
    assert dropped["rerank_components"]["centrality"] == 0.1
