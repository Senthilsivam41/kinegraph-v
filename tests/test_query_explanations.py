import asyncio

from backend.app.models import DocumentChunk
from backend.core.langgraph_workflow import HybridRAGWorkflow


def test_formatted_result_explains_selection_from_existing_provenance():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    state = {
        "intent": "comparison",
        "reranked_results": [
            {
                "content": "Neo4j stores graph relationships.",
                "source": "graph",
                "score": 0.82,
                "candidate_id": "graph-7",
                "source_channels": ["graph", "vector"],
                "channel_ranks": {"graph": 1, "vector": 3},
                "original_scores": {"graph": 0.91, "vector": 0.74},
                "rrf_contributions": {"graph": 0.0164, "vector": 0.0159},
                "rerank_mode": "graph_aware_keyword",
                "rerank_score": 0.88,
                "semantic_score": 0.8,
                "graph_signal_score": 0.2,
                "graph_signals_applied": True,
                "rerank_components": {"semantic": 0.8, "graph": 0.2},
                "graph_paths": {"graph": [{"relationship": "USES"}]},
            }
        ],
    }

    formatted = asyncio.run(workflow._format_results(state))["final_results"][0]

    assert formatted.explanation == {
        "candidate_id": "graph-7",
        "source_channels": ["graph", "vector"],
        "channel_ranks": {"graph": 1, "vector": 3},
        "original_scores": {"graph": 0.91, "vector": 0.74},
        "rrf_contributions": {"graph": 0.0164, "vector": 0.0159},
        "reranking": {
            "mode": "graph_aware_keyword",
            "score": 0.88,
            "semantic_score": 0.8,
            "graph_signal_score": 0.2,
            "graph_signals_applied": True,
            "components": {"semantic": 0.8, "graph": 0.2},
        },
        "graph_paths": {"graph": [{"relationship": "USES"}]},
    }


def test_formatted_result_explanation_has_safe_defaults():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    state = {
        "intent": "conceptual",
        "reranked_results": [{"content": "Evidence", "source": "vector", "score": 0.5}],
    }

    formatted = asyncio.run(workflow._format_results(state))["final_results"][0]

    assert formatted.explanation["source_channels"] == ["vector"]
    assert formatted.explanation["channel_ranks"] == {}
    assert formatted.explanation["graph_paths"] == {}


def test_explanation_is_documented_in_the_public_schema():
    schema = DocumentChunk.model_json_schema()

    assert "no additional model call" in schema["properties"]["explanation"]["description"]
