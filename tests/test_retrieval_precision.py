import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.models import QueryMode, QueryRequest
from backend.core.langgraph_workflow import HybridRAGWorkflow
from backend.core.rrf import deduplicate_results
from backend.graph_retrieval.multi_hop import TraversalStrategy
from backend.services.chroma_service import ChromaService


def test_near_duplicate_chunks_are_removed_in_rank_order():
    results = [
        {"content": "alpha beta gamma delta", "rrf_score": 0.9},
        {"content": "delta gamma beta alpha", "rrf_score": 0.8},
        {"content": "unrelated epsilon evidence", "rrf_score": 0.7},
    ]

    deduplicated = deduplicate_results(results, similarity_threshold=0.95)

    assert [item["content"] for item in deduplicated] == [
        "alpha beta gamma delta",
        "unrelated epsilon evidence",
    ]


def test_stored_embeddings_drive_near_duplicate_filtering_when_available():
    results = [
        {"content": "wording one", "embedding": [1.0, 0.0]},
        {"content": "different wording", "embedding": [0.999, 0.01]},
        {"content": "independent evidence", "embedding": [0.0, 1.0]},
    ]

    deduplicated = deduplicate_results(results, similarity_threshold=0.95)

    assert [item["content"] for item in deduplicated] == [
        "wording one",
        "independent evidence",
    ]


def test_chroma_retrieval_exposes_stored_embeddings_for_internal_dedup():
    service = ChromaService.__new__(ChromaService)
    service.embeddings = MagicMock()
    service.embeddings.embed_query.return_value = [0.1, 0.2]
    collection = MagicMock()
    collection.query.return_value = {
        "documents": [["source chunk"]],
        "metadatas": [[{"chunk_id": "chunk-1"}]],
        "distances": [[0.1]],
        "embeddings": [[[0.3, 0.4]]],
    }
    service.get_or_create_collection = MagicMock(return_value=collection)

    results = asyncio.run(service.similarity_search("query", n_results=25))

    assert "embeddings" in collection.query.call_args.kwargs["include"]
    assert results[0]["embedding"] == [0.3, 0.4]

def test_precision_defaults_retrieve_wide_and_generate_narrow():
    request = QueryRequest(query="How does Kinegraph retrieve context?")

    assert request.candidate_pool_size == 25
    assert request.max_results == 6
    assert request.max_hops == 2


def test_parallel_fetch_uses_candidate_pool_independent_of_generator_top_k():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    workflow.chroma = MagicMock()
    workflow.chroma.similarity_search = AsyncMock(return_value=[])
    workflow.graph_retriever_node = MagicMock()
    workflow.graph_retriever_node.retrieve_chunks = AsyncMock(return_value=[])
    state = {
        "rewritten_query": "kinegraph retrieval",
        "max_results": 6,
        "candidate_pool_size": 25,
        "max_hops": 2,
        "traversal_strategy": TraversalStrategy.BFS,
        "community_id": None,
        "filters": None,
        "latency_breakdown": {},
    }

    asyncio.run(workflow._parallel_fetch(state))

    assert workflow.chroma.similarity_search.await_args.kwargs["n_results"] == 25
    assert workflow.graph_retriever_node.retrieve_chunks.await_args.kwargs["n_results"] == 25
    assert workflow.graph_retriever_node.retrieve_chunks.await_args.kwargs["max_hops"] == 2


def test_opt_in_lexical_channel_enters_weighted_fusion():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    workflow.chroma = MagicMock()
    workflow.chroma.similarity_search = AsyncMock(return_value=[
        {"content": "vector evidence", "score": 0.9, "source": "vector"}
    ])
    workflow.graph_retriever_node = MagicMock()
    workflow.graph_retriever_node.retrieve_chunks = AsyncMock(return_value=[
        {"content": "graph evidence", "score": 0.8, "source": "graph"}
    ])
    state = {
        "rewritten_query": "kinegraph retrieval",
        "max_results": 6,
        "candidate_pool_size": 25,
        "max_hops": 2,
        "traversal_strategy": TraversalStrategy.BFS,
        "community_id": None,
        "filters": None,
        "enable_lexical_fusion": True,
        "latency_breakdown": {},
    }

    with patch("backend.core.langgraph_workflow.VectorlessService") as lexical_cls:
        lexical_cls.return_value.search_chunks.return_value = [
            {"content": "lexical evidence", "score": 0.7, "source": "vectorless"}
        ]
        fetched = asyncio.run(workflow._parallel_fetch(state))

    fetched.update({
        "mode": QueryMode.HYBRID,
        "vector_fusion_weight": 1.0,
        "graph_fusion_weight": 1.2,
        "lexical_fusion_weight": 0.6,
    })
    fused = asyncio.run(workflow._fusion_node(fetched))

    assert fetched["lexical_results"][0]["content"] == "lexical evidence"
    assert fused["fused_results"][0]["content"] == "graph evidence"
    assert fused["fused_results"][0]["rrf_contributions"]["graph"] > 0
