from unittest.mock import MagicMock

import pytest

from backend.app.models import QueryRequest
from backend.graph_retrieval.multi_hop import MultiHopGraphRetriever, TraversalStrategy


def _node(node_id, community="community-a", centrality=0.5):
    return {
        "element_id": node_id,
        "node_id": node_id,
        "properties": {
            "id": node_id,
            "name": node_id,
            "description": f"Description for {node_id}",
            "community_id": community,
            "centrality_score": centrality,
        },
    }


class FakeTraversal(MultiHopGraphRetriever):
    def __init__(self, adjacency, *args, **kwargs):
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = MagicMock()
        super().__init__(driver, *args, **kwargs)
        self.adjacency = adjacency
        self.communities_seen = []

    def _find_seeds(self, session, query, seed_node_ids, limit, community_id):
        return [_node("A")]

    def _neighbors(self, session, element_id, community_id):
        self.communities_seen.append(community_id)
        neighbors = self.adjacency.get(element_id, [])
        if community_id:
            neighbors = [n for n in neighbors if n["properties"]["community_id"] == community_id]
        return neighbors


def _edge(target, relationship="RELATES_TO", community="community-a", weight=0.8):
    return {
        **_node(target, community=community),
        "relationship_type": relationship,
        "weight": weight,
        "evidence_text": f"Evidence leading to {target}",
        "direction": "OUTGOING",
    }


def test_three_hop_bfs_returns_relationship_paths_and_depth():
    retriever = FakeTraversal({
        "A": [_edge("B")],
        "B": [_edge("C")],
        "C": [_edge("D")],
    }, max_hops=3)

    results = retriever.retrieve("A relationships", n_results=10, strategy="bfs")

    assert [r["metadata"]["traversal_depth"] for r in results] == [1, 2, 3]
    assert results[-1]["metadata"]["max_hops"] == 3
    assert len(results[-1]["metadata"]["relationship_path"]) == 3
    assert "A -[RELATES_TO]-> B" in results[-1]["content"]
    assert results[-1]["source"] == "graph_traversal"


def test_bfs_and_dfs_have_distinct_ordering():
    adjacency = {
        "A": [_edge("B"), _edge("C")],
        "B": [_edge("D")],
        "C": [_edge("E")],
    }
    bfs = FakeTraversal(adjacency).retrieve("A", n_results=5, strategy=TraversalStrategy.BFS)
    dfs = FakeTraversal(adjacency).retrieve("A", n_results=5, strategy=TraversalStrategy.DFS)

    assert [r["metadata"]["id"] for r in bfs] == ["B", "C", "D", "E"]
    assert [r["metadata"]["id"] for r in dfs] == ["B", "D", "C", "E"]


def test_community_strategy_stays_inside_seed_community():
    retriever = FakeTraversal({
        "A": [_edge("B", community="community-a"), _edge("X", community="community-b")],
        "B": [_edge("C", community="community-a")],
    })

    results = retriever.retrieve("A", n_results=5, strategy="community")

    assert [r["metadata"]["id"] for r in results] == ["B", "C"]
    assert all(r["metadata"]["community_id"] == "community-a" for r in results)
    assert all(value == "community-a" for value in retriever.communities_seen)


def test_max_hops_is_exposed_and_validated_by_api_model():
    request = QueryRequest(query="trace dependencies", max_hops=5, traversal_strategy="dfs")
    assert request.max_hops == 5
    assert request.traversal_strategy == TraversalStrategy.DFS

    with pytest.raises(ValueError):
        QueryRequest(query="invalid", max_hops=6)
    with pytest.raises(ValueError, match="max_hops"):
        MultiHopGraphRetriever(MagicMock(), max_hops=0)
