import json

import pytest

from eval.provenance import (
    SCHEMA_VERSION,
    build_provenance_record,
    diagnostic_summary,
    write_provenance_jsonl,
)


SCORES = {
    "faithfulness": 0.8,
    "answer_relevancy": 0.7,
    "context_precision": 0.9,
    "context_recall": 0.7,
    "answer_correctness": 0.7,
    "ragas_failed": False,
    "ragas_error": None,
}
PROFILE = {
    "name": "hybrid",
    "requested_mode": "hybrid",
    "declared_effective_modes": ["hybrid"],
}
SAMPLE = {
    "sample_id": "benchmark-001",
    "question": "What is RRF?",
    "categories": ["single_hop"],
}


def _result(**overrides):
    candidate = {
        "content": "RRF evidence OPENAI_API_KEY=sk-secretvalue",
        "score": 0.9,
        "source": "vector",
        "metadata": {"chunk_id": "chunk-1", "file_name": "guide.md"},
        "embedding": [0.1, 0.2],
    }
    result = {
        "answer": "RRF uses evidence sk-secretvalue.",
        "confidence": 0.9,
        "grounded_claims": [{"claim_id": "claim-1", "chunk_ids": ["chunk-1"]}],
        "citation_validation": {"structured_output_valid": True},
        "grounding_critique": {"completed": True},
        "answer_relevancy": {"question_coverage": "complete"},
        "trace": {
            "requested_mode": "hybrid",
            "effective_mode": "hybrid",
            "rewritten_query": "reciprocal rank fusion",
            "routing": {"facets": ["What is RRF?"], "decision": "hybrid retained"},
            "initial_candidates": {"vector": [candidate], "graph": [], "lexical": []},
            "channel_candidates": {"vector": [candidate], "graph": [], "lexical": [], "vectorless": []},
            "retrieval_failures": {},
            "recovery": {"structured_recovery_used": False},
            "fusion": {"candidates": [candidate], "deduplication": {"removed_candidate_ids": []}},
            "reranking": {"candidates": [candidate], "mode": "cross_encoder", "fallback": False},
            "final_contexts": [candidate],
            "latency_ms": {"retrieval": 10.0},
        },
    }
    result.update(overrides)
    return result


def _record(result=None, scores=None, workflow_error=None):
    return build_provenance_record(
        sample=SAMPLE,
        result=result or _result(),
        scores=scores or SCORES,
        profile=PROFILE,
        workflow_error=workflow_error,
        workflow_latency_ms=20.0,
        eval_latency_ms=30.0,
        judge_model="judge-model",
        embedding_model="embedding-model",
    )


def test_provenance_is_versioned_reconstructable_and_redacts_secrets():
    record = _record()

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["profile"]["effective_mode"] == "hybrid"
    assert record["retrieval"]["initial_candidates"]["vector"][0]["candidate_id"] == "chunk-1"
    assert record["retrieval"]["initial_candidates"]["vector"][0]["embedding_present"] is True
    assert "sk-secretvalue" not in json.dumps(record)
    assert record["failure"]["first_stage"] is None
    assert record["retrieval"]["candidate_lifecycle"][0]["sent_to_generation"] is True


def test_graph_paths_and_retriever_behavior_are_audited():
    result = _result()
    graph_candidate = {
        "content": "A uses B",
        "source": "graph_traversal",
        "score": 0.8,
        "metadata": {
            "chunk_id": "graph-1",
            "seed_node_id": "A",
            "traversal_depth": 1,
            "max_hops": 2,
            "traversal_strategy": "bfs",
            "relationship_path": [{
                "from_node_id": "A",
                "to_node_id": "B",
                "relationship_type": "USES",
                "direction": "OUTGOING",
                "weight": 0.9,
                "evidence_text": "A uses B",
            }],
        },
    }
    result["trace"]["channel_candidates"]["graph"] = [graph_candidate]
    result["trace"]["graph_retrieval_diagnostics"] = {
        "seed_count": 1,
        "cycle_prevention_count": 2,
        "missing_evidence_edge_count": 0,
    }

    record = _record(result=result)
    audit = record["retrieval"]["graph_path_audit"]

    assert audit["all_paths_complete"] is True
    assert audit["paths"][0]["seed_node_id"] == "A"
    assert audit["paths"][0]["cycle_detected"] is False
    assert audit["retriever_diagnostics"]["cycle_prevention_count"] == 2


@pytest.mark.parametrize(
    ("result", "scores", "workflow_error", "expected_stage"),
    [
        ({}, SCORES, "Neo4j unavailable", "workflow"),
        (_result(trace={**_result()["trace"], "retrieval_failures": {"graph": "down"}}), SCORES, None, "retrieval_channel"),
        (_result(trace={**_result()["trace"], "final_contexts": []}), SCORES, None, "empty_retrieval"),
        (_result(trace={**_result()["trace"], "reranking": {"candidates": [], "fallback": True}}), SCORES, None, "reranker_fallback"),
        (_result(grounding_critique={"reason": "critic_failed"}), SCORES, None, "grounding_critic"),
        (_result(), {**SCORES, "ragas_failed": True, "ragas_error": "judge down"}, None, "ragas_judge"),
    ],
)
def test_offline_failure_fixtures_are_schema_valid(result, scores, workflow_error, expected_stage):
    record = _record(result=result, scores=scores, workflow_error=workflow_error)

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["failure"]["first_stage"] == expected_stage


def test_profile_escape_is_recorded_and_grouped_as_failure():
    result = _result()
    result["trace"]["effective_mode"] = "vector"

    record = _record(result=result)

    assert record["profile"]["valid"] is False
    assert record["failure"]["first_stage"] == "mode_profile"


def test_jsonl_is_deterministic_and_diagnostics_group_failures(tmp_path):
    first = _record()
    second = _record(
        scores={**SCORES, "ragas_failed": True, "ragas_error": "judge down"},
    )
    second["sample_id"] = "benchmark-002"
    destination = tmp_path / "trace.jsonl"

    write_provenance_jsonl(destination, [second, first])
    first_payload = destination.read_text()
    write_provenance_jsonl(destination, [first, second])

    assert destination.read_text() == first_payload
    assert [json.loads(line)["sample_id"] for line in first_payload.splitlines()] == [
        "benchmark-001",
        "benchmark-002",
    ]
    summary = diagnostic_summary([first, second])
    assert summary["total_records"] == 2
    assert summary["ragas_failed_count"] == 1
    assert summary["first_failure_stage_counts"] == {"ragas_judge": 1}
