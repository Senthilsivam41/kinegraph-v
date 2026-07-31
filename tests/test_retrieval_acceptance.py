from eval.retrieval_acceptance import (
    assess_cross_encoder_experiment,
    assess_retrieval_benchmark,
)


def _accepted_profile():
    return {
        "ragas_accepted": True,
        "precision_at_5": 0.8,
        "recall_at_5": 0.7,
        "ndcg_at_5": 0.75,
        "context_precision": 0.9,
        "context_recall": 0.7,
        "p95_latency_ms": 500,
        "candidate_provenance_completeness": 1.0,
    }


def test_retrieval_acceptance_requires_all_profiles_and_real_ragas():
    reports = {name: _accepted_profile() for name in (
        "hybrid", "hybrid_lexical", "vectorless"
    )}
    assert assess_retrieval_benchmark(reports)["accepted"] is True

    reports["vectorless"]["ragas_accepted"] = False
    rejected = assess_retrieval_benchmark(reports)
    assert rejected["accepted"] is False
    assert any("RAGAS result is not accepted" in item for item in rejected["failures"])


def test_cross_encoder_gate_requires_one_lever_and_slice_gain():
    baseline = {
        "configuration": {"enable_cross_encoder_reranking": False, "max_hops": 2},
        "p95_latency_ms": 100,
        "slices": {"multi_hop": {"context_precision": 0.8, "ndcg_at_5": 0.7}},
    }
    candidate = {
        "configuration": {"enable_cross_encoder_reranking": True, "max_hops": 2},
        "p95_latency_ms": 130,
        "slices": {"multi_hop": {"context_precision": 0.84, "ndcg_at_5": 0.74}},
    }

    assert assess_cross_encoder_experiment(
        baseline=baseline, candidate=candidate
    )["accepted"] is True

    candidate["configuration"]["max_hops"] = 3
    assert assess_cross_encoder_experiment(
        baseline=baseline, candidate=candidate
    )["accepted"] is False


def test_cross_encoder_gate_accepts_persisted_manifest_shape():
    def manifest(enabled, precision, latency):
        return {
            "pipeline_config": {
                "reranking": {"enable_cross_encoder_reranking": enabled},
            },
            "report": {
                "per_category": {
                    "multi_hop": {
                        "metrics": {"context_precision": precision},
                        "retrieval_metrics": {"ndcg_at_5": precision},
                    }
                },
                "retrieval_diagnostics": {"p95_latency_ms": latency},
            },
        }

    result = assess_cross_encoder_experiment(
        baseline=manifest(False, 0.80, 100),
        candidate=manifest(True, 0.84, 130),
    )

    assert result["accepted"] is True
    assert result["changed_levers"] == [
        "reranking.enable_cross_encoder_reranking"
    ]
