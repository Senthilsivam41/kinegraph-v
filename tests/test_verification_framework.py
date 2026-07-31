from backend.core.verification import (
    KINETIC_SCORE_POLICY_VERSION,
    apply_response_policy,
    build_verification_outcome,
    calibrate_kinetic_score,
    compute_kinetic_score,
)


def _validated_claim_state():
    return {
        "claims": [{"claim_id": "claim-1", "text": "Supported", "chunk_ids": ["c-1"]}],
        "contexts": [{
            "candidate_id": "c-1",
            "content": "Supported",
            "semantic_score": 0.9,
            "source_channels": ["vector"],
            "original_scores": {"vector": 0.8},
            "channel_ranks": {"vector": 1},
        }],
        "citation_validation": {
            "structured_output_valid": True,
            "total_claims": 1,
            "accepted_claims": 1,
            "rejected_claims": [],
        },
        "grounding_critique": {
            "completed": True,
            "removed_unsupported_claim_ids": [],
            "removed_irrelevant_claim_ids": [],
        },
        "answer_relevancy": {
            "question_coverage": "complete",
            "missing_question_facets": [],
        },
    }


def test_complete_verified_evidence_returns_answer_unchanged():
    state = _validated_claim_state()
    outcome = build_verification_outcome(**state)

    assert outcome["status"] == "verified"
    assert apply_response_policy("Supported [c-1]", outcome) == "Supported [c-1]"


def test_missing_facet_produces_bounded_partial_answer():
    state = _validated_claim_state()
    state["answer_relevancy"] = {
        "question_coverage": "partial",
        "missing_question_facets": ["deployment procedure"],
    }
    outcome = build_verification_outcome(**state)

    answer = apply_response_policy("Supported [c-1]", outcome)
    assert outcome["status"] == "partial"
    assert "Supported [c-1]" in answer
    assert "deployment procedure" in answer


def test_no_supported_claims_forces_refusal_even_when_context_exists():
    state = _validated_claim_state()
    state["claims"] = []
    outcome = build_verification_outcome(**state)

    assert outcome["status"] == "refused"
    assert "cannot answer from the retrieved evidence" in apply_response_policy("", outcome)


def test_missing_semantic_verification_fails_closed():
    state = _validated_claim_state()
    state["grounding_critique"] = {"completed": False, "reason": "critic_failed"}
    outcome = build_verification_outcome(**state)

    assert outcome["status"] == "refused"
    assert outcome["reason"] == "semantic_verification_unavailable"


def test_kinetic_score_is_versioned_shadow_evidence_confidence():
    state = _validated_claim_state()
    outcome = build_verification_outcome(**state)
    score = compute_kinetic_score(outcome=outcome, **state)

    assert score["policy_version"] == KINETIC_SCORE_POLICY_VERSION
    assert 0 <= score["kinetic_score"] <= 100
    assert score["mode"] == "shadow"
    assert score["calibration"]["promotion_allowed"] is False


def test_score_calibration_requires_labeled_sample_floor():
    insufficient = calibrate_kinetic_score(
        [{"kinetic_score": 80, "acceptable": True}], minimum_samples=2
    )
    calibrated = calibrate_kinetic_score([
        {"kinetic_score": 20, "acceptable": False},
        {"kinetic_score": 80, "acceptable": True},
    ], minimum_samples=2)

    assert insufficient["status"] == "insufficient_labeled_samples"
    assert calibrated["status"] == "calibrated"
    assert calibrated["promotion_allowed"] is False
