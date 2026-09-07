"""Evidence-only response policy and shadow Kinetic Score for ADR-004."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from backend.core.retrieval_orchestration import candidate_identity


VERIFICATION_POLICY_VERSION = "kinegraph.verification.v1"
KINETIC_SCORE_POLICY_VERSION = "kinegraph.kinetic-score.v1"
KINETIC_SCORE_CALIBRATION_VERSION = "kinegraph.kinetic-score-calibration.v1"

_SCORE_WEIGHTS = {
    "evidence_coverage": 0.20,
    "citation_validity": 0.25,
    "verification_success": 0.20,
    "answer_relevance": 0.15,
    "reranking_quality": 0.10,
    "source_diversity": 0.05,
    "metadata_link_consistency": 0.05,
}


def _bounded(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return default


def _explicit_conflicts(contexts: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Report only conflict metadata emitted by retrieval; never infer conflicts."""
    conflicts = []
    for context in contexts:
        metadata = context.get("metadata") or {}
        conflict_flag = any(
            bool(context.get(key, metadata.get(key)))
            for key in ("conflict", "conflicting", "conflict_flag", "conflicts_with")
        )
        if conflict_flag:
            conflicts.append({
                "candidate_id": candidate_identity(context),
                "conflicts_with": context.get(
                    "conflicts_with", metadata.get("conflicts_with")
                ),
            })
    return conflicts


def build_verification_outcome(
    *,
    claims: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
    citation_validation: Mapping[str, Any],
    grounding_critique: Mapping[str, Any],
    answer_relevancy: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose verified, partial, or refusal using only observable evidence state."""
    missing_facets = [
        str(facet).strip()
        for facet in answer_relevancy.get("missing_question_facets", [])
        if str(facet).strip()
    ]
    coverage = str(answer_relevancy.get("question_coverage") or "unverified").lower()
    critique_completed = bool(grounding_critique.get("completed"))
    conflicts = _explicit_conflicts(contexts)
    gaps = list(dict.fromkeys(missing_facets))

    if not citation_validation.get("structured_output_valid", False):
        status, reason = "refused", "citation_validation_unavailable"
    elif not contexts:
        status, reason = "refused", "no_retrieved_evidence"
    elif not claims:
        status, reason = "refused", "no_supported_claims"
    elif not critique_completed:
        status, reason = "refused", "semantic_verification_unavailable"
    elif coverage == "complete" and critique_completed and not conflicts:
        status, reason = "verified", "all_returned_claims_verified"
    else:
        status, reason = "partial", "bounded_answer_from_incomplete_evidence"

    if claims and not critique_completed:
        gaps.append("semantic grounding critique was not completed")
    if coverage in {"none", "partial"} and not gaps:
        gaps.append("the retrieved evidence does not cover every question facet")
    if conflicts:
        gaps.append("retrieved evidence contains an explicit conflict marker")
    if not citation_validation.get("structured_output_valid", False):
        gaps.append("structured citation validation did not complete")

    return {
        "policy_version": VERIFICATION_POLICY_VERSION,
        "status": status,
        "reason": reason,
        "evidence_confidence_only": True,
        "supported_claim_ids": [str(claim.get("claim_id")) for claim in claims],
        "citation_rejections": list(citation_validation.get("rejected_claims") or []),
        "unsupported_claim_ids": list(
            grounding_critique.get("removed_unsupported_claim_ids") or []
        ),
        "irrelevant_claim_ids": list(
            grounding_critique.get("removed_irrelevant_claim_ids") or []
        ),
        "question_coverage": coverage,
        "missing_question_facets": missing_facets,
        "gaps": list(dict.fromkeys(gaps)),
        "conflicts": conflicts,
    }


def apply_response_policy(answer: str, outcome: Mapping[str, Any]) -> str:
    """Render a bounded answer without adding any factual claim."""
    status = outcome.get("status")
    if status == "refused":
        if outcome.get("reason") == "semantic_verification_unavailable":
            return (
                "I cannot return an answer because semantic grounding verification "
                "did not complete."
            )
        return (
            "I cannot answer from the retrieved evidence because no supported "
            "claim is available."
        )
    if status != "partial":
        return answer
    gaps = [str(gap).strip() for gap in outcome.get("gaps", []) if str(gap).strip()]
    suffix = "; ".join(gaps) if gaps else "some question facets remain unverified"
    return f"{answer}\n\nEvidence gaps: {suffix}."


def _source_diversity(contexts: Sequence[Mapping[str, Any]]) -> float:
    if not contexts:
        return 0.0
    source_ids = set()
    for context in contexts:
        metadata = context.get("metadata") or {}
        source_id = next((
            context.get(key, metadata.get(key))
            for key in ("document_id", "source_document", "source_id", "file_name")
            if context.get(key, metadata.get(key)) not in (None, "")
        ), None)
        if source_id is None:
            channels = context.get("source_channels") or [context.get("source", "unknown")]
            source_id = "channel:" + ",".join(sorted(str(item) for item in channels))
        source_ids.add(str(source_id))
    return _bounded(len(source_ids) / len(contexts))


def _metadata_link_consistency(contexts: Sequence[Mapping[str, Any]]) -> float:
    if not contexts:
        return 0.0
    complete = 0
    for context in contexts:
        metadata = context.get("metadata") or {}
        channels = list(context.get("source_channels") or [])
        has_core = bool(
            candidate_identity(context)
            and channels
            and context.get("original_scores")
            and context.get("channel_ranks")
        )
        try:
            traversal_depth = int(metadata.get("traversal_depth") or 0)
        except (TypeError, ValueError):
            traversal_depth = 0
        graph_path_required = bool(
            "graph" in channels
            and (context.get("source") == "graph_traversal" or traversal_depth > 0)
        )
        graph_complete = not graph_path_required or bool(context.get("graph_paths"))
        if has_core and graph_complete:
            complete += 1
    return complete / len(contexts)


def compute_kinetic_score(
    *,
    outcome: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
    citation_validation: Mapping[str, Any],
    grounding_critique: Mapping[str, Any],
    answer_relevancy: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute a versioned 0–100 shadow score from observable pipeline signals."""
    total_claims = int(citation_validation.get("total_claims") or 0)
    accepted_claims = int(citation_validation.get("accepted_claims") or 0)
    citation_validity = accepted_claims / total_claims if total_claims else 0.0

    critique_completed = bool(grounding_critique.get("completed"))
    retained_count = len(claims)
    verification_success = (
        retained_count / accepted_claims
        if critique_completed and accepted_claims
        else 0.0
    )
    irrelevant_count = len(
        grounding_critique.get("removed_irrelevant_claim_ids") or []
    )
    relevant_total = retained_count + irrelevant_count
    answer_relevance = (
        retained_count / relevant_total
        if critique_completed and relevant_total
        else 0.0
    )
    coverage_map = {"complete": 1.0, "partial": 0.5, "none": 0.0}
    evidence_coverage = coverage_map.get(
        str(answer_relevancy.get("question_coverage") or "unverified"),
        0.0,
    )
    semantic_scores = [
        _bounded(context.get("semantic_score"))
        for context in contexts
        if context.get("semantic_score") is not None
    ]
    reranking_quality = (
        sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
    )
    components = {
        "evidence_coverage": evidence_coverage,
        "citation_validity": citation_validity,
        "verification_success": verification_success,
        "answer_relevance": answer_relevance,
        "reranking_quality": reranking_quality,
        "source_diversity": _source_diversity(contexts),
        "metadata_link_consistency": _metadata_link_consistency(contexts),
    }
    weighted = sum(
        _SCORE_WEIGHTS[name] * _bounded(value)
        for name, value in components.items()
    )
    conflicts = len(outcome.get("conflicts") or [])
    missing_facets = len(outcome.get("missing_question_facets") or [])
    penalties = {
        "explicit_conflicts": min(conflicts / 3.0, 1.0) * 0.20,
        "missing_question_facets": min(missing_facets / 3.0, 1.0) * 0.15,
    }
    score = round(100.0 * max(0.0, min(weighted - sum(penalties.values()), 1.0)), 2)
    return {
        "policy_version": KINETIC_SCORE_POLICY_VERSION,
        "mode": "shadow",
        "kinetic_score": score,
        "evidence_confidence_only": True,
        "disclaimer": "This score measures evidence support, not factual correctness.",
        "response_status": outcome.get("status"),
        "components": {name: round(_bounded(value), 4) for name, value in components.items()},
        "weights": dict(_SCORE_WEIGHTS),
        "penalties": {name: round(value, 4) for name, value in penalties.items()},
        "calibration": {
            "policy_version": KINETIC_SCORE_CALIBRATION_VERSION,
            "status": "uncalibrated_shadow",
            "promotion_allowed": False,
        },
    }


def calibrate_kinetic_score(
    records: Sequence[Mapping[str, Any]],
    *,
    minimum_samples: int = 20,
) -> dict[str, Any]:
    """Fit an acceptance threshold to human labels without changing runtime policy."""
    samples = []
    for record in records:
        try:
            score = float(record["kinetic_score"])
            acceptable = record["acceptable"]
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(acceptable, bool) and 0.0 <= score <= 100.0:
            samples.append((score, acceptable))
    positive_samples = sum(label for _, label in samples)
    negative_samples = len(samples) - positive_samples
    if len(samples) < minimum_samples or not {label for _, label in samples} == {False, True}:
        return {
            "policy_version": KINETIC_SCORE_CALIBRATION_VERSION,
            "status": "insufficient_labeled_samples",
            "samples": len(samples),
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
            "minimum_samples": minimum_samples,
            "promotion_allowed": False,
        }

    candidates = sorted({score for score, _ in samples})
    best = None
    for threshold in candidates:
        tp = sum(score >= threshold and label for score, label in samples)
        fp = sum(score >= threshold and not label for score, label in samples)
        tn = sum(score < threshold and not label for score, label in samples)
        fn = sum(score < threshold and label for score, label in samples)
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        balanced_accuracy = (sensitivity + specificity) / 2.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        candidate = (balanced_accuracy, precision, threshold, sensitivity, specificity)
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    balanced_accuracy, precision, threshold, sensitivity, specificity = best
    return {
        "policy_version": KINETIC_SCORE_CALIBRATION_VERSION,
        "status": "calibrated",
        "samples": len(samples),
        "positive_samples": positive_samples,
        "negative_samples": negative_samples,
        "threshold": threshold,
        "balanced_accuracy": round(balanced_accuracy, 4),
        "precision": round(precision, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "promotion_allowed": False,
        "note": "Benchmark governance must approve promotion separately.",
    }
