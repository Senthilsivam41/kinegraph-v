"""Shadow-mode Kinetic Score calibration helpers (ADR-004 prep)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


SCORE_POLICY_VERSION = "kinegraph.kinetic-score.shadow.v1"


def compute_kinetic_score_shadow(
    *,
    ragas_scores: Mapping[str, Any],
    ir_metrics: Mapping[str, Any],
    citation_validity: float | None = None,
    path_completeness: float | None = None,
) -> dict[str, Any]:
    """
    Compose a 0–100 shadow Kinetic Score from observable components.

    This never overrides refusals and is not exposed as correctness.
    """
    faithfulness = float(ragas_scores.get("faithfulness") or 0.0)
    context_precision = float(ragas_scores.get("context_precision") or 0.0)
    context_recall = float(ragas_scores.get("context_recall") or 0.0)
    answer_relevancy = float(ragas_scores.get("answer_relevancy") or 0.0)
    precision_at_k = float(ir_metrics.get("precision_at_5") or ir_metrics.get("precision_at_k") or 0.0)
    recall_at_k = float(ir_metrics.get("recall_at_5") or ir_metrics.get("recall_at_k") or 0.0)
    citation = 0.0 if citation_validity is None else max(0.0, min(1.0, citation_validity))
    path = 0.0 if path_completeness is None else max(0.0, min(1.0, path_completeness))

    components = {
        "evidence_coverage": 0.30 * context_recall + 0.20 * recall_at_k,
        "retrieval_precision": 0.25 * context_precision + 0.25 * precision_at_k,
        "groundedness": 0.60 * faithfulness + 0.40 * citation,
        "answer_focus": answer_relevancy,
        "graph_path_completeness": path,
    }
    weighted = (
        0.30 * components["evidence_coverage"]
        + 0.25 * components["retrieval_precision"]
        + 0.25 * components["groundedness"]
        + 0.15 * components["answer_focus"]
        + 0.05 * components["graph_path_completeness"]
    )
    score = round(100.0 * max(0.0, min(1.0, weighted)), 2)
    return {
        "policy_version": SCORE_POLICY_VERSION,
        "mode": "shadow",
        "kinetic_score": score,
        "components": {key: round(value, 4) for key, value in components.items()},
        "inputs": {
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_relevancy": answer_relevancy,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "citation_validity": citation_validity,
            "path_completeness": path_completeness,
        },
    }


def calibrate_shadow_scores(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [
        float((record.get("kinetic_score_shadow") or {}).get("kinetic_score") or 0.0)
        for record in records
        if isinstance(record.get("kinetic_score_shadow"), Mapping)
    ]
    if not scores:
        return {
            "policy_version": SCORE_POLICY_VERSION,
            "samples": 0,
            "mean": None,
            "min": None,
            "max": None,
        }
    return {
        "policy_version": SCORE_POLICY_VERSION,
        "samples": len(scores),
        "mean": round(sum(scores) / len(scores), 2),
        "min": round(min(scores), 2),
        "max": round(max(scores), 2),
    }
