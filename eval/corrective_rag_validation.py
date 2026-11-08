"""Use case that checks whether corrective RAG beats weak term-frequency retrieval.

The question is a parental-leave policy lookup. A repeated-token bereavement
policy shares "paid leave" and "caregiver" and outranks the real policy under
raw term frequency. Corrective RAG treats the missing term "weeks" as
distinctive, drops the distractor, and — when the real policy was never
retrieved — searches once more for that gap.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from backend.core.corrective_rag import apply_corrective_rag, tokenize
from eval.ir_metrics import score_retrieval


PARENTAL_LEAVE_QUERY = "How many weeks of paid leave does a primary caregiver receive?"

PARENTAL_LEAVE_POLICY = (
    "Primary caregivers receive 16 weeks of paid parental leave after the "
    "birth or adoption of a child."
)

PARENTAL_LEAVE_CORPUS: list[dict[str, Any]] = [
    {
        "id": "parental-leave",
        "content": PARENTAL_LEAVE_POLICY,
        "metadata": {"file_name": "leave-policy.md"},
        "source": "corpus",
    },
    {
        "id": "bereavement-leave",
        "content": (
            "A primary caregiver may receive paid leave. A primary caregiver "
            "may receive paid leave when supporting family. Paid leave requests "
            "from a caregiver receive manager approval. The caregiver receives "
            "paid leave documentation within five days."
        ),
        "metadata": {"file_name": "bereavement-policy.md"},
        "source": "corpus",
    },
    {
        "id": "parking-access",
        "content": (
            "Primary badge holders receive paid parking access. Paid parking "
            "is billed every thirty days."
        ),
        "metadata": {"file_name": "workplace.md"},
        "source": "corpus",
    },
    {
        "id": "project-timeline",
        "content": (
            "The project timeline spans 12 weeks of paid contractor work and "
            "does not describe employee benefits."
        ),
        "metadata": {"file_name": "delivery-plan.md"},
        "source": "corpus",
    },
    {
        "id": "remote-stipend",
        "content": (
            "Remote employees receive a home office stipend. Caregivers may "
            "request a flexible schedule during school holidays."
        ),
        "metadata": {"file_name": "remote-work.md"},
        "source": "corpus",
    },
]


def weak_term_frequency_retrieve(
    query: str,
    corpus: Sequence[Mapping[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    """Rank by how often query terms are repeated, which rewards padded distractors."""
    query_terms = list(dict.fromkeys(tokenize(query)))
    scored: list[tuple[int, dict[str, Any]]] = []
    for document in corpus:
        counts: dict[str, int] = {}
        for token in tokenize(str(document.get("content") or "")):
            counts[token] = counts.get(token, 0) + 1
        score = sum(counts.get(term, 0) for term in query_terms)
        scored.append((score, dict(document)))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("id"))))
    return [document for score, document in scored[:k] if score > 0]


def _follow_up_retrieve(corrective_query: str) -> list[dict[str, Any]]:
    """Return corpus rows that contain the gap term appended by the rewrite."""
    tokens = tokenize(corrective_query)
    required = tokens[-1] if tokens else ""
    hits = []
    for document in PARENTAL_LEAVE_CORPUS:
        if required and required in set(tokenize(str(document["content"]))):
            hits.append(dict(document))
    return hits


def _bundle_metrics(contexts: Sequence[str], k: int) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for probe in sorted({1, k}):
        metrics.update(score_retrieval(
            retrieved_contexts=contexts,
            reference_contexts=[PARENTAL_LEAVE_POLICY],
            k=probe,
        ))
    return metrics


def _verdict(baseline: Mapping[str, float], corrected: Mapping[str, float]) -> dict[str, Any]:
    deltas = {
        key: round(float(corrected[key]) - float(baseline[key]), 4)
        for key in baseline
    }
    improved_keys = [key for key, delta in deltas.items() if delta > 0.01]
    regressed_keys = [key for key, delta in deltas.items() if delta < -0.01]
    if improved_keys and not regressed_keys:
        verdict = "improved"
    elif regressed_keys and not improved_keys:
        verdict = "regressed"
    elif improved_keys and regressed_keys:
        verdict = "mixed"
    else:
        verdict = "unchanged"
    return {
        "deltas": deltas,
        "improved_metrics": improved_keys,
        "regressed_metrics": regressed_keys,
        "verdict": verdict,
        "improved": verdict == "improved",
    }


def _case(
    *,
    name: str,
    description: str,
    baseline: Sequence[Mapping[str, Any]],
    corrected: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    k: int,
) -> dict[str, Any]:
    baseline_texts = [str(item.get("content") or "") for item in baseline]
    corrected_texts = [str(item.get("content") or "") for item in corrected]
    baseline_metrics = _bundle_metrics(baseline_texts, k)
    corrected_metrics = _bundle_metrics(corrected_texts, k)
    comparison = _verdict(baseline_metrics, corrected_metrics)
    return {
        "use_case": name,
        "description": description,
        "query": PARENTAL_LEAVE_QUERY,
        "reference": PARENTAL_LEAVE_POLICY,
        "baseline_ids": [str(item.get("id")) for item in baseline],
        "corrected_ids": [str(item.get("id")) for item in corrected],
        "baseline_metrics": baseline_metrics,
        "corrected_metrics": corrected_metrics,
        "corrective_rag": {
            "action": report.get("action"),
            "distinctive_terms": report.get("distinctive_terms"),
            "corrective_query": report.get("corrective_query"),
            "needs_supplementary": report.get("needs_supplementary"),
        },
        **comparison,
    }


def run_distractor_validation(k: int = 3) -> dict[str, Any]:
    """The true policy is retrieved, but a padded bereavement policy ranks first."""
    baseline = weak_term_frequency_retrieve(PARENTAL_LEAVE_QUERY, PARENTAL_LEAVE_CORPUS, k)
    corrected, report = apply_corrective_rag(PARENTAL_LEAVE_QUERY, baseline, top_k=k)
    return _case(
        name="parental_leave_distractor",
        description=(
            "Term frequency ranks a repeated bereavement policy ahead of the "
            "16-week parental leave policy. Corrective RAG drops that distractor."
        ),
        baseline=baseline,
        corrected=corrected,
        report=report,
        k=k,
    )


def run_missed_context_validation() -> dict[str, Any]:
    """Top-1 retrieval misses the policy, and one corrective search brings it back."""
    baseline = weak_term_frequency_retrieve(PARENTAL_LEAVE_QUERY, PARENTAL_LEAVE_CORPUS, 1)
    corrected, report = apply_corrective_rag(
        PARENTAL_LEAVE_QUERY,
        baseline,
        top_k=3,
        retrieve=_follow_up_retrieve,
    )
    return _case(
        name="parental_leave_missed_context",
        description=(
            "The first hit is only the bereavement policy. Corrective RAG "
            "searches again for the missing term and keeps the parental leave policy."
        ),
        baseline=baseline,
        corrected=corrected,
        report=report,
        k=1,
    )


def run_parental_leave_validation() -> dict[str, Any]:
    cases = [run_distractor_validation(), run_missed_context_validation()]
    return {
        "use_case": "parental_leave",
        "query": PARENTAL_LEAVE_QUERY,
        "improved": all(case["improved"] for case in cases),
        "cases": cases,
    }


def main() -> None:
    report = run_parental_leave_validation()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["improved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
