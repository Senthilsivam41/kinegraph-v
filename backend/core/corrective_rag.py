"""Corrective RAG: grade retrieved context and replace what does not answer the query.

The evaluator is lexical and deterministic. It treats query terms that the
initial hit list mostly missed as distinctive, then labels each chunk:

- correct: covers the question and every distinctive term
- ambiguous: covers part of the question, including a distinctive term
- incorrect: misses the distinctive terms, even when generic tokens overlap

Incorrect chunks are dropped. Ambiguous chunks are reduced to the sentences
that carry those terms. When nothing correct survives, the caller can run one
follow-up retrieval with a query that repeats the missing terms.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from backend.core.retrieval_orchestration import candidate_identity


CORRECTIVE_RAG_POLICY_VERSION = "kinegraph.corrective-rag.v1"

_TOKEN = re.compile(r"[a-z0-9]+")
_STRIP = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = frozenset({
    "about", "after", "also", "and", "are", "before", "but", "can", "does",
    "during", "for", "from", "has", "have", "her", "his", "how", "into", "its",
    "many", "may", "not", "our", "over", "per", "than", "that", "the", "their",
    "them", "then", "they", "this", "under", "was", "were", "what", "when",
    "where", "which", "who", "with", "within", "you", "your",
})


@dataclass(frozen=True)
class RetrievalGrade:
    candidate_id: str
    label: str
    score: float
    coverage: float
    distinctive_coverage: float
    origin: str


def tokenize(text: str) -> list[str]:
    """Return content tokens, with a light plural fold so caregiver matches caregivers."""
    tokens: list[str] = []
    for token in _TOKEN.findall(str(text).lower()):
        if len(token) <= 2 or token in _STOPWORDS:
            continue
        if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
            token = token[:-1]
        tokens.append(token)
    return tokens


def _term_set(text: str) -> set[str]:
    return set(tokenize(text))


def distinctive_terms(query_terms: set[str], documents: Sequence[str]) -> set[str]:
    """Query terms the initial hit list mostly failed to contain."""
    if not query_terms:
        return set()
    if not documents:
        return set(query_terms)
    counts: Counter[str] = Counter()
    for document in documents:
        counts.update(_term_set(document) & query_terms)
    total = float(len(documents))
    return {term for term in query_terms if counts[term] / total < 0.5}


def _coverage(present: set[str], required: set[str]) -> float:
    if not required:
        return 0.0
    return len(present & required) / float(len(required))


def label_relevance(
    *,
    coverage: float,
    distinctive_coverage: float,
    has_distinctive: bool,
) -> str:
    if has_distinctive:
        if distinctive_coverage == 1.0 and coverage >= 0.5:
            return "correct"
        if distinctive_coverage > 0.0 and coverage >= 0.5:
            return "ambiguous"
        return "incorrect"
    if coverage >= 0.67:
        return "correct"
    if coverage >= 0.4:
        return "ambiguous"
    return "incorrect"


def _grade_text(
    *,
    candidate_id: str,
    text: str,
    query_terms: set[str],
    distinctive: set[str],
    origin: str,
) -> RetrievalGrade:
    present = _term_set(text)
    coverage = _coverage(present, query_terms)
    distinctive_coverage = _coverage(present, distinctive) if distinctive else 0.0
    label = label_relevance(
        coverage=coverage,
        distinctive_coverage=distinctive_coverage,
        has_distinctive=bool(distinctive),
    )
    if distinctive:
        score = (0.25 * coverage) + (0.75 * distinctive_coverage)
    else:
        score = coverage
    return RetrievalGrade(
        candidate_id=candidate_id,
        label=label,
        score=round(score, 4),
        coverage=round(coverage, 4),
        distinctive_coverage=round(distinctive_coverage, 4),
        origin=origin,
    )


def split_knowledge_strips(text: str) -> list[str]:
    parts = _STRIP.split(str(text).strip())
    return [part.strip() for part in parts if part.strip()]


def refine_knowledge(text: str, query_terms: set[str], distinctive: set[str]) -> str:
    """Keep sentences that carry the terms the retrieval evaluator trusted."""
    strips = split_knowledge_strips(text)
    if len(strips) <= 1:
        return str(text).strip()
    kept: list[str] = []
    for strip in strips:
        tokens = _term_set(strip)
        if distinctive and tokens & distinctive:
            kept.append(strip)
        elif not distinctive and tokens & query_terms:
            kept.append(strip)
    if not kept:
        return str(text).strip()
    return " ".join(kept)


def rewrite_corrective_query(query: str, initial_texts: Sequence[str]) -> str:
    """Repeat the query terms the incorrect context never contained."""
    query_terms = _term_set(query)
    if not initial_texts:
        return query.strip()
    seen: set[str] = set()
    for text in initial_texts:
        seen.update(_term_set(text) & query_terms)
    ordered = list(dict.fromkeys(tokenize(query)))
    missing = [term for term in ordered if term not in seen]
    if not missing:
        missing = ordered[:3]
    emphasis = " ".join(missing)
    return f"{query.strip()} {emphasis} {emphasis}".strip()


def _action_for(labels: Sequence[str]) -> str:
    if not labels:
        return "retrieve"
    if "correct" in labels:
        return "refine"
    if "ambiguous" in labels:
        return "refine_and_retrieve"
    return "retrieve"


def _copy_candidate(candidate: Mapping[str, Any], grade: RetrievalGrade, content: str) -> dict[str, Any]:
    updated = dict(candidate)
    if "original_score" not in updated and updated.get("score") is not None:
        updated["original_score"] = updated.get("score")
    updated["candidate_id"] = grade.candidate_id
    updated["content"] = content
    updated["score"] = grade.score
    updated["corrective_grade"] = grade.label
    updated["corrective_score"] = grade.score
    return updated


def correct_retrieval(
    query: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    supplementary: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Grade one retrieval list and, when provided, one follow-up list.

    Distinctive terms are frozen from the initial list so a later hit cannot
    wash out the signal that made the first retrieval look wrong.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")

    query_terms = _term_set(query)
    initial = [dict(candidate) for candidate in candidates]
    initial_texts = [str(candidate.get("content") or "") for candidate in initial]
    distinctive = distinctive_terms(query_terms, initial_texts)
    follow_up = [dict(candidate) for candidate in (supplementary or [])]

    graded: list[tuple[Mapping[str, Any], RetrievalGrade, str]] = []
    seen: set[str] = set()
    for origin, group in (("initial", initial), ("supplementary", follow_up)):
        for candidate in group:
            identity = candidate_identity(candidate)
            if identity in seen:
                continue
            seen.add(identity)
            text = str(candidate.get("content") or "")
            grade = _grade_text(
                candidate_id=identity,
                text=text,
                query_terms=query_terms,
                distinctive=distinctive,
                origin=origin,
            )
            graded.append((candidate, grade, text))

    initial_labels = [grade.label for _, grade, _ in graded if grade.origin == "initial"]
    action = _action_for(initial_labels)
    needs_supplementary = action in {"retrieve", "refine_and_retrieve"} and supplementary is None
    corrective_query = None
    if needs_supplementary or supplementary is not None:
        corrective_query = rewrite_corrective_query(query, initial_texts)

    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    if not needs_supplementary:
        ranked = sorted(
            enumerate(graded),
            key=lambda item: (
                0 if item[1][1].label == "correct" else 1 if item[1][1].label == "ambiguous" else 2,
                -item[1][1].score,
                item[0],
            ),
        )
        for _, (candidate, grade, text) in ranked:
            keep = grade.label in {"correct", "ambiguous"}
            reason = grade.label
            content = refine_knowledge(text, query_terms, distinctive) if keep else text
            if keep and len(selected) < top_k:
                selected.append(_copy_candidate(candidate, grade, content))
                decision = "survived"
            else:
                decision = "dropped"
                if keep and len(selected) >= top_k:
                    reason = "context_budget_exceeded"
            decisions.append({
                "candidate_id": grade.candidate_id,
                "stage": "corrective_rag",
                "decision": decision,
                "reason": reason,
                "grade": grade.label,
                "score": grade.score,
                "origin": grade.origin,
            })
    else:
        for _, grade, _ in graded:
            decisions.append({
                "candidate_id": grade.candidate_id,
                "stage": "corrective_rag",
                "decision": "pending",
                "reason": "awaiting_corrective_retrieval",
                "grade": grade.label,
                "score": grade.score,
                "origin": grade.origin,
            })

    report = {
        "policy_version": CORRECTIVE_RAG_POLICY_VERSION,
        "enabled": True,
        "action": action,
        "needs_supplementary": needs_supplementary,
        "retrieval_completed": supplementary is not None or not needs_supplementary,
        "corrective_query": corrective_query,
        "distinctive_terms": sorted(distinctive),
        "input_count": len(initial),
        "supplementary_count": len(follow_up),
        "output_count": len(selected),
        "top_k": top_k,
        "grades": [
            {
                "candidate_id": grade.candidate_id,
                "label": grade.label,
                "score": grade.score,
                "coverage": grade.coverage,
                "distinctive_coverage": grade.distinctive_coverage,
                "origin": grade.origin,
            }
            for _, grade, _ in graded
        ],
        "decisions": decisions,
    }
    return selected, report


def apply_corrective_rag(
    query: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    retrieve: Callable[[str], Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run grading and, when the trigger fires, exactly one corrective retrieval."""
    selected, report = correct_retrieval(query, candidates, top_k=top_k)
    if report["needs_supplementary"] and retrieve is not None:
        extra = list(retrieve(str(report["corrective_query"] or query)))
        selected, report = correct_retrieval(
            query,
            candidates,
            top_k=top_k,
            supplementary=extra,
        )
        report["supplementary_count"] = len(extra)
    return selected, report


def disabled_corrective_report(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "policy_version": CORRECTIVE_RAG_POLICY_VERSION,
        "enabled": False,
        "action": "passthrough",
        "needs_supplementary": False,
        "retrieval_completed": False,
        "reason": "corrective RAG disabled",
        "corrective_query": None,
        "distinctive_terms": [],
        "input_count": len(candidates),
        "supplementary_count": 0,
        "output_count": len(candidates),
        "grades": [],
        "decisions": [],
    }
