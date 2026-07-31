"""Retrieval IR metrics for Kinegraph live evaluation reports."""
from __future__ import annotations

import ast
import math
from typing import Any, Mapping, Sequence


def parse_reference_contexts(value: Any) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            items = parsed if isinstance(parsed, list) else [value]
        except (SyntaxError, ValueError):
            items = [value]
    else:
        items = []
    return [str(item).strip() for item in items if str(item).strip()]


def normalize_text(value: str) -> str:
    return " ".join(str(value).lower().split())


def context_identity(text: str) -> str:
    return normalize_text(text)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = list(retrieved)[:k]
    if not top:
        return 0.0
    relevant_ids = {context_identity(item) for item in relevant}
    hits = sum(1 for item in top if context_identity(item) in relevant_ids)
    return hits / float(k)


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    relevant_ids = {context_identity(item) for item in relevant}
    if not relevant_ids:
        return 0.0
    top = list(retrieved)[:k]
    hits = {context_identity(item) for item in top if context_identity(item) in relevant_ids}
    return len(hits) / float(len(relevant_ids))


def average_precision(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    relevant_ids = {context_identity(item) for item in relevant}
    if not relevant_ids:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(list(retrieved)[:k], 1):
        if context_identity(item) in relevant_ids:
            hits += 1
            score += hits / float(rank)
    return score / float(min(len(relevant_ids), k))


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    relevant_ids = {context_identity(item) for item in relevant}
    if not relevant_ids or k <= 0:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(list(retrieved)[:k], 1):
        rel = 1.0 if context_identity(item) in relevant_ids else 0.0
        if rel:
            dcg += rel / math.log2(rank + 1.0)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def retrieve_texts_from_provenance(provenance: Mapping[str, Any]) -> list[str]:
    retrieval = provenance.get("retrieval") or {}
    finals = retrieval.get("final_contexts") or []
    texts: list[str] = []
    for item in finals:
        if isinstance(item, Mapping):
            content = item.get("content") or item.get("text") or ""
            if str(content).strip():
                texts.append(str(content))
        elif str(item).strip():
            texts.append(str(item))
    return texts


def score_retrieval(
    *,
    retrieved_contexts: Sequence[str],
    reference_contexts: Any,
    k: int = 5,
) -> dict[str, float]:
    relevant = parse_reference_contexts(reference_contexts)
    return {
        f"precision_at_{k}": round(precision_at_k(retrieved_contexts, relevant, k), 4),
        f"recall_at_{k}": round(recall_at_k(retrieved_contexts, relevant, k), 4),
        f"ndcg_at_{k}": round(ndcg_at_k(retrieved_contexts, relevant, k), 4),
        f"map_at_{k}": round(average_precision(retrieved_contexts, relevant, k), 4),
    }


def aggregate_ir_metrics(rows: Sequence[Mapping[str, Any]], k: int = 5) -> dict[str, Any]:
    keys = [f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", f"map_at_{k}"]
    values = {key: [] for key in keys}
    for row in rows:
        for key in keys:
            if key in row and row[key] is not None:
                values[key].append(float(row[key]))
    return {
        key: {
            "mean": round(sum(items) / len(items), 4) if items else None,
            "samples": len(items),
        }
        for key, items in values.items()
    }
