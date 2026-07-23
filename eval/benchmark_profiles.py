"""Explicit, non-default-changing benchmark execution profiles."""
from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from backend.app.models import QueryMode


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    requested_mode: QueryMode
    enable_lexical_fusion: bool
    declared_effective_modes: tuple[str, ...]
    allow_mode_downgrade: bool = False
    allow_vectorless_auto_route: bool = False
    use_reference_context_as_attachment: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["requested_mode"] = self.requested_mode.value
        payload["declared_effective_modes"] = list(self.declared_effective_modes)
        return payload


PROFILES = {
    "adaptive_hybrid": BenchmarkProfile(
        name="adaptive_hybrid",
        requested_mode=QueryMode.HYBRID,
        enable_lexical_fusion=False,
        declared_effective_modes=(
            QueryMode.HYBRID.value,
            QueryMode.VECTOR.value,
            QueryMode.GRAPH.value,
        ),
        allow_mode_downgrade=True,
    ),
    "hybrid": BenchmarkProfile(
        name="hybrid",
        requested_mode=QueryMode.HYBRID,
        enable_lexical_fusion=False,
        declared_effective_modes=(QueryMode.HYBRID.value,),
    ),
    "hybrid_lexical": BenchmarkProfile(
        name="hybrid_lexical",
        requested_mode=QueryMode.HYBRID,
        enable_lexical_fusion=True,
        declared_effective_modes=(QueryMode.HYBRID.value,),
    ),
    "vectorless": BenchmarkProfile(
        name="vectorless",
        requested_mode=QueryMode.VECTORLESS,
        enable_lexical_fusion=False,
        declared_effective_modes=(QueryMode.VECTORLESS.value,),
        use_reference_context_as_attachment=True,
    ),
}


def get_profile(name: str) -> BenchmarkProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark profile: {name}") from exc


def _reference_contexts(value: Any) -> list[str]:
    if isinstance(value, list):
        contexts = value
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            contexts = parsed if isinstance(parsed, list) else [value]
        except (SyntaxError, ValueError):
            contexts = [value]
    else:
        contexts = []
    return [str(context) for context in contexts if str(context).strip()]


def _query_categories(row: Any) -> list[str]:
    existing = row.get("categories")
    if isinstance(existing, list) and existing:
        return sorted(str(category) for category in existing)
    query = str(row.get("user_input", ""))
    synthesizer = str(row.get("synthesizer_name", ""))
    categories = ["multi_hop" if synthesizer.startswith("multi_hop") else "single_hop"]
    if re.search(r"OPENAI_API_KEY|https?://|\.pdf|\.env", query, re.I):
        categories.append("exact_token")
    if re.search(r"\b(and|compared? to|versus|vs\.?|alternatives?)\b", query, re.I):
        categories.append("compound")
    if re.search(r"\b(documnt|opne)\b", query, re.I) or row.get("query_style") == "MISSPELLED":
        categories.append("misspelled")
    if "<2-hop>" in str(row.get("reference_contexts", "")):
        categories.append("two_reference_facets")
    return categories


def build_profile_dataset(rows: Iterable[Any], profile: BenchmarkProfile) -> list[dict[str, Any]]:
    """Build comparable samples; Vectorless receives the same source contexts as attachments."""
    rows = list(rows)
    source_contexts = []
    for row in rows:
        for context in _reference_contexts(row.get("reference_contexts")):
            if context.strip() and context not in source_contexts:
                source_contexts.append(context)
    vectorless_corpus = "\n\n--- SOURCE CONTEXT ---\n\n".join(source_contexts)
    corpus_sha256 = hashlib.sha256(vectorless_corpus.encode("utf-8")).hexdigest()
    dataset = []
    for index, row in enumerate(rows, 1):
        sample = {
            "sample_id": str(row.get("benchmark_id") or f"benchmark-{index:03d}"),
            "question": str(row["user_input"]),
            "ground_truth": str(row["reference"]),
            "categories": _query_categories(row),
            "query_style": str(row.get("query_style", "") or "unknown"),
            "synthesizer_name": str(row.get("synthesizer_name", "") or "unknown"),
            "dataset_version": str(row.get("dataset_version", "unversioned")),
            "reference_status": str(row.get("reference_status", "legacy")),
        }
        if profile.use_reference_context_as_attachment:
            sample["attachment_content"] = vectorless_corpus
            sample["attachment_name"] = "kinegraph-benchmark-corpus.txt"
            sample["source_corpus_sha256"] = corpus_sha256
        dataset.append(sample)
    return dataset
