"""Versioned, redacted per-query benchmark provenance artifacts."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "kinegraph.eval.provenance.v1"
METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)
_SECRET_PATTERNS = (
    (re.compile(r"(?i)(OPENAI(?:_AI)?_KEY\s*=\s*)[^\s]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+\-/=]+"), "Bearer [REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "[REDACTED_API_KEY]"),
)
_METADATA_KEYS = {
    "chunk_id",
    "doc_id",
    "document_id",
    "file_name",
    "node_id",
    "vector_record_id",
    "community_id",
    "centrality_score",
    "traversal_depth",
    "traversal_strategy",
    "max_hops",
    "seed_node_id",
    "relationship_path",
    "relationships_json",
    "recovery_stage",
    "retrieval_query",
    "original_query",
    "hypothesis_is_evidence",
    "citation_id",
}


def _candidate_lifecycle(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    initial = trace.get("initial_candidates") or {}
    initial_ids = {
        candidate_id(candidate)
        for candidates in initial.values()
        for candidate in (candidates or [])
    }
    fusion_ids = {
        candidate_id(candidate)
        for candidate in ((trace.get("fusion") or {}).get("candidates") or [])
    }
    reranked_ids = {
        candidate_id(candidate)
        for candidate in ((trace.get("reranking") or {}).get("candidates") or [])
    }
    final_ids = {candidate_id(candidate) for candidate in (trace.get("final_contexts") or [])}
    all_ids = sorted(initial_ids | fusion_ids | reranked_ids | final_ids)
    lifecycle = []
    for item_id in all_ids:
        if item_id not in fusion_ids:
            dropped_at = "pre_fusion"
        elif item_id not in reranked_ids:
            dropped_at = "reranking"
        elif item_id not in final_ids:
            dropped_at = "final_truncation"
        else:
            dropped_at = None
        lifecycle.append({
            "candidate_id": item_id,
            "initial": item_id in initial_ids,
            "fused": item_id in fusion_ids,
            "reranked": item_id in reranked_ids,
            "sent_to_generation": item_id in final_ids,
            "dropped_at": dropped_at,
        })
    return lifecycle


def _graph_path_audit(trace: Mapping[str, Any]) -> dict[str, Any]:
    candidates = (trace.get("channel_candidates") or {}).get("graph") or []
    traversal_candidates = [
        candidate for candidate in candidates if candidate.get("source") == "graph_traversal"
    ]
    paths = []
    for candidate in traversal_candidates:
        metadata = candidate.get("metadata") or {}
        relationship_path = metadata.get("relationship_path") or []
        depth = int(metadata.get("traversal_depth") or 0)
        node_sequence = []
        missing_edge_fields = []
        for edge_index, edge in enumerate(relationship_path):
            if edge_index == 0:
                node_sequence.append(edge.get("from_node_id"))
            node_sequence.append(edge.get("to_node_id"))
            missing = [
                key for key in (
                    "from_node_id", "to_node_id", "relationship_type",
                    "direction", "weight", "evidence_text",
                )
                if edge.get(key) in (None, "")
            ]
            if missing:
                missing_edge_fields.append({"edge_index": edge_index, "fields": missing})
        paths.append({
            "candidate_id": candidate_id(candidate),
            "seed_node_id": metadata.get("seed_node_id"),
            "traversal_depth": depth,
            "path_length": len(relationship_path),
            "path_complete": depth > 0 and len(relationship_path) == depth and not missing_edge_fields,
            "cycle_detected": len([node for node in node_sequence if node is not None]) != len(set(node for node in node_sequence if node is not None)),
            "missing_edge_fields": missing_edge_fields,
        })
    return {
        "traversal_candidate_count": len(traversal_candidates),
        "complete_path_count": sum(path["path_complete"] for path in paths),
        "all_paths_complete": bool(paths) and all(path["path_complete"] for path in paths),
        "paths": paths,
        "retriever_diagnostics": _sanitize(trace.get("graph_retrieval_diagnostics") or {}),
    }


def redact_text(value: Any, limit: int = 500) -> str:
    text = str(value)
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text[:limit]


def _sanitize(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize(item, depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) != "embedding"
        }
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(item, depth + 1) for item in value]
    return redact_text(value)


def candidate_id(candidate: Mapping[str, Any]) -> str:
    metadata = candidate.get("metadata") or {}
    for key in ("chunk_id", "vector_record_id", "node_id", "id", "citation_id"):
        value = candidate.get(key, metadata.get(key))
        if value not in (None, ""):
            return str(value)
    digest = hashlib.sha256(str(candidate.get("content", "")).encode("utf-8")).hexdigest()
    return f"content-sha256:{digest}"


def candidate_snapshot(candidate: Mapping[str, Any], rank: int) -> dict[str, Any]:
    metadata = candidate.get("metadata") or {}
    selected_metadata = {
        key: value for key, value in metadata.items() if key in _METADATA_KEYS
    }
    scores = {
        key: candidate.get(key)
        for key in (
            "score",
            "original_score",
            "rrf_score",
            "rrf_contributions",
            "rerank_score",
            "semantic_score",
            "graph_signal_score",
            "rerank_components",
        )
        if candidate.get(key) is not None
    }
    return {
        "candidate_id": candidate_id(candidate),
        "rank": rank,
        "source": str(candidate.get("source", "unknown")),
        "scores": _sanitize(scores),
        "metadata": _sanitize(selected_metadata),
        "excerpt": redact_text(candidate.get("content", ""), limit=180),
        "embedding_present": candidate.get("embedding") is not None,
    }


def _candidate_channels(channels: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        str(channel): [
            candidate_snapshot(candidate, rank)
            for rank, candidate in enumerate(candidates or [], 1)
        ]
        for channel, candidates in sorted(channels.items())
    }


def _first_failure_stage(
    *,
    workflow_error: str | None,
    trace: Mapping[str, Any],
    result: Mapping[str, Any],
    scores: Mapping[str, Any],
    profile_error: str | None,
) -> str | None:
    if workflow_error:
        return "workflow"
    if profile_error:
        return "mode_profile"
    if trace.get("retrieval_failures"):
        return "retrieval_channel"
    if not trace.get("final_contexts"):
        return "empty_retrieval"
    citation = result.get("citation_validation") or {}
    if citation and not citation.get("structured_output_valid", False):
        return "citation_validation"
    critic = result.get("grounding_critique") or {}
    if critic.get("reason") == "critic_failed":
        return "grounding_critic"
    if (trace.get("reranking") or {}).get("fallback"):
        return "reranker_fallback"
    if scores.get("ragas_failed"):
        return "ragas_judge"
    return None


def build_provenance_record(
    *,
    sample: Mapping[str, Any],
    result: Mapping[str, Any],
    scores: Mapping[str, Any],
    profile: Mapping[str, Any],
    workflow_error: str | None,
    workflow_latency_ms: float,
    eval_latency_ms: float,
    judge_model: str | None,
    embedding_model: str | None,
) -> dict[str, Any]:
    trace = result.get("trace") or {}
    routing = trace.get("routing") or result.get("routing") or {}
    effective_mode = str(trace.get("effective_mode") or result.get("effective_mode") or "unknown")
    declared_modes = list(profile.get("declared_effective_modes", []))
    profile_error = None
    if declared_modes and effective_mode not in declared_modes:
        profile_error = (
            f"effective mode {effective_mode!r} is not declared by profile "
            f"{profile.get('name')!r}: {declared_modes}"
        )
    initial_candidates = trace.get("initial_candidates") or {}
    if effective_mode == "vectorless" and initial_candidates.get("lexical"):
        initial_candidates = {**initial_candidates, "vectorless": initial_candidates.get("lexical", [])}
        initial_candidates["lexical"] = []
    first_failure = _first_failure_stage(
        workflow_error=workflow_error,
        profile_error=profile_error,
        trace=trace,
        result=result,
        scores=scores,
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": str(sample.get("sample_id", "unknown")),
        "categories": sorted(str(category) for category in sample.get("categories", [])),
        "query": {
            "original": redact_text(sample.get("question", ""), limit=1000),
            "rewritten": redact_text(trace.get("rewritten_query", ""), limit=1000),
            "facets": _sanitize(routing.get("facets", [])),
        },
        "profile": {
            "name": str(profile.get("name", "unspecified")),
            "requested_mode": str(profile.get("requested_mode", trace.get("requested_mode", "unknown"))),
            "effective_mode": effective_mode,
            "declared_effective_modes": declared_modes,
            "source_corpus_sha256": sample.get("source_corpus_sha256"),
            "valid": profile_error is None,
            "error": profile_error,
        },
        "routing": _sanitize(routing),
        "retrieval": {
            "initial_candidates": _candidate_channels(initial_candidates),
            "channel_candidates": _candidate_channels(trace.get("channel_candidates") or {}),
            "failures": _sanitize(trace.get("retrieval_failures") or {}),
            "recovery": _sanitize(trace.get("recovery") or result.get("recovery") or {}),
            "candidate_lifecycle": _candidate_lifecycle(trace),
            "graph_path_audit": _graph_path_audit(trace),
            "fusion": {
                **_sanitize({
                    key: value for key, value in (trace.get("fusion") or {}).items()
                    if key != "candidates"
                }),
                "candidates": [
                    candidate_snapshot(candidate, rank)
                    for rank, candidate in enumerate((trace.get("fusion") or {}).get("candidates", []), 1)
                ],
            },
            "reranking": {
                **_sanitize({
                    key: value for key, value in (trace.get("reranking") or {}).items()
                    if key != "candidates"
                }),
                "candidates": [
                    candidate_snapshot(candidate, rank)
                    for rank, candidate in enumerate((trace.get("reranking") or {}).get("candidates", []), 1)
                ],
            },
            "final_contexts": [
                candidate_snapshot(candidate, rank)
                for rank, candidate in enumerate(trace.get("final_contexts") or [], 1)
            ],
        },
        "generation": {
            "answer": redact_text(result.get("answer", ""), limit=4000),
            "confidence": result.get("confidence"),
            "grounded_claims": _sanitize(result.get("grounded_claims") or []),
            "citation_validation": _sanitize(result.get("citation_validation") or {}),
            "grounding_critique": _sanitize(result.get("grounding_critique") or {}),
            "answer_relevancy": _sanitize(result.get("answer_relevancy") or {}),
        },
        "evaluation": {
            "metrics": {
                metric: scores.get(metric) for metric in METRIC_NAMES if metric in scores
            },
            "ragas_failed": bool(scores.get("ragas_failed", True)),
            "ragas_error": redact_text(scores.get("ragas_error", ""), limit=1000) or None,
            "judge_model": judge_model,
            "embedding_model": embedding_model,
            "judge_rationale": None,
            "judge_rationale_available": False,
        },
        "latency_ms": {
            "workflow": workflow_latency_ms,
            "evaluation": eval_latency_ms,
            "stages": _sanitize(trace.get("latency_ms") or result.get("latency") or {}),
        },
        "failure": {
            "first_stage": first_failure,
            "workflow_error": redact_text(workflow_error, limit=1000) if workflow_error else None,
        },
    }
    validate_provenance_record(record)
    return record


def validate_provenance_record(record: Mapping[str, Any]) -> None:
    required = {"schema_version", "sample_id", "query", "profile", "routing", "retrieval", "generation", "evaluation", "latency_ms", "failure"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"provenance record missing keys: {', '.join(missing)}")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported provenance schema: {record.get('schema_version')}")
    if not str(record.get("sample_id", "")).strip():
        raise ValueError("provenance sample_id cannot be empty")


def write_provenance_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted((dict(record) for record in records), key=lambda item: item["sample_id"])
    for record in ordered:
        validate_provenance_record(record)
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in ordered)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)


def diagnostic_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(records)
    modes = Counter(str(record["profile"]["effective_mode"]) for record in records)
    categories = Counter(category for record in records for category in record.get("categories", []))
    failures = Counter(
        str(record["failure"]["first_stage"])
        for record in records
        if record.get("failure", {}).get("first_stage")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "total_records": len(records),
        "effective_mode_counts": dict(sorted(modes.items())),
        "category_counts": dict(sorted(categories.items())),
        "first_failure_stage_counts": dict(sorted(failures.items())),
        "profile_valid": all(record["profile"]["valid"] for record in records),
        "ragas_failed_count": sum(record["evaluation"]["ragas_failed"] for record in records),
    }


def write_diagnostic_summary(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(diagnostic_summary(records), indent=2, sort_keys=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)
