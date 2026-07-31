"""Versioned provenance and context-selection contracts for ADR-003."""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable, Mapping


ORCHESTRATION_POLICY_VERSION = "kinegraph.retrieval-orchestration.v1"


def candidate_identity(result: Mapping[str, Any]) -> str:
    """Return a stable store identifier, falling back to a full-content hash."""
    metadata = result.get("metadata") or {}
    for key in (
        "candidate_id",
        "chunk_id",
        "vector_record_id",
        "node_id",
        "id",
        "citation_id",
    ):
        value = result.get(key, metadata.get(key))
        if value not in (None, ""):
            return str(value)
    content = str(result.get("content") or "")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"content-sha256:{digest}"


def _original_score(result: Mapping[str, Any]) -> float | None:
    value = result.get("original_score", result.get("score"))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _graph_path(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = result.get("metadata") or {}
    raw_path = metadata.get("relationship_path") or result.get("relationship_path")
    return list(raw_path) if isinstance(raw_path, list) else []


def annotate_channel_candidates(
    results: Iterable[Mapping[str, Any]],
    channel: str,
) -> list[dict[str, Any]]:
    """Attach channel rank, score, stable identity, and graph-path provenance."""
    annotated: list[dict[str, Any]] = []
    for rank, raw_result in enumerate(results, start=1):
        result = dict(raw_result)
        candidate_id = candidate_identity(result)
        source_channels = list(dict.fromkeys([
            *result.get("source_channels", []),
            channel,
        ]))
        original_scores = dict(result.get("original_scores") or {})
        original_scores[channel] = _original_score(result)
        channel_ranks = dict(result.get("channel_ranks") or {})
        channel_ranks[channel] = rank
        graph_paths = dict(result.get("graph_paths") or {})
        path = _graph_path(result)
        if path:
            graph_paths[channel] = path

        provenance = dict(result.get("retrieval_provenance") or {})
        provenance.update({
            "policy_version": ORCHESTRATION_POLICY_VERSION,
            "candidate_id": candidate_id,
            "source_channels": source_channels,
            "original_scores": original_scores,
            "channel_ranks": channel_ranks,
            "graph_paths": graph_paths,
        })
        result.update({
            "candidate_id": candidate_id,
            "source_channels": source_channels,
            "original_scores": original_scores,
            "channel_ranks": channel_ranks,
            "graph_paths": graph_paths,
            "retrieval_provenance": provenance,
        })
        annotated.append(result)
    return annotated


def merge_candidate_provenance(
    primary: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge two store representations without losing graph or channel evidence."""
    merged = dict(primary)
    primary_metadata = dict(primary.get("metadata") or {})
    incoming_metadata = dict(incoming.get("metadata") or {})
    for key, value in incoming_metadata.items():
        if key not in primary_metadata or primary_metadata[key] in (None, "", [], {}):
            primary_metadata[key] = value
    merged["metadata"] = primary_metadata

    merged["source_channels"] = list(dict.fromkeys([
        *primary.get("source_channels", []),
        *incoming.get("source_channels", []),
    ]))
    merged["original_scores"] = {
        **(primary.get("original_scores") or {}),
        **(incoming.get("original_scores") or {}),
    }
    merged["channel_ranks"] = {
        **(primary.get("channel_ranks") or {}),
        **(incoming.get("channel_ranks") or {}),
    }
    merged["graph_paths"] = {
        **(primary.get("graph_paths") or {}),
        **(incoming.get("graph_paths") or {}),
    }
    merged["candidate_id"] = candidate_identity(primary)
    merged["retrieval_provenance"] = {
        "policy_version": ORCHESTRATION_POLICY_VERSION,
        "candidate_id": merged["candidate_id"],
        "source_channels": merged["source_channels"],
        "original_scores": merged["original_scores"],
        "channel_ranks": merged["channel_ranks"],
        "graph_paths": merged["graph_paths"],
    }
    return merged


def _group_value(candidate: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    metadata = candidate.get("metadata") or {}
    for key in keys:
        value = candidate.get(key, metadata.get(key))
        if value not in (None, ""):
            return str(value)
    return None


def optimize_context(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    max_per_source: int = 0,
    max_per_community: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a bounded, diverse context set while preserving rank order."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if max_per_source < 0 or max_per_community < 0:
        raise ValueError("context diversity caps cannot be negative")

    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    community_counts: Counter[str] = Counter()
    for rank, candidate in enumerate(candidates, start=1):
        candidate_id = candidate_identity(candidate)
        source_id = _group_value(
            candidate,
            ("document_id", "source_document", "source_id", "file_name"),
        )
        community_id = _group_value(candidate, ("community_id",))
        reason = "selected_within_context_budget"
        decision = "survived"
        if len(selected) >= top_k:
            decision, reason = "dropped", "context_budget_exceeded"
        elif max_per_source and source_id and source_counts[source_id] >= max_per_source:
            decision, reason = "dropped", "source_diversity_cap"
        elif (
            max_per_community
            and community_id
            and community_counts[community_id] >= max_per_community
        ):
            decision, reason = "dropped", "community_diversity_cap"
        else:
            selected.append(candidate)
            if source_id:
                source_counts[source_id] += 1
            if community_id:
                community_counts[community_id] += 1

        decisions.append({
            "candidate_id": candidate_id,
            "stage": "context_optimization",
            "decision": decision,
            "reason": reason,
            "input_rank": rank,
            "source_id": source_id,
            "community_id": community_id,
        })

    return selected, {
        "policy_version": ORCHESTRATION_POLICY_VERSION,
        "enabled": True,
        "top_k": top_k,
        "max_per_source": max_per_source,
        "max_per_community": max_per_community,
        "input_count": len(candidates),
        "output_count": len(selected),
        "decisions": decisions,
    }


def passthrough_context_report(
    candidates: list[dict[str, Any]],
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "policy_version": ORCHESTRATION_POLICY_VERSION,
        "enabled": False,
        "reason": reason,
        "input_count": len(candidates),
        "output_count": len(candidates),
        "decisions": [
            {
                "candidate_id": candidate_identity(candidate),
                "stage": "context_optimization",
                "decision": "survived",
                "reason": "optimization_disabled_passthrough",
                "input_rank": rank,
            }
            for rank, candidate in enumerate(candidates, start=1)
        ],
    }


def build_candidate_lifecycle(
    *,
    channel_candidates: Mapping[str, list[dict[str, Any]]],
    fused_candidates: list[dict[str, Any]],
    final_candidates: list[dict[str, Any]],
    stage_reports: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a compact, per-candidate survival ledger for persisted traces."""
    records: dict[str, dict[str, Any]] = {}

    def ensure(candidate: Mapping[str, Any]) -> dict[str, Any]:
        candidate_id = candidate_identity(candidate)
        record = records.setdefault(candidate_id, {
            "candidate_id": candidate_id,
            "source_channels": [],
            "original_scores": {},
            "channel_ranks": {},
            "graph_paths": {},
            "graph_path_required": False,
            "events": [],
        })
        record["source_channels"] = list(dict.fromkeys([
            *record["source_channels"],
            *candidate.get("source_channels", []),
        ]))
        record["original_scores"].update(candidate.get("original_scores") or {})
        record["channel_ranks"].update(candidate.get("channel_ranks") or {})
        record["graph_paths"].update(candidate.get("graph_paths") or {})
        metadata = candidate.get("metadata") or {}
        try:
            traversal_depth = int(metadata.get("traversal_depth") or 0)
        except (TypeError, ValueError):
            traversal_depth = 0
        record["graph_path_required"] = bool(
            record["graph_path_required"]
            or candidate.get("source") == "graph_traversal"
            or traversal_depth > 0
        )
        return record

    for channel, candidates in channel_candidates.items():
        for rank, candidate in enumerate(candidates, start=1):
            annotated = annotate_channel_candidates([candidate], channel)[0]
            annotated["channel_ranks"][channel] = rank
            record = ensure(annotated)
            record["events"].append({
                "stage": "retrieval",
                "decision": "survived",
                "reason": "retrieved_by_channel",
                "channel": channel,
                "rank": rank,
            })

    fused_ids = set()
    for rank, candidate in enumerate(fused_candidates, start=1):
        record = ensure(candidate)
        fused_ids.add(record["candidate_id"])
        record["events"].append({
            "stage": "fusion",
            "decision": "survived",
            "reason": "ranked_by_fusion_or_passthrough",
            "rank": rank,
            "rrf_score": candidate.get("rrf_score"),
        })

    for record in records.values():
        if record["candidate_id"] not in fused_ids:
            record["events"].append({
                "stage": "fusion",
                "decision": "dropped",
                "reason": "channel_not_active_or_zero_weight",
            })

    for report in stage_reports:
        for event in report.get("decisions", []):
            candidate_id = str(event.get("candidate_id") or "")
            if not candidate_id:
                continue
            record = records.setdefault(candidate_id, {
                "candidate_id": candidate_id,
                "source_channels": [],
                "original_scores": {},
                "channel_ranks": {},
                "graph_paths": {},
                "graph_path_required": False,
                "events": [],
            })
            record["events"].append(dict(event))

    final_ids = {candidate_identity(candidate) for candidate in final_candidates}
    complete = 0
    for record in records.values():
        sent_to_generation = record["candidate_id"] in final_ids
        record["final_status"] = (
            "selected_for_generation" if sent_to_generation else "dropped_before_generation"
        )
        record["initial"] = any(
            event.get("stage") == "retrieval" for event in record["events"]
        )
        record["fused"] = any(
            event.get("stage") == "fusion" and event.get("decision") == "survived"
            for event in record["events"]
        )
        record["reranked"] = any(
            event.get("stage") == "semantic_reranking"
            and event.get("decision") == "survived"
            for event in record["events"]
        )
        record["sent_to_generation"] = sent_to_generation
        first_drop = next((
            event.get("stage")
            for event in record["events"]
            if event.get("decision") == "dropped"
        ), None)
        record["dropped_at"] = first_drop if not sent_to_generation else None
        missing = []
        if not record["source_channels"]:
            missing.append("source_channels")
        if not record["original_scores"]:
            missing.append("original_scores")
        if not record["channel_ranks"]:
            missing.append("channel_ranks")
        if record["graph_path_required"] and not record["graph_paths"]:
            missing.append("graph_path")
        if not record["events"] or any(not event.get("reason") for event in record["events"]):
            missing.append("stage_reason")
        record["missing_provenance_fields"] = missing
        if not missing:
            complete += 1

    total = len(records)
    return {
        "policy_version": ORCHESTRATION_POLICY_VERSION,
        "candidate_count": total,
        "complete_candidate_count": complete,
        "candidate_provenance_completeness": round(complete / total, 4) if total else 1.0,
        "candidates": sorted(records.values(), key=lambda item: item["candidate_id"]),
    }
