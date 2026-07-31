"""Deterministic citation contracts for grounded answer generation."""
from __future__ import annotations

import json
import re
from typing import Any


def _parse_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def assign_citation_ids(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign a unique stable ID to every retrieved context passed to generation."""
    cited = []
    used: set[str] = set()
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        raw_id = (
            chunk.get("citation_id")
            or chunk.get("candidate_id")
            or chunk.get("id")
            or metadata.get("chunk_id")
            or metadata.get("id")
            or metadata.get("node_id")
            or f"ctx-{index}"
        )
        base = str(raw_id).strip() or f"ctx-{index}"
        citation_id = base
        suffix = 2
        while citation_id in used:
            citation_id = f"{base}#{suffix}"
            suffix += 1
        used.add(citation_id)
        enriched = dict(chunk)
        enriched["citation_id"] = citation_id
        cited.append(enriched)
    return cited


def build_citation_context(chunks: list[dict[str, Any]]) -> tuple[str, dict[str, str]]:
    """Build the exact ID-to-content map presented to generation and critique."""
    context_map = {
        str(chunk["citation_id"]): str(chunk.get("content") or "").strip()
        for chunk in chunks
    }
    rendered = "\n\n".join(
        f"[{citation_id}] {content}" for citation_id, content in context_map.items()
    )
    return rendered, context_map


def format_grounded_claims(claims: list[dict[str, Any]]) -> str:
    """Render only accepted claims, with every claim carrying verified citations."""
    return "\n".join(
        f"{claim['text'].strip()} {' '.join(f'[{citation}]' for citation in claim['chunk_ids'])}"
        for claim in claims
    )


def validate_grounded_response(
    content: str,
    valid_chunk_ids: set[str],
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    """Reject any generated claim missing citations or citing unavailable chunks."""
    payload = _parse_json_object(content)
    claims_shape_valid = isinstance(payload.get("claims"), list)
    raw_claims = payload.get("claims") if claims_shape_valid else []
    accepted = []
    rejected = []
    for index, raw_claim in enumerate(raw_claims, start=1):
        if not isinstance(raw_claim, dict):
            rejected.append({"claim_index": index, "reason": "invalid_claim_shape"})
            continue
        text = str(raw_claim.get("text") or "").strip()
        raw_ids = raw_claim.get("chunk_ids")
        chunk_ids = [str(item).strip() for item in raw_ids] if isinstance(raw_ids, list) else []
        invalid_ids = [chunk_id for chunk_id in chunk_ids if chunk_id not in valid_chunk_ids]
        if not text:
            rejected.append({"claim_index": index, "reason": "empty_claim"})
        elif not chunk_ids:
            rejected.append({"claim_index": index, "reason": "missing_citation"})
        elif invalid_ids:
            rejected.append({
                "claim_index": index,
                "reason": "invalid_citation",
                "invalid_chunk_ids": invalid_ids,
            })
        else:
            accepted.append({
                "claim_id": f"claim-{index}",
                "text": text,
                "chunk_ids": list(dict.fromkeys(chunk_ids)),
            })

    try:
        confidence = max(0.0, min(float(payload.get("confidence", 0.0)), 1.0))
    except (TypeError, ValueError):
        confidence = 0.0
    acceptance_ratio = len(accepted) / len(raw_claims) if raw_claims else 0.0
    confidence = round(confidence * acceptance_ratio, 4)
    details = {
        "structured_output_valid": bool(payload) and claims_shape_valid,
        "total_claims": len(raw_claims),
        "accepted_claims": len(accepted),
        "rejected_claims": rejected,
        "valid_chunk_ids": sorted(valid_chunk_ids),
    }
    return accepted, confidence, details


def apply_critic_response(
    claims: list[dict[str, Any]],
    content: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Allow the critic only to retain existing claims; it cannot add or rewrite."""
    payload = _parse_json_object(content)
    raw_supported = payload.get("supported_claim_ids")
    raw_relevant = payload.get("directly_relevant_claim_ids")
    if not isinstance(raw_supported, list) or not isinstance(raw_relevant, list):
        return claims, {
            "completed": False,
            "reason": "invalid_critic_output",
            "removed_claim_ids": [],
            "removed_irrelevant_claim_ids": [],
        }
    supported = {str(claim_id) for claim_id in raw_supported}
    relevant = {str(claim_id) for claim_id in raw_relevant}
    known = {claim["claim_id"] for claim in claims}
    retained_ids = known & supported & relevant
    retained = [claim for claim in claims if claim["claim_id"] in retained_ids]
    removed_unsupported = sorted(known - supported)
    removed_irrelevant = sorted((known & supported) - relevant)
    removed = sorted(known - retained_ids)
    coverage = str(payload.get("question_coverage") or "unverified").lower()
    if coverage not in {"complete", "partial", "none"}:
        coverage = "unverified"
    raw_missing_facets = payload.get("missing_question_facets")
    missing_facets = (
        [str(facet).strip() for facet in raw_missing_facets if str(facet).strip()]
        if isinstance(raw_missing_facets, list)
        else []
    )
    return retained, {
        "completed": True,
        "retained_claim_ids": [claim["claim_id"] for claim in retained],
        "removed_claim_ids": removed,
        "removed_unsupported_claim_ids": removed_unsupported,
        "removed_irrelevant_claim_ids": removed_irrelevant,
        "unsupported_reasons": payload.get("unsupported_reasons") or {},
        "irrelevant_reasons": payload.get("irrelevant_reasons") or {},
        "question_coverage": coverage,
        "missing_question_facets": missing_facets,
    }
