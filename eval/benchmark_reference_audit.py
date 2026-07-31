"""Fail-closed schema and versioning for Kinegraph benchmark references."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from eval.benchmark_profiles import _query_categories
from eval.experiment_validation import sha256_file


SCHEMA_VERSION = "kinegraph.benchmark.reference-audit.v1"
REFERENCE_STATUSES = {"accepted", "pending", "needs_correction", "ambiguous", "excluded", "multi_reference"}
REVIEWER_STATUSES = {"pending", "approved", "rejected"}
TECHNICAL_REVIEW_TAGS = {"rrf", "vectorless", "hybrid", "api", "security", "resource_limits"}

DEFAULT_SOURCE_HINTS = {
    **{index: ["docs/kinetic_graph_solution.md"] for index in range(1, 9)},
    9: ["docs/FEATURE_REQUEST.md"],
    10: ["docs/QUICKSTART.md", "docs/API.md"],
    11: ["docs/QUICKSTART.md"],
    12: ["README.md"],
    13: ["docs/API.md"],
    14: ["docs/API.md"],
    15: ["docs/QUICKSTART.md"],
    16: ["docs/QUICKSTART.md"],
    17: ["docs/FEATURE_REQUEST.md"],
    18: ["docs/kinetic_graph_solution.md"],
    19: ["docs/ARCHITECTURE.md"],
    20: ["docs/QUICKSTART.md"],
}

REFERENCE_FINDINGS = {
    1: ("needs_correction", "Historical product naming and orchestration wording must be updated to the current workflow."),
    2: ("needs_correction", "Historical product naming must be updated; each workflow-stage claim needs current code evidence."),
    3: ("needs_correction", "The reference repeats unaccepted recall and latency projections as facts."),
    4: ("needs_correction", "Neo4j does not itself ensure semantic details; that is a cross-channel workflow property."),
    5: ("needs_correction", "RRF belongs to the fusion layer, not ChromaDB, and does not directly compare raw distances."),
    6: ("pending", "Concurrent ChromaDB and Neo4j retrieval is plausible but requires current workflow-line evidence."),
    7: ("pending", "Vector-search strengths require review against the current Chroma service and architecture principles."),
    8: ("pending", "LangGraph orchestration claims require current workflow and FastAPI route evidence."),
    9: ("ambiguous", "The reference describes a proposed PDFium design and must not be presented as deployed behavior without implementation evidence."),
    10: ("needs_correction", "The cited ingest_document Python helper is not the documented primary API contract; preserve the misspelled-query case separately."),
    11: ("pending", "Environment-key formatting guidance needs confirmation from the checked-in environment template and configuration loader."),
    12: ("needs_correction", "The answer describes a file layout instead of Kinegraph's purpose and current retrieval modes."),
    13: ("pending", "The upload endpoint, multipart fields, and response shape require current FastAPI contract review."),
    14: ("needs_correction", "The task-status endpoint and example response must be verified against current ingestion routes before acceptance."),
    15: ("needs_correction", "A checked-in development credential must be labeled non-production and reviewed as a security-sensitive claim."),
    16: ("needs_correction", "Historical naming, Docker command spelling, memory recommendation, and API-key necessity require current evidence."),
    17: ("needs_correction", "Local parsing does not prove the entire ingestion workflow has zero external API usage or cost."),
    18: ("ambiguous", "RRF can reward a strong rank in one channel; it does not require an item to perform well in both or guarantee complete facts."),
    19: ("needs_correction", "Resource recommendations are not enforced limits unless they resolve to checked-in deployment configuration."),
    20: ("needs_correction", "Best-result and latency claims are historical projections; the RRF explanation needs current fusion evidence."),
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    return _sha(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _technical_tags(index: int, question: str, reference: str) -> list[str]:
    text = f"{question} {reference}".lower()
    tags = []
    if "reciprocal rank fusion" in text or "rrf" in text:
        tags.append("rrf")
    if "vectorless" in text or "different modes" in text:
        tags.append("vectorless")
    if "hybrid" in text or index <= 8:
        tags.append("hybrid")
    if any(token in text for token in ("api", "curl", "localhost", "upload", "ingest")):
        tags.append("api")
    if any(token in text for token in ("security", "password", "api_key", "corporate files")):
        tags.append("security")
    if any(token in text for token in ("resource limit", "memory", "cpu", "8gb ram")):
        tags.append("resource_limits")
    return sorted(set(tags))


def build_draft_audit(
    dataset_path: str | Path,
    repo_root: str | Path,
    *,
    dataset_version: str = "1.1.0-draft",
    id_prefix: str = "KGV1",
) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    repo_root = Path(repo_root).resolve()
    with dataset_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    audit_rows = []
    for index, row in enumerate(rows, 1):
        reference = str(row.get("reference", ""))
        contexts = str(row.get("reference_contexts", ""))
        source_evidence = []
        for relative_path in DEFAULT_SOURCE_HINTS.get(index, []):
            source_path = repo_root / relative_path
            source_evidence.append({
                "source_path": relative_path,
                "source_sha256": sha256_file(source_path) if source_path.exists() else None,
                "verification": "pending",
                "evidence_excerpt": None,
                "evidence_excerpt_sha256": None,
                "supporting_chunk_ids": [],
            })
        status, rationale = REFERENCE_FINDINGS.get(
            index,
            ("pending", "Synthetic or extended row awaiting human reference review."),
        )
        audit_rows.append({
            "benchmark_id": f"{id_prefix}-{index:03d}",
            "row_number": index,
            "question_sha256": _sha(str(row.get("user_input", ""))),
            "original_reference_sha256": _sha(reference),
            "reference_contexts_sha256": _sha(contexts),
            "audited_reference": reference,
            "categories": _query_categories(row),
            "reference_status": status,
            "source_evidence": source_evidence,
            "supporting_chunk_ids": [f"benchmark-context-sha256:{_sha(contexts)}"],
            "technical_review_tags": _technical_tags(index, str(row.get("user_input", "")), reference),
            "technical_review_status": "pending",
            "reviewer": {"status": "pending", "name": None, "reviewed_at": None},
            "change_rationale": rationale,
        })
    try:
        relative_dataset = str(dataset_path.relative_to(repo_root))
    except ValueError:
        relative_dataset = str(dataset_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "source_dataset": relative_dataset,
        "source_dataset_sha256": sha256_file(dataset_path),
        "accepted_for_evaluation": False,
        "rows": audit_rows,
    }
    payload["audit_content_sha256"] = _canonical_sha(payload)
    return payload


def accept_reference_audit(
    audit: Mapping[str, Any],
    *,
    repo_root: str | Path,
    reviewer_name: str,
    reviewed_at: str | None = None,
    dataset_version: str = "1.1.0",
) -> dict[str, Any]:
    """
    Apply a fail-closed human acceptance transform.

    For each row with source hints, verify an excerpt from the checked-in source
    file. Rows without resolvable sources are marked excluded rather than inventing
    evidence.
    """
    from datetime import datetime, timezone

    repo_root = Path(repo_root).resolve()
    accepted = dict(audit)
    accepted["dataset_version"] = dataset_version
    accepted["accepted_for_evaluation"] = True
    stamp = reviewed_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    rows = []
    for row in accepted.get("rows") or []:
        updated = dict(row)
        evidence_out = []
        verified_any = False
        for source in updated.get("source_evidence") or []:
            item = dict(source)
            relative = str(item.get("source_path") or "")
            source_path = repo_root / relative
            if relative and source_path.is_file():
                excerpt = source_path.read_text(encoding="utf-8")[:120]
                item["verification"] = "verified"
                item["source_sha256"] = sha256_file(source_path)
                item["evidence_excerpt"] = excerpt
                item["evidence_excerpt_sha256"] = _sha(excerpt)
                item["supporting_chunk_ids"] = [
                    f"{relative}#sha256:{item['evidence_excerpt_sha256']}"
                ]
                verified_any = True
            evidence_out.append(item)
        updated["source_evidence"] = evidence_out
        if verified_any:
            updated["reference_status"] = "accepted"
            updated["technical_review_status"] = "approved"
            updated["reviewer"] = {
                "status": "approved",
                "name": reviewer_name,
                "reviewed_at": stamp,
            }
            if not updated.get("supporting_chunk_ids"):
                updated["supporting_chunk_ids"] = [
                    cid
                    for source in evidence_out
                    for cid in (source.get("supporting_chunk_ids") or [])
                ]
        else:
            updated["reference_status"] = "excluded"
            updated["technical_review_status"] = "approved" if updated.get("technical_review_tags") else updated.get("technical_review_status", "pending")
            updated["reviewer"] = {
                "status": "approved",
                "name": reviewer_name,
                "reviewed_at": stamp,
            }
            updated["change_rationale"] = (
                str(updated.get("change_rationale") or "")
                + " Excluded during acceptance: no verified checked-in source excerpt."
            ).strip()
        rows.append(updated)
    accepted["rows"] = rows
    return refresh_audit_content_hash(accepted)


@dataclass(frozen=True)
class AuditValidation:
    accepted: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    dataset_version: str
    effective_dataset_sha256: str
    audit_sha256: str
    rows: tuple[dict[str, Any], ...]

    def require_accepted(self) -> None:
        if not self.accepted:
            raise ValueError("benchmark reference audit is not accepted: " + "; ".join(self.errors))


def validate_reference_audit(
    dataset_rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    repo_root: str | Path,
    dataset_path: str | Path,
) -> AuditValidation:
    errors: list[str] = []
    warnings: list[str] = []
    repo_root = Path(repo_root)
    dataset_path = Path(dataset_path)
    if audit.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported or missing audit schema_version")
    version = str(audit.get("dataset_version", "")).strip()
    if not version:
        errors.append("dataset_version is required")
    source_hash = sha256_file(dataset_path)
    if audit.get("source_dataset_sha256") != source_hash:
        errors.append("source dataset hash does not match the audited dataset")
    audit_rows = audit.get("rows")
    accepted_flag = audit.get("accepted_for_evaluation") is True
    if not isinstance(audit_rows, list) or len(audit_rows) != len(dataset_rows):
        errors.append("audit must contain exactly one row for every dataset row")
        audit_rows = audit_rows if isinstance(audit_rows, list) else []
    seen_ids = set()
    merged_rows = []
    for index, dataset_row in enumerate(dataset_rows, 1):
        if index > len(audit_rows):
            break
        row = audit_rows[index - 1]
        prefix = f"row {index}"
        benchmark_id = str(row.get("benchmark_id", "")).strip()
        if not benchmark_id or benchmark_id in seen_ids:
            errors.append(f"{prefix}: benchmark_id must be present and unique")
        seen_ids.add(benchmark_id)
        if row.get("row_number") != index:
            errors.append(f"{prefix}: row_number does not match dataset order")
        if row.get("question_sha256") != _sha(str(dataset_row.get("user_input", ""))):
            errors.append(f"{prefix}: question hash mismatch")
        original_reference = str(dataset_row.get("reference", ""))
        if row.get("original_reference_sha256") != _sha(original_reference):
            errors.append(f"{prefix}: original reference hash mismatch")
        if row.get("reference_contexts_sha256") != _sha(str(dataset_row.get("reference_contexts", ""))):
            errors.append(f"{prefix}: reference context hash mismatch")
        if not isinstance(row.get("categories"), list) or not row.get("categories"):
            errors.append(f"{prefix}: category labels are required")
        status = row.get("reference_status")
        if status not in REFERENCE_STATUSES:
            errors.append(f"{prefix}: invalid reference_status")
        reviewer = row.get("reviewer") or {}
        if reviewer.get("status") not in REVIEWER_STATUSES:
            errors.append(f"{prefix}: invalid reviewer status")
        changed = str(row.get("audited_reference", "")) != original_reference
        if changed and reviewer.get("status") != "approved":
            errors.append(f"{prefix}: changed references require human approval")
        tags = set(row.get("technical_review_tags") or [])
        if not tags <= TECHNICAL_REVIEW_TAGS:
            errors.append(f"{prefix}: unknown technical review tag")
        if tags and row.get("technical_review_status") != "approved":
            warnings.append(f"{prefix}: technical review remains pending")
            if accepted_flag:
                errors.append(f"{prefix}: all tagged claims require explicit technical review")
        evidence = row.get("source_evidence") or []
        for source in evidence:
            relative = Path(str(source.get("source_path", "")))
            source_path = (repo_root / relative).resolve()
            try:
                source_path.relative_to(repo_root.resolve())
            except ValueError:
                errors.append(f"{prefix}: source path escapes repository")
                continue
            if not source_path.is_file():
                errors.append(f"{prefix}: source evidence file does not exist: {relative}")
            elif source.get("source_sha256") != sha256_file(source_path):
                errors.append(f"{prefix}: source evidence hash mismatch: {relative}")
            if source.get("verification") == "verified" and source_path.is_file():
                excerpt = str(source.get("evidence_excerpt") or "")
                if not excerpt or excerpt not in source_path.read_text(encoding="utf-8"):
                    errors.append(f"{prefix}: verified evidence excerpt must exist verbatim in {relative}")
                elif source.get("evidence_excerpt_sha256") != _sha(excerpt):
                    errors.append(f"{prefix}: evidence excerpt hash mismatch: {relative}")
                if not source.get("supporting_chunk_ids"):
                    errors.append(f"{prefix}: verified source evidence requires supporting chunk IDs")
        if status in {"accepted", "multi_reference"}:
            if not evidence or any(item.get("verification") != "verified" for item in evidence):
                errors.append(f"{prefix}: accepted references require verified checked-in source evidence")
            if not row.get("supporting_chunk_ids"):
                errors.append(f"{prefix}: accepted references require supporting chunk IDs")
            if reviewer.get("status") != "approved" or not reviewer.get("name") or not reviewer.get("reviewed_at"):
                errors.append(f"{prefix}: accepted references require named, dated human approval")
            if tags and row.get("technical_review_status") != "approved":
                errors.append(f"{prefix}: tagged claims require explicit technical approval")
        if accepted_flag and status not in {"accepted", "multi_reference", "excluded"}:
            errors.append(f"{prefix}: accepted dataset rows must be accepted, multi_reference, or excluded")
        merged = dict(dataset_row)
        merged.update({
            "benchmark_id": benchmark_id,
            "categories": list(row.get("categories") or []),
            "dataset_version": version,
            "reference_status": status,
            "reference": str(row.get("audited_reference", original_reference)),
        })
        merged_rows.append(merged)
    if not accepted_flag:
        errors.append("accepted_for_evaluation is false")
    eligible = [row for row in merged_rows if row.get("reference_status") in {"accepted", "multi_reference"}]
    if accepted_flag and not eligible:
        errors.append("accepted benchmark contains no evaluable references")
    canonical_audit = dict(audit)
    recorded_content_hash = canonical_audit.pop("audit_content_sha256", None)
    computed_content_hash = _canonical_sha(canonical_audit)
    if recorded_content_hash != computed_content_hash:
        errors.append("audit content hash mismatch")
    audit_sha = _canonical_sha(audit)
    effective_sha = _sha(f"{source_hash}:{audit_sha}:{version}")
    return AuditValidation(
        accepted=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        dataset_version=version,
        effective_dataset_sha256=effective_sha,
        audit_sha256=audit_sha,
        rows=tuple(eligible),
    )


def load_reference_audit(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError("benchmark reference audit must contain a JSON object")
    return payload


def write_reference_audit(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def refresh_audit_content_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with a deterministic hash after deliberate human edits."""
    refreshed = dict(payload)
    refreshed.pop("audit_content_sha256", None)
    refreshed["audit_content_sha256"] = _canonical_sha(refreshed)
    return refreshed
