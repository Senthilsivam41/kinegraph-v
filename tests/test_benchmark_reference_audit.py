import csv
import hashlib
import json
from pathlib import Path

import pytest

from eval.benchmark_reference_audit import (
    build_draft_audit,
    validate_reference_audit,
)


REPO_ROOT = Path(__file__).parents[1]
DATASET = REPO_ROOT / "eval" / "kinegraph_benchmark_v1.csv"


def _rows():
    with DATASET.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _rehash(audit):
    payload = dict(audit)
    payload.pop("audit_content_sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    audit["audit_content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()


def _accepted_audit():
    audit = build_draft_audit(DATASET, REPO_ROOT)
    audit["dataset_version"] = "1.1.0"
    audit["accepted_for_evaluation"] = True
    for row in audit["rows"]:
        row["reference_status"] = "accepted"
        row["technical_review_status"] = "approved"
        row["reviewer"] = {
            "status": "approved",
            "name": "Human Reviewer",
            "reviewed_at": "2026-07-22T00:00:00Z",
        }
        for source in row["source_evidence"]:
            source["verification"] = "verified"
            excerpt = (REPO_ROOT / source["source_path"]).read_text(encoding="utf-8")[:120]
            source["evidence_excerpt"] = excerpt
            source["evidence_excerpt_sha256"] = hashlib.sha256(excerpt.encode()).hexdigest()
            source["supporting_chunk_ids"] = [
                f"{source['source_path']}#sha256:{source['evidence_excerpt_sha256']}"
            ]
    _rehash(audit)
    return audit


def test_checked_in_audit_covers_all_rows_but_is_deliberately_unaccepted():
    audit = json.loads((REPO_ROOT / "eval" / "kinegraph_benchmark_v1.audit.json").read_text())

    validation = validate_reference_audit(_rows(), audit, REPO_ROOT, DATASET)

    assert len(audit["rows"]) == 20
    assert len({row["benchmark_id"] for row in audit["rows"]}) == 20
    assert [
        row["benchmark_id"] for row in audit["rows"]
        if "two_reference_facets" in row["categories"]
    ] == ["KGV1-009", "KGV1-014", "KGV1-017", "KGV1-018", "KGV1-019", "KGV1-020"]
    assert validation.accepted is False
    assert "accepted_for_evaluation is false" in validation.errors


def test_fully_reviewed_versioned_audit_is_accepted_and_filters_rows():
    audit = _accepted_audit()

    validation = validate_reference_audit(_rows(), audit, REPO_ROOT, DATASET)

    assert validation.accepted is True
    assert validation.dataset_version == "1.1.0"
    assert len(validation.rows) == 20
    assert validation.rows[0]["benchmark_id"] == "KGV1-001"


def test_changed_reference_without_human_review_is_rejected():
    audit = _accepted_audit()
    audit["rows"][0]["audited_reference"] = "A corrected answer."
    audit["rows"][0]["reviewer"] = {"status": "pending", "name": None, "reviewed_at": None}
    _rehash(audit)

    validation = validate_reference_audit(_rows(), audit, REPO_ROOT, DATASET)

    assert validation.accepted is False
    assert any("changed references require human approval" in error for error in validation.errors)


def test_schema_rejects_missing_stable_id_and_stale_dataset_hash():
    audit = _accepted_audit()
    audit["rows"][0]["benchmark_id"] = ""
    audit["source_dataset_sha256"] = "stale"
    _rehash(audit)

    validation = validate_reference_audit(_rows(), audit, REPO_ROOT, DATASET)

    assert validation.accepted is False
    assert any("dataset hash" in error for error in validation.errors)
    assert any("benchmark_id" in error for error in validation.errors)
