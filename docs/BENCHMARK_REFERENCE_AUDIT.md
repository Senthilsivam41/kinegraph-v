# Kinegraph Benchmark Reference Audit

## Current state

The 20-row benchmark now has a versioned audit sidecar at
`eval/kinegraph_benchmark_v1.audit.json`. Version `1.1.0-draft` is intentionally
not accepted for evaluation: 13 rows need correction, 5 are pending evidence
review, and 2 are ambiguous. This prevents stale or debatable reference answers
from being mistaken for retrieval failures.

The audit is evidence metadata, not a score-improvement device. Correcting a
reference changes the benchmark identity and invalidates comparisons with runs
using a previous audit hash.

## Row-level findings

| Stable ID | Status | Primary finding |
|---|---|---|
| KGV1-001 | Needs correction | Historical name and orchestration wording |
| KGV1-002 | Needs correction | Historical name and current workflow-stage evidence |
| KGV1-003 | Needs correction | Unaccepted recall and latency projections stated as facts |
| KGV1-004 | Needs correction | Neo4j incorrectly credited with cross-channel semantic coverage |
| KGV1-005 | Needs correction | RRF incorrectly attributed to ChromaDB |
| KGV1-006 | Pending | Verify concurrent retrieval against current workflow |
| KGV1-007 | Pending | Verify vector-search strengths against current service |
| KGV1-008 | Pending | Verify LangGraph/FastAPI orchestration contract |
| KGV1-009 | Ambiguous | Proposed PDFium behavior may not be deployed behavior |
| KGV1-010 | Needs correction | Reference uses an unverified Python ingestion helper |
| KGV1-011 | Pending | Verify key formatting against `.env` and settings loading |
| KGV1-012 | Needs correction | Answer describes a file layout instead of the project and modes |
| KGV1-013 | Pending | Verify endpoint, multipart fields, and response schema |
| KGV1-014 | Needs correction | Verify task-status endpoint and example response |
| KGV1-015 | Needs correction | Development credential needs explicit non-production treatment |
| KGV1-016 | Needs correction | Historical name, Docker command, memory, and key claims |
| KGV1-017 | Needs correction | Local parsing does not imply zero external API usage overall |
| KGV1-018 | Ambiguous | RRF does not require strong rank in both channels or guarantee recall |
| KGV1-019 | Needs correction | Recommendations are not enforced resource limits |
| KGV1-020 | Needs correction | Historical latency/best-result claims and incomplete RRF explanation |

The JSON sidecar contains the detailed rationale, category labels, source-file
hints and hashes, supporting context IDs, and technical-review tags for RRF,
Vectorless, Hybrid, API, security, and resource-limit claims.

## Human review workflow

1. For each row, resolve every claim to current checked-in source evidence.
2. Correct `audited_reference` based on that evidence—not based on a generated
   answer or desired metric score.
3. Mark subjective or non-unique answers `excluded` or `multi_reference`.
4. Add a verbatim checked-in evidence excerpt, its SHA-256, and exact supporting
   chunk IDs; only then set the evidence entry to `verification: verified`.
5. Complete explicit technical review for every tagged claim.
6. For accepted rows, record a named human reviewer and timestamp.
7. Change the dataset version, set `accepted_for_evaluation: true`, and refresh
   `audit_content_sha256` with `--rehash`.

Validate at any time:

```bash
PYTHONPATH=. venv/bin/python scripts/audit_benchmark_references.py
PYTHONPATH=. venv/bin/python scripts/audit_benchmark_references.py --rehash
```

The validator rejects missing or duplicate row IDs, stale question/reference/
context hashes, missing categories, invalid reviewer states, changed answers
without human approval, missing checked-in sources, missing chunk IDs, pending
technical review, path traversal outside the repository, and a stale audit hash.
