"""Run every synthetic HR case in Hybrid and Vector modes."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError

import client


MODES = ("hybrid", "vector")


def summarize(rows):
    successes = [row for row in rows if row["status"] == "success"]
    return {
        "total_runs": len(rows),
        "successful_runs": len(successes),
        "failed_runs": len(rows) - len(successes),
        "mode_downgrades": sum(row["checks"]["mode_downgrade"] for row in successes),
        "runs_missing_evidence": sum(not row["checks"]["evidence_present"] for row in successes),
        "runs_without_cited_claims": sum(not row["checks"]["cited_claims_present"] for row in successes),
        "runs_missing_expected_facts": sum(bool(row["checks"]["missing_expected_facts"]) for row in successes),
    }


def build_report(query, code_revision, environment, captured_at=None):
    captured_at = captured_at or datetime.now(timezone.utc).isoformat()
    rows = []
    for case in client.CASES:
        for mode in MODES:
            started = perf_counter()
            try:
                response = query(case, mode)
            except (OSError, ValueError) as error:
                detail = {}
                if isinstance(error, HTTPError):
                    with error:
                        try:
                            payload = json.load(error)
                            if isinstance(payload, dict) and isinstance(payload.get("detail"), dict):
                                detail = payload["detail"]
                        except (OSError, ValueError):
                            pass
                artifact = client.build_artifact(case, mode, detail, code_revision, captured_at)
                artifact.update(
                    case_id=case["id"],
                    status="failure",
                    error=f"{type(error).__name__}: {error}",
                    client_elapsed_ms=round((perf_counter() - started) * 1000, 2),
                )
                rows.append(artifact)
                continue

            artifact = client.build_artifact(case, mode, response, code_revision, captured_at)
            answer = (response.get("generated_answer") or "").casefold()
            artifact["status"] = "success"
            artifact["client_elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
            claims = response.get("grounded_claims") or []
            artifact["checks"] = {
                "mode_downgrade": response.get("effective_mode") != mode,
                "evidence_present": any(response.get(key) for key in client.EVIDENCE_KEYS),
                "cited_claims_present": any(isinstance(claim, dict) and claim.get("chunk_ids") for claim in claims),
                "missing_expected_facts": [fact for fact in case["expected_facts"] if fact.casefold() not in answer],
            }
            rows.append(artifact)

    return {
        "schema_version": "kinegraph.demo.comparison.v1",
        "captured_at": captured_at,
        "environment": environment,
        "kinegraph_code_revision": code_revision,
        "source_corpus_sha256": client.CONTRACT["corpus_sha256"],
        "ingestion_pdf_sha256": client.PDF_SHA256,
        "results": rows,
        "summary": {
            **summarize(rows),
            "by_mode": {
                mode: summarize([row for row in rows if row["requested_mode"] == mode])
                for mode in MODES
            },
            "comparative_claim": "not_evaluated",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("KINEGRAPH_API_URL", "http://localhost:8000"))
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--environment", default="local")
    parser.add_argument("--timeout", type=int, default=60, help="Seconds to wait for each query")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be at least one second")

    report = build_report(
        lambda case, mode: client.query_case(args.base_url, case, mode, timeout=args.timeout),
        args.code_revision,
        args.environment,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
