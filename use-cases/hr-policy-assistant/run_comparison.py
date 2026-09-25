"""Run every synthetic HR case in Hybrid and Vector modes."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import client


MODES = ("hybrid", "vector")


def build_report(query, code_revision, environment, captured_at=None):
    captured_at = captured_at or datetime.now(timezone.utc).isoformat()
    rows = []
    for case in client.CASES:
        for mode in MODES:
            try:
                response = query(case, mode)
            except Exception as error:
                rows.append({
                    "case_id": case["id"],
                    "requested_mode": mode,
                    "status": "failure",
                    "error": f"{type(error).__name__}: {error}",
                })
                continue

            artifact = client.build_artifact(case, mode, response, code_revision, captured_at)
            answer = str(response.get("generated_answer", "")).casefold()
            artifact["status"] = "success"
            artifact["checks"] = {
                "mode_downgrade": response.get("effective_mode") != mode,
                "evidence_present": any(key in response for key in ("citation_validation", "grounded_claims")),
                "missing_expected_facts": [fact for fact in case["expected_facts"] if fact.casefold() not in answer],
            }
            rows.append(artifact)

    successes = [row for row in rows if row["status"] == "success"]
    return {
        "schema_version": "kinegraph.demo.comparison.v1",
        "captured_at": captured_at,
        "environment": environment,
        "kinegraph_code_revision": code_revision,
        "source_corpus_sha256": client.CONTRACT["corpus_sha256"],
        "results": rows,
        "summary": {
            "total_runs": len(rows),
            "successful_runs": len(successes),
            "failed_runs": len(rows) - len(successes),
            "mode_downgrades": sum(row["checks"]["mode_downgrade"] for row in successes),
            "runs_missing_evidence": sum(not row["checks"]["evidence_present"] for row in successes),
            "runs_missing_expected_facts": sum(bool(row["checks"]["missing_expected_facts"]) for row in successes),
            "comparative_claim": "not_evaluated",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("KINEGRAPH_API_URL", "http://localhost:8000"))
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--environment", default="local")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(
        lambda case, mode: client.query_case(args.base_url, case, mode),
        args.code_revision,
        args.environment,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
