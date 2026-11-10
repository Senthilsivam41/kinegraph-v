import importlib.util
import io
import sys
from pathlib import Path
from urllib.error import HTTPError


USE_CASE_DIR = Path(__file__).parents[1] / "use-cases" / "hr-policy-assistant"


def load_runner():
    sys.path.insert(0, str(USE_CASE_DIR))
    try:
        spec = importlib.util.spec_from_file_location("hr_policy_comparison", USE_CASE_DIR / "run_comparison.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_comparison_runs_every_case_and_retains_failures():
    runner = load_runner()

    def query(case, mode):
        if case["id"] == "HR-006" and mode == "vector":
            raise HTTPError(
                "http://localhost:8000/api/v1/query/", 503, "Unavailable", {},
                io.BytesIO(b'{"detail":{"code":"store_unavailable","query_id":"failed-query"}}'),
            )
        answer = " ".join(case["expected_facts"])
        return {
            "generated_answer": answer,
            "effective_mode": "vector" if case["id"] == "HR-005" else mode,
            "citation_validation": None if case["id"] == "HR-004" else {"valid": True},
            "grounded_claims": [] if case["id"] == "HR-004" else [{"chunk_ids": ["chunk-1"]}],
            "execution_time_ms": 10,
        }

    report = runner.build_report(query, "abc123", "test", "2026-09-25T00:00:00+00:00")

    assert report["schema_version"] == "kinegraph.demo.comparison.v1"
    summary = report["summary"]
    assert {key: value for key, value in summary.items() if key != "by_mode"} == {
        "total_runs": 12,
        "successful_runs": 11,
        "failed_runs": 1,
        "mode_downgrades": 1,
        "runs_missing_evidence": 2,
        "runs_without_cited_claims": 2,
        "runs_missing_expected_facts": 0,
        "comparative_claim": "not_evaluated",
    }
    assert summary["by_mode"]["hybrid"]["mode_downgrades"] == 1
    assert summary["by_mode"]["vector"]["failed_runs"] == 1
    failure = next(row for row in report["results"] if row["status"] == "failure")
    assert [(row["case"]["id"], row["requested_mode"]) for row in report["results"]] == [
        (case["id"], mode) for case in runner.client.CASES for mode in ("hybrid", "vector")
    ]
    assert {row["ingestion_pdf_sha256"] for row in report["results"]} == {runner.client.PDF_SHA256}
    assert failure["case_id"] == "HR-006"
    assert failure["requested_mode"] == "vector"
    assert failure["error"] == "HTTPError: HTTP Error 503: Unavailable"
    assert failure["query_id"] == "failed-query"
    assert failure["client_elapsed_ms"] >= 0
