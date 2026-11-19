import importlib.util
import io
import json
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest


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
        "graph_participation": {
            "required_case_ids": ["HR-001", "HR-004"],
            "missing_case_ids": ["HR-001", "HR-004"],
            "passed": False,
        },
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


@pytest.mark.parametrize(
    ("initial_count", "selected", "evidence", "missing"),
    [
        (3, True, "The policy applies to Sales.", []),
        (0, True, "The policy applies to Sales.", ["HR-004"]),
        (3, False, "The policy applies to Sales.", ["HR-004"]),
        (3, True, " ", ["HR-004"]),
    ],
)
def test_graph_participation_requires_an_initial_candidate_and_a_selected_evidence_path(
    initial_count, selected, evidence, missing,
):
    runner = load_runner()

    def query(case, mode):
        expense_case = case["id"] == "HR-004"
        count = initial_count if expense_case else 3
        path = [{"evidence_text": "Source evidence."}]
        if expense_case:
            path.append({"evidence_text": evidence})
        chunks = [{"source": "vector", "metadata": {"graph_paths": {"graph": path}}}]
        return {
            "effective_mode": mode,
            "recovery_details": {"route_escalation": {"initial_channel_counts": {"graph": count}}},
            "results": chunks if selected or not expense_case else [],
        }

    report = runner.build_report(query, "abc123", "test")
    checks = {
        row["case"]["id"]: row["checks"]
        for row in report["results"] if row["requested_mode"] == "hybrid"
    }

    assert checks["HR-001"]["initial_graph_candidate_count"] == 3
    assert checks["HR-001"]["selected_evidence_graph_path_count"] == 1
    assert checks["HR-004"]["initial_graph_candidate_count"] == initial_count
    assert checks["HR-004"]["selected_evidence_graph_path_count"] == int(selected and bool(evidence.strip()))
    assert report["summary"]["graph_participation"]["missing_case_ids"] == missing
    assert report["summary"]["graph_participation"]["passed"] is (not missing)
    assert report["summary"]["comparative_claim"] == "not_evaluated"


def test_november_smoke_responses_do_not_pass_graph_participation():
    runner = load_runner()
    artifact = USE_CASE_DIR.parents[1] / "reports/test-runs/2026-11-11/hr-policy/comparison.json"
    saved = json.loads(artifact.read_text())
    responses = {(row["case"]["id"], row["requested_mode"]): row["response"] for row in saved["results"]}

    report = runner.build_report(lambda case, mode: responses[case["id"], mode], "abc123", "test")

    assert report["summary"]["successful_runs"] == 12
    assert report["summary"]["runs_missing_expected_facts"] == 10
    assert report["summary"]["graph_participation"]["passed"] is False
    for row in report["results"]:
        if row["requested_mode"] == "hybrid":
            assert row["checks"]["initial_graph_candidate_count"] == 0
            assert row["checks"]["selected_evidence_graph_path_count"] == 0
