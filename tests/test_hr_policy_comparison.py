import importlib.util
import sys
from pathlib import Path


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
            raise ConnectionError("offline")
        answer = " ".join(case["expected_facts"])
        return {
            "generated_answer": answer,
            "effective_mode": "vector" if case["id"] == "HR-005" else mode,
            "citation_validation": {"valid": True},
            "execution_time_ms": 10,
        }

    report = runner.build_report(query, "abc123", "test", "2026-09-25T00:00:00+00:00")

    assert report["schema_version"] == "kinegraph.demo.comparison.v1"
    assert report["summary"] == {
        "total_runs": 12,
        "successful_runs": 11,
        "failed_runs": 1,
        "mode_downgrades": 1,
        "runs_missing_evidence": 0,
        "runs_missing_expected_facts": 0,
        "comparative_claim": "not_evaluated",
    }
    failure = next(row for row in report["results"] if row["status"] == "failure")
    assert failure["case_id"] == "HR-006"
    assert failure["requested_mode"] == "vector"
    assert failure["error"] == "ConnectionError: offline"
