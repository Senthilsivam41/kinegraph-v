import json
import os
import subprocess
import sys

import pytest

from eval.regression_gate import (
    RUN_OUTPUT_SCHEMA,
    RatchetCase,
    load_baseline_score,
    load_current_run,
    promote_baseline_if_passed,
    write_run_output,
)


def test_write_and_load_batch_run_output(tmp_path):
    output = tmp_path / "run_output.json"
    run = write_run_output(
        output,
        [
            {
                "question": "What is RRF?",
                "answer": "It combines ranked lists.",
                "contexts": ["RRF combines ranked lists."],
                "composite_score": 0.81,
            },
            {
                "question": "What is graph retrieval?",
                "answer": "It follows relationships.",
                "contexts": [{"content": "Graph retrieval follows relationships."}],
                "composite_score": 0.79,
            },
        ],
        0.80,
    )

    loaded = load_current_run(output)

    assert loaded.ragas_composite_score == pytest.approx(0.80)
    assert len(loaded.citation_cases()) == 2
    assert loaded.citation_cases()[1].retrieved_chunks == [
        "Graph retrieval follows relationships."
    ]
    assert json.loads(output.read_text())["schema_version"] == RUN_OUTPUT_SCHEMA
    assert isinstance(run.cases[0], RatchetCase)


def test_first_run_baseline_and_atomic_promotion(tmp_path):
    baseline = tmp_path / "baseline_ref.json"
    assert load_baseline_score(baseline) == 0.0

    output = tmp_path / "run_output.json"
    run = write_run_output(
        output,
        [{
            "question": "Q",
            "answer": "A",
            "contexts": ["C"],
            "composite_score": 0.76,
        }],
        0.76,
    )
    promote_baseline_if_passed(run, baseline)

    assert load_baseline_score(baseline) == pytest.approx(0.76)
    assert json.loads(baseline.read_text())["schema_version"] == RUN_OUTPUT_SCHEMA


def test_run_output_rejects_invalid_scores(tmp_path):
    with pytest.raises(ValueError, match="between 0 and 1"):
        write_run_output(
            tmp_path / "run_output.json",
            [{"question": "Q", "answer": "A", "contexts": [], "composite_score": 1.2}],
            0.9,
        )


def test_run_output_loader_rejects_unknown_schema(tmp_path):
    path = tmp_path / "run_output.json"
    path.write_text(json.dumps({"schema_version": "other"}))
    with pytest.raises(ValueError, match="unsupported run output schema"):
        load_current_run(path)


def test_run_output_loader_explains_missing_accepted_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run the accepted benchmark"):
        load_current_run(tmp_path / "missing.json")


def test_regression_gate_import_does_not_bootstrap_live_settings():
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.getcwd()}
    result = subprocess.run(
        [sys.executable, "-c", "import eval.regression_gate; import regression_gate"],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
