"""Persisted-score contract for the thin DeepEval regression gate."""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

RUN_OUTPUT_SCHEMA = "kinegraph.regression-gate.v1"


@dataclass(frozen=True)
class RatchetCase:
    query: str
    answer: str
    retrieved_chunks: list[str]
    ragas_composite_score: float


@dataclass(frozen=True)
class RatchetRun:
    ragas_composite_score: float
    cases: tuple[RatchetCase, ...]

    @property
    def query(self) -> str:
        return self.cases[0].query

    @property
    def answer(self) -> str:
        return self.cases[0].answer

    @property
    def retrieved_chunks(self) -> list[str]:
        return self.cases[0].retrieved_chunks

    def citation_cases(self) -> tuple[RatchetCase, ...]:
        return self.cases


def _score(value: Any, field: str = "ragas_composite_score") -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"{field} must be finite and between 0 and 1")
    return score


def _texts(value: Any, field: str = "retrieved_chunks") -> list[str]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(x, str) for x in value):
        raise ValueError(f"{field} must be a list of strings")
    return list(value)


def _case(data: Mapping[str, Any]) -> RatchetCase:
    query = str(data["query"])
    if not query.strip():
        raise ValueError("every regression case must have a non-empty query")
    return RatchetCase(
        query=query,
        answer=str(data["answer"]),
        retrieved_chunks=_texts(data["retrieved_chunks"]),
        ragas_composite_score=_score(data["ragas_composite_score"]),
    )


def load_current_run(path: str | Path = "reports/run_output.json") -> RatchetRun:
    run_path = Path(path)
    if not run_path.exists():
        raise FileNotFoundError(
            f"Accepted RAGAS run output not found at {run_path}. "
            "Run the accepted benchmark before invoking regression_gate.py."
        )
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("run output must be a JSON object")
    schema = payload.get("schema_version")
    if schema not in (None, RUN_OUTPUT_SCHEMA):
        raise ValueError(f"unsupported run output schema: {schema}")
    raw_cases = payload.get("cases") or [payload]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("run output must contain at least one case")
    return RatchetRun(
        ragas_composite_score=_score(payload["ragas_composite_score"]),
        cases=tuple(_case(item) for item in raw_cases),
    )


def load_baseline_score(path: str | Path = "baseline_ref.json") -> float:
    baseline = Path(path)
    if not baseline.exists():
        return 0.0
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("baseline reference must be a JSON object")
    return _score(payload["ragas_composite_score"])


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        temporary = Path(f.name)
    temporary.replace(path)


def promote_baseline_if_passed(run: RatchetRun, path: str | Path = "baseline_ref.json") -> None:
    _atomic_json(Path(path), {
        "schema_version": RUN_OUTPUT_SCHEMA,
        "ragas_composite_score": run.ragas_composite_score,
    })


def _context_texts(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [
        item if isinstance(item, str) else item["content"]
        for item in value
        if isinstance(item, str) or isinstance(item, Mapping) and isinstance(item.get("content"), str)
    ]


def write_run_output(
    path: str | Path,
    results: Iterable[Mapping[str, Any]],
    overall_composite_score: float,
) -> RatchetRun:
    cases = tuple(
        RatchetCase(
            query=str(row.get("question", row.get("query", ""))),
            answer=str(row.get("answer", "")),
            retrieved_chunks=_context_texts(row.get("contexts", row.get("retrieved_chunks", []))),
            ragas_composite_score=_score(row["composite_score"], "composite_score"),
        )
        for row in results
    )
    if not cases:
        raise ValueError("cannot write regression output without result rows")
    run = RatchetRun(_score(overall_composite_score), cases)
    _atomic_json(Path(path), {
        "schema_version": RUN_OUTPUT_SCHEMA,
        "query": run.query,
        "answer": run.answer,
        "retrieved_chunks": run.retrieved_chunks,
        "ragas_composite_score": run.ragas_composite_score,
        "cases": [case.__dict__ for case in cases],
    })
    return run


def build_ragas_composite_metric(baseline_score: float, min_floor: float = 0.75):
    """Create the zero-LLM composite assertion lazily."""
    try:
        from deepeval.metrics import BaseMetric
    except ImportError as exc:
        raise RuntimeError("DeepEval is required for regression_gate.py") from exc

    class RagasCompositeGate(BaseMetric):
        def __init__(self) -> None:
            self.threshold = max(_score(baseline_score, "baseline_score"), _score(min_floor, "min_floor"))
            self.name = "RagasCompositeGate"
            self.score = 0.0
            self.success = False
            self.reason = ""

        def measure(self, test_case: Any) -> float:
            self.score = _score((getattr(test_case, "additional_metadata", None) or {})["ragas_composite_score"])
            self.success = self.score >= self.threshold
            self.reason = f"composite={self.score:.4f} vs ratchet_floor={self.threshold:.4f}"
            return self.score

        async def a_measure(self, test_case: Any) -> float:
            return self.measure(test_case)

        def is_successful(self) -> bool:
            return self.success

        def get_score(self) -> float:
            return self.score

        def get_reason(self) -> str:
            return self.reason

    return RagasCompositeGate()


def default_gate_paths() -> tuple[Path, Path]:
    return (
        Path(os.getenv("KINEGRAPH_RUN_OUTPUT", "reports/run_output.json")),
        Path(os.getenv("KINEGRAPH_BASELINE_REF", "baseline_ref.json")),
    )
