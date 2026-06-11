"""
LangSmith Tracing Integration — KineticGraph-Vectra
Wraps LangGraph workflow with end-to-end trace capture, feedback collection,
and per-step latency/token logging. Degrades gracefully when LANGSMITH_API_KEY is absent.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional LangSmith import
# ---------------------------------------------------------------------------
try:
    from langsmith import Client as LangSmithClient
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False
    logger.warning(
        "langsmith not installed — tracing disabled. pip install langsmith"
    )


# ---------------------------------------------------------------------------
# Token-cost estimation (GPT-4o-mini pricing as of 2024)
# ---------------------------------------------------------------------------
_COST_PER_1K_INPUT = 0.00015   # USD per 1k input tokens
_COST_PER_1K_OUTPUT = 0.00060  # USD per 1k output tokens


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        (input_tokens * _COST_PER_1K_INPUT + output_tokens * _COST_PER_1K_OUTPUT) / 1000,
        6,
    )


# ---------------------------------------------------------------------------
# LangSmithTracer
# ---------------------------------------------------------------------------

class LangSmithTracer:
    """
    End-to-end tracing for the KineticGraph-Vectra hybrid RAG pipeline.

    Captures per-step latency (router → vector/graph agent → fusion → generation),
    tags traces with query mode / session metadata, logs token usage and cost,
    and exposes a feedback endpoint for thumbs-up/down scoring.

    When LangSmith is unavailable the class operates in a no-op / local-log mode
    so the rest of the application continues to work.

    Usage::

        tracer = LangSmithTracer()

        with tracer.trace_query(query="...", mode="hybrid") as run_ctx:
            tracer.log_step(run_ctx["run_id"], "vector_agent", latency_ms=45.2)
            tracer.log_step(run_ctx["run_id"], "graph_agent",  latency_ms=62.1)
            tracer.log_step(run_ctx["run_id"], "fusion_node",  latency_ms=5.3)
            tracer.log_generation(run_ctx["run_id"], tokens_input=312, tokens_output=128)

        tracer.log_feedback(run_ctx["run_id"], score=1.0, comment="Great answer!")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_name: str = "kinegraph-vectra",
        endpoint: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY", "")
        self.project_name = project_name
        self.endpoint = endpoint or os.getenv(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        )
        self._client: Optional[Any] = None
        self._local_runs: Dict[str, Dict[str, Any]] = {}  # fallback in-memory store
        self._setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        if not _LANGSMITH_AVAILABLE:
            logger.info("LangSmith tracing not available — local-log mode active.")
            return
        if not self.api_key:
            logger.warning("LANGSMITH_API_KEY not set — local-log mode active.")
            return
        try:
            self._client = LangSmithClient(
                api_url=self.endpoint,
                api_key=self.api_key,
            )
            # Activate env vars for automatic tracing via @traceable
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            os.environ.setdefault("LANGCHAIN_PROJECT", self.project_name)
            os.environ.setdefault("LANGCHAIN_API_KEY", self.api_key)
            logger.info("LangSmith client connected. Project: %s", self.project_name)
        except Exception as exc:
            logger.error("LangSmith init error: %s", exc)
            self._client = None

    @property
    def is_active(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------
    # Context manager — wraps a full RAG query
    # ------------------------------------------------------------------

    @contextmanager
    def trace_query(
        self,
        query: str,
        mode: str,
        session_id: Optional[str] = None,
        document_id: Optional[str] = None,
        extra_tags: Optional[Dict[str, str]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager that opens a LangSmith run (or a local trace entry)
        for a full query execution.

        Yields a dict with::

            {
                "run_id": str,
                "start_time": float,
                "mode": str,
                "session_id": str,
            }

        Usage::

            with tracer.trace_query(query="...", mode="hybrid") as ctx:
                ...  # execute pipeline
        """
        run_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        start_time = time.perf_counter()

        tags = [f"mode:{mode}", f"session:{session_id}"]
        if document_id:
            tags.append(f"doc:{document_id}")
        if extra_tags:
            tags.extend([f"{k}:{v}" for k, v in extra_tags.items()])

        metadata = {
            "query_mode": mode,
            "session_id": session_id,
            "document_id": document_id,
            "project": self.project_name,
        }

        # Initialise local tracking entry
        self._local_runs[run_id] = {
            "run_id": run_id,
            "query": query,
            "mode": mode,
            "session_id": session_id,
            "start_time": start_time,
            "steps": [],
            "tokens": {"input": 0, "output": 0},
            "status": "running",
        }

        ctx: Dict[str, Any] = {
            "run_id": run_id,
            "start_time": start_time,
            "mode": mode,
            "session_id": session_id,
        }

        if self._client:
            try:
                run_obj = self._client.create_run(
                    name=f"kinegraph_query_{mode}",
                    run_type="chain",
                    inputs={"query": query, "mode": mode},
                    tags=tags,
                    extra={"metadata": metadata},
                    project_name=self.project_name,
                )
                ctx["langsmith_run_id"] = str(run_obj.id) if run_obj else run_id
            except Exception as exc:
                logger.debug("LangSmith create_run error (non-fatal): %s", exc)
                ctx["langsmith_run_id"] = run_id

        try:
            yield ctx
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            self._local_runs[run_id]["status"] = "success"
            self._local_runs[run_id]["total_latency_ms"] = elapsed_ms

            if self._client:
                try:
                    ls_run_id = ctx.get("langsmith_run_id", run_id)
                    self._client.update_run(
                        run_id=ls_run_id,
                        outputs={"total_latency_ms": elapsed_ms},
                        end_time=time.time(),
                    )
                except Exception as exc:
                    logger.debug("LangSmith update_run error (non-fatal): %s", exc)

        except Exception as exc:
            self._local_runs[run_id]["status"] = "error"
            self._local_runs[run_id]["error"] = str(exc)
            if self._client:
                try:
                    self._client.update_run(
                        run_id=ctx.get("langsmith_run_id", run_id),
                        error=str(exc),
                        end_time=time.time(),
                    )
                except Exception:
                    pass
            raise

    # ------------------------------------------------------------------
    # Step-level logging
    # ------------------------------------------------------------------

    def log_step(
        self,
        run_id: str,
        step_name: str,
        latency_ms: float,
        result_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a pipeline step (e.g., vector_agent, graph_agent, fusion_node)."""
        step = {
            "step": step_name,
            "latency_ms": latency_ms,
            "result_count": result_count,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        if run_id in self._local_runs:
            self._local_runs[run_id]["steps"].append(step)

        logger.debug("[%s] step=%s latency=%.1fms results=%d",
                     run_id[:8], step_name, latency_ms, result_count)

    def log_generation(
        self,
        run_id: str,
        tokens_input: int,
        tokens_output: int,
        model: str = "gpt-4o-mini",
    ) -> None:
        """Log LLM token usage and estimated cost for a run."""
        cost = _estimate_cost(tokens_input, tokens_output)
        if run_id in self._local_runs:
            self._local_runs[run_id]["tokens"] = {
                "input": tokens_input,
                "output": tokens_output,
                "total": tokens_input + tokens_output,
                "estimated_cost_usd": cost,
                "model": model,
            }
        logger.debug("[%s] tokens in=%d out=%d cost=$%.6f",
                     run_id[:8], tokens_input, tokens_output, cost)

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def log_feedback(
        self,
        run_id: str,
        score: float,
        comment: Optional[str] = None,
        key: str = "user_rating",
    ) -> None:
        """
        Log thumbs-up/down or numeric feedback for a completed run.

        Args:
            run_id: The run ID returned by trace_query context.
            score: 1.0 = thumbs-up, 0.0 = thumbs-down (or any 0–1 float).
            comment: Optional textual feedback.
            key: Feedback key name in LangSmith.
        """
        if run_id in self._local_runs:
            self._local_runs[run_id]["feedback"] = {
                "score": score,
                "comment": comment,
                "key": key,
                "timestamp": time.time(),
            }

        if self._client:
            try:
                ls_run_id = self._local_runs.get(run_id, {}).get(
                    "langsmith_run_id", run_id
                )
                self._client.create_feedback(
                    run_id=ls_run_id,
                    key=key,
                    score=score,
                    comment=comment or "",
                )
                logger.info("Feedback logged: run=%s score=%.1f", run_id[:8], score)
            except Exception as exc:
                logger.error("LangSmith feedback error: %s", exc)
        else:
            logger.info("[LOCAL] Feedback: run=%s score=%.1f comment=%s",
                        run_id[:8], score, comment)

    # ------------------------------------------------------------------
    # Metrics retrieval
    # ------------------------------------------------------------------

    def get_run_metrics(self, run_id: str) -> Dict[str, Any]:
        """
        Return metrics for a completed run.

        Checks local in-memory store first; falls back to LangSmith API.
        """
        if run_id in self._local_runs:
            run = self._local_runs[run_id]
            return {
                "run_id": run_id,
                "query": run.get("query"),
                "mode": run.get("mode"),
                "session_id": run.get("session_id"),
                "status": run.get("status"),
                "total_latency_ms": run.get("total_latency_ms"),
                "steps": run.get("steps", []),
                "tokens": run.get("tokens", {}),
                "feedback": run.get("feedback"),
                "source": "local",
            }

        if self._client:
            try:
                run_obj = self._client.read_run(run_id)
                return {
                    "run_id": run_id,
                    "name": run_obj.name,
                    "status": run_obj.status,
                    "start_time": str(run_obj.start_time),
                    "end_time": str(run_obj.end_time),
                    "inputs": run_obj.inputs,
                    "outputs": run_obj.outputs,
                    "tags": run_obj.tags,
                    "source": "langsmith",
                }
            except Exception as exc:
                logger.error("LangSmith read_run error: %s", exc)

        return {"run_id": run_id, "error": "Run not found"}

    def list_recent_runs(self, limit: int = 20) -> list:
        """List the most recent runs (from local store + LangSmith if available)."""
        local = [
            {
                "run_id": rid,
                "query": r.get("query"),
                "mode": r.get("mode"),
                "status": r.get("status"),
                "latency_ms": r.get("total_latency_ms"),
                "source": "local",
            }
            for rid, r in list(self._local_runs.items())[-limit:]
        ]
        return local
