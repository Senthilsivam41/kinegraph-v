"""
RAGAS Evaluation Module — KineticGraph-Vectra.

Heuristic scores are available only for explicitly requested diagnostic runs.
Accepted benchmark runs fail closed when the RAGAS judge cannot be configured.
"""
from __future__ import annotations

import os
import sys
# Path patch to support running directly or without PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import json
import re
import time
from typing import Any, Dict, List, Mapping, Optional

from backend.app.models import QueryMode
from backend.core.config import settings
from eval.benchmark_reference_audit import load_reference_audit, validate_reference_audit
from eval.benchmark_profiles import PROFILES, build_profile_dataset, get_profile
from eval.experiment_validation import (
    ValidationPolicy,
    bootstrap_mean_interval,
    build_manifest,
    compare_manifests,
    current_git_revision,
    load_manifest,
    validate_metric_values,
    weighted_composite,
    working_tree_is_clean,
    write_manifest,
)
from eval.provenance import (
    build_provenance_record,
    write_diagnostic_summary,
    write_provenance_jsonl,
)
from eval.ir_metrics import aggregate_ir_metrics, score_retrieval
from eval.kinetic_score import calibrate_shadow_scores, compute_kinetic_score_shadow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False
    logger.warning(
        "ragas is not installed; accepted evaluation is unavailable. "
        "Install the pinned requirements to enable it."
    )

try:
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
ALL_METRICS = DEFAULT_METRICS + ["answer_correctness"]
JUDGE_MAX_RETRIES = 2
JUDGE_RETRY_BACKOFF_SECONDS = 1.0
JUDGE_MAX_TOKENS = 4096

METRIC_DESCRIPTIONS = {
    "faithfulness": "Fraction of answer claims supported by retrieved context",
    "answer_relevancy": "How well the answer addresses the question",
    "context_precision": "Signal-to-noise: most relevant chunks ranked first",
    "context_recall": "Fraction of ground-truth info present in context",
    "answer_correctness": "Answer accuracy vs reference ground truth",
}


class RAGASValidationError(RuntimeError):
    """Raised when a benchmark contains fallback scores presented as RAGAS."""


class RAGASConfigurationError(RuntimeError):
    """Raised when the configured RAGAS judge cannot be initialized."""


def require_successful_ragas(
    results: pd.DataFrame,
    expected_rows: Optional[int] = None,
    required_metrics: Optional[List[str]] = None,
) -> None:
    """Fail closed unless every evaluated row is a successful RAGAS result."""
    if results.empty:
        raise RAGASValidationError("No evaluation rows were produced.")
    if expected_rows is not None and len(results) != expected_rows:
        raise RAGASValidationError(
            f"Expected {expected_rows} evaluation rows but received {len(results)}."
        )
    if "workflow_error" in results.columns:
        workflow_failures = results[results["workflow_error"].fillna("").astype(str).str.strip() != ""]
        if not workflow_failures.empty:
            raise RAGASValidationError(
                f"Rejected {len(workflow_failures)}/{len(results)} rows because the live workflow failed."
            )
    if "mode_profile_error" in results.columns:
        profile_failures = results[
            results["mode_profile_error"].fillna("").astype(str).str.strip() != ""
        ]
        if not profile_failures.empty:
            raise RAGASValidationError(
                f"Rejected {len(profile_failures)}/{len(results)} rows because the effective mode "
                "left the declared benchmark profile."
            )
    if "ragas_failed" not in results.columns:
        raise RAGASValidationError(
            "Missing ragas_failed provenance; this run cannot be accepted as RAGAS."
        )

    failed = results[results["ragas_failed"].fillna(True).astype(bool)]
    if not failed.empty:
        errors = sorted({
            str(error) for error in failed.get("ragas_error", pd.Series(dtype=str)).dropna()
        })
        detail = f" Errors: {'; '.join(errors)}" if errors else ""
        raise RAGASValidationError(
            f"Rejected {len(failed)}/{len(results)} rows because RAGAS failed and "
            f"heuristic fallback scores are not benchmark evidence.{detail}"
        )

    metrics = required_metrics or DEFAULT_METRICS
    invalid_rows = []
    for index, row in results.iterrows():
        try:
            validate_metric_values(row.to_dict(), metrics)
        except ValueError as exc:
            invalid_rows.append(f"row {index}: {exc}")
    if invalid_rows:
        raise RAGASValidationError(
            "Rejected invalid metric output. " + "; ".join(invalid_rows[:5])
        )


# ---------------------------------------------------------------------------
# Heuristic fallback helpers
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return len(wa & wb) / len(wa | wb) if (wa | wb) else 0.0


def _keyword_faithfulness(answer: str, contexts: List[str]) -> float:
    if not contexts or not answer:
        return 0.0
    ctx = " ".join(contexts).lower()
    words = [w.strip(".,!?") for w in answer.lower().split() if len(w) > 3]
    return round(sum(1 for w in words if w in ctx) / len(words), 4) if words else 0.0


def _fallback_evaluate(
    question: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None
) -> Dict[str, float]:
    scores: Dict[str, float] = {
        "faithfulness": _keyword_faithfulness(answer, contexts),
        "answer_relevancy": round(_jaccard(answer, question), 4),
        "context_precision": float(len(contexts) > 0),
        "context_recall": _keyword_faithfulness(ground_truth or "", contexts),
    }
    if ground_truth:
        scores["answer_correctness"] = round(_jaccard(answer, ground_truth), 4)
    return scores


# ---------------------------------------------------------------------------
# RAGASEvaluator
# ---------------------------------------------------------------------------

class RAGASEvaluator:
    """
    Production-ready RAGAS evaluator for the KineticGraph-Vectra pipeline.

    Usage::

        ev = RAGASEvaluator(openai_api_key="sk-...")
        scores = ev.evaluate_single(question=..., answer=..., contexts=[...])
        df = ev.evaluate_batch(dataset=[...])
        report = ev.generate_report(df)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = settings.LLM_MODEL,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        metrics: Optional[List[str]] = None,
        critic_model: Optional[str] = None,
        critic_api_key: Optional[str] = None,
        critic_base_url: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        local_embeddings_only: bool = True,
        allow_heuristic_fallback: bool = False,
        judge_max_retries: int = JUDGE_MAX_RETRIES,
        judge_retry_backoff_seconds: float = JUDGE_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self.provider = (provider or os.getenv("RAGAS_JUDGE_PROVIDER") or "").strip().lower()
        self.openai_api_key = (
            openai_api_key
            or (os.getenv("FIREWORKS_API_KEY") if self.provider == "fireworks" else None)
            or (os.getenv("NVIDIA_API_KEY") if self.provider == "nvidia" else None)
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_AI_KEY")
            or os.getenv("OPENAI_API_KEY")
            or getattr(settings, "OPENAI_API_KEY", None)
        )
        self.model = model
        self.embedding_model = embedding_model
        self.metrics_names = metrics or DEFAULT_METRICS
        self.critic_model = critic_model
        self.critic_api_key = critic_api_key
        self.critic_base_url = critic_base_url
        self.base_url = base_url or os.getenv("RAGAS_JUDGE_BASE_URL")
        if not self.base_url and self.provider == "nvidia":
            self.base_url = "https://integrate.api.nvidia.com/v1"
        if not self.base_url and self.provider == "fireworks":
            self.base_url = "https://api.fireworks.ai/inference/v1"
        self.local_embeddings_only = local_embeddings_only
        self.allow_heuristic_fallback = allow_heuristic_fallback
        self.judge_max_retries = max(0, int(judge_max_retries))
        self.judge_retry_backoff_seconds = max(0.0, float(judge_retry_backoff_seconds))
        self._llm = None
        self._critic_llm = None
        self._embeddings = None
        self._ragas_metrics: List[Any] = []
        self._setup_error: Optional[str] = None
        self._setup()
        if not self.allow_heuristic_fallback:
            self.require_ready()

    def _setup(self) -> None:
        if not _RAGAS_AVAILABLE:
            self._setup_error = "ragas and datasets are not installed"
            return
        if not _OPENAI_AVAILABLE:
            self._setup_error = (
                "langchain-openai and langchain-huggingface are not installed"
            )
            return
        try:
            if not self.openai_api_key:
                raise ValueError(
                    "No judge API key configured. Set FIREWORKS_API_KEY for the "
                    "Fireworks provider, NVIDIA_API_KEY for NVIDIA, "
                    "NVIDIA provider, OPENROUTER_API_KEY, or OPENAI_API_KEY."
                )

            is_openrouter = (
                self.provider == "openrouter"
                or bool(self.base_url and "openrouter.ai" in self.base_url)
                or self.openai_api_key.startswith("sk-or-")
            )
            if self.provider and self.provider not in {"openai", "openrouter", "nvidia", "fireworks"}:
                raise ValueError(
                    f"Unsupported RAGAS judge provider '{self.provider}'. "
                    "Use 'openai', 'openrouter', 'nvidia', or 'fireworks'."
                )

            judge_base_url = self.base_url
            if is_openrouter and not judge_base_url:
                judge_base_url = "https://openrouter.ai/api/v1"
            kw: Dict[str, Any] = {
                "model": self.model,
                "temperature": 0,
                # Bound judge output so OpenRouter cannot reserve an
                # unnecessarily large context budget for each metric call.
                "max_tokens": max(
                    256,
                    int(os.getenv("RAGAS_JUDGE_MAX_TOKENS", str(JUDGE_MAX_TOKENS))),
                ),
                "openai_api_key": self.openai_api_key,
            }
            if judge_base_url:
                kw["base_url"] = judge_base_url
            self._llm = ChatOpenAI(**kw)

            # RAGAS semantic metrics need embeddings, but OpenRouter is an LLM
            # router and should not be treated as an embedding endpoint. Keeping
            # this model local also makes baseline runs reproducible.
            model_kwargs: Dict[str, Any] = {}
            if self.local_embeddings_only:
                model_kwargs["local_files_only"] = True
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs=model_kwargs,
            )

            # Setup separate evaluation model (critic) if requested
            if self.critic_model:
                ckw: Dict[str, Any] = {"model": self.critic_model, "temperature": 0}
                c_key = self.critic_api_key or os.getenv("CRITIC_API_KEY") or self.openai_api_key
                if c_key:
                    ckw["openai_api_key"] = c_key
                if self.critic_base_url:
                    ckw["base_url"] = self.critic_base_url
                elif c_key and (c_key.startswith("sk-or-") or "openrouter" in c_key):
                    ckw["base_url"] = "https://openrouter.ai/api/v1"

                if "claude" in self.critic_model.lower():
                    try:
                        from langchain_anthropic import ChatAnthropic
                        self._critic_llm = ChatAnthropic(
                            model=self.critic_model, 
                            temperature=0, 
                            anthropic_api_key=c_key or os.getenv("ANTHROPIC_API_KEY")
                        )
                    except ImportError:
                        logger.warning("langchain-anthropic not installed. Defaulting to ChatOpenAI for critic.")
                        self._critic_llm = ChatOpenAI(**ckw)
                else:
                    self._critic_llm = ChatOpenAI(**ckw)
            else:
                self._critic_llm = self._llm

            _map = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "answer_correctness": answer_correctness,
            }
            unknown_metrics = sorted(set(self.metrics_names) - set(_map))
            if unknown_metrics:
                raise ValueError(
                    f"Unsupported RAGAS metric(s): {', '.join(unknown_metrics)}"
                )
            self._ragas_metrics = [_map[m] for m in self.metrics_names]
            self._setup_error = None
            logger.info("RAGASEvaluator ready with metrics: %s", self.metrics_names)
        except Exception as exc:
            self._llm = None
            self._critic_llm = None
            self._embeddings = None
            self._ragas_metrics = []
            self._setup_error = f"{type(exc).__name__}: {exc}"
            logger.exception("RAGAS setup failed")

    def readiness(self) -> Dict[str, Any]:
        """Return a secret-free, persistable description of judge readiness."""
        if self.provider in {"nvidia", "fireworks"}:
            provider_name = self.provider
        elif (
            self.provider == "openrouter"
            or bool(self.base_url and "openrouter.ai" in self.base_url)
            or bool(self.openai_api_key and self.openai_api_key.startswith("sk-or-"))
        ):
            provider_name = "openrouter"
        else:
            provider_name = self.provider or "openai"
        return {
            "ready": self._setup_error is None and bool(self._ragas_metrics),
            "provider": provider_name,
            "judge_model": self.critic_model or self.model,
            "embedding_model": self.embedding_model,
            "local_embeddings_only": self.local_embeddings_only,
            "metrics": list(self.metrics_names),
            "setup_error": self._setup_error,
        }

    def require_ready(self) -> None:
        """Fail before retrieval starts when judge configuration is invalid."""
        status = self.readiness()
        if not status["ready"]:
            raise RAGASConfigurationError(
                "RAGAS judge preflight failed: "
                + (self._setup_error or "no metrics were initialized")
            )

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _judge_exception_details(exc: BaseException) -> str:
        """Return actionable, secret-free details for a failed judge call."""
        details = [f"{type(exc).__module__}.{type(exc).__name__}", repr(exc)]
        status = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        if status is not None:
            details.append(f"status_code={status}")
        request_id = getattr(exc, "request_id", None)
        if request_id is None and response is not None:
            headers = getattr(response, "headers", {}) or {}
            request_id = headers.get("x-request-id") or headers.get("request-id")
        if request_id:
            details.append(f"request_id={request_id}")
        return "; ".join(details)

    @staticmethod
    def _is_transient_judge_error(exc: BaseException) -> bool:
        """Retry transport, timeout, rate-limit, and server-side failures only."""
        status = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        if status is not None:
            try:
                return int(status) == 408 or int(status) == 429 or int(status) >= 500
            except (TypeError, ValueError):
                pass
        name = type(exc).__name__.lower()
        return any(token in name for token in (
            "timeout", "timeouterror", "connecterror", "connectionerror",
            "rate", "temporar", "serviceunavailable", "internalserver",
        ))

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate one QA sample. Returns metric_name → score (0-1) and metadata."""
        if not _RAGAS_AVAILABLE or not self._ragas_metrics:
            scores = _fallback_evaluate(question, answer, contexts, ground_truth)
            return {
                **scores,
                "ragas_failed": True,
                "ragas_error": (
                    f"RAGAS setup failed: {self._setup_error}"
                    if self._setup_error
                    else "RAGAS not available or not configured"
                ),
            }

        from datasets import Features, Sequence, Value
        row: Dict[str, Any] = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        features = Features({
            "question": Value("string"),
            "answer": Value("string"),
            "contexts": Sequence(Value("string")),
        })
        if ground_truth:
            row["ground_truth"] = [ground_truth]
            features["ground_truth"] = Value("string")

        dataset = Dataset.from_dict(row, features=features)
        active = self._ragas_metrics
        if not ground_truth:
            active = [m for m in active if m not in (answer_correctness, context_recall)]

        try:
            result = None
            last_error: Optional[BaseException] = None
            for attempt in range(self.judge_max_retries + 1):
                try:
                    # Keep this strict: a retry may recover a transient judge
                    # failure, but no exception is converted into a score.
                    result = ragas_evaluate(
                        dataset=dataset,
                        metrics=active,
                        llm=self._critic_llm or self._llm,
                        embeddings=self._embeddings,
                        raise_exceptions=True,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if (
                        attempt >= self.judge_max_retries
                        or not self._is_transient_judge_error(exc)
                    ):
                        raise
                    delay = self.judge_retry_backoff_seconds * (2 ** attempt)
                    logger.warning(
                        "Transient RAGAS judge failure (attempt %d/%d): %s; retrying in %.1fs",
                        attempt + 1,
                        self.judge_max_retries + 1,
                        self._judge_exception_details(exc),
                        delay,
                    )
                    if delay:
                        time.sleep(delay)
            if result is None and last_error is not None:
                raise last_error
            scores_dict = {}
            nan_metrics = []
            import math
            for m in active:
                val = result[m.name]
                if isinstance(val, list):
                    val = val[0] if val else 0.0
                elif hasattr(val, "iloc"):
                    val = val.iloc[0] if len(val) > 0 else 0.0
                
                # Check for NaN
                is_nan = False
                try:
                    fval = float(val) if val is not None else float('nan')
                    if math.isnan(fval):
                        is_nan = True
                except (ValueError, TypeError):
                    is_nan = True
                
                if is_nan:
                    nan_metrics.append(m.name)
                    scores_dict[m.name] = float('nan')
                else:
                    scores_dict[m.name] = round(fval, 4)
            
            if nan_metrics:
                scores_dict["ragas_failed"] = True
                scores_dict["ragas_error"] = f"RAGAS returned NaN for metric(s): {', '.join(nan_metrics)}"
            else:
                scores_dict["ragas_failed"] = False
                scores_dict["ragas_error"] = None
                
            return scores_dict
        except Exception as exc:
            details = self._judge_exception_details(exc)
            logger.error("RAGAS evaluate_single error: %s", details)
            scores = _fallback_evaluate(question, answer, contexts, ground_truth)
            return {
                **scores,
                "ragas_failed": True,
                "ragas_error": f"Exception: {details}"
            }

    def evaluate_batch(
        self, dataset: List[Dict[str, Any]], show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate a batch of samples.

        Each dict must have: question, answer, contexts (List[str]).
        Optional: ground_truth.
        """
        records = []
        for idx, sample in enumerate(dataset):
            if show_progress:
                print(f"  [{idx+1}/{len(dataset)}] Evaluating …", end="\r")
            t0 = time.perf_counter()
            scores = self.evaluate_single(
                question=sample["question"],
                answer=sample["answer"],
                contexts=sample.get("contexts", []),
                ground_truth=sample.get("ground_truth"),
            )
            if scores.get("ragas_failed"):
                logger.warning(
                    "RAGAS evaluation failed for query '%s'. Error: %s",
                    sample["question"],
                    scores.get("ragas_error"),
                )
            records.append({
                "question": sample["question"],
                "answer": sample["answer"],
                "n_contexts": len(sample.get("contexts", [])),
                "has_ground_truth": bool(sample.get("ground_truth")),
                "eval_latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                **scores,
            })
        if show_progress:
            print(f"  Evaluated {len(dataset)} samples.        ")
        return pd.DataFrame(records)

    async def evaluate_live_single(
        self,
        workflow: Any,
        question: str,
        ground_truth: Optional[str] = None,
        mode: QueryMode = QueryMode.HYBRID,
        max_results: int = settings.CONTEXT_TOP_K,
        max_hops: int = settings.GRAPH_MAX_HOPS,
        candidate_pool_size: int = settings.RETRIEVAL_CANDIDATE_LIMIT,
        enable_lexical_fusion: bool = False,
        vector_fusion_weight: float = settings.FUSION_VECTOR_WEIGHT,
        graph_fusion_weight: float = settings.FUSION_GRAPH_WEIGHT,
        lexical_fusion_weight: float = settings.FUSION_LEXICAL_WEIGHT,
        enable_adaptive_routing: bool = settings.ADAPTIVE_ROUTING_ENABLED,
        enable_conservative_routing: bool = settings.CONSERVATIVE_ROUTING_ENABLED,
        enable_retrieval_orchestration: bool = settings.RETRIEVAL_ORCHESTRATION_ENABLED,
        enable_cross_encoder_reranking: bool = settings.CROSS_ENCODER_RERANK_ENABLED,
        enable_verification_framework: bool = settings.VERIFICATION_FRAMEWORK_ENABLED,
        allow_mode_downgrade: bool = True,
        allow_vectorless_auto_route: bool = True,
        attachment_content: Optional[str] = None,
        attachment_name: Optional[str] = None,
        sample_id: str = "single-001",
        categories: Optional[List[str]] = None,
        profile: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the live workflow on a single query and evaluate the output with RAGAS.
        """
        t0 = time.perf_counter()
        res: Dict[str, Any] = {}
        try:
            res = await workflow.execute_with_answer(
                query=question,
                mode=mode,
                max_results=max_results,
                max_hops=max_hops,
                candidate_pool_size=candidate_pool_size,
                enable_lexical_fusion=enable_lexical_fusion,
                vector_fusion_weight=vector_fusion_weight,
                graph_fusion_weight=graph_fusion_weight,
                lexical_fusion_weight=lexical_fusion_weight,
                enable_adaptive_routing=enable_adaptive_routing,
                enable_conservative_routing=enable_conservative_routing,
                enable_retrieval_orchestration=enable_retrieval_orchestration,
                enable_cross_encoder_reranking=enable_cross_encoder_reranking,
                enable_verification_framework=enable_verification_framework,
                allow_mode_downgrade=allow_mode_downgrade,
                allow_vectorless_auto_route=allow_vectorless_auto_route,
                attachment_content=attachment_content,
                attachment_name=attachment_name,
                include_trace=True,
            )
            answer = res.get("answer", "")
            chunks = res.get("chunks", [])
            contexts = [c.content for c in chunks]
            run_latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            workflow_error = None
        except Exception as e:
            logger.error("Workflow execution failed for query '%s': %s", question, e)
            answer = "Error: Workflow execution failed"
            contexts = []
            run_latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            workflow_error = str(e)

        t_eval = time.perf_counter()
        scores = self.evaluate_single(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )
        eval_latency_ms = round((time.perf_counter() - t_eval) * 1000, 1)

        profile_payload = dict(profile or {
            "name": "ad_hoc",
            "requested_mode": mode.value,
            "declared_effective_modes": [],
        })
        provenance = build_provenance_record(
            sample={
                "sample_id": sample_id,
                "question": question,
                "ground_truth": ground_truth,
                "categories": categories or [],
            },
            result=res,
            scores=scores,
            profile=profile_payload,
            workflow_error=workflow_error,
            workflow_latency_ms=run_latency_ms,
            eval_latency_ms=eval_latency_ms,
            judge_model=getattr(self, "model", None),
            embedding_model=getattr(self, "embedding_model", None),
        )
        reference_contexts = []
        if isinstance(profile, Mapping):
            reference_contexts = list(profile.get("reference_contexts") or [])
        ir_metrics = score_retrieval(
            retrieved_contexts=contexts,
            reference_contexts=reference_contexts,
            k=min(5, max(1, max_results)),
        )
        path_audit = ((provenance.get("retrieval") or {}).get("graph_path_audit") or {})
        complete = float(path_audit.get("complete_path_count") or 0)
        total_paths = float(path_audit.get("traversal_candidate_count") or 0)
        path_completeness = (complete / total_paths) if total_paths else None
        kinetic_score_shadow = compute_kinetic_score_shadow(
            ragas_scores=scores,
            ir_metrics=ir_metrics,
            path_completeness=path_completeness,
        )

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "n_contexts": len(contexts),
            "has_ground_truth": bool(ground_truth),
            "ground_truth": ground_truth,
            "reference_contexts": reference_contexts,
            "workflow_latency_ms": run_latency_ms,
            "eval_latency_ms": eval_latency_ms,
            "workflow_error": workflow_error,
            "requested_mode": provenance["profile"]["requested_mode"],
            "effective_mode": provenance["profile"]["effective_mode"],
            "mode_profile_error": provenance["profile"]["error"],
            "profile_name": provenance["profile"]["name"],
            "provenance": provenance,
            "max_results": max_results,
            "max_hops": max_hops,
            "candidate_pool_size": candidate_pool_size,
            "enable_lexical_fusion": enable_lexical_fusion,
            "vector_fusion_weight": vector_fusion_weight,
            "graph_fusion_weight": graph_fusion_weight,
            "lexical_fusion_weight": lexical_fusion_weight,
            "enable_adaptive_routing": enable_adaptive_routing,
            "enable_retrieval_orchestration": enable_retrieval_orchestration,
            "enable_cross_encoder_reranking": enable_cross_encoder_reranking,
            "enable_verification_framework": enable_verification_framework,
            "candidate_provenance_completeness": (
                (provenance.get("retrieval") or {}).get(
                    "candidate_provenance_completeness"
                )
            ),
            "kinetic_score_shadow": kinetic_score_shadow,
            **ir_metrics,
            **scores,
        }

    async def evaluate_live_workflow(
        self,
        workflow: Any,
        dataset: List[Dict[str, Any]],
        mode: QueryMode = QueryMode.HYBRID,
        max_results: int = settings.CONTEXT_TOP_K,
        max_hops: int = settings.GRAPH_MAX_HOPS,
        candidate_pool_size: int = settings.RETRIEVAL_CANDIDATE_LIMIT,
        enable_lexical_fusion: bool = False,
        vector_fusion_weight: float = settings.FUSION_VECTOR_WEIGHT,
        graph_fusion_weight: float = settings.FUSION_GRAPH_WEIGHT,
        lexical_fusion_weight: float = settings.FUSION_LEXICAL_WEIGHT,
        enable_adaptive_routing: bool = settings.ADAPTIVE_ROUTING_ENABLED,
        enable_conservative_routing: bool = settings.CONSERVATIVE_ROUTING_ENABLED,
        enable_retrieval_orchestration: bool = settings.RETRIEVAL_ORCHESTRATION_ENABLED,
        enable_cross_encoder_reranking: bool = settings.CROSS_ENCODER_RERANK_ENABLED,
        enable_verification_framework: bool = settings.VERIFICATION_FRAMEWORK_ENABLED,
        allow_mode_downgrade: bool = True,
        allow_vectorless_auto_route: bool = True,
        profile: Optional[Mapping[str, Any]] = None,
        concurrency_limit: int = 3,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Run the live workflow on a dataset and evaluate the output with RAGAS.
        
        Args:
            workflow: An instance of HybridRAGWorkflow.
            dataset: List of dicts, each containing at least 'question' and optionally 'ground_truth'.
            mode: QueryMode to run the workflow (e.g. QueryMode.HYBRID).
            max_results: Maximum retrieved context chunks.
            max_hops: Maximum graph traversal depth.
            candidate_pool_size: Per-channel candidates fetched before reranking.
            enable_lexical_fusion: Include the local BM25 channel in hybrid fusion.
            concurrency_limit: Concurrency limit for executing queries against the LLM/databases.
            show_progress: Whether to print progress to stdout.
        """
        import asyncio
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def evaluate_sample(idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
            question = sample["question"]
            ground_truth = sample.get("ground_truth")
            res: Dict[str, Any] = {}
            
            async with semaphore:
                if show_progress:
                    print(f"  [{idx+1}/{len(dataset)}] Querying workflow: {question}")
                t0 = time.perf_counter()
                try:
                    res = await workflow.execute_with_answer(
                        query=question,
                        mode=mode,
                        max_results=max_results,
                        max_hops=max_hops,
                        candidate_pool_size=candidate_pool_size,
                        enable_lexical_fusion=enable_lexical_fusion,
                        vector_fusion_weight=vector_fusion_weight,
                        graph_fusion_weight=graph_fusion_weight,
                        lexical_fusion_weight=lexical_fusion_weight,
                        enable_adaptive_routing=enable_adaptive_routing,
                        enable_conservative_routing=enable_conservative_routing,
                        enable_retrieval_orchestration=enable_retrieval_orchestration,
                        enable_cross_encoder_reranking=enable_cross_encoder_reranking,
                        enable_verification_framework=enable_verification_framework,
                        allow_mode_downgrade=allow_mode_downgrade,
                        allow_vectorless_auto_route=allow_vectorless_auto_route,
                        attachment_content=sample.get("attachment_content"),
                        attachment_name=sample.get("attachment_name"),
                        include_trace=True,
                    )
                    answer = res.get("answer", "")
                    chunks = res.get("chunks", [])
                    contexts = [c.content for c in chunks]
                    run_latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                    workflow_error = None
                except Exception as e:
                    logger.error("Workflow execution failed for query '%s': %s", question, e)
                    answer = "Error: Workflow execution failed"
                    contexts = []
                    run_latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                    workflow_error = str(e)
            
            if show_progress:
                print(f"  [{idx+1}/{len(dataset)}] Evaluating output: {question}")
                
            t_eval = time.perf_counter()
            scores = self.evaluate_single(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )
            eval_latency_ms = round((time.perf_counter() - t_eval) * 1000, 1)
            profile_payload = dict(profile or {
                "name": "ad_hoc",
                "requested_mode": mode.value,
                "declared_effective_modes": [],
            })
            provenance = build_provenance_record(
                sample=sample,
                result=res,
                scores=scores,
                profile=profile_payload,
                workflow_error=workflow_error,
                workflow_latency_ms=run_latency_ms,
                eval_latency_ms=eval_latency_ms,
                judge_model=getattr(self, "model", None),
                embedding_model=getattr(self, "embedding_model", None),
            )
            reference_contexts = list(sample.get("reference_contexts") or [])
            ir_metrics = score_retrieval(
                retrieved_contexts=contexts,
                reference_contexts=reference_contexts,
                k=min(5, max(1, max_results)),
            )
            path_audit = ((provenance.get("retrieval") or {}).get("graph_path_audit") or {})
            complete = float(path_audit.get("complete_path_count") or 0)
            total_paths = float(path_audit.get("traversal_candidate_count") or 0)
            path_completeness = (complete / total_paths) if total_paths else None
            kinetic_score_shadow = compute_kinetic_score_shadow(
                ragas_scores=scores,
                ir_metrics=ir_metrics,
                path_completeness=path_completeness,
            )

            return {
                "question": question,
                "answer": answer,
                "n_contexts": len(contexts),
                "has_ground_truth": bool(ground_truth),
                "ground_truth": ground_truth,
                "reference_contexts": reference_contexts,
                "workflow_latency_ms": run_latency_ms,
                "eval_latency_ms": eval_latency_ms,
                "workflow_error": workflow_error,
                "sample_id": provenance["sample_id"],
                "categories": provenance["categories"],
                "requested_mode": provenance["profile"]["requested_mode"],
                "effective_mode": provenance["profile"]["effective_mode"],
                "mode_profile_error": provenance["profile"]["error"],
                "profile_name": provenance["profile"]["name"],
                "provenance": provenance,
                "max_results": max_results,
                "max_hops": max_hops,
                "candidate_pool_size": candidate_pool_size,
                "enable_lexical_fusion": enable_lexical_fusion,
                "vector_fusion_weight": vector_fusion_weight,
                "graph_fusion_weight": graph_fusion_weight,
                "lexical_fusion_weight": lexical_fusion_weight,
                "enable_retrieval_orchestration": enable_retrieval_orchestration,
                "enable_cross_encoder_reranking": enable_cross_encoder_reranking,
                "enable_verification_framework": enable_verification_framework,
                "candidate_provenance_completeness": (
                    (provenance.get("retrieval") or {}).get(
                        "candidate_provenance_completeness"
                    )
                ),
                "kinetic_score_shadow": kinetic_score_shadow,
                **ir_metrics,
                **scores,
            }
            
        tasks = []
        for idx, sample in enumerate(dataset):
            tasks.append(evaluate_sample(idx, sample))
            
        if show_progress:
            mode_str = mode.value if hasattr(mode, 'value') else str(mode)
            print(f"🚀 Running live evaluation for {len(dataset)} samples in {mode_str} mode (concurrency={concurrency_limit})...")
            
        records = await asyncio.gather(*tasks)
        
        if show_progress:
            print(f"🎉 Evaluation completed for {len(dataset)} samples.")
            
        return pd.DataFrame(records)

    def generate_report(
        self,
        results: pd.DataFrame,
        validation_policy: Optional[ValidationPolicy] = None,
    ) -> Dict[str, Any]:
        """Summarise batch results into a structured report dict."""
        policy = validation_policy or ValidationPolicy()
        policy.validate()
        metric_cols = [c for c in ALL_METRICS if c in results.columns]
        if not metric_cols:
            return {"error": "No metric columns found in results DataFrame."}

        per_metric: Dict[str, Any] = {}
        for col in metric_cols:
            s = results[col].dropna()
            per_metric[col] = {
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "description": METRIC_DESCRIPTIONS.get(col, ""),
            }

        per_category: Dict[str, Any] = {}
        if "categories" in results.columns:
            category_names = sorted({
                str(category)
                for categories in results["categories"]
                for category in (categories if isinstance(categories, list) else [])
            })
            for category in category_names:
                subset = results[results["categories"].apply(
                    lambda categories: category in categories if isinstance(categories, list) else False
                )]
                per_category[category] = {
                    "samples": len(subset),
                    "sample_ids": (
                        sorted(str(value) for value in subset["sample_id"])
                        if "sample_id" in subset.columns else []
                    ),
                    "metrics": {
                        metric: round(float(subset[metric].mean()), 4)
                        for metric in metric_cols
                    },
                    "retrieval_metrics": {
                        metric: round(float(subset[metric].mean()), 4)
                        for metric in ("precision_at_5", "recall_at_5", "ndcg_at_5")
                        if metric in subset.columns
                    },
                    "mean_workflow_latency_ms": (
                        round(float(subset["workflow_latency_ms"].mean()), 2)
                        if "workflow_latency_ms" in subset.columns else None
                    ),
                }

        provenance_records = [
            record for record in results.get("provenance", [])
            if isinstance(record, Mapping)
        ]
        graph_latencies = []
        path_audits = []
        graph_diagnostics = []
        lifecycle = []
        provenance_completeness = []
        for record in provenance_records:
            latency = record.get("latency_ms", {}).get("stages", {})
            graph_latency = latency.get("graph_retrieval_ms", latency.get("graph_agent_ms"))
            if isinstance(graph_latency, (int, float)):
                graph_latencies.append(float(graph_latency))
            retrieval = record.get("retrieval", {})
            path_audit = retrieval.get("graph_path_audit") or {}
            path_audits.append(path_audit)
            graph_diagnostics.append(path_audit.get("retriever_diagnostics") or {})
            lifecycle.extend(retrieval.get("candidate_lifecycle") or [])
            completeness = retrieval.get("candidate_provenance_completeness")
            if isinstance(completeness, (int, float)):
                provenance_completeness.append(float(completeness))

        def _percentile(values: List[float], quantile: float) -> Optional[float]:
            if not values:
                return None
            return round(float(np.percentile(values, quantile)), 2)

        traversal_path_count = sum(
            int(audit.get("traversal_candidate_count", 0)) for audit in path_audits
        )
        complete_path_count = sum(
            int(audit.get("complete_path_count", 0)) for audit in path_audits
        )
        retrieval_diagnostics = {
            "p95_latency_ms": (
                _percentile(results["workflow_latency_ms"].dropna().astype(float).tolist(), 95)
                if "workflow_latency_ms" in results.columns
                else None
            ),
            "candidate_provenance_completeness": (
                round(sum(provenance_completeness) / len(provenance_completeness), 4)
                if provenance_completeness
                else None
            ),
            "graph_stage_latency_ms": {
                "samples": len(graph_latencies),
                "p50": _percentile(graph_latencies, 50),
                "p95": _percentile(graph_latencies, 95),
            },
            "graph_paths": {
                "traversal_candidate_count": traversal_path_count,
                "complete_path_count": complete_path_count,
                "all_complete": traversal_path_count > 0 and complete_path_count == traversal_path_count,
                "empty_seed_count": sum(bool(item.get("empty_seed")) for item in graph_diagnostics),
                "traversal_failure_count": sum(bool(item.get("traversal_failure")) for item in graph_diagnostics),
                "cycle_prevention_count": sum(int(item.get("cycle_prevention_count", 0)) for item in graph_diagnostics),
                "missing_evidence_edge_count": sum(int(item.get("missing_evidence_edge_count", 0)) for item in graph_diagnostics),
            },
            "candidate_survival": {
                "observed_candidates": len(lifecycle),
                "sent_to_generation": sum(bool(item.get("sent_to_generation")) for item in lifecycle),
                "dropped_pre_fusion": sum(
                    item.get("dropped_at") in {"pre_fusion", "fusion"}
                    for item in lifecycle
                ),
                "dropped_deduplication": sum(
                    item.get("dropped_at") == "identity_deduplication"
                    for item in lifecycle
                ),
                "dropped_reranking": sum(
                    item.get("dropped_at") in {"reranking", "semantic_reranking"}
                    for item in lifecycle
                ),
                "dropped_final_truncation": sum(
                    item.get("dropped_at") in {"final_truncation", "context_optimization"}
                    for item in lifecycle
                ),
            },
        }

        results = results.copy()
        results["composite_score"] = results.apply(
            lambda row: weighted_composite(row.to_dict(), policy), axis=1
        )
        ci_low, ci_high = bootstrap_mean_interval(
            results["composite_score"].tolist(), policy
        )

        def _tier(score: float) -> str:
            if score >= 0.8: return "excellent"
            if score >= 0.6: return "good"
            if score >= 0.4: return "fair"
            return "poor"

        results["quality_tier"] = results["composite_score"].apply(_tier)

        recs = self._recommendations(per_metric)

        workflow_accepted = (
            "workflow_error" not in results.columns
            or not results["workflow_error"].fillna("").astype(str).str.strip().ne("").any()
        )
        ragas_accepted = (
            "ragas_failed" in results.columns
            and not results["ragas_failed"].fillna(True).astype(bool).any()
            and workflow_accepted
        )
        try:
            for _, row in results.iterrows():
                validate_metric_values(row.to_dict(), tuple(policy.metric_weights))
            metric_values_valid = True
        except ValueError:
            metric_values_valid = False
            ragas_accepted = False

        ir_rows = results.to_dict(orient="records")
        ir_summary = aggregate_ir_metrics(ir_rows, k=5)
        kinetic_calibration = calibrate_shadow_scores(ir_rows)
        cost_summary = {
            "estimated_cost_usd": None,
            "total_tokens": None,
            "cost_complete": False,
            "note": "per-request token cost is reported when UsageTracker is populated",
        }

        return {
            "summary": {
                "total_samples": len(results),
                "overall_composite_score": round(float(results["composite_score"].mean()), 4),
                "composite_confidence_interval_95": [round(ci_low, 4), round(ci_high, 4)],
                "composite_metric_weights": dict(policy.metric_weights),
                "metrics_evaluated": metric_cols,
                "quality_distribution": results["quality_tier"].value_counts().to_dict(),
                "eval_mode": "ragas" if ragas_accepted else "heuristic_or_mixed",
                "accepted_as_ragas": ragas_accepted,
                "metric_values_valid": metric_values_valid,
                "profile": (
                    str(results["profile_name"].iloc[0])
                    if "profile_name" in results.columns and not results.empty
                    else "unspecified"
                ),
                "profile_preference": "not established; compare separate accepted manifests",
            },
            "per_metric": per_metric,
            "per_category": per_category,
            "ir_metrics": ir_summary,
            "kinetic_score_shadow": kinetic_calibration,
            "cost": cost_summary,
            "retrieval_diagnostics": retrieval_diagnostics,
            "worst_samples": results.nsmallest(5, "composite_score")[
                ["question", "composite_score"] + metric_cols
            ].reset_index(drop=True).to_dict(orient="records"),
            "best_samples": results.nlargest(5, "composite_score")[
                ["question", "composite_score"] + metric_cols
            ].reset_index(drop=True).to_dict(orient="records"),
            "recommendations": recs,
        }

    @staticmethod
    def _recommendations(per_metric: Dict[str, Any]) -> List[str]:
        recs = []
        def _s(n: str) -> float:
            return per_metric.get(n, {}).get("mean", 1.0)

        if _s("faithfulness") < 0.7:
            recs.append("Low faithfulness: LLM hallucinating. Tighten system prompt with 'only use provided context'.")
        if _s("answer_relevancy") < 0.7:
            recs.append("Low answer relevancy: Answers drift off-topic. Add query intent classification.")
        if _s("context_precision") < 0.6:
            recs.append("Low context precision: Noise chunks ranked too high. Tune RRF k-constant or add a reranker.")
        if _s("context_recall") < 0.6:
            recs.append("Low context recall: Missing relevant chunks. Increase chunk overlap or add BM25 fallback.")
        if _s("answer_correctness") < 0.7:
            recs.append("Low answer correctness: Verify knowledge-base freshness and consider fine-tuning.")
        if not recs:
            recs.append("All metrics are within acceptable ranges. Keep monitoring!")
        return recs

if __name__ == "__main__":
    import argparse
    import os
    import sys
    import asyncio
    from dotenv import load_dotenv
    
    # Load dotenv
    load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")))

    parser = argparse.ArgumentParser(description="Run the accepted live Kinegraph RAGAS benchmark.")
    parser.add_argument("--max-hops", type=int, default=settings.GRAPH_MAX_HOPS)
    parser.add_argument("--max-results", type=int, default=settings.CONTEXT_TOP_K)
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=settings.RETRIEVAL_CANDIDATE_LIMIT,
    )
    parser.add_argument("--run-label", default="latest", help="Safe label used for persisted result files")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Maximum concurrent workflow/judge samples (use 1 for deterministic sequential runs)",
    )
    parser.add_argument(
        "--benchmark-audit",
        default="eval/kinegraph_benchmark_v1.audit.json",
        help="Versioned reference audit; unaccepted or stale audits fail closed",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="hybrid",
        help="Explicit benchmark execution profile; profiles never change production defaults",
    )
    parser.add_argument("--generation-model", default=settings.LLM_MODEL)
    parser.add_argument("--judge-model", default=settings.LLM_MODEL)
    parser.add_argument(
        "--judge-provider",
        choices=["openrouter", "openai", "nvidia", "fireworks"],
        default=os.getenv("RAGAS_JUDGE_PROVIDER", "openrouter"),
        help="OpenAI-compatible provider used only by the RAGAS judge",
    )
    parser.add_argument(
        "--judge-base-url",
        default=os.getenv("RAGAS_JUDGE_BASE_URL"),
        help="Optional OpenAI-compatible judge endpoint override",
    )
    parser.add_argument(
        "--judge-embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Locally cached embedding model used by semantic RAGAS metrics",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate judge and local embedding initialization, then exit without retrieval",
    )
    parser.add_argument(
        "--judge-smoke-test",
        action="store_true",
        help="Run one small paid RAGAS judge request, then exit without databases",
    )
    parser.add_argument(
        "--baseline-manifest",
        help="Optional accepted manifest to compare using the one-lever ratchet gate",
    )
    parser.add_argument(
        "--enable-adaptive-routing",
        action="store_true",
        help="Experimental ADR-001 execution-plan policy; use with --profile adaptive_hybrid",
    )
    parser.add_argument(
        "--enable-conservative-routing",
        action="store_true",
        help="Deprecated alias for --enable-adaptive-routing",
    )
    parser.add_argument(
        "--enable-retrieval-orchestration",
        action="store_true",
        help="Experimental ADR-003 provenance ledger and context optimization",
    )
    parser.add_argument(
        "--enable-cross-encoder-reranking",
        action="store_true",
        help="Controlled cross-encoder experiment; compare against an identical baseline",
    )
    parser.add_argument(
        "--enable-verification-framework",
        action="store_true",
        help="Experimental ADR-004 partial/refusal policy and shadow Kinetic Score",
    )
    parser.add_argument(
        "--enable-lexical-fusion",
        action="store_true",
        help="Deprecated compatibility alias for --profile hybrid_lexical",
    )
    parser.add_argument("--vector-weight", type=float, default=settings.FUSION_VECTOR_WEIGHT)
    parser.add_argument("--graph-weight", type=float, default=settings.FUSION_GRAPH_WEIGHT)
    parser.add_argument("--lexical-weight", type=float, default=settings.FUSION_LEXICAL_WEIGHT)
    args = parser.parse_args()
    if args.preflight_only and args.judge_smoke_test:
        parser.error("--preflight-only and --judge-smoke-test are mutually exclusive")
    if args.enable_lexical_fusion:
        if args.profile == "vectorless":
            parser.error("dedicated vectorless cannot enable Hybrid lexical fusion")
        args.profile = "hybrid_lexical"
    profile = get_profile(args.profile)
    if args.enable_conservative_routing:
        args.enable_adaptive_routing = True
    if args.enable_adaptive_routing and profile.name != "adaptive_hybrid":
        parser.error("--enable-adaptive-routing requires --profile adaptive_hybrid")
    if not 1 <= args.max_hops <= 5:
        parser.error("--max-hops must be between 1 and 5")
    if not 1 <= args.max_results <= 100:
        parser.error("--max-results must be between 1 and 100")
    if not 5 <= args.candidate_pool_size <= 100:
        parser.error("--candidate-pool-size must be between 5 and 100")
    weights = [args.vector_weight, args.graph_weight]
    if args.enable_lexical_fusion:
        weights.append(args.lexical_weight)
    if any(weight < 0 or weight > 5 for weight in weights) or not any(weights):
        parser.error("active fusion weights must be between 0 and 5 with at least one positive value")
    run_label = re.sub(r"[^a-zA-Z0-9_-]+", "-", args.run_label).strip("-") or "latest"
    artifact_label = f"{run_label}-{profile.name}"
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_git_revision = current_git_revision(repo_root)
    run_working_tree_clean = working_tree_is_clean(repo_root)
    
    print("Checking RAGAS evaluator configuration...")
    try:
        evaluator = RAGASEvaluator(
            model=args.judge_model,
            embedding_model=args.judge_embedding_model,
            metrics=ALL_METRICS,
            provider=args.judge_provider,
            base_url=args.judge_base_url,
        )
    except RAGASConfigurationError as exc:
        print(f"RAGAS PREFLIGHT FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
    readiness = evaluator.readiness()
    print(json.dumps(readiness, indent=2, sort_keys=True))
    if args.preflight_only:
        print("RAGAS preflight passed; no benchmark queries were executed.")
        sys.exit(0)
    if args.judge_smoke_test:
        smoke_scores = evaluator.evaluate_single(
            question="What does Reciprocal Rank Fusion combine?",
            answer="Reciprocal Rank Fusion combines multiple ranked result lists.",
            contexts=[
                "Reciprocal Rank Fusion combines multiple ranked result lists "
                "using reciprocal-rank contributions."
            ],
            ground_truth=(
                "Reciprocal Rank Fusion combines multiple ranked result lists."
            ),
        )
        smoke_results = pd.DataFrame([smoke_scores])
        try:
            require_successful_ragas(
                smoke_results,
                expected_rows=1,
                required_metrics=ALL_METRICS,
            )
        except RAGASValidationError as exc:
            print(f"RAGAS JUDGE SMOKE TEST FAILED: {exc}", file=sys.stderr)
            sys.exit(2)
        print(json.dumps(smoke_scores, indent=2, sort_keys=True))
        print("RAGAS judge smoke test passed; no benchmark queries were executed.")
        sys.exit(0)
    
    # Run the live pipeline
    from backend.services.chroma_service import ChromaService
    from backend.services.neo4j_service import Neo4jService
    from backend.core.langgraph_workflow import HybridRAGWorkflow
    
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "kinegraph_benchmark_v1.csv"))
    if not os.path.exists(csv_path):
        # Try relative to project root
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval", "kinegraph_benchmark_v1.csv"))
        
    if not os.path.exists(csv_path):
        print(f"Benchmark CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    print(f"\nFound benchmark dataset at '{csv_path}'.")
    df = pd.read_csv(csv_path)
    required_columns = {"user_input", "reference"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Benchmark CSV missing columns: {sorted(missing_columns)}", file=sys.stderr)
        sys.exit(2)
    if len(df) != 20:
        print(
            f"Benchmark rejected: expected 20 rows but found {len(df)}. "
            "Generate the complete testset before running the live benchmark.",
            file=sys.stderr,
        )
        sys.exit(2)
    audit_path = os.path.abspath(os.path.join(repo_root, args.benchmark_audit))
    try:
        audit_validation = validate_reference_audit(
            df.to_dict(orient="records"),
            load_reference_audit(audit_path),
            repo_root,
            csv_path,
        )
        audit_validation.require_accepted()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"BENCHMARK REJECTED: {exc}", file=sys.stderr)
        print(
            "Review eval/kinegraph_benchmark_v1.audit.json with a named human reviewer; "
            "reference corrections are benchmark changes, not retrieval gains.",
            file=sys.stderr,
        )
        sys.exit(2)
    raw_data = build_profile_dataset(audit_validation.rows, profile)

    chroma = ChromaService()
    neo4j = Neo4jService()
    workflow = HybridRAGWorkflow(
        chroma_service=chroma,
        neo4j_service=neo4j,
        generation_model=args.generation_model,
    )

    print("\nRunning concurrent live workflow and RAGAS evaluation...")
    try:
        results_df = asyncio.run(evaluator.evaluate_live_workflow(
            workflow=workflow,
            dataset=raw_data,
            mode=profile.requested_mode,
            max_results=args.max_results,
            max_hops=args.max_hops,
            candidate_pool_size=args.candidate_pool_size,
            enable_lexical_fusion=profile.enable_lexical_fusion,
            vector_fusion_weight=args.vector_weight,
            graph_fusion_weight=args.graph_weight,
            lexical_fusion_weight=args.lexical_weight,
            enable_adaptive_routing=args.enable_adaptive_routing,
            enable_conservative_routing=args.enable_conservative_routing,
            enable_retrieval_orchestration=args.enable_retrieval_orchestration,
            enable_cross_encoder_reranking=args.enable_cross_encoder_reranking,
            enable_verification_framework=args.enable_verification_framework,
            allow_mode_downgrade=profile.allow_mode_downgrade,
            allow_vectorless_auto_route=profile.allow_vectorless_auto_route,
            profile=profile.to_dict(),
            concurrency_limit=max(1, args.concurrency),
        ))
    finally:
        try:
            neo4j.close()
        except Exception:
            pass
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
    os.makedirs(reports_dir, exist_ok=True)
    provenance_records = results_df["provenance"].tolist()
    provenance_path = os.path.join(reports_dir, f"ragas_{artifact_label}_provenance.jsonl")
    diagnostics_path = os.path.join(reports_dir, f"ragas_{artifact_label}_diagnostics.json")
    write_provenance_jsonl(provenance_path, provenance_records)
    write_diagnostic_summary(diagnostics_path, provenance_records)

    try:
        require_successful_ragas(
            results_df,
            expected_rows=len(raw_data),
            required_metrics=evaluator.metrics_names,
        )
    except RAGASValidationError as exc:
        print(f"\nBENCHMARK REJECTED: {exc}", file=sys.stderr)
        print(
            f"Diagnostic provenance was saved to {provenance_path}; no accepted report or spider graph was updated.",
            file=sys.stderr,
        )
        sys.exit(2)
    report = evaluator.generate_report(results_df)

    results_path = os.path.join(reports_dir, f"ragas_{artifact_label}_results.csv")
    report_path = os.path.join(reports_dir, f"ragas_{artifact_label}_report.json")
    serializable_results = results_df.copy()
    for column in ("categories", "provenance"):
        if column in serializable_results.columns:
            serializable_results[column] = serializable_results[column].apply(
                lambda value: json.dumps(value, sort_keys=True)
            )
    serializable_results.to_csv(results_path, index=False)
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    
    print("\n=== PER-METRIC AVERAGE SCORES ===")
    for metric, stats in report['per_metric'].items():
        print(f"  {metric:25s}: {stats['mean']:.4f}")
        
    print(f"\nOverall Composite Score: {report['summary']['overall_composite_score']:.4f}")
    
    # Save the spider graph
    import matplotlib.pyplot as plt
    from math import pi

    metric_cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']
    means = [report['per_metric'].get(m, {}).get('mean', 0.0) for m in metric_cols]
    N = len(metric_cols)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [2 * pi]
    means_plot = means + [means[0]]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, means_plot, 'o-', linewidth=2, color='#7c3aed')
    ax.fill(angles, means_plot, alpha=0.25, color='#7c3aed')
    ax.set_xticks(angles[:-1])
    display_names = [m.replace('_', ' ').title() for m in metric_cols]
    ax.set_xticklabels(display_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('PropertyGraphIndex Hybrid RAG Scores', size=13, pad=20)
    plt.tight_layout()
    
    graph_path = os.path.join(reports_dir, f"spider_graph_ragas_{artifact_label}.png")
    plt.savefig(graph_path, dpi=300)
    policy = ValidationPolicy()
    pipeline_config = {
        "retrieval": {
            "profile": profile.to_dict(),
            "mode": profile.requested_mode.value,
            "enable_adaptive_routing": args.enable_adaptive_routing,
            "enable_conservative_routing": args.enable_conservative_routing,
            "max_results": args.max_results,
            "max_hops": args.max_hops,
            "candidate_pool_size": args.candidate_pool_size,
            "enable_lexical_fusion": profile.enable_lexical_fusion,
            "enable_retrieval_orchestration": args.enable_retrieval_orchestration,
            "vector_fusion_weight": args.vector_weight,
            "graph_fusion_weight": args.graph_weight,
            "lexical_fusion_weight": args.lexical_weight,
            "rrf_k": settings.RRF_K,
        },
        "benchmark": {
            "dataset_version": audit_validation.dataset_version,
            "reference_audit_sha256": audit_validation.audit_sha256,
        },
        "reranking": {
            "enabled": True,
            "enable_cross_encoder_reranking": args.enable_cross_encoder_reranking,
            "model": settings.RERANKER_MODEL,
            "minimum_relevance": settings.RERANKER_MIN_RELEVANCE,
            "dedup_threshold": settings.RETRIEVAL_DEDUP_THRESHOLD,
        },
        "recovery": {
            "conditional_recovery": True,
            "hyde_fallback": False,
        },
        "generation": {
            "temperature": settings.GENERATION_TEMPERATURE,
            "grounding_critique": True,
            "grounding_critic_temperature": settings.FAITHFULNESS_CRITIC_TEMPERATURE,
            "enable_verification_framework": args.enable_verification_framework,
        },
        "evaluation": readiness,
    }
    manifest = build_manifest(
        run_label=artifact_label,
        repo_root=repo_root,
        dataset_path=csv_path,
        pipeline_config=pipeline_config,
        models={
            "generation": args.generation_model,
            "grounding_critic": settings.FAITHFULNESS_CRITIC_MODEL,
            "judge": args.judge_model,
            "embedding": args.judge_embedding_model,
        },
        report=report,
        artifacts={
            "results_csv": results_path,
            "report_json": report_path,
            "spider_graph": graph_path,
            "provenance_jsonl": provenance_path,
            "diagnostics_json": diagnostics_path,
            "reference_audit": audit_path,
        },
        policy=policy,
        git_revision=run_git_revision,
        working_tree_clean=run_working_tree_clean,
        dataset_identity={
            "effective_dataset_sha256": audit_validation.effective_dataset_sha256,
            "dataset_version": audit_validation.dataset_version,
            "audit_sha256": audit_validation.audit_sha256,
            "audit_path": audit_path,
        },
    )
    comparison = None
    if args.baseline_manifest:
        comparison = compare_manifests(
            load_manifest(args.baseline_manifest), manifest, policy
        )
        manifest["baseline_comparison"] = comparison
    manifest_path = os.path.join(reports_dir, f"ragas_{artifact_label}_manifest.json")
    write_manifest(manifest_path, manifest)
    print(f"\nResults saved to {results_path}")
    print(f"Report saved to {report_path}")
    print(f"Spider graph saved to {graph_path}")
    print(f"Experiment manifest saved to {manifest_path}")
    if comparison:
        print(
            "Baseline decision: "
            f"{comparison['decision'].upper()} "
            f"(delta={comparison['composite_delta']})"
        )
        if comparison["decision"] != "keep":
            print("Candidate did not pass the experiment ratchet gate.", file=sys.stderr)
            sys.exit(3)
