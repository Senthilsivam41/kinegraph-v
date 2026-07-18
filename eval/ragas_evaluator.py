"""
RAGAS Evaluation Module — KineticGraph-Vectra
Metrics: faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
Falls back to keyword-heuristics when ragas/openai are not installed.
"""
from __future__ import annotations

import os
import sys
# Path patch to support running directly or without PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import time
from typing import Any, Dict, List, Optional

from backend.app.models import QueryMode
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
    logger.warning("ragas not installed — using heuristic fallback. pip install ragas datasets")

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
ALL_METRICS = DEFAULT_METRICS + ["answer_correctness"]

METRIC_DESCRIPTIONS = {
    "faithfulness": "Fraction of answer claims supported by retrieved context",
    "answer_relevancy": "How well the answer addresses the question",
    "context_precision": "Signal-to-noise: most relevant chunks ranked first",
    "context_recall": "Fraction of ground-truth info present in context",
    "answer_correctness": "Answer accuracy vs reference ground truth",
}


class RAGASValidationError(RuntimeError):
    """Raised when a benchmark contains fallback scores presented as RAGAS."""


def require_successful_ragas(
    results: pd.DataFrame,
    expected_rows: Optional[int] = None,
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
    if "ragas_failed" not in results.columns:
        raise RAGASValidationError(
            "Missing ragas_failed provenance; this run cannot be accepted as RAGAS."
        )

    failed = results[results["ragas_failed"].fillna(True).astype(bool)]
    if failed.empty:
        return

    errors = sorted({
        str(error) for error in failed.get("ragas_error", pd.Series(dtype=str)).dropna()
    })
    detail = f" Errors: {'; '.join(errors)}" if errors else ""
    raise RAGASValidationError(
        f"Rejected {len(failed)}/{len(results)} rows because RAGAS failed and "
        f"heuristic fallback scores are not benchmark evidence.{detail}"
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
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        metrics: Optional[List[str]] = None,
        critic_model: Optional[str] = None,
        critic_api_key: Optional[str] = None,
        critic_base_url: Optional[str] = None,
    ) -> None:
        import os
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_AI_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.embedding_model = embedding_model
        self.metrics_names = metrics or DEFAULT_METRICS
        self.critic_model = critic_model
        self.critic_api_key = critic_api_key
        self.critic_base_url = critic_base_url
        self._llm = None
        self._critic_llm = None
        self._embeddings = None
        self._ragas_metrics: List[Any] = []
        self._setup()

    def _setup(self) -> None:
        if not (_RAGAS_AVAILABLE and _OPENAI_AVAILABLE):
            return
        import os
        try:
            is_openrouter = self.openai_api_key and (
                self.openai_api_key.startswith("sk-or-") or "openrouter" in self.openai_api_key
            )

            kw: Dict[str, Any] = {"model": self.model, "temperature": 0}
            ekw: Dict[str, Any] = {"model": self.embedding_model}

            if self.openai_api_key:
                kw["openai_api_key"] = self.openai_api_key
                ekw["openai_api_key"] = self.openai_api_key

            if is_openrouter:
                kw["base_url"] = "https://openrouter.ai/api/v1"
                self._llm = ChatOpenAI(**kw)
                
                # Use local free embeddings to avoid OpenRouter 402/payment limit errors
                from langchain_huggingface import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            else:
                self._llm = ChatOpenAI(**kw)
                self._embeddings = OpenAIEmbeddings(**ekw)

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
            self._ragas_metrics = [_map[m] for m in self.metrics_names if m in _map]
            logger.info("RAGASEvaluator ready with metrics: %s", self.metrics_names)
        except Exception as exc:
            logger.error("RAGAS setup failed: %s", exc)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

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
                "ragas_error": "RAGAS not available or not configured"
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
            result = ragas_evaluate(
                dataset=dataset,
                metrics=active,
                llm=self._critic_llm or self._llm,
                embeddings=self._embeddings,
                raise_exceptions=False,
            )
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
            logger.error("RAGAS evaluate_single error: %s", exc)
            scores = _fallback_evaluate(question, answer, contexts, ground_truth)
            return {
                **scores,
                "ragas_failed": True,
                "ragas_error": f"Exception: {str(exc)}"
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
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """
        Run the live workflow on a single query and evaluate the output with RAGAS.
        """
        t0 = time.perf_counter()
        try:
            res = await workflow.execute_with_answer(
                query=question,
                mode=mode,
                max_results=max_results,
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

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "n_contexts": len(contexts),
            "has_ground_truth": bool(ground_truth),
            "ground_truth": ground_truth,
            "workflow_latency_ms": run_latency_ms,
            "eval_latency_ms": eval_latency_ms,
            "workflow_error": workflow_error,
            **scores,
        }

    async def evaluate_live_workflow(
        self,
        workflow: Any,
        dataset: List[Dict[str, Any]],
        mode: QueryMode = QueryMode.HYBRID,
        max_results: int = 10,
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
            concurrency_limit: Concurrency limit for executing queries against the LLM/databases.
            show_progress: Whether to print progress to stdout.
        """
        import asyncio
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def evaluate_sample(idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
            question = sample["question"]
            ground_truth = sample.get("ground_truth")
            
            async with semaphore:
                if show_progress:
                    print(f"  [{idx+1}/{len(dataset)}] Querying workflow: {question}")
                t0 = time.perf_counter()
                try:
                    res = await workflow.execute_with_answer(
                        query=question,
                        mode=mode,
                        max_results=max_results,
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
            
            return {
                "question": question,
                "answer": answer,
                "n_contexts": len(contexts),
                "has_ground_truth": bool(ground_truth),
                "ground_truth": ground_truth,
                "workflow_latency_ms": run_latency_ms,
                "eval_latency_ms": eval_latency_ms,
                "workflow_error": workflow_error,
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

    def generate_report(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Summarise batch results into a structured report dict."""
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

        results = results.copy()
        results["composite_score"] = results[metric_cols].mean(axis=1)

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

        return {
            "summary": {
                "total_samples": len(results),
                "overall_composite_score": round(float(results["composite_score"].mean()), 4),
                "metrics_evaluated": metric_cols,
                "quality_distribution": results["quality_tier"].value_counts().to_dict(),
                "eval_mode": "ragas" if ragas_accepted else "heuristic_or_mixed",
                "accepted_as_ragas": ragas_accepted,
            },
            "per_metric": per_metric,
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
    import os
    import sys
    import asyncio
    from dotenv import load_dotenv
    
    # Load dotenv
    load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")))
    
    print("Checking RAGAS Evaluator configuration...")
    print(f"RAGAS Available: {_RAGAS_AVAILABLE}")
    print(f"OpenAI API Key Set: {bool(os.getenv('OPENAI_API_KEY'))}")
    
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
    raw_data = [
        {"question": row["user_input"], "ground_truth": row["reference"]}
        for _, row in df.iterrows()
    ]

    chroma = ChromaService()
    neo4j = Neo4jService()
    workflow = HybridRAGWorkflow(chroma_service=chroma, neo4j_service=neo4j)

    evaluator = RAGASEvaluator(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        metrics=['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']
    )
    print("\nRunning concurrent live workflow and RAGAS evaluation...")
    try:
        results_df = asyncio.run(evaluator.evaluate_live_workflow(
            workflow=workflow,
            dataset=raw_data,
            mode=QueryMode.HYBRID,
            concurrency_limit=3,
        ))
    finally:
        try:
            neo4j.close()
        except Exception:
            pass
    try:
        require_successful_ragas(results_df, expected_rows=len(raw_data))
    except RAGASValidationError as exc:
        print(f"\nBENCHMARK REJECTED: {exc}", file=sys.stderr)
        print(
            "No report or spider graph was updated. Resolve the judge failure and rerun.",
            file=sys.stderr,
        )
        sys.exit(2)
    report = evaluator.generate_report(results_df)
    
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
    
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
    os.makedirs(reports_dir, exist_ok=True)
    graph_path = os.path.join(reports_dir, 'spider_graph_ragas_score.png')
    plt.savefig(graph_path, dpi=300)
    print(f"\nSpider graph saved to {graph_path}")
