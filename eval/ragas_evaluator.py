"""
RAGAS Evaluation Module — KineticGraph-Vectra
Metrics: faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
Falls back to keyword-heuristics when ragas/openai are not installed.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

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
    ) -> None:
        self.openai_api_key = openai_api_key
        self.model = model
        self.embedding_model = embedding_model
        self.metrics_names = metrics or DEFAULT_METRICS
        self._llm = None
        self._embeddings = None
        self._ragas_metrics: List[Any] = []
        self._setup()

    def _setup(self) -> None:
        if not (_RAGAS_AVAILABLE and _OPENAI_AVAILABLE):
            return
        try:
            kw: Dict[str, Any] = {"model": self.model, "temperature": 0}
            if self.openai_api_key:
                kw["openai_api_key"] = self.openai_api_key
            self._llm = ChatOpenAI(**kw)

            ekw: Dict[str, Any] = {"model": self.embedding_model}
            if self.openai_api_key:
                ekw["openai_api_key"] = self.openai_api_key
            self._embeddings = OpenAIEmbeddings(**ekw)

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
    ) -> Dict[str, float]:
        """Evaluate one QA sample. Returns metric_name → score (0-1)."""
        if not _RAGAS_AVAILABLE or not self._ragas_metrics:
            return _fallback_evaluate(question, answer, contexts, ground_truth)

        row: Dict[str, Any] = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            row["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(row)
        active = self._ragas_metrics
        if not ground_truth:
            active = [m for m in active if m not in (answer_correctness, context_recall)]

        try:
            result = ragas_evaluate(
                dataset=dataset,
                metrics=active,
                llm=self._llm,
                embeddings=self._embeddings,
                raise_exceptions=False,
            )
            return {m.name: round(float(result[m.name] or 0), 4) for m in active}
        except Exception as exc:
            logger.error("RAGAS evaluate_single error: %s", exc)
            return _fallback_evaluate(question, answer, contexts, ground_truth)

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

        return {
            "summary": {
                "total_samples": len(results),
                "overall_composite_score": round(float(results["composite_score"].mean()), 4),
                "metrics_evaluated": metric_cols,
                "quality_distribution": results["quality_tier"].value_counts().to_dict(),
                "eval_mode": "ragas" if _RAGAS_AVAILABLE else "heuristic",
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
