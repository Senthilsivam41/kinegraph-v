"""Conditional query decomposition, vocabulary expansion, and constrained HyDE."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class WeaknessAssessment:
    weak: bool
    reasons: list[str]
    result_count: int
    top_score: float
    graph_result_count: int
    source_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecoveryPlan:
    subqueries: list[str]
    vocabulary: list[str]


class QueryRecoveryEngine:
    """Plan recovery only after ordinary retrieval produces weak candidates."""

    def __init__(
        self,
        llm: Any,
        min_results: int = 3,
        min_top_score: float = 0.35,
    ) -> None:
        self.llm = llm
        self.min_results = min_results
        self.min_top_score = min_top_score

    def assess(
        self,
        vector_results: list[dict[str, Any]],
        graph_results: list[dict[str, Any]],
        require_graph: bool = False,
        require_source_diversity: bool = False,
    ) -> WeaknessAssessment:
        results = [*vector_results, *graph_results]
        scores = [
            float(result.get("original_score", result.get("score")) or 0.0)
            for result in results
        ]
        sources = {result.get("source") for result in results if result.get("source")}
        reasons = []
        if len(results) < self.min_results:
            reasons.append("insufficient_results")
        if not scores or max(scores) < self.min_top_score:
            reasons.append("low_top_score")
        if require_graph and not graph_results:
            reasons.append("no_graph_seed_or_path")
        if len(results) >= 2 and len(sources) < 2 and require_source_diversity:
            reasons.append("low_source_diversity")
        return WeaknessAssessment(
            weak=bool(reasons),
            reasons=reasons,
            result_count=len(results),
            top_score=max(scores, default=0.0),
            graph_result_count=len(graph_results),
            source_count=len(sources),
        )

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z0-9_-]{3,}", text.lower())
            if token not in {"and", "are", "does", "how", "the", "what", "when", "where", "which"}
        }

    @classmethod
    def _valid_subquery(cls, original: str, candidate: str) -> bool:
        candidate = candidate.strip()
        if not candidate or len(candidate) > 240:
            return False
        original_tokens = cls._tokens(original)
        candidate_tokens = cls._tokens(candidate)
        if not original_tokens or not candidate_tokens:
            return False
        return len(original_tokens & candidate_tokens) / len(candidate_tokens) >= 0.3

    @staticmethod
    def _heuristic_subqueries(query: str) -> list[str]:
        parts = [part.strip(" ,?.") for part in re.split(r"\b(?:and|versus|vs\.?|then)\b|[;]", query, flags=re.I)]
        meaningful = [part for part in parts if len(part.split()) >= 2]
        if len(meaningful) < 2:
            return []
        return [f"{part} in the context of: {query}" for part in meaningful[:3]]

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
        try:
            return json.loads(content)
        except (TypeError, json.JSONDecodeError):
            match = re.search(r"\{.*\}", content, flags=re.S)
            return json.loads(match.group(0)) if match else {}

    async def create_plan(self, query: str, intent: str) -> RecoveryPlan:
        prompt = f"""Create a retrieval recovery plan for the query below.
Return JSON only with keys "subqueries" (maximum 3 strings) and "vocabulary" (maximum 8 short search terms).
Subqueries must decompose the user's actual request. Do not introduce named entities, products, facts, or relationships absent from the query.
Vocabulary may contain synonyms or technical equivalents, but no factual claims.

Intent: {intent}
Query: {query}
"""
        payload: dict[str, Any] = {}
        try:
            response = await self.llm.ainvoke(prompt)
            payload = self._parse_json(response.content)
        except Exception:
            payload = {}

        candidates = payload.get("subqueries") if isinstance(payload.get("subqueries"), list) else []
        subqueries = []
        for candidate in candidates:
            if isinstance(candidate, str) and self._valid_subquery(query, candidate) and candidate not in subqueries:
                subqueries.append(candidate.strip())
        if not subqueries:
            subqueries = self._heuristic_subqueries(query)

        vocabulary = []
        raw_vocabulary = payload.get("vocabulary") if isinstance(payload.get("vocabulary"), list) else []
        for term in raw_vocabulary:
            if not isinstance(term, str):
                continue
            clean = term.strip()
            if 1 <= len(clean.split()) <= 4 and len(clean) <= 80 and clean.lower() not in {v.lower() for v in vocabulary}:
                vocabulary.append(clean)
        return RecoveryPlan(subqueries=subqueries[:3], vocabulary=vocabulary[:8])

    async def generate_hypothesis(self, query: str) -> str:
        prompt = f"""Write one concise hypothetical passage solely to improve vector retrieval for this query:
{query}

Constraints:
- Maximum 80 words.
- Do not introduce named entities absent from the query.
- Do not state dates, numbers, citations, or unverifiable facts.
- This passage is a search probe, never evidence and never an answer.
Return only the passage.
"""
        try:
            response = await self.llm.ainvoke(prompt)
            hypothesis = response.content.strip()
        except Exception:
            return ""
        hypothesis = " ".join(hypothesis.split()[:80])
        if not hypothesis or re.search(r"\d|https?://|\[[0-9]+\]", hypothesis):
            return ""

        # Reject newly introduced proper-name tokens. This is intentionally
        # conservative: a rejected hypothesis simply leaves ordinary recovery
        # results in place and never blocks the request.
        query_names = set(re.findall(r"\b[A-Z][A-Za-z0-9_-]+\b", query))
        hypothesis_names = set(re.findall(r"\b[A-Z][A-Za-z0-9_-]+\b", hypothesis))
        sentence_starters = set(re.findall(r"(?:^|[.!?]\s+)([A-Z][a-z]+)\b", hypothesis))
        if hypothesis_names - sentence_starters - query_names:
            return ""
        return hypothesis

    @staticmethod
    def annotate_results(
        results: list[dict[str, Any]],
        original_query: str,
        retrieval_query: str,
        stage: str,
    ) -> list[dict[str, Any]]:
        annotated = []
        for result in results:
            enriched = dict(result)
            enriched["metadata"] = {
                **result.get("metadata", {}),
                "recovery_stage": stage,
                "retrieval_query": retrieval_query,
                "original_query": original_query,
                "hypothesis_is_evidence": False if stage == "hyde" else None,
            }
            annotated.append(enriched)
        return annotated
