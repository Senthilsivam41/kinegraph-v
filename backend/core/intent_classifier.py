"""
Query Intent Classifier — KineticGraph-Vectra
Classifies a query into intent categories and suggests the optimal retrieval mode.
Runs fast via keyword heuristics first; falls back to LLM only when ambiguous.
"""
from __future__ import annotations

import re
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------
# Maps intent label → (description, preferred_mode)
INTENT_CATALOG: Dict[str, Tuple[str, str]] = {
    "definition":      ("What-is / define questions",                        "vector"),
    "comparison":      ("Compare / differ / vs questions",                   "hybrid"),
    "how_to":          ("How-does / step-by-step questions",                 "hybrid"),
    "relationship":    ("How entities relate / connect / link questions",     "graph"),
    "debugging":       ("Why / root-cause / error / fix questions",          "hybrid"),
    "factual_lookup":  ("Who / when / where / numerical fact questions",     "graph"),
    "procedural":      ("How-to-configure / setup / install questions",      "vector"),
    "conceptual":      ("Explain / describe / overview questions",           "vector"),
}

# Keyword triggers per intent (lower-cased)
_TRIGGERS: Dict[str, list] = {
    "definition":     ["what is", "define", "definition of", "meaning of", "what are"],
    "comparison":     ["differ", "difference", "compare", "vs ", "versus", "unlike", "over", "better than"],
    "how_to":         ["how does", "how do", "how can", "how to", "how is", "how are"],
    "relationship":   ["relate", "relationship", "connect", "link", "depend", "interact", "between"],
    "debugging":      ["why", "debug", "root cause", "fix", "error", "fail", "issue", "problem"],
    "factual_lookup": ["who", "when", "where", "which", "how many", "how much", "list of"],
    "procedural":     ["configure", "setup", "install", "deploy", "run", "start", "enable"],
    "conceptual":     ["explain", "describe", "overview", "concept", "purpose", "role of"],
}


def classify_intent(query: str) -> Dict[str, str]:
    """
    Classify the query intent using keyword heuristics.

    Returns::

        {
            "intent": "comparison",
            "description": "Compare / differ / vs questions",
            "suggested_mode": "hybrid",
            "confidence": "high" | "low",
        }
    """
    q = query.lower().strip()

    scores: Dict[str, int] = {intent: 0 for intent in INTENT_CATALOG}
    for intent, keywords in _TRIGGERS.items():
        for kw in keywords:
            if kw in q:
                scores[intent] += 1

    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]

    if best_score == 0:
        # No clear signal — default to hybrid
        return {
            "intent": "conceptual",
            "description": INTENT_CATALOG["conceptual"][0],
            "suggested_mode": "hybrid",
            "confidence": "low",
        }

    desc, mode = INTENT_CATALOG[best_intent]
    return {
        "intent": best_intent,
        "description": desc,
        "suggested_mode": mode,
        "confidence": "high" if best_score >= 2 else "medium",
    }


def rewrite_query_for_retrieval(query: str, intent: str) -> str:
    """
    Expand / rewrite the query to improve recall.

    Adds intent-specific context words that anchor vector search to the right
    semantic neighbourhood and improve context_recall.
    """
    expansions: Dict[str, str] = {
        "definition":     f"definition explanation concept: {query}",
        "comparison":     f"comparison difference similarities: {query}",
        "how_to":         f"process steps mechanism workflow: {query}",
        "relationship":   f"relationship connection graph entities: {query}",
        "debugging":      f"root cause debugging solution fix: {query}",
        "factual_lookup": f"fact data information: {query}",
        "procedural":     f"procedure configuration setup steps: {query}",
        "conceptual":     f"concept overview purpose: {query}",
    }
    return expansions.get(intent, query)
