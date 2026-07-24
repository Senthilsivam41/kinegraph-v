"""
Query Intent Classifier — KineticGraph-Vectra
Classifies a query into intent categories and suggests the optimal retrieval mode.
Runs fast via keyword heuristics first; falls back to LLM only when ambiguous.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Tuple

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
    "comparison":     ["differ", "difference", "compare", "compared to", "vs ", "versus", "unlike", "over", "better than"],
    "how_to":         ["how does", "how do", "how can", "how to", "how is", "how are"],
    "relationship":   ["relate", "relationship", "connect", "link", "depend", "interact", "between"],
    "debugging":      ["why", "debug", "root cause", "fix", "error", "fail", "issue", "problem"],
    "factual_lookup": ["who", "when", "where", "which", "how many", "how much", "list of"],
    "procedural":     ["configure", "setup", "install", "deploy", "run", "start", "enable"],
    "conceptual":     ["explain", "describe", "overview", "concept", "purpose", "role of"],
}

_EXACT_TOKEN_PATTERNS = (
    (r"https?://[^\s)>\]]+", re.IGNORECASE),
    (r"(?<!\w)--[a-z][a-z0-9-]*", re.IGNORECASE),
    (r"\b[A-Z][A-Z0-9_]{2,}\b", 0),
    (r"\b[\w.-]+\.(?:pdf|md|txt|json|ya?ml|toml|env|py|sh|csv)\b", re.IGNORECASE),
    (r"(?:^|\s)(?:docker(?:\s+compose)?|pip|python|curl|git)\s+[^\n,;]+", re.IGNORECASE),
)

_ENTITY_STOP_WORDS = {
    "How", "What", "When", "Where", "Which", "Who", "Why",
    "Compare", "Define", "Describe", "Explain", "List", "Show",
}


def analyze_query_signals(query: str) -> Dict[str, Any]:
    """Extract observable routing signals without generating graph seeds."""
    exact_tokens = []
    for pattern, flags in _EXACT_TOKEN_PATTERNS:
        for match in re.findall(pattern, query, flags=flags | re.MULTILINE):
            token = str(match).strip()
            if token and token not in exact_tokens:
                exact_tokens.append(token)
    entity_candidates = []
    for token in re.findall(r"\b(?:[A-Z][A-Za-z0-9.+-]{2,}|[A-Z]{2,})\b", query):
        if token not in _ENTITY_STOP_WORDS and token not in entity_candidates:
            entity_candidates.append(token)
    return {
        "exact_tokens": exact_tokens[:12],
        "has_exact_tokens": bool(exact_tokens),
        # Candidates are routing hints only. They are never authoritative graph seeds.
        "entity_candidates": entity_candidates[:12],
    }


def _trigger_matches(query: str, trigger: str) -> bool:
    """Match whole trigger phrases instead of arbitrary substrings."""
    phrase = trigger.strip()
    if not phrase:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", query))


def extract_query_facets(query: str) -> list[str]:
    """Return explicit question/request facets without generating new intent."""
    parts = re.split(
        r"[;]|\s+and\s+(?=(?:what|how|why|which|where|when|who|"
        r"query|compare|report|identify|list|describe|explain|show|provide)\b)",
        query,
        flags=re.IGNORECASE,
    )
    facets = [part.strip(" ,?.") for part in parts if len(part.strip(" ,?.").split()) >= 2]
    return facets if facets else [query.strip()]


def classify_intent(query: str) -> Dict[str, Any]:
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
    matched_triggers: Dict[str, list[str]] = {intent: [] for intent in INTENT_CATALOG}
    for intent, keywords in _TRIGGERS.items():
        for kw in keywords:
            if _trigger_matches(q, kw):
                scores[intent] += 1
                matched_triggers[intent].append(kw.strip())

    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]
    if best_score > 0 and scores["comparison"] == best_score:
        best_intent = "comparison"
    facets = extract_query_facets(query)
    query_signals = analyze_query_signals(query)

    if best_score == 0:
        # No clear signal — default to hybrid
        return {
            "intent": "conceptual",
            "description": INTENT_CATALOG["conceptual"][0],
            "suggested_mode": "hybrid",
            "confidence": "low",
            "scores": scores,
            "matched_triggers": {},
            "tied_intents": [],
            "facets": facets,
            "coverage_sensitive": len(facets) > 1,
            "route_confidence": 0.25,
            "relationship_signal": False,
            **query_signals,
            "route_rationale": "no trigger matched; retain broad hybrid coverage",
        }

    desc, mode = INTENT_CATALOG[best_intent]
    tied_intents = [intent for intent, score in scores.items() if score == best_score]
    coverage_sensitive = scores["comparison"] > 0 or len(facets) > 1
    runner_up = max(
        (score for intent, score in scores.items() if intent != best_intent),
        default=0,
    )
    route_confidence = min(0.95, 0.50 + (0.18 * best_score) + (0.08 * (best_score - runner_up)))
    if len(tied_intents) > 1:
        route_confidence -= 0.20
    if coverage_sensitive:
        route_confidence -= 0.15
    route_confidence = round(max(0.10, route_confidence), 2)
    confidence = (
        "high" if route_confidence >= 0.80
        else "medium" if route_confidence >= 0.55
        else "low"
    )
    return {
        "intent": best_intent,
        "description": desc,
        "suggested_mode": mode,
        "confidence": confidence,
        "scores": scores,
        "matched_triggers": {
            intent: triggers for intent, triggers in matched_triggers.items() if triggers
        },
        "tied_intents": tied_intents if len(tied_intents) > 1 else [],
        "facets": facets,
        "coverage_sensitive": coverage_sensitive,
        "route_confidence": route_confidence,
        "relationship_signal": scores["relationship"] > 0,
        **query_signals,
        "route_rationale": (
            f"selected {best_intent} from deterministic trigger scores; "
            f"confidence={confidence} ({route_confidence:.2f}); "
            f"coverage_sensitive={coverage_sensitive}"
        ),
    }


def rewrite_query_for_retrieval(query: str, intent: str) -> str:
    """
    Expand / rewrite the query to improve recall.

    Adds intent-specific context words, resolves common abbreviations (e.g., RRF, DB),
    and filters out conversational noise to improve context_recall.
    """
    # 1. Lowercase for mapping
    q = query.lower().strip()
    
    # 2. Resolve common abbreviations
    abbreviations = {
        r"\brrf\b": "reciprocal rank fusion",
        r"\bdb\b": "database",
        r"\bapi\b": "application programming interface",
        r"\brag\b": "retrieval augmented generation",
    }
    for abbrev, full in abbreviations.items():
        q = re.sub(abbrev, full, q)
        
    # 3. Clean common conversational filler
    fillers = [
        r"^how does\b", r"^how do\b", r"^how can\b", r"^what is the role of\b",
        r"^what is the purpose of\b", r"^what is the\b", r"^what is\b", r"^what are\b",
        r"^why use\b", r"^why do we use\b", r"^can you explain\b", r"^explain\b"
    ]
    for filler in fillers:
        q = re.sub(filler, "", q)
    q = q.strip("? ").strip()
    
    # Re-apply cleaned query to expansions
    cleaned_query = q or query
    
    expansions: Dict[str, str] = {
        "definition":     f"definition explanation concept: {cleaned_query}",
        "comparison":     f"comparison difference similarities: {cleaned_query}",
        "how_to":         f"process steps mechanism workflow: {cleaned_query}",
        "relationship":   f"relationship connection graph entities: {cleaned_query}",
        "debugging":      f"root cause debugging solution fix: {cleaned_query}",
        "factual_lookup": f"fact data information: {cleaned_query}",
        "procedural":     f"procedure configuration setup steps: {cleaned_query}",
        "conceptual":     f"concept overview purpose: {cleaned_query}",
    }
    return expansions.get(intent, cleaned_query)
