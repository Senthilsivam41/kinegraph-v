"""Deterministic, provenance-first execution planning for ADR-001."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


ROUTING_POLICY_VERSION = "kinegraph.adaptive-routing.v1"
SINGLE_CHANNEL_MODES = {"vector", "graph"}
ALL_MODES = ("hybrid", "vector", "graph", "vectorless")


@dataclass(frozen=True)
class ExecutionPlan:
    """Auditable retrieval plan; it never generates evidence or an answer."""

    policy_version: str
    policy: str
    requested_mode: str
    effective_mode: str
    route_confidence: float
    confidence_label: str
    pinned: bool
    signals: dict[str, Any]
    required_channels: list[str]
    recommended_channels: list[str]
    alternatives: list[dict[str, str]]
    decision: str
    fallback_mode: str | None = None
    fallback_trigger: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _channels(mode: str, lexical_enabled: bool) -> list[str]:
    if mode == "hybrid":
        channels = ["vector", "graph"]
        if lexical_enabled:
            channels.append("lexical")
        return channels
    if mode == "vectorless":
        return ["vectorless"]
    return [mode]


def _alternatives(effective_mode: str, reasons: Mapping[str, str]) -> list[dict[str, str]]:
    return [
        {"mode": mode, "rejected_reason": reasons.get(mode, "not selected by policy")}
        for mode in ALL_MODES
        if mode != effective_mode
    ]


def build_execution_plan(
    *,
    intent_result: Mapping[str, Any],
    requested_mode: str,
    allow_mode_downgrade: bool,
    adaptive_enabled: bool,
    lexical_enabled: bool,
    vectorless_eligible: bool,
    vectorless_reason: str | None,
    minimum_confidence: float = 0.80,
) -> ExecutionPlan:
    """Build an execution plan while preserving explicit modes and weak signals."""
    suggested_mode = str(intent_result.get("suggested_mode", "hybrid"))
    confidence = float(intent_result.get("route_confidence", 0.0))
    confidence_label = str(intent_result.get("confidence", "low"))
    exact_tokens = list(intent_result.get("exact_tokens", []))
    coverage_sensitive = bool(intent_result.get("coverage_sensitive", False))
    tied_intents = list(intent_result.get("tied_intents", []))
    signals = {
        "intent": str(intent_result.get("intent", "unknown")),
        "facets": list(intent_result.get("facets", [])),
        "coverage_sensitive": coverage_sensitive,
        "exact_tokens": exact_tokens,
        "entity_candidates": list(intent_result.get("entity_candidates", [])),
        "relationship_signal": bool(intent_result.get("relationship_signal", False)),
        "attachment_eligible": vectorless_eligible,
        "tied_intents": tied_intents,
    }

    if requested_mode != "hybrid":
        effective_mode = requested_mode
        decision = f"explicit caller mode '{requested_mode}' preserved"
        return ExecutionPlan(
            policy_version=ROUTING_POLICY_VERSION,
            policy="adaptive" if adaptive_enabled else "legacy",
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            route_confidence=confidence,
            confidence_label=confidence_label,
            pinned=True,
            signals=signals,
            required_channels=_channels(effective_mode, lexical_enabled),
            recommended_channels=_channels(effective_mode, lexical_enabled),
            alternatives=_alternatives(
                effective_mode,
                {mode: "explicit caller mode is authoritative" for mode in ALL_MODES},
            ),
            decision=decision,
        )

    if vectorless_eligible:
        decision = vectorless_reason or "eligible attachment or local-document route"
        return ExecutionPlan(
            policy_version=ROUTING_POLICY_VERSION,
            policy="adaptive" if adaptive_enabled else "legacy",
            requested_mode=requested_mode,
            effective_mode="vectorless",
            route_confidence=max(confidence, 0.9),
            confidence_label="high",
            pinned=False,
            signals=signals,
            required_channels=["vectorless"],
            recommended_channels=["vectorless"],
            alternatives=_alternatives(
                "vectorless",
                {
                    "hybrid": "eligible bounded document content is directly searchable",
                    "vector": "attachment/local-document source should remain explicit",
                    "graph": "attachment/local-document source should remain explicit",
                },
            ),
            decision=decision,
        )

    if not allow_mode_downgrade:
        decision = "requested Hybrid mode pinned by caller or benchmark profile"
        return ExecutionPlan(
            policy_version=ROUTING_POLICY_VERSION,
            policy="adaptive" if adaptive_enabled else "legacy",
            requested_mode=requested_mode,
            effective_mode="hybrid",
            route_confidence=confidence,
            confidence_label=confidence_label,
            pinned=True,
            signals=signals,
            required_channels=_channels("hybrid", lexical_enabled),
            recommended_channels=_channels("hybrid", lexical_enabled or bool(exact_tokens)),
            alternatives=_alternatives(
                "hybrid",
                {
                    "vector": "requested Hybrid mode is pinned",
                    "graph": "requested Hybrid mode is pinned",
                    "vectorless": "no eligible attachment or local document",
                },
            ),
            decision=decision,
        )

    if not adaptive_enabled:
        effective_mode = suggested_mode if suggested_mode in SINGLE_CHANNEL_MODES else "hybrid"
        decision = (
            f"legacy classifier selected {effective_mode}"
            if effective_mode != "hybrid"
            else "legacy classifier retained Hybrid"
        )
        return ExecutionPlan(
            policy_version=ROUTING_POLICY_VERSION,
            policy="legacy",
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            route_confidence=confidence,
            confidence_label=confidence_label,
            pinned=False,
            signals=signals,
            required_channels=_channels(effective_mode, lexical_enabled),
            recommended_channels=_channels(effective_mode, lexical_enabled),
            alternatives=_alternatives(
                effective_mode,
                {"vectorless": "no eligible attachment or local document"},
            ),
            decision=decision,
        )

    retain_reasons = []
    if coverage_sensitive:
        retain_reasons.append("compound or coverage-sensitive query")
    if exact_tokens:
        retain_reasons.append("exact-token evidence benefits from lexical coverage")
    if tied_intents:
        retain_reasons.append("intent classification is tied")
    if confidence < minimum_confidence:
        retain_reasons.append(
            f"route confidence {confidence:.2f} is below {minimum_confidence:.2f}"
        )
    if suggested_mode not in SINGLE_CHANNEL_MODES:
        retain_reasons.append("classifier did not recommend a single-channel route")

    if retain_reasons:
        decision = "adaptive policy retained hybrid: " + "; ".join(retain_reasons)
        return ExecutionPlan(
            policy_version=ROUTING_POLICY_VERSION,
            policy="adaptive",
            requested_mode=requested_mode,
            effective_mode="hybrid",
            route_confidence=confidence,
            confidence_label=confidence_label,
            pinned=False,
            signals=signals,
            required_channels=_channels("hybrid", lexical_enabled),
            recommended_channels=_channels("hybrid", lexical_enabled or bool(exact_tokens)),
            alternatives=_alternatives(
                "hybrid",
                {
                    "vector": "; ".join(retain_reasons),
                    "graph": "; ".join(retain_reasons),
                    "vectorless": "no eligible attachment or local document",
                },
            ),
            decision=decision,
        )

    decision = (
        f"adaptive policy selected high-confidence single-channel {suggested_mode}"
    )
    return ExecutionPlan(
        policy_version=ROUTING_POLICY_VERSION,
        policy="adaptive",
        requested_mode=requested_mode,
        effective_mode=suggested_mode,
        route_confidence=confidence,
        confidence_label=confidence_label,
        pinned=False,
        signals=signals,
        required_channels=[suggested_mode],
        recommended_channels=[suggested_mode],
        alternatives=_alternatives(
            suggested_mode,
            {
                "hybrid": "single-facet route met confidence threshold",
                "vectorless": "no eligible attachment or local document",
            },
        ),
        decision=decision,
        fallback_mode="hybrid",
        fallback_trigger="measurable_initial_retrieval_weakness",
    )
