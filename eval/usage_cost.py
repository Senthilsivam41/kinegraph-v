"""Best-effort token/cost tracking hooks for generation and live evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UsageTracker:
    """Accumulates optional usage metadata without inventing costs."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        label: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        prompt = int(prompt_tokens or 0)
        completion = int(completion_tokens or 0)
        total = int(total_tokens if total_tokens is not None else prompt + completion)
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        if estimated_cost_usd is not None:
            self.estimated_cost_usd = float(
                (self.estimated_cost_usd or 0.0) + estimated_cost_usd
            )
        self.events.append(
            {
                "label": label,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
                "estimated_cost_usd": estimated_cost_usd,
                "raw": raw or {},
            }
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "event_count": len(self.events),
            "cost_complete": self.estimated_cost_usd is not None and self.total_tokens > 0,
        }


def extract_openai_usage(response: Any, *, label: str = "llm") -> dict[str, Any]:
    """Best-effort extraction from OpenAI-compatible response objects."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {
            "label": label,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cost_complete": False,
        }
    if hasattr(usage, "model_dump"):
        payload = usage.model_dump()
    elif isinstance(usage, dict):
        payload = usage
    else:
        payload = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    return {
        "label": label,
        "prompt_tokens": payload.get("prompt_tokens"),
        "completion_tokens": payload.get("completion_tokens"),
        "total_tokens": payload.get("total_tokens"),
        "cost_complete": payload.get("total_tokens") is not None,
        "raw": payload,
    }
