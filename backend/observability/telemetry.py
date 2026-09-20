"""Low-cardinality OpenTelemetry metrics and traces for query execution.

Query IDs are trace-only correlation attributes. They deliberately never become
metric labels, and prompts, answers, source content, and credentials are never
added to telemetry.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping

from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


ROUTE = "/api/v1/query"


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def normalize_usage(raw_usage: Any) -> dict[str, int]:
    """Normalize OpenAI/LangChain token metadata without retaining raw payloads."""
    if not isinstance(raw_usage, Mapping):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = _as_int(raw_usage.get("prompt_tokens", raw_usage.get("input_tokens")))
    completion = _as_int(raw_usage.get("completion_tokens", raw_usage.get("output_tokens")))
    total = _as_int(raw_usage.get("total_tokens")) or prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def estimate_cost_usd(
    usage: Mapping[str, Any],
    *,
    input_per_million: float | None,
    output_per_million: float | None,
) -> float | None:
    """Return a transparent configured estimate, or None when pricing is unknown."""
    if input_per_million is None or output_per_million is None:
        return None
    normalized = normalize_usage(usage)
    return (
        normalized["prompt_tokens"] * input_per_million
        + normalized["completion_tokens"] * output_per_million
    ) / 1_000_000


@dataclass
class QueryObservation(AbstractContextManager["QueryObservation"]):
    telemetry: "QueryTelemetry"
    query_id: str
    requested_mode: str

    def __post_init__(self) -> None:
        self._span = self.telemetry.tracer.start_span("kinegraph.query")
        self._span.set_attribute("kinegraph.query.id", self.query_id)
        self._span.set_attribute("http.route", ROUTE)
        self._span.set_attribute("kinegraph.query.requested_mode", self.requested_mode)

    def __enter__(self) -> "QueryObservation":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None:
            self.record_failure(exc)
        self._span.end()
        return False

    def record_success(self, result: Mapping[str, Any], execution_time_ms: float) -> None:
        effective_mode = str(result.get("effective_mode", self.requested_mode))
        attrs = {
            "http.route": ROUTE,
            "kinegraph.query.requested_mode": self.requested_mode,
            "kinegraph.query.effective_mode": effective_mode,
            "kinegraph.query.outcome": "success",
        }
        self.telemetry.query_requests.add(1, attrs)
        self.telemetry.query_duration.record(execution_time_ms, attrs)
        self._span.set_attribute("kinegraph.query.effective_mode", effective_mode)
        self._span.set_attribute("kinegraph.query.outcome", "success")
        self._span.set_attribute("kinegraph.query.result_count", len(result.get("chunks") or []))

        for stage, duration_ms in (result.get("latency") or {}).items():
            if isinstance(duration_ms, (int, float)) and duration_ms >= 0:
                self.telemetry.stage_duration.record(
                    duration_ms, {"kinegraph.query.stage": str(stage)}
                )

        self.telemetry.record_llm_usage(result.get("llm_usage") or [])
        self.telemetry.record_evidence_quality(
            result.get("citation_validation") or {}, result.get("grounding_critique") or {}
        )

    def record_failure(self, error: BaseException) -> None:
        error_type = type(error).__name__
        attrs = {"http.route": ROUTE, "error.type": error_type}
        self.telemetry.query_failures.add(1, attrs)
        self.telemetry.query_requests.add(
            1,
            {
                "http.route": ROUTE,
                "kinegraph.query.requested_mode": self.requested_mode,
                "kinegraph.query.effective_mode": "unknown",
                "kinegraph.query.outcome": "failure",
            },
        )
        self._span.set_attribute("kinegraph.query.outcome", "failure")
        self._span.record_exception(error)
        self._span.set_status(trace.Status(trace.StatusCode.ERROR, error_type))


class QueryTelemetry:
    """Owns the OTel instruments used by the FastAPI query route."""

    def __init__(
        self,
        *,
        service_name: str,
        service_version: str,
        environment: str,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
        otlp_endpoint: str | None = None,
    ) -> None:
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": service_version,
                "deployment.environment": environment,
            }
        )
        metric_readers = []
        span_processors = []
        if otlp_endpoint:
            # Imported only for configured exports so local development and
            # tests do not require a collector or exporter package at import.
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            metric_readers.append(
                PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=f"{otlp_endpoint.rstrip('/')}/v1/metrics"))
            )
            span_processors.append(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{otlp_endpoint.rstrip('/')}/v1/traces"))
            )
        self.meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
        self.tracer_provider = TracerProvider(resource=resource)
        for processor in span_processors:
            self.tracer_provider.add_span_processor(processor)
        self.meter = self.meter_provider.get_meter("kinegraph.query")
        self.tracer = self.tracer_provider.get_tracer("kinegraph.query")
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million

        self.query_requests = self.meter.create_counter("kinegraph.query.requests", unit="{query}")
        self.query_failures = self.meter.create_counter("kinegraph.query.failures", unit="{failure}")
        self.query_duration = self.meter.create_histogram("kinegraph.query.duration", unit="ms")
        self.stage_duration = self.meter.create_histogram("kinegraph.query.stage.duration", unit="ms")
        self.llm_tokens = self.meter.create_counter("kinegraph.llm.tokens", unit="{token}")
        self.llm_cost = self.meter.create_histogram("kinegraph.llm.estimated_cost", unit="USD")
        self.usage_missing = self.meter.create_counter("kinegraph.llm.usage_missing", unit="{call}")
        # This is an evidence-quality proxy, not a deterministic hallucination rate.
        self.evidence_quality_failures = self.meter.create_counter(
            "kinegraph.evidence_quality.failures", unit="{failure}"
        )

    def observe(self, query_id: str, requested_mode: str) -> QueryObservation:
        return QueryObservation(self, query_id, requested_mode)

    def record_llm_usage(self, usages: list[Mapping[str, Any]]) -> None:
        for usage in usages:
            operation = str(usage.get("operation", "generation"))
            model = str(usage.get("model", "unknown"))
            normalized = normalize_usage(usage)
            attrs = {"kinegraph.llm.operation": operation, "gen_ai.request.model": model}
            if normalized["total_tokens"] == 0:
                self.usage_missing.add(1, attrs)
                continue
            self.llm_tokens.add(normalized["prompt_tokens"], {**attrs, "gen_ai.token.type": "input"})
            self.llm_tokens.add(normalized["completion_tokens"], {**attrs, "gen_ai.token.type": "output"})
            estimate = estimate_cost_usd(
                normalized,
                input_per_million=self.input_cost_per_million,
                output_per_million=self.output_cost_per_million,
            )
            if estimate is not None:
                self.llm_cost.record(estimate, attrs)

    def record_evidence_quality(
        self, citation_validation: Mapping[str, Any], grounding_critique: Mapping[str, Any]
    ) -> None:
        if not citation_validation.get("structured_output_valid", False):
            self.evidence_quality_failures.add(1, {"kinegraph.evidence_quality.proxy": "citation_validation"})
        elif citation_validation.get("rejected_claims"):
            self.evidence_quality_failures.add(1, {"kinegraph.evidence_quality.proxy": "citation_rejection"})
        if grounding_critique.get("reason") == "critic_failed":
            self.evidence_quality_failures.add(1, {"kinegraph.evidence_quality.proxy": "grounding_check_unavailable"})
        if grounding_critique.get("removed_unsupported_claim_ids") or grounding_critique.get("removed_claim_ids"):
            self.evidence_quality_failures.add(1, {"kinegraph.evidence_quality.proxy": "grounding_rejection"})

    def shutdown(self) -> None:
        self.meter_provider.shutdown()
        self.tracer_provider.shutdown()
