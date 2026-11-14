from unittest.mock import Mock

import pytest
import requests

from backend.core.config import Settings
from backend.observability.telemetry import (
    ALLOWED_ERROR_CODES,
    QueryTelemetry,
    estimate_cost_usd,
    normalize_usage,
    sanitized_error_code,
)


@pytest.mark.parametrize("disable_metrics", [False, True])
def test_otlp_metrics_export_can_be_disabled(monkeypatch, disable_metrics):
    monkeypatch.delenv("OTEL_EXPORT_METRICS_ENABLED", raising=False)
    if disable_metrics:
        monkeypatch.setenv("OTEL_EXPORT_METRICS_ENABLED", "false")
    settings = Settings(_env_file=None, OPENAI_API_KEY="test", NEO4J_PASSWORD="unused-test-password")
    posted_urls = set()

    def post(_session, url, **_kwargs):
        posted_urls.add(url)
        return Mock(ok=True)

    monkeypatch.setattr(requests.Session, "post", post)

    telemetry = QueryTelemetry(
        service_name="test",
        service_version="1",
        environment="test",
        otlp_endpoint="http://collector:4318/",
        otlp_metrics_enabled=settings.OTEL_EXPORT_METRICS_ENABLED,
    )
    try:
        with telemetry.tracer.start_as_current_span("query"):
            telemetry.query_requests.add(1)
        assert telemetry.tracer_provider.force_flush()
        assert telemetry.meter_provider.force_flush()
    finally:
        telemetry.shutdown()

    expected = {"http://collector:4318/v1/traces"}
    if not disable_metrics:
        expected.add("http://collector:4318/v1/metrics")
    assert posted_urls == expected


def test_normalize_usage_supports_openai_and_langchain_names():
    assert normalize_usage({"input_tokens": 12, "output_tokens": 3}) == {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
    }


def test_estimated_cost_requires_explicit_pricing():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}
    assert estimate_cost_usd(usage, input_per_million=None, output_per_million=4.0) is None
    assert estimate_cost_usd(usage, input_per_million=2.0, output_per_million=4.0) == 4.0


def test_usage_normalization_does_not_retain_provider_payload_fields():
    assert normalize_usage({"prompt_tokens": 3, "completion_tokens": 2, "prompt": "secret"}) == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }


def test_failure_metric_uses_the_wrapped_root_cause():
    telemetry = QueryTelemetry(service_name="test", service_version="1", environment="test")
    telemetry.query_requests = _Capture()
    telemetry.query_failures = _Capture()
    observation = telemetry.observe("query-123", "hybrid")
    wrapper = RuntimeError("public error")
    wrapper.__cause__ = ValueError("provider failure")

    observation.record_failure(wrapper)

    assert telemetry.query_failures.calls == [(1, {
        "http.route": "/api/v1/query",
        "error.type": "validation_error",
    })]
    assert sanitized_error_code(wrapper) in ALLOWED_ERROR_CODES
    rendered = _span_text(observation)
    assert "public error" not in rendered
    assert "provider failure" not in rendered
    assert all(event.name != "exception" for event in observation._span.events)
    observation.__exit__(None, None, None)
    telemetry.shutdown()


def test_unclassified_failure_code_hides_exception_text():
    telemetry = QueryTelemetry(service_name="test", service_version="1", environment="test")
    telemetry.query_requests = _Capture()
    telemetry.query_failures = _Capture()
    observation = telemetry.observe("query-123", "hybrid")
    secret = "credential-shaped-value"

    observation.record_failure(RuntimeError(f"provider returned {secret}"))

    assert telemetry.query_failures.calls[0][1]["error.type"] == "internal_error"
    assert secret not in _span_text(observation)
    observation.__exit__(None, None, None)
    telemetry.shutdown()


def _span_text(observation) -> str:
    span = observation._span
    events = [(event.name, dict(event.attributes)) for event in span.events]
    return repr(dict(span.attributes)) + repr(events) + repr(span.status)


class _Capture:
    def __init__(self):
        self.calls = []

    def add(self, value, attributes):
        self.calls.append((value, attributes))

    def record(self, value, attributes):
        self.calls.append((value, attributes))


def test_query_metrics_are_bounded_and_query_id_stays_in_the_trace_only():
    telemetry = QueryTelemetry(
        service_name="test", service_version="1", environment="test", input_cost_per_million=1.0, output_cost_per_million=2.0
    )
    telemetry.query_requests = _Capture()
    telemetry.query_duration = _Capture()
    telemetry.stage_duration = _Capture()
    telemetry.llm_tokens = _Capture()
    telemetry.llm_cost = _Capture()
    telemetry.usage_missing = _Capture()
    telemetry.evidence_quality_failures = _Capture()
    observation = telemetry.observe("query-123", "hybrid")
    observation.record_success(
        {
            "effective_mode": "hybrid",
            "chunks": [object()],
            "latency": {"generation_ms": 25.0},
            "llm_usage": [{"operation": "generation", "model": "model-a", "input_tokens": 10, "output_tokens": 5}],
            "citation_validation": {"structured_output_valid": True, "rejected_claims": [{}]},
            "grounding_critique": {"completed": True},
        },
        31.0,
    )

    emitted = [attributes for _, attributes in telemetry.query_requests.calls]
    assert emitted == [{
        "http.route": "/api/v1/query",
        "kinegraph.query.requested_mode": "hybrid",
        "kinegraph.query.effective_mode": "hybrid",
        "kinegraph.query.outcome": "success",
    }]
    assert all("query-123" not in str(attributes) for _, attributes in telemetry.llm_tokens.calls)
    assert telemetry.evidence_quality_failures.calls[0][1] == {
        "kinegraph.evidence_quality.proxy": "citation_rejection"
    }
    observation.__exit__(None, None, None)
    telemetry.shutdown()
