from backend.observability.telemetry import (
    ALLOWED_ERROR_CODES,
    QueryTelemetry,
    estimate_cost_usd,
    normalize_usage,
    sanitized_error_code,
)


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
