from backend.observability.telemetry import QueryTelemetry, estimate_cost_usd, normalize_usage


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
