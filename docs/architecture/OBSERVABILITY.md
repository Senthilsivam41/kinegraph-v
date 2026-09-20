# Query OpenTelemetry observability

The query API emits vendor-neutral OpenTelemetry traces and metrics. Configure
`OTEL_EXPORTER_OTLP_ENDPOINT` with an OTLP/HTTP collector base URL (for example,
`http://otel-collector:4318`) to export `/v1/traces` and `/v1/metrics`.

Each `POST /api/v1/query` produces the `kinegraph.query` span. Its generated
query UUID is an attribute for trace correlation only; it is never a metric
label. Prompt text, answer text, evidence content, credentials, and attachment
content are not sent as telemetry attributes.

| Signal | Type | Bounded dimensions | Purpose |
| --- | --- | --- | --- |
| `kinegraph.query.requests` | Counter | route, requested/effective mode, outcome | Query rate and success/failure split |
| `kinegraph.query.failures` | Counter | route, exception type | Failed query diagnosis |
| `kinegraph.query.duration` | Histogram (ms) | route, modes, outcome | Query latency percentiles |
| `kinegraph.query.stage.duration` | Histogram (ms) | stage | Retrieval, rerank, generation, and critique latency |
| `kinegraph.llm.tokens` | Counter | operation, configured model, input/output | Provider-reported LLM token usage |
| `kinegraph.llm.estimated_cost` | Histogram (USD) | operation, configured model | Estimated spend when both configured per-million prices are present |
| `kinegraph.llm.usage_missing` | Counter | operation, configured model | Provider did not return token metadata |
| `kinegraph.evidence_quality.failures` | Counter | fixed proxy category | Citation validation/rejection and grounding-check outcomes |

`kinegraph.evidence_quality.failures` is an evidence-quality proxy. It is not a
deterministic hallucination rate: it reports contract-level citation and
grounding-check failures/rejections, not a measurement of model truthfulness.

Token and cost values are aggregate provider metadata from generation and the
grounding critic. Cost is only emitted when both
`LLM_INPUT_COST_PER_MILLION_TOKENS` and
`LLM_OUTPUT_COST_PER_MILLION_TOKENS` are explicitly configured for the deployed
model. This avoids representing unknown pricing as zero.
