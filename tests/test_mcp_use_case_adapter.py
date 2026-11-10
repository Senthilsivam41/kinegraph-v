import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

CLIENT_PATH = Path(__file__).parents[1] / "use-cases/hr-policy-assistant/mcp_client.py"
ENDPOINT = "http://localhost:9000/mcp"


@pytest.fixture
def client():
    spec = importlib.util.spec_from_file_location("hr_policy_mcp_client", CLIENT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse(io.BytesIO):
    def __init__(self, data, content_type="application/json"):
        super().__init__(data)
        self.headers = {"Content-Type": content_type}


def test_query_constructs_stateless_request_and_normalizes_sse_evidence(client):
    case = client.CONTRACT["cases"][0]
    args = {"query": case["question"], "mode": "hybrid", "allow_mode_downgrade": False}
    result = {
        "query_id": "query-1",
        "effective_mode": "hybrid",
        "citation_validation": {"accepted_claims": 1},
        "grounded_claims": [{"chunk_ids": ["chunk-1"]}],
        "execution_time_ms": 42,
        "latency_breakdown": {"generation_ms": 12},
    }
    message = {
        "jsonrpc": "2.0", "id": "request-1",
        "result": {"resultType": "complete", "structuredContent": result},
    }

    def opener(request, timeout):
        assert request.full_url == ENDPOINT
        assert request.get_method() == "POST"
        assert timeout == 45
        headers = {key.lower(): value for key, value in request.header_items()}
        assert headers["mcp-protocol-version"] == client.PROTOCOL_VERSION
        assert headers["mcp-method"] == "tools/call"
        assert headers["mcp-name"] == "kinegraph.query"
        assert headers["accept"] == "application/json, text/event-stream"
        payload = json.loads(request.data)
        assert payload["id"] == "request-1"
        assert payload["method"] == "tools/call"
        assert payload["params"]["name"] == "kinegraph.query"
        assert payload["params"]["arguments"] == args
        assert payload["params"]["_meta"] == {
            "io.modelcontextprotocol/protocolVersion": client.PROTOCOL_VERSION,
            "io.modelcontextprotocol/clientInfo": {"name": "kinegraph-use-case-client", "version": "1.0.0"},
            "io.modelcontextprotocol/clientCapabilities": {},
        }
        # A notification precedes the matching response. Subsequent events need not be read.
        stream = b'data: {"jsonrpc":"2.0","method":"notifications/progress"}\r\n\r\n'
        stream += f"data: {json.dumps(message)}\r\n\r\ndata: invalid-json\r\n\r\n".encode()
        return FakeResponse(stream, "text/event-stream; charset=utf-8")

    evidence = client.call_tool(ENDPOINT, client.QUERY_TOOL, args, opener=opener, request_id="request-1", timeout=45)

    assert evidence["status"] == "success"
    assert evidence["request_id"] == "request-1"
    assert evidence["response"] == result
    assert evidence["query_id"] == "query-1"
    assert evidence["requested_mode"] == evidence["effective_mode"] == "hybrid"
    assert evidence["citations"] == result["citation_validation"]
    assert evidence["grounded_claims"] == result["grounded_claims"]
    assert evidence["execution_time_ms"] == 42
    assert evidence["latency_breakdown"] == result["latency_breakdown"]


def test_ingest_sends_the_pinned_pdf_and_reads_a_text_result(client):
    arguments = client.ingestion_arguments()
    task = {"task_id": "task-1", "status": "PENDING", "message": "Queued"}

    def opener(request, timeout):
        payload = json.loads(request.data)
        assert payload["params"]["name"] == "kinegraph.ingest_document"
        sent = payload["params"]["arguments"]
        pdf = base64.b64decode(sent["content_base64"], validate=True)
        assert pdf.startswith(b"%PDF-")
        assert sent["metadata"] == {
            "demo_pdf_sha256": hashlib.sha256(pdf).hexdigest(),
            "source_corpus_sha256": client.CONTRACT["corpus_sha256"],
        }
        assert sent["metadata"]["demo_pdf_sha256"] == client.CONTRACT["ingestion_pdf_sha256"]
        assert sent["file_name"] == "northwind-analytics-employee-handbook.pdf"
        return FakeResponse(json.dumps({
            "jsonrpc": "2.0", "id": payload["id"],
            "result": {"resultType": "complete", "content": [{"type": "text", "text": json.dumps(task)}]},
        }).encode())

    evidence = client.call_tool(ENDPOINT, client.INGEST_TOOL, arguments, opener=opener)
    assert evidence["status"] == "success"
    assert evidence["response"] == task


@pytest.mark.parametrize(("message", "kind"), [
    ({"id": "request-2", "error": {"code": -32602, "message": "Invalid params"}}, "jsonrpc"),
    ({"id": "request-2", "result": {"resultType": "complete", "isError": True, "content": []}}, "tool"),
    ({"id": "request-2", "result": {"resultType": "input_required"}}, "input_required"),
    ({"id": "other-request", "result": {"resultType": "complete", "structuredContent": {"x": 1}}}, "protocol"),
    ({"id": "request-2", "result": {"resultType": "complete", "structuredContent": "invalid"}}, "protocol"),
])
def test_protocol_and_tool_failures_keep_request_identity(client, message, kind):
    def opener(request, timeout):
        return FakeResponse(json.dumps({"jsonrpc": "2.0", **message}).encode())

    evidence = client.call_tool(ENDPOINT, client.QUERY_TOOL, {}, opener=opener, request_id="request-2")
    assert evidence["status"] == "failure"
    assert evidence["request_id"] == "request-2"
    assert evidence["tool"] == "kinegraph.query"
    assert evidence["error"]["kind"] == kind
    assert evidence["response"] is not None


def test_http_and_connection_failures_produce_evidence(client):
    payload = {"jsonrpc": "2.0", "id": "request-3", "error": {
        "code": -32602, "message": "Rejected", "data": {"query_id": "query-3"},
    }}

    def failed_http(request, timeout):
        raise HTTPError(ENDPOINT, 400, "Bad Request", {"Content-Type": "application/json"},
                        io.BytesIO(json.dumps(payload).encode()))

    evidence = client.call_tool(ENDPOINT, client.QUERY_TOOL, {}, opener=failed_http, request_id="request-3")
    assert evidence["status"] == "failure"
    assert evidence["error"] == {"kind": "http", "status_code": 400}
    assert evidence["query_id"] == "query-3"
    assert evidence["response"] == payload

    def offline(request, timeout):
        raise URLError("connection refused")

    evidence = client.call_tool(ENDPOINT, client.QUERY_TOOL, {}, opener=offline, request_id="request-4")
    assert evidence["status"] == "failure"
    assert evidence["request_id"] == "request-4"
    assert evidence["error"]["kind"] == "transport"


def test_stream_reads_are_bounded(client, monkeypatch):
    monkeypatch.setattr(client, "MAX_RESPONSE_BYTES", 10)

    def opener(request, timeout):
        return FakeResponse(b":" + b"x" * 100, "text/event-stream")

    evidence = client.call_tool(ENDPOINT, client.QUERY_TOOL, {}, opener=opener)
    assert evidence["status"] == "failure"
    assert evidence["error"]["kind"] == "protocol"
    assert "exceeds" in evidence["error"]["message"]
