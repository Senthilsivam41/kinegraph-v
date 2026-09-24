import importlib.util
import json
from pathlib import Path


CLIENT_PATH = Path(__file__).parents[1] / "use-cases" / "hr-policy-assistant" / "mcp_client.py"


def load_client():
    spec = importlib.util.spec_from_file_location("hr_policy_mcp_client", CLIENT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload, content_type="application/json"):
        self.payload = payload
        self.headers = {"Content-Type": content_type}

    def read(self, _limit):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def test_mcp_query_request_and_evidence_normalization():
    client = load_client()
    captured = {}
    result = {
        "jsonrpc": "2.0",
        "id": "request-1",
        "result": {
            "structuredContent": {
                "effective_mode": "hybrid",
                "query_id": "query-1",
                "citation_validation": {"valid": True},
                "latency_breakdown": {"total_ms": 42},
            }
        },
    }

    def opener(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["timeout"] = timeout
        event = f"data: {json.dumps(result)}\n\n".encode()
        return FakeResponse(event, "text/event-stream")

    message = client.call_tool(
        "http://localhost:9000/mcp",
        "kinegraph.query",
        {"query": "Who approves?", "mode": "hybrid", "allow_mode_downgrade": False},
        opener=opener,
        request_id="request-1",
    )
    evidence = client.normalize("kinegraph.query", message)

    assert captured["body"]["method"] == "tools/call"
    assert captured["body"]["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"] == "2026-07-28"
    assert captured["headers"]["mcp-protocol-version"] == "2026-07-28"
    assert captured["headers"]["mcp-name"] == "kinegraph.query"
    assert captured["timeout"] == 60
    assert evidence["status"] == "success"
    assert evidence["query_id"] == "query-1"
    assert evidence["citations"] == {"valid": True}
    assert evidence["latency"] == {"total_ms": 42}


def test_mcp_failure_is_retained_as_evidence():
    client = load_client()
    evidence = client.normalize(
        "kinegraph.ingest_document",
        {"jsonrpc": "2.0", "id": "request-2", "error": {"code": -32602, "message": "Invalid params"}},
    )

    assert evidence["status"] == "failure"
    assert evidence["request_id"] == "request-2"
    assert evidence["error"]["code"] == -32602
