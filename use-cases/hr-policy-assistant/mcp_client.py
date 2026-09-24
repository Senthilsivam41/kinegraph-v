"""Standalone MCP adapter for the synthetic HR policy use case."""

import argparse
import base64
import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


PROTOCOL_VERSION = "2026-07-28"
ROOT = Path(__file__).parent
CONTRACT = json.loads((ROOT / "contract" / "prompts.json").read_text())
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_PDF_BYTES = 20 * 1024 * 1024


def _decode_response(data, content_type, request_id):
    text = data.decode("utf-8")
    if "text/event-stream" not in content_type:
        return json.loads(text)
    for event in text.replace("\r\n", "\n").split("\n\n"):
        payload = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
        if payload and payload != "[DONE]":
            message = json.loads(payload)
            if message.get("id") == request_id:
                return message
    raise ValueError("MCP response did not contain the requested JSON-RPC result")


def call_tool(endpoint, tool_name, arguments, *, opener=urlopen, request_id=None):
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("MCP endpoint must be an absolute HTTP(S) URL")
    request_id = request_id or str(uuid.uuid4())
    meta = {
        "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientInfo": {"name": "kinegraph-use-case-client", "version": "1.0.0"},
        "io.modelcontextprotocol/clientCapabilities": {},
    }
    body = {"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments, "_meta": meta}}
    request = Request(endpoint, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": PROTOCOL_VERSION,
        "Mcp-Method": "tools/call",
        "Mcp-Name": tool_name,
    })
    with opener(request, timeout=60) as response:
        data = response.read(MAX_RESPONSE_BYTES + 1)
        if len(data) > MAX_RESPONSE_BYTES:
            raise ValueError("MCP response exceeds 4 MiB")
        return _decode_response(data, response.headers.get("Content-Type", ""), request_id)


def normalize(tool_name, message):
    result = message.get("result", {})
    structured = result.get("structuredContent") if isinstance(result, dict) else None
    if structured is None and isinstance(result, dict):
        for item in result.get("content", []):
            if item.get("type") == "text":
                try:
                    structured = json.loads(item["text"])
                except (KeyError, json.JSONDecodeError):
                    pass
                break
    structured = structured if isinstance(structured, dict) else {}
    error = message.get("error") or (result.get("content") if isinstance(result, dict) and result.get("isError") else None)
    return {
        "schema_version": "kinegraph.mcp.evidence.v1",
        "transport": "mcp-streamable-http",
        "protocol_version": PROTOCOL_VERSION,
        "tool": tool_name,
        "request_id": message.get("id"),
        "status": "failure" if error else "success",
        "error": error,
        "query_id": structured.get("query_id"),
        "effective_mode": structured.get("effective_mode"),
        "citations": structured.get("citation_validation"),
        "latency": structured.get("latency_breakdown", structured.get("execution_time_ms")),
        "response": structured or result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.getenv("KINEGRAPH_MCP_URL"), required=not os.getenv("KINEGRAPH_MCP_URL"))
    commands = parser.add_subparsers(dest="command", required=True)
    query = commands.add_parser("query")
    query.add_argument("--case", choices=[case["id"] for case in CONTRACT["cases"]], required=True)
    query.add_argument("--mode", choices=("hybrid", "vector", "graph", "vectorless"), default="hybrid")
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "query":
        case = next(case for case in CONTRACT["cases"] if case["id"] == args.case)
        tool, arguments = "kinegraph.query", {"query": case["question"], "mode": args.mode, "allow_mode_downgrade": False}
    else:
        if args.pdf.suffix.lower() != ".pdf" or not args.pdf.is_file():
            parser.error("--pdf must name an existing PDF")
        content = args.pdf.read_bytes()
        if len(content) > MAX_PDF_BYTES:
            parser.error("PDF exceeds the 20 MiB use-case limit")
        tool, arguments = "kinegraph.ingest_document", {"file_name": args.pdf.name, "content_base64": base64.b64encode(content).decode(), "metadata": {"source_corpus_sha256": CONTRACT["corpus_sha256"]}}

    print(json.dumps(normalize(tool, call_tool(args.endpoint, tool, arguments)), indent=2))


if __name__ == "__main__":
    main()
