"""Call the standalone HR demo's versioned MCP tools over Streamable HTTP."""

import argparse
import base64
import hashlib
import json
import os
import uuid
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

PROTOCOL_VERSION = "2026-07-28"
CONTRACT_DIR = Path(__file__).parent / "contract"
CONTRACT = json.loads((CONTRACT_DIR / "prompts.json").read_text(encoding="utf-8"))
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
QUERY_TOOL = "kinegraph.query"
INGEST_TOOL = "kinegraph.ingest_document"
MODES = ("hybrid", "vector", "graph", "vectorless")


def read_message(response, request_id):
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if content_type == "application/json":
        data = response.read(MAX_RESPONSE_BYTES + 1)
        if len(data) > MAX_RESPONSE_BYTES:
            raise ValueError("MCP response exceeds 4 MiB")
        return json.loads(data)
    if content_type != "text/event-stream":
        raise ValueError("Expected a JSON or SSE MCP response")

    remaining = MAX_RESPONSE_BYTES
    event = []
    while line := response.readline(remaining + 1):
        remaining -= len(line)
        if remaining < 0:
            raise ValueError("MCP response exceeds 4 MiB")
        line = line.rstrip(b"\r\n")
        if line.startswith(b"data:"):
            event.append(line[6:] if line.startswith(b"data: ") else line[5:])
        elif not line and event:
            message = json.loads(b"\n".join(event))
            if isinstance(message, dict) and message.get("id") == request_id:
                return message
            event.clear()
    raise ValueError("MCP stream ended before the matching JSON-RPC response")


def normalize(message, request_id):
    if (
        not isinstance(message, dict)
        or message.get("jsonrpc") != "2.0"
        or message.get("id") != request_id
        or ("error" in message) == ("result" in message)
    ):
        return message, {"kind": "protocol", "message": "Invalid or mismatched JSON-RPC response"}
    if "error" in message:
        return message, {"kind": "jsonrpc", "detail": message["error"]}

    result = message["result"]
    if isinstance(result, dict):
        if result.get("isError") is True:
            return result, {"kind": "tool", "detail": result.get("content")}
        if result.get("resultType") == "input_required":
            return result, {"kind": "input_required", "message": "The tool requires interactive input"}
        if result.get("resultType") == "complete":
            structured = result.get("structuredContent")
            if isinstance(structured, dict):
                return structured, None
            if "structuredContent" not in result and isinstance(result.get("content"), list):
                for item in result["content"]:
                    if not isinstance(item, dict) or item.get("type") != "text":
                        continue
                    try:
                        structured = json.loads(item.get("text", ""))
                    except (TypeError, ValueError):
                        continue
                    if isinstance(structured, dict):
                        return structured, None
    return result, {"kind": "protocol", "message": "Expected a complete tool result with a JSON object"}


def call_tool(endpoint, tool, arguments, *, opener=urlopen, request_id=None, timeout=60):
    request_id = request_id or str(uuid.uuid4())
    result = None
    try:
        url = urlsplit(endpoint)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.fragment:
            raise ValueError("Provide an absolute HTTP(S) MCP endpoint without URL credentials or fragments")
        body = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": tool,
                "arguments": arguments,
                "_meta": {
                    "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
                    "io.modelcontextprotocol/clientInfo": {"name": "kinegraph-use-case-client", "version": "1.0.0"},
                    "io.modelcontextprotocol/clientCapabilities": {},
                },
            },
        }
        request = Request(endpoint, data=json.dumps(body).encode("utf-8"), headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
            "Mcp-Method": "tools/call",
            "Mcp-Name": tool,
        })
        with opener(request, timeout=timeout) as response:
            result, error = normalize(read_message(response, request_id), request_id)
    except HTTPError as exc:
        with exc:
            try:
                result = read_message(exc, request_id)
            except (HTTPException, OSError, ValueError):
                pass
        error = {"kind": "http", "status_code": exc.code}
    except (HTTPException, OSError) as exc:
        error = {"kind": "transport", "message": str(exc)}
    except ValueError as exc:
        error = {"kind": "protocol", "message": str(exc)}

    fields = result if isinstance(result, dict) else {}
    if error is not None:
        rpc_error = fields.get("error")
        detail = rpc_error.get("data") if isinstance(rpc_error, dict) else fields.get("detail")
        structured = fields.get("structuredContent")
        if isinstance(structured, dict):
            fields = structured
        elif isinstance(detail, dict):
            fields = detail
    return {
        "schema_version": "kinegraph.mcp.evidence.v1",
        "transport": "mcp-streamable-http",
        "protocol_version": PROTOCOL_VERSION,
        "tool": tool,
        "request_id": request_id,
        "status": "failure" if error is not None else "success",
        "error": error,
        "requested_mode": arguments.get("mode"),
        "effective_mode": fields.get("effective_mode"),
        "query_id": fields.get("query_id"),
        "citations": fields.get("citations", fields.get("citation_validation")),
        "grounded_claims": fields.get("grounded_claims"),
        "grounding_critique": fields.get("grounding_critique"),
        "execution_time_ms": fields.get("execution_time_ms"),
        "latency_breakdown": fields.get("latency_breakdown"),
        "response": result,
    }


def ingestion_arguments():
    pdf = CONTRACT_DIR / CONTRACT["ingestion_pdf"]
    content = pdf.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != CONTRACT["ingestion_pdf_sha256"]:
        raise ValueError("The bundled PDF SHA-256 differs from the HR prompt contract")
    return {
        "file_name": pdf.name,
        "content_base64": base64.b64encode(content).decode("ascii"),
        "metadata": {"source_corpus_sha256": CONTRACT["corpus_sha256"], "demo_pdf_sha256": digest},
    }


def main():
    cases = {case["id"]: case for case in CONTRACT["cases"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.getenv("KINEGRAPH_MCP_URL"), help="Streamable HTTP MCP URL")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")
    commands = parser.add_subparsers(dest="command", required=True)
    query = commands.add_parser("query")
    query.add_argument("--case", choices=cases, required=True)
    query.add_argument("--mode", choices=MODES, default="hybrid")
    commands.add_parser("ingest", help="Upload the bundled synthetic handbook")
    args = parser.parse_args()
    if not args.endpoint or args.timeout < 1:
        parser.error("Set --endpoint or KINEGRAPH_MCP_URL and use a positive timeout")

    if args.command == "query":
        case = cases[args.case]
        arguments = {"query": case["question"], "mode": args.mode, "allow_mode_downgrade": False}
        evidence = call_tool(args.endpoint, QUERY_TOOL, arguments, timeout=args.timeout)
        evidence["case"] = case
    else:
        try:
            arguments = ingestion_arguments()
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        evidence = call_tool(args.endpoint, INGEST_TOOL, arguments, timeout=args.timeout)
        evidence.update(arguments["metadata"])

    print(json.dumps(evidence, indent=2))
    return int(evidence["status"] == "failure")


if __name__ == "__main__":
    raise SystemExit(main())
