# Kinegraph use-case MCP contract v1

Status: client contract. A live Kinegraph MCP server has not yet been verified.

Use-case clients pin MCP protocol `2026-07-28` over Streamable HTTP. Each
request carries the protocol version, client identity, and capabilities in its
`_meta` object. The adapter calls tools directly; `server/discover` is optional
for this pinned, stateless protocol revision.

## Tools

### `kinegraph.query`

Arguments:

- `query` (string, required)
- `mode` (`hybrid`, `vector`, `graph`, or `vectorless`; required)
- `allow_mode_downgrade` (boolean, required)

The tool returns the public query response as `structuredContent`. It should
include requested and effective mode, generated answer, citations or grounding
fields, and latency. `query_id` is optional until the public response exposes
one.

### `kinegraph.ingest_document`

Arguments:

- `file_name` (PDF basename, required)
- `content_base64` (base64-encoded PDF bytes, required)
- `metadata` (object, required)

The tool returns the public ingestion task response as `structuredContent`,
including `task_id`, status, and message.

## Failure contract

JSON-RPC errors and MCP results with `isError: true` are failures. Clients must
retain the tool name, request ID, and returned error without reporting a
successful Kinegraph operation.
