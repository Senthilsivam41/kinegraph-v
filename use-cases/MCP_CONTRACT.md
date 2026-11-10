# Kinegraph use-case MCP contract v1

This is the client contract for a separately configured MCP server.
Compatibility with a live Kinegraph MCP server is unverified.

Use-case clients pin MCP protocol `2026-07-28` over Streamable HTTP. Each
request carries the protocol version, client identity, and capabilities in its
`_meta` object. The adapter calls tools directly; `server/discover` is optional
for this pinned, stateless protocol revision.
See the [MCP transport specification](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports).

## Tools

### `kinegraph.query`

The tool accepts these arguments.

- `query` (string, required)
- `mode` (`hybrid`, `vector`, `graph`, or `vectorless`; required)
- `allow_mode_downgrade` (boolean, required)

The tool returns the public query response as `structuredContent`, including
`effective_mode`, `generated_answer`, `citation_validation`,
`grounded_claims`, `grounding_critique`, and `execution_time_ms` or
`latency_breakdown` when available. It may supply a `query_id`. The server
must honor `allow_mode_downgrade` as the public API does.

### `kinegraph.ingest_document`

The tool accepts these arguments.

- `file_name` (PDF basename, required)
- `content_base64` (base64-encoded PDF bytes, required)
- `metadata` (object, required)

The tool returns the public ingestion task response as `structuredContent`,
including `task_id`, status, and message.

The HR adapter submits the bundled PDF after checking its SHA-256. It sends the
source hash as `source_corpus_sha256` and the PDF hash as `demo_pdf_sha256`
in `metadata`. The v1 query tool has no corpus filter. Run the HR smoke in an
isolated index containing only the synthetic handbook.

Both tools may additionally return a JSON encoding of the structured object
in a text content block. The adapter uses that block if `structuredContent`
is absent.

## Failure contract

JSON-RPC errors and MCP results with `isError: true` are failures. Clients must
retain the tool name, request ID, and returned error without reporting a
successful Kinegraph operation. The HR adapter also reports transport failures,
invalid response envelopes, and `input_required` results as failures. The
adapter does not send interactive input responses.
