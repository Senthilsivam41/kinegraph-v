# Ask HR policy demo

The Northwind Analytics handbook and all six questions are fictional. Use them
to exercise retrieval and source verification. Do not use the demo for real
employment, benefits, or expense decisions.

## Five-minute walkthrough

1. Start a local Kinegraph development stack with an ingestion worker,
   Neo4j with APOC, and working chat and embedding models. See the
   [repository setup guide](../../README.md).
   Start with fresh demo storage so the two routes see the same corpus.
2. Start the standalone browser client from this directory.

   ```shell
   python -m http.server 8080 --bind 127.0.0.1
   ```

   Open <http://localhost:8080/web/> and set the Kinegraph API URL (by default
   `http://localhost:8000`). If port 8080 is occupied, use another port and add
   that exact browser origin to `CORS_ALLOWED_ORIGINS` on the API.
3. Open the bundled handbook if you want to read it, then select **Ingest
   handbook**. The client verifies the PDF's SHA-256, uploads it through
   `POST /api/v1/ingest/document`, and waits for
   `GET /api/v1/ingest/task/{task_id}` to report success.
4. Run HR-001 and HR-002 in both **Hybrid** and **Vector**. Review the answer,
   effective route, latency, citations, and grounding evidence against
   [the prompt contract](contract/prompts.json). HR-001 combines the team roster
   with the transfer approval rule. HR-002 tests precedence between a named
   exception and a general rule. HR-006 should state the handbook's contractor
   PTO limit and decline to infer terms from the missing addendum.

The [Markdown source](data/northwind-analytics-employee-handbook.md) and its
[ingestion PDF](data/northwind-analytics-employee-handbook.pdf) each have a hash
in the contract. The browser attaches the verified PDF hash as
`demo_pdf_sha256` when ingesting. The browser and CLI use that same metadata
filter for queries. If you ingested the handbook before this metadata was
introduced, use the browser to ingest the bundled copy.

## Record all twelve results

Run the comparison from this directory once the browser ingestion completes.

```shell
python run_comparison.py \
  --base-url http://localhost:8000 \
  --code-revision <running-kinegraph-commit> \
  --environment "local; describe chat and embedding models and relevant settings" \
  --output /tmp/hr-policy-comparison.json
```

The runner prints overall and per-route counts and writes JSON with the two corpus hashes,
service revision, timestamp, expected facts and sections, and the full query
response or failure for each case and route. It records the query ID when the
API supplies it. It flags route changes, missing evidence fields, uncited responses, and
expected phrases absent from the answer. The phrase check is literal. The report
also records the initial graph-candidate count and the number of candidates sent
to generation with a graph path that carries evidence text on every edge. Missing
trace fields are recorded as `null`.

The graph participation check covers HR-001 (transfer approval chain) and HR-004
(role classification and expense rule). Each Hybrid response needs an initial
graph candidate and a selected candidate with an evidence-bearing graph path.
For a comparison of retrieval quality, inspect whether the selected paths connect
the entities and sections the contract calls for. Hold the corpus, models, and
context budget fixed. Compare coverage and rank of the expected evidence against
Vector, and review paraphrases, citations, and refusals separately from retrieval.
The runner leaves comparative retrieval quality unevaluated.

Keep the successful ingestion task response with
the report to document what went into the index. Use `--timeout` to increase
the per-query timeout from 60 seconds when running a slower local model.

To capture one case, use the same API URL and revision.

```shell
python client.py --case HR-001 --mode hybrid \
  --code-revision <running-kinegraph-commit> > hr-001-hybrid.json
```

## Benchmark retrieval features

The [benchmark contract](contract/benchmark.json) adds 14 handbook-grounded
questions. The original six remain available for reproducing the smoke run.

| Questions | Feature exercised |
| --- | --- |
| HR-B01 | One-hop relationship between Customer Success and its team lead. Adaptive routing can select Graph. |
| HR-B02–B04 | Cross-department transfer chains, paraphrased client entertainment lookup, and policy precedence. The transfer also compares one-hop depth, BFS, DFS, and community traversal. |
| HR-B05–B06 | Exact `401(k)` and `FTE` lookup, misspelled names, lexical fusion, and cross-encoder reranking. |
| HR-B07–B08 | Compound and indirect queries for decomposition, vocabulary expansion, conditional HyDE, and corrective retrieval. |
| HR-B09–B10 | Partial answer with an unavailable addendum and refusal on a policy absent from the handbook. |
| HR-B11–B12 | Table lookup through scoped Vectorless and Markdown attachment routing, plus a high-confidence adaptive Vector route. |
| HR-B13–B14 | Approval thresholds and competing policy rules for distractor rejection, reranking, and context selection. |

From the repository root, validate the question references and compare the
expected graph entities against the original ingestion receipt.

```shell
python use-cases/hr-policy-assistant/run_benchmark.py --plan \
  --ingest-receipt reports/test-runs/2026-11-11/hr-policy/browser-smoke.json
```

The archived receipt misses Customer Success in HR-B01, Data Science,
Engineering, Customer Success, and Revenue in HR-B02, and Sales in HR-B03.
The receipt checks entity presence. To establish graph readiness, verify the
listed paths and their source evidence in the newly ingested graph. A Hybrid
request exercises the intended relation when a matching path survives into the
returned context. Keep the PDF scope filter enabled during the comparison.

An offline check against the archived six-chunk index ran the local Vectorless
BM25 retrieval stage directly. At three chunks, it covered all reference
sections for 10 of 13 answerable or partially answerable questions. It missed
the roster for HR-B01 and HR-B02 and the expense rule for HR-B13. The full
refusal case has no expected evidence section.

After ingesting into the local stack, run a three-way check for the graph
questions before the full 63-request matrix.

```shell
python use-cases/hr-policy-assistant/run_benchmark.py \
  --base-url http://localhost:8000 \
  --code-revision "$(git rev-parse HEAD)" \
  --environment 'Record the actual chat and embedding models, parser, and index/chunking settings' \
  --case HR-B01 --case HR-B02 --case HR-B03 \
  --profile vector --profile hybrid --profile graph \
  --output /tmp/hr-graph-benchmark.json
```

Omit the case and profile selectors for the full matrix. The report records
the request settings, corpus and contract hashes, expected and observed routes,
initial channel counts in the API response, graph paths, gold evidence coverage
at rank 1 and across returned contexts, cited evidence coverage, and latency.
Evidence coverage uses reviewed handbook excerpts and tolerates whitespace and
punctuation differences. Use the reference answer and cited chunks to review
answer correctness, unsupported claims, and refusals. The benchmark sets a
three-chunk context limit and a ten-candidate pool per channel. Feature profiles
use matched baselines. In particular, HyDE compares against recovery alone,
and attachment auto-routing compares against explicit Vectorless on the same
Markdown attachment.

For a chunking comparison, repeat the matrix against two clean indexes built
with `ADAPTIVE_CHUNKING_ENABLED=false` and `true`, respectively. Keep the
parser, models, and PDF fixed and record both ingestion receipts. The one-file
handbook can exercise source caps and provide a regression baseline. Measuring
community diversity or generalizing quality or latency requires a larger,
multi-document corpus and repeated runs. The public response lacks pre-filter
traversal diagnostics, so the report leaves traversal execution unconfirmed
unless separately verified in the workflow trace.

## Troubleshooting

- **Failed to fetch**. Confirm the API base URL and the API's `/health/`
  response. Verify that the page's origin is explicitly allowed by
  `CORS_ALLOWED_ORIGINS`.
- **Ingestion is pending or timed out**. Use the displayed task ID to check
  `/api/v1/ingest/task/{task_id}`. Confirm the worker and its Redis queue are
  running before retrying. For failed tasks, inspect the task error and worker
  logs before running a query. If LiteParse returns empty text for the bundled
  PDF, check its API version. For a disposable local demo you can use the
  built-in PyMuPDF fallback by pointing the worker's `PARSER_URL` at an unused
  loopback port and restarting that worker.
- **No citations or an unexpected route**. Check the effective mode and the
  returned grounding fields. Confirm the task succeeded against the same API
  URL and review the API response for unavailable retrieval channels.
- **Graph participation check fails**. Use the initial graph count to see
  whether the graph channel produced candidates. The scoped retriever can
  reload a graph-discovered chunk from Chroma while dropping its relationship
  path. To pass the check, the response needs to carry a path verified against
  the same corpus scope.

## Verification

The focused checks use deterministic responses and require no model service.
Run them from the repository root.

```shell
python -m pytest -q tests/test_hr_policy_demo.py tests/test_hr_policy_comparison.py tests/test_hr_policy_benchmark.py tests/test_hr_policy_web.py
node --test tests/hr_policy_web.test.mjs
```

For a live smoke run, retain the browser's upload response, terminal task
status, and one query response. Record the deployed revision, corpus hashes,
API URL and browser origin, model names, requested and effective routes, and
the returned latency and provenance. Use the batch command above to preserve
the Hybrid and Vector observations for all six cases.

See the [November 11 local smoke report](../../reports/test-runs/2026-11-11/hr-policy/REPORT.md)
for the captured browser and comparison results.

## Optional MCP transport

Point the standalone adapter at a Streamable HTTP endpoint that implements
the [Kinegraph use-case MCP contract v1](../MCP_CONTRACT.md). Use a disposable
index containing only the synthetic handbook. The v1 MCP query tool does
not accept the corpus filter used by the HTTP clients.

```shell
python mcp_client.py --endpoint http://localhost:9000/mcp ingest > hr-mcp-ingest.json
# Wait for the returned task ID to succeed through the server's ingestion status API.
python mcp_client.py --endpoint http://localhost:9000/mcp \
  query --case HR-001 --mode hybrid > hr-mcp-query.json
```

Set `KINEGRAPH_MCP_URL` to reuse the endpoint. The adapter prints a JSON
evidence record and exits nonzero for tool, protocol, or transport failures.
The normal browser, CLI, and comparison runner use the public HTTP API.

Run `python -m pytest -q tests/test_mcp_use_case_adapter.py` from the
repository root for deterministic request and evidence checks. For a live
smoke test, retain the ingestion response, terminal task status, query
response, server revision, and test environment. Compatibility with a live
Kinegraph MCP server remains unverified.
