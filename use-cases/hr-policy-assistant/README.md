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
expected phrases absent from the answer. The phrase check is literal. Review
whether paraphrases, citations, and refusals satisfy the contract before
making a comparative claim. Keep the successful ingestion task response with
the report to document what went into the index. Use `--timeout` to increase
the per-query timeout from 60 seconds when running a slower local model.

To capture one case, use the same API URL and revision.

```shell
python client.py --case HR-001 --mode hybrid \
  --code-revision <running-kinegraph-commit> > hr-001-hybrid.json
```

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

## Verification

The focused checks use deterministic responses and require no model service.
Run them from the repository root.

```shell
python -m pytest -q tests/test_hr_policy_demo.py tests/test_hr_policy_comparison.py tests/test_hr_policy_web.py
node --test tests/hr_policy_web.test.mjs
```

For a live smoke run, retain the browser's upload response, terminal task
status, and one query response. Record the deployed revision, corpus hashes,
API URL and browser origin, model names, requested and effective routes, and
the returned latency and provenance. Use the batch command above to preserve
the Hybrid and Vector observations for all six cases.

See the [November 11 local smoke report](../../reports/test-runs/2026-11-11/hr-policy/REPORT.md)
for the captured browser and comparison results.
