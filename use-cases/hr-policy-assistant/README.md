# Ask HR policy demo

This is fictional data for a five-minute retrieval demo. It is not HR advice
and it must not be used for real workplace decisions.

`client.py` is a standalone standard-library HTTP client for Kinegraph's public
`POST /api/v1/query/` API. It does not import or modify the core service.

## Run it

1. Convert `data/northwind-analytics-employee-handbook.md` to a PDF with a local
   tool you trust. The document-ingest endpoint accepts PDFs only.
2. Start Kinegraph with the Hybrid route enabled, then upload that PDF through
   the UI or `POST /api/v1/ingest/document`. Wait for the queued ingestion task
   to complete.
3. Run `python client.py --case HR-001` (or use its API request as a template)
   for the six questions in `contract/prompts.json`. Check every answer against its
   `expected_facts` and `source_sections`; `HR-006` should state the handbook
   limit and decline details that are only in the missing addendum.
4. For a comparison, ingest the same PDF into an isolated Vector-only run and
   ask the same six questions. Record the route, answer, cited sections, query
   ID, and latency for each case. Do not claim a retrieval advantage without
   this recorded run.

The contract pins the input corpus with SHA-256
`95119b5dab08c01ea39adbd5bb8a2987f4b0153f278ce7ead37898d7f41504a8`.

## What to look for

`HR-001` joins team-to-department facts with the cross-department transfer
rule. `HR-002` resolves a named exception over a general rule. `HR-006` is an
evidence-quality check: a grounded partial refusal, not a hallucination-rate
claim.
