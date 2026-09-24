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
3. Run a case and name the revision of the Kinegraph service you started:

   ```shell
   python client.py --case HR-001 --mode hybrid \
     --code-revision <running-kinegraph-commit> > hr-001-hybrid.json
   ```

   Repeat for the six questions in `contract/prompts.json`. Check every answer
   against its `expected_facts` and `source_sections`; `HR-006` should state the
   handbook limit and decline details that are only in the missing addendum.
4. For a comparison, ingest the same PDF into an isolated Vector-only run and
   ask the same six questions with `--mode vector`. The client records the
   corpus hash, code revision, requested and effective route, full response,
   latency fields, and query ID when the API supplies one. Do not claim a
   retrieval advantage without these recorded runs.

The contract pins the input corpus with SHA-256
`95119b5dab08c01ea39adbd5bb8a2987f4b0153f278ce7ead37898d7f41504a8`.

## What to look for

`HR-001` joins team-to-department facts with the cross-department transfer
rule. `HR-002` resolves a named exception over a general rule. `HR-006` is an
evidence-quality check: a grounded partial refusal, not a hallucination-rate
claim.
