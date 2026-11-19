# Ask HR local smoke run

On November 11, 2026 (IST), Chrome uploaded the bundled Northwind Analytics
PDF, polled the ingestion task to `SUCCESS`, and rendered a live Hybrid
response to HR-001. The batch runner then sent the same six contract questions
through both requested modes against that same index. All 12 requests returned
HTTP 200 and used their requested modes.

- [Browser upload, task receipt, and query](browser-smoke.json)
- [All 12 API responses and per-mode summary](comparison.json)

## Environment

- Kinegraph API and Celery worker ran from commit
  `976426197a11f70e462b725f52c06580586fb025`.
- Chrome ran headlessly at `http://localhost:18088`. The API listened on
  `http://127.0.0.1:18081` and allowed that browser origin.
- Redis 7, Chroma 0.5.20, and Neo4j 5.16 Community with APOC 5.16.1 ran in
  disposable local containers. The corpus was the only document ingested.
  The task recorded six chunks, 18 entities, and 30 relationships.
- LM Studio served cached Nemotron 3 Nano 4B for chat and Nomic Embed Text
  v1.5 for embeddings. A loopback adapter decoded legacy embedding token IDs
  to text and supplied a JSON schema for entity extraction. The inference
  calls used local models and synthetic input.
- LiteParse returned empty text on the first attempt, so the worker retried
  using its PyMuPDF fallback. The first browser query failed while APOC was
  missing. After APOC was enabled on the same graph store, Chrome reran the
  query and rendered two cited claims in 22,348.63 ms.

The Markdown SHA-256 is
`95119b5dab08c01ea39adbd5bb8a2987f4b0153f278ce7ead37898d7f41504a8`.
The ingested PDF SHA-256 is
`c9cad04fd762536011a7bc6e481c95c10916cd339c49e0592a7613d4c1fb782b`.
Every returned chunk in the comparison carried that PDF hash. The API omitted
query IDs on successful responses. The JSON preserves the responses and uses
the current runner's per-mode summary format.

## Observed comparison

| Case | Hybrid API time | Vector API time | Observed answer in both modes |
| --- | ---: | ---: | --- |
| HR-001 Transfer approvals | 12,534 ms | 11,514 ms | Cited both approval rules and omitted all four approver names. |
| HR-002 Notice-period PTO | 19,753 ms | 6,858 ms | Cited Department Head approval and omitted the explicit override of Team Lead approval. |
| HR-003 401(k) waiver | 23,550 ms | 7,568 ms | Cited waiver eligibility and omitted the normal match and waiting period and the required approver. |
| HR-004 Client dinner | 25,218 ms | 10,454 ms | Grounding verification removed both draft claims as irrelevant in each mode. |
| HR-005 Permanent remote status | 17,820 ms | 4,708 ms | Cited the Team Lead and Department Head requirements. |
| HR-006 Contractor PTO | 17,162 ms | 4,825 ms | Cited the handbook's PTO limit and omitted the refusal about terms in the missing addendum. |

Each Hybrid response reported zero initial graph candidates and returned
vector-sourced chunks. A follow-up Hybrid transfer query against the
synthetic-only index with the metadata filter removed and retrieval
orchestration enabled also reported zero initial graph candidates. Ten
comparison responses missed at least one literal expected phrase. Comparative
retrieval quality remains unevaluated.

HR-001's returned context already contained all four approver names in both
modes. Both answers omitted the names while retaining the approval rules. The
batch establishes that the API served both routes and that Hybrid fell back to
vector evidence for this corpus. A follow-up comparison needs a demonstrated
graph contribution. Start with HR-001 and HR-004, verify that the graph contains
the needed relationships and source evidence, and confirm those paths reach
generation before scoring the paired responses.

## Offline checks

The Python suite passed 410 tests with five skipped. Four deterministic Node
tests passed. These checks ran separately from the live smoke run.
