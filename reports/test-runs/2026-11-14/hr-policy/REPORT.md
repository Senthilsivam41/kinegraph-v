# HR policy live retrieval benchmark

On November 14, 2026 (IST), all 63 Kinegraph API requests completed with HTTP
200 across 14 questions and 17 profiles. The responses contained no chunks
outside the benchmark scope.

- [Full responses, request settings, and scores](benchmark.json)
- [Ingestion receipt](ingest-receipt.json)
- [Relationships checked directly in Neo4j](graph-inventory.json)

## Vector versus Hybrid

| Measure | Vector | Hybrid |
| --- | ---: | ---: |
| Successful requests | 14/14 | 14/14 |
| Requests with initial graph candidates | 0/14 | 14/14 |
| Complete expected evidence in the returned context | 12/13 | 9/13 |
| Mean expected section recall in returned context | 96.2% | 73.1% |
| Expected graph paths returned on annotated questions | 0/3 | 0/3 |

Evidence recall measures the proportion of expected handbook sections covered
by the returned chunks using the contract's text anchors. It excludes HR-B10
because the handbook does not establish a parental leave allocation. Hybrid
tied Vector on 10 scored questions and scored lower on HR-B08 (carryover),
HR-B09 (contractor PTO), and HR-B13 (the entertainment spending limit).
Both modes matched the roster anchors and missed the full approval-rule
anchors on HR-B02.

The expanded questions exercised graph retrieval. Each Hybrid request started
with four or five graph candidates. HR-B01's adaptive route selected Graph.
Graph-only retrieval moved the roster evidence from rank three to rank two on
that case. No response exposed an evidence-bearing relationship path, so the
benchmark cannot attribute evidence changes to traversal.

A direct Neo4j check confirmed the Customer Success team-lead edge and both
department-head chains for HR-B01 and HR-B02. The scoped graph retriever reloads
matching source chunks from Chroma and omits relationship-path metadata in the
returned record. HR-B03 also had an ingestion gap. Neo4j contained Sales →
Client-Facing Role and lacked the link to Expense Approval.

## Feature observations

| Feature | Observation |
| --- | --- |
| Adaptive routing | All five profiles matched the planned route, including a Graph route, a Vector route, and retained Hybrid. |
| Lexical fusion | Both runs had lexical candidates. Both tied fixed Hybrid on recall. |
| Cross-encoder | Ran in all four profiles. Recall improved once, fell once, and tied twice against fixed Hybrid. |
| Vectorless | All three scoped runs retrieved every expected section. Explicit attachment Vectorless and attachment auto-routing both retrieved the table evidence. |
| Structured recovery and HyDE | Neither activated. Initial retrieval was assessed as adequate even for HR-B08, where the final context missed carryover evidence. |
| Corrective retrieval | Requested follow-up retrieval in all three cases. It failed to recover expected evidence in the two answerable cases. |
| Verification | Correctly refused the missing parental leave question. It also refused the contractor PTO question, which expected a partial answer. |
| Context optimization | Ran in all three profiles. One case lost expected evidence against fixed Hybrid. |

## Run conditions

The shared Docker API could reach Neo4j. Its Redis container had only a
loopback interface and restarting it did not restore the network attachment.
Its Chroma client was version 1.5.9 against server 0.5.20, and the collection
initialization call returned `KeyError('_type')`. This run used a separate
temporary Neo4j 5.16.0 store with APOC 5.16.1, a Chroma 1.5.9 server that matched
the client, and a separate API process inside the local Docker environment.
The Celery task body ran directly without Redis. LiteParse returned empty
text on the first attempt, so the run used PyMuPDF on the same pinned PDF.
Ingestion produced six chunks, 30 entities, and 28 extracted relationships.
The configured chat model was Qwen 3.6 27B, with
`text-embedding-ada-002` for embeddings and `gpt-4o-mini` for graph synonyms.

The matrix used up to eight concurrent requests and one run per case and
profile. The recorded latency reflects shared load. The scored measures
describe retrieved and cited evidence. Answer correctness needs separate
review. A replay of the saved responses corrected the benchmark's
cross-encoder detection for the API's graph-aware mode label.
