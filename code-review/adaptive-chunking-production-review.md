# Adaptive Chunking — Production Readiness Review

**Date:** 2026-07-25  
**Scope:** Uncommitted ADR-002 work — `adaptive_chunking.py`, `ingest.py`, `document_processor.py`, `tasks.py`, config, scripts, tests  
**Canvas:** [`adaptive-chunking-production-readiness.canvas.tsx`](./adaptive-chunking-production-readiness.canvas.tsx)

## Verdict

**Not production-ready as a default.** Safe to experiment with `ADAPTIVE_CHUNKING_ENABLED=false`; do not promote adaptive chunking or treat ingestion as production-hardened yet.

Dual ingest paths disagree on idempotency and metadata, Celery partial-write retries can wedge the worker, and content-hash skip is both fail-open and cross-document.

| Severity | Count |
|----------|------:|
| CRITICAL | 3 |
| HIGH | 5 |
| MEDIUM | 7 |
| LOW | 2 |
| NIT | 1 |

---

## Findings

### CRITICAL

#### F1 — Celery path: Chroma add + Neo4j retry creates stuck partial writes
- **Area:** Reliability
- **Where:** `backend/workers/tasks.py:60-66,189-235` · `backend/services/chroma_service.py:69-74`
- **Risk:** Chroma uses `collection.add` (not upsert). If Chroma succeeds and Neo4j fails, retry re-adds the same chunk IDs, `add` fails, and the task retries until `max_retries`. The upload file is kept, but the document never completes and the worker burns retries.
- **Fix:** Use upsert/update-by-id; make persist transactional across stores (or compensate); treat duplicate-ID as success when content matches; delete/quarantine the file only after durable commit of both stores.

#### F2 — Content-hash skip is global and fail-open
- **Area:** Idempotency
- **Where:** `backend/graph_ingestion/ingest.py:77-92,146-148`
- **Risk:** `is_chunk_ingested` returns `False` on Neo4j errors → duplicate ingest under outage. Hash is text-only (no `document_id`), so identical prose in Doc B is skipped because Doc A already has it — silent evidence loss across documents.
- **Fix:** Fail closed on check errors (abort or retry). Scope uniqueness as `(document_id, chunk_hash)` or use stable `chunk_id`. Batch existence checks instead of N+1.

#### F3 — Directory ingest can stamp one `document_id` on every file
- **Area:** Correctness
- **Where:** `backend/graph_ingestion/ingest.py:296-306`
- **Risk:** `metadata.document_id` is reused for all files in the directory. Stable chunk IDs collide across files; provenance and `section_path` attach to the wrong document.
- **Fix:** Always derive per-file `document_id` (path/content hash). Treat `metadata.document_id` as override only for single-file ingest.

### HIGH

#### F4 — ADR claims Celery has hash idempotency + validation; code does not
- **Area:** Contract
- **Where:** `docs/architecture/ADR-002-Adaptive-Chunking.md:39-41` · `backend/workers/tasks.py:164-230` · `backend/services/neo4j_service.py:190-200`
- **Risk:** Celery path never checks `chunk_hash`, never emits `ingestion_validation`, and Neo4j `MERGE` does not `SET chunk_hash` — so `IdempotentGraphIngester` cannot see Celery-written chunks. Two pipelines, incompatible contracts.
- **Fix:** Persist `chunk_hash` + ADR metadata on Celery Chunk nodes; add pre-write idempotency or upsert; return `build_validation_report`; or route API uploads through `IdempotentGraphIngester`.

#### F5 — Policy version label disagrees with persisted metadata
- **Area:** Contract
- **Where:** `backend/graph_ingestion/adaptive_chunking.py:45,548-569` · `backend/graph_ingestion/ingest.py:213-215` · `backend/workers/tasks.py:223-227`
- **Risk:** `recursive_only` still stamps `policy_version=kinegraph.adaptive-chunking.v1` on every `ChunkRecord`, while API/task summaries say `legacy.recursive`. Consumers cannot trust the flag for migrations or A/B retrieval.
- **Fix:** Set `policy_version` from active policy (e.g. `kinegraph.recursive.v1` vs adaptive v1). Keep summary field identical to metadata.

#### F6 — Tables and long sentences never oversized-split
- **Area:** Correctness
- **Where:** `backend/graph_ingestion/adaptive_chunking.py:328-369,458-481`
- **Risk:** Table blocks and single sentences longer than `chunk_size` become one chunk. Huge LiteParse tables can blow embedding token limits, Chroma metadata, and retrieval context.
- **Fix:** Apply oversized fallback to table bodies (preserve headers on each slice). After semantic packing, recursively split any piece still > `chunk_size`.

#### F7 — `CHUNK_OVERLAP` can be >= `CHUNK_SIZE`
- **Area:** Config
- **Where:** `backend/core/config.py:79-80`
- **Risk:** No `model_validator`. `RecursiveCharacterTextSplitter` with overlap >= size can loop or emit pathological chunks in production if env is mis-set.
- **Fix:** Add validator: overlap < size; also set a sane upper bound on `CHUNK_SIZE` (e.g. `le=8000`).

#### F8 — Per-chunk Neo4j existence queries (N+1)
- **Area:** Performance
- **Where:** `backend/graph_ingestion/ingest.py:133-162,77-89`
- **Risk:** Large docs / directory batches do one `session.run` per chunk before write. Latency and Neo4j load scale linearly with chunk count.
- **Fix:** `UNWIND` batch of hashes; return existing set; filter in process.

### MEDIUM

#### F9 — `document_id` tied to ephemeral upload path
- **Area:** Reliability
- **Where:** `backend/workers/document_processor.py:189-200` · `backend/workers/tasks.py:162` · `backend/graph_ingestion/ingest.py:64-75`
- **Risk:** API uses UUID filename; `generate_document_id` hashes that path. Re-upload of the same PDF → new `doc_id` → new `chunk_ids` → full duplicate corpus. Stable IDs only help within one path.
- **Fix:** Hash content (or content + `original_file_name` + tenant) for `document_id`; keep upload path ephemeral.

#### F10 — Image caption heuristic steals following prose
- **Area:** Correctness
- **Where:** `backend/graph_ingestion/adaptive_chunking.py:262-267`
- **Risk:** Any non-heading next line under 240 chars becomes a caption. Normal paragraphs after images are attached to image chunks and omitted from structural sections.
- **Fix:** Only accept explicit caption prefixes (`Figure` / `Fig.` / `Caption`) or blank-line+italic patterns; never length-only.

#### F11 — No structured ingest metrics / correlation fields
- **Area:** Observability
- **Where:** `backend/workers/tasks.py:169-176` · `backend/graph_ingestion/ingest.py:219-225`
- **Risk:** Logs are free-text. No counters for `chunk_type`, skipped vs ingested, incomplete enrichment, policy version, latency, or `doc_id` correlation across Celery/API.
- **Fix:** Emit structured extras (`task_id`, `document_id`, policy, counts, bytes, `duration_ms`). Prometheus/OTel counters for incomplete/failed enrichment and retry reasons.

#### F12 — Whole-document embed batch with no backpressure
- **Area:** Performance
- **Where:** `backend/workers/tasks.py:60-64` · `backend/services/chroma_service.py:65-66` · `ingest_directory` `all_nodes`
- **Risk:** Huge PDFs load fully, chunk fully, `embed_documents(all)` in one call. Memory spikes; worker OOM risk; directory batch compounds.
- **Fix:** Cap max chars / max chunks; embed in batches; stream directory files or chunk-batch `PropertyGraphIndex` writes.

#### F13 — `coordinates_json` is Python repr, not JSON
- **Area:** API contract
- **Where:** `backend/graph_ingestion/adaptive_chunking.py:86-87`
- **Risk:** `str(dict)` is not `json.loads`-safe and breaks non-Python consumers reading Chroma metadata.
- **Fix:** `json.dumps(coordinates, separators=(',', ':'), sort_keys=True)`.

#### F14 — Untrusted document text injected into LLM prompts
- **Area:** Security
- **Where:** `backend/workers/document_processor.py:133-162` · `backend/workers/tasks.py:95`
- **Risk:** PDF/Markdown content is interpolated into the extraction prompt. Prompt-injection can invent entities/relationships that pollute the graph.
- **Fix:** Delimiter-fence content; instruct model to treat as data; schema-validate JSON; rate-limit; never trust entity names without allowlists for critical edges.

#### F15 — UTF-8 decode can crash non-UTF text files
- **Area:** Correctness
- **Where:** `backend/graph_ingestion/ingest.py:182-183,290-291`
- **Risk:** `open(..., encoding='utf-8')` with no `errors=` raises `UnicodeDecodeError` mid-directory batch after other files already staged in memory.
- **Fix:** `errors='replace'` or detect encoding; isolate per-file failures in `ingest_directory`.

### LOW

#### F16 — Mutable default `metadata=None` and weak Dict typing
- **Area:** Type safety
- **Where:** `backend/graph_ingestion/ingest.py:164,260`
- **Risk:** Pre-existing Optional default pattern; return payloads are untyped `Dict[str, Any]` so incomplete vs success is easy to ignore at call sites.
- **Fix:** TypedDict / Pydantic models for ingest results; `metadata: dict | None = None` is fine on 3.12.

#### F17 — Duplicate chunking entry points still diverge
- **Area:** Python 3.12+
- **Where:** `backend/workers/document_processor.py:54-72` · `backend/graph_ingestion/ingest.py:94-103` · `adaptive_chunking.py`
- **Risk:** `chunk_text` / `IdempotentGraphIngester.chunk_text` remain parallel recursive splitters. Drift risk vs `chunk_document`.
- **Fix:** Deprecate string-only helpers; route all paths through `chunk_document_text`.

### NIT

#### F18 — Feature flags good; no staged rollout knobs
- **Area:** Ops
- **Where:** `backend/core/config.py:77-78` · `.env.example`
- **Risk:** Boolean flags only. Cannot canary by tenant/% traffic or freeze policy version per corpus.
- **Fix:** Optional `policy_version` pin + sample rate once benchmarks land.

---

## Prioritized action list

1. Fix Celery Chroma upsert + cross-store commit/compensate (F1)
2. Fail-closed, document-scoped existence checks; batch Neo4j (F2, F8)
3. Per-file `document_id` in `ingest_directory` (F3)
4. Unify Celery ↔ `IdempotentGraphIngester` contract + `chunk_hash` (F4, F5)
5. Oversized table/sentence split + overlap validator (F6, F7)
6. Content-based `document_id` + structured ingest metrics (F9, F11)
7. Tighten caption heuristic, JSON coordinates, encoding (F10, F13, F15)
8. Add integration tests: retry/partial write, dual-path, huge table

---

## Quick wins vs deeper work

### Quick wins (<1 day)

- **F3:** per-file `document_id` in `ingest_directory`
- **F5:** stamp real `policy_version` on records
- **F7:** pydantic validator `overlap < size`
- **F10:** caption prefix-only heuristic
- **F13:** `json.dumps` for coordinates
- **F15:** encoding `errors=replace` + per-file try/except
- **F2 (partial):** fail closed when Neo4j check errors

### Deeper work (multi-day)

- **F1 / F4:** unified durable ingest + upsert semantics
- **F2 / F8:** batched, document-scoped idempotency
- **F6:** table/sentence oversized policy
- **F9 / F12:** content doc IDs + embed batching/limits
- **F11:** metrics + tracing for ingest pipeline
- **F14:** hardened extraction prompt + schema validation
- Integration tests across Celery retry & dual path

---

## Testing gaps

| Gap | Why it matters | Suggested test |
|-----|----------------|----------------|
| Celery partial failure | Chroma OK / Neo4j fail wedges retries | Mock add success then graph fail; assert upsert/idempotent second run |
| Fail-open Neo4j check | Duplicates during outages | `is_chunk_ingested` raises → ingest aborts or retries, no write |
| Cross-document same text | Silent skip of valid evidence | Two docs share paragraph; both persist under own `document_id` |
| Directory `document_id` override | ID collisions across files | `ingest_directory` with `metadata.document_id` → unique per file |
| Huge table / long sentence | Embedding / context blowups | `chunk_size=200`, 5k-char table → multiple table/recursive slices |
| `overlap >= size` | Pathological splitter | Settings validation rejects `CHUNK_OVERLAP=1000`, `CHUNK_SIZE=1000` |
| Policy metadata when flag off | Migration/A-B confusion | `ADAPTIVE=false` → metadata `policy_version` is legacy, not adaptive v1 |
| Caption theft | Lost prose after images | Image followed by normal paragraph stays structural, not caption |

---

## Deep-dive notes

### Dual ingest pipelines (CRITICAL)

- **API → Celery `process_document`** writes Chroma then Neo4j without `chunk_hash` idempotency or `ingestion_validation`.
- **Scripts / `IdempotentGraphIngester`** use `PropertyGraphIndex` + content-hash skip + validation report.
- ADR-002 Implementation bullets claim both paths share the contract; only the graph-ingester path does. Cross-path reingest and enrichment verification will disagree in production.

### Idempotency semantics (CRITICAL)

- Skip key is SHA-256(text) only. Same boilerplate across docs → second doc silently drops chunks.
- On Neo4j check failure, code returns `False` and proceeds to write — classic fail-open duplicate amplifier during incidents.
- Prefer stable `chunk_id` existence (includes `document_id` + policy) or composite `(document_id, chunk_hash)`, and abort/retry when the graph is unreachable.

### Chunk boundary edge cases (HIGH)

- Empty/whitespace docs return `[]` / skipped — OK.
- Tables never go through `_split_oversized`; semantic mode can emit pieces larger than `chunk_size` when a sentence is huge.
- Image next-line heuristic (<240 chars) can absorb real paragraphs into image chunks.
- Code fences and lists are treated as structure triggers even without headings — good — but unstructured PDF plain text from PyMuPDF fallback stays `recursive_only` effectively.

### Operability & rollout (HIGH)

- Defaults off (`ADAPTIVE_CHUNKING_ENABLED=false`) — correct safe default. Semantic double-gated.
- Missing: `overlap < size` validator, `CHUNK_SIZE` upper bound, max document bytes, per-policy version pin for migrations.
- Keep flag off until frozen-corpus acceptance (ADR) plus the dual-path contract fixes land. Then canary one corpus, compare Recall@K / storage / incomplete enrichment rate.

### Stack alignment

- FastAPI upload path hardening (UUID store names, basename validation) is solid and should stay. Adaptive chunking itself is clean Python 3.12 (PEP 604 unions, frozen dataclasses, Literal types).
- Celery sync task + `asyncio.run` for persist matches existing project pattern; the production gap is store consistency, not the event-loop choice. Prefer aligning with existing `ChromaService` / `Neo4jService` rather than introducing a new orchestration framework.

---

*Source: git working tree diff for ADR-002 files · not a live runtime profile. Severity tags: CRITICAL / HIGH / MEDIUM / LOW / NIT.*
