import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  TodoListCard,
  useHostTheme,
} from "cursor/canvas";

type Severity = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "NIT";

type Finding = {
  id: string;
  severity: Severity;
  area: string;
  title: string;
  where: string;
  risk: string;
  fix: string;
};

const FINDINGS: Finding[] = [
  {
    id: "F1",
    severity: "CRITICAL",
    area: "Reliability",
    title: "Celery path: Chroma add + Neo4j retry creates stuck partial writes",
    where: "backend/workers/tasks.py:60-66,189-235 · backend/services/chroma_service.py:69-74",
    risk: "Chroma uses collection.add (not upsert). If Chroma succeeds and Neo4j fails, retry re-adds the same chunk IDs, add fails, task retries until max_retries. Upload file is kept, but document never completes and worker burns retries.",
    fix: "Use upsert/update-by-id; make persist transactional across stores (or compensate); treat duplicate-ID as success when content matches; delete/quarantine file only after durable commit of both stores.",
  },
  {
    id: "F2",
    severity: "CRITICAL",
    area: "Idempotency",
    title: "Content-hash skip is global and fail-open",
    where: "backend/graph_ingestion/ingest.py:77-92,146-148",
    risk: "is_chunk_ingested returns False on Neo4j errors → duplicate ingest under outage. Hash is text-only (no document_id), so identical prose in Doc B is skipped because Doc A already has it — silent evidence loss across documents.",
    fix: "Fail closed on check errors (abort or retry). Scope uniqueness as (document_id, chunk_hash) or use stable chunk_id. Batch existence checks instead of N+1.",
  },
  {
    id: "F3",
    severity: "CRITICAL",
    area: "Correctness",
    title: "Directory ingest can stamp one document_id on every file",
    where: "backend/graph_ingestion/ingest.py:296-306",
    risk: "metadata.document_id is reused for all files in the directory. Stable chunk IDs collide across files; provenance and section_path attach to the wrong document.",
    fix: "Always derive per-file document_id (path/content hash). Treat metadata.document_id as override only for single-file ingest.",
  },
  {
    id: "F4",
    severity: "HIGH",
    area: "Contract",
    title: "ADR claims Celery has hash idempotency + validation; code does not",
    where: "docs/.../ADR-002:39-41 · tasks.py:164-230 · neo4j_service.py:190-200",
    risk: "Celery path never checks chunk_hash, never emits ingestion_validation, and Neo4j MERGE does not SET chunk_hash — so IdempotentGraphIngester cannot see Celery-written chunks. Two pipelines, incompatible contracts.",
    fix: "Persist chunk_hash + ADR metadata on Celery Chunk nodes; add pre-write idempotency or upsert; return build_validation_report; or route API uploads through IdempotentGraphIngester.",
  },
  {
    id: "F5",
    severity: "HIGH",
    area: "Contract",
    title: "Policy version label disagrees with persisted metadata",
    where: "adaptive_chunking.py:45,548-569 · ingest.py:213-215 · tasks.py:223-227",
    risk: "recursive_only still stamps policy_version=kinegraph.adaptive-chunking.v1 on every ChunkRecord, while API/task summaries say legacy.recursive. Consumers cannot trust the flag for migrations or A/B retrieval.",
    fix: "Set policy_version from active policy (e.g. kinegraph.recursive.v1 vs adaptive v1). Keep summary field identical to metadata.",
  },
  {
    id: "F6",
    severity: "HIGH",
    area: "Correctness",
    title: "Tables and long sentences never oversized-split",
    where: "adaptive_chunking.py:328-369,458-481",
    risk: "Table blocks and single sentences longer than chunk_size become one chunk. Huge LiteParse tables can blow embedding token limits, Chroma metadata, and retrieval context.",
    fix: "Apply oversized fallback to table bodies (preserve headers on each slice). After semantic packing, recursively split any piece still > chunk_size.",
  },
  {
    id: "F7",
    severity: "HIGH",
    area: "Config",
    title: "CHUNK_OVERLAP can be >= CHUNK_SIZE",
    where: "backend/core/config.py:79-80",
    risk: "No model_validator. RecursiveCharacterTextSplitter with overlap >= size can loop or emit pathological chunks in production if env is mis-set.",
    fix: "Add validator: overlap < size; also set sane upper bound on CHUNK_SIZE (e.g. le=8000).",
  },
  {
    id: "F8",
    severity: "HIGH",
    area: "Performance",
    title: "Per-chunk Neo4j existence queries (N+1)",
    where: "ingest.py:133-162,77-89",
    risk: "Large docs / directory batches do one session.run per chunk before write. Latency and Neo4j load scale linearly with chunk count.",
    fix: "UNWIND batch of hashes; return existing set; filter in process.",
  },
  {
    id: "F9",
    severity: "MEDIUM",
    area: "Reliability",
    title: "document_id tied to ephemeral upload path",
    where: "document_processor.py:189-200 · tasks.py:162 · ingest.py:64-75",
    risk: "API uses UUID filename; generate_document_id hashes that path. Re-upload of same PDF → new doc_id → new chunk_ids → full duplicate corpus. Stable IDs only help within one path.",
    fix: "Hash content (or content + original_file_name + tenant) for document_id; keep upload path ephemeral.",
  },
  {
    id: "F10",
    severity: "MEDIUM",
    area: "Correctness",
    title: "Image caption heuristic steals following prose",
    where: "adaptive_chunking.py:262-267",
    risk: "Any non-heading next line under 240 chars becomes caption. Normal paragraphs after images are attached to image chunks and omitted from structural sections.",
    fix: "Only accept explicit caption prefixes (Figure/Fig./Caption) or blank-line+italic patterns; never length-only.",
  },
  {
    id: "F11",
    severity: "MEDIUM",
    area: "Observability",
    title: "No structured ingest metrics / correlation fields",
    where: "tasks.py:169-176 · ingest.py:219-225",
    risk: "Logs are free-text. No counters for chunk_type, skipped vs ingested, incomplete enrichment, policy version, latency, or doc_id correlation across Celery/API.",
    fix: "Emit structured extras (task_id, document_id, policy, counts, bytes, duration_ms). Prometheus/OTel counters for incomplete/failed enrichment and retry reasons.",
  },
  {
    id: "F12",
    severity: "MEDIUM",
    area: "Performance",
    title: "Whole-document embed batch with no backpressure",
    where: "tasks.py:60-64 · chroma_service.py:65-66 · ingest_directory all_nodes",
    risk: "Huge PDFs load fully, chunk fully, embed_documents(all) in one call. Memory spikes; worker OOM risk; directory batch compounds.",
    fix: "Cap max chars / max chunks; embed in batches; stream directory files or chunk-batch PropertyGraphIndex writes.",
  },
  {
    id: "F13",
    severity: "MEDIUM",
    area: "API contract",
    title: "coordinates_json is Python repr, not JSON",
    where: "adaptive_chunking.py:86-87",
    risk: "str(dict) is not json.loads-safe and breaks non-Python consumers reading Chroma metadata.",
    fix: "json.dumps(coordinates, separators=(',', ':'), sort_keys=True).",
  },
  {
    id: "F14",
    severity: "MEDIUM",
    area: "Security",
    title: "Untrusted document text injected into LLM prompts",
    where: "document_processor.py:133-162 · tasks.py:95",
    risk: "PDF/Markdown content is interpolated into the extraction prompt. Prompt-injection can invent entities/relationships that pollute the graph.",
    fix: "Delimiter-fence content; instruct model to treat as data; schema-validate JSON; rate-limit; never trust entity names without allowlists for critical edges.",
  },
  {
    id: "F15",
    severity: "MEDIUM",
    area: "Correctness",
    title: "UTF-8 decode can crash non-UTF text files",
    where: "ingest.py:182-183,290-291",
    risk: "open(..., encoding='utf-8') with no errors= raises UnicodeDecodeError mid-directory batch after other files already staged in memory.",
    fix: "errors='replace' or detect encoding; isolate per-file failures in ingest_directory.",
  },
  {
    id: "F16",
    severity: "LOW",
    area: "Type safety",
    title: "Mutable default metadata=None and weak Dict typing",
    where: "ingest.py:164,260",
    risk: "Pre-existing Optional default pattern; return payloads are untyped Dict[str, Any] so incomplete vs success is easy to ignore at call sites.",
    fix: "TypedDict / Pydantic models for ingest results; metadata: dict | None = None is fine on 3.12.",
  },
  {
    id: "F17",
    severity: "LOW",
    area: "Python 3.12+",
    title: "Duplicate chunking entry points still diverge",
    where: "document_processor.py:54-72 · ingest.py:94-103 · adaptive_chunking.py",
    risk: "chunk_text / IdempotentGraphIngester.chunk_text remain parallel recursive splitters. Drift risk vs chunk_document.",
    fix: "Deprecate string-only helpers; route all paths through chunk_document_text.",
  },
  {
    id: "F18",
    severity: "NIT",
    area: "Ops",
    title: "Feature flags good; no staged rollout knobs",
    where: "config.py:77-78 · .env.example",
    risk: "Boolean flags only. Cannot canary by tenant/% traffic or freeze policy version per corpus.",
    fix: "Optional policy_version pin + sample rate once benchmarks land.",
  },
];

function countBySeverity(severity: Severity): number {
  return FINDINGS.filter((f) => f.severity === severity).length;
}

function toneForSeverity(
  severity: Severity,
): "danger" | "warning" | "info" | "success" | undefined {
  switch (severity) {
    case "CRITICAL":
      return "danger";
    case "HIGH":
      return "warning";
    case "MEDIUM":
      return "info";
    default:
      return undefined;
  }
}

function rowTone(
  severity: Severity,
): "danger" | "warning" | "info" | "neutral" | undefined {
  switch (severity) {
    case "CRITICAL":
      return "danger";
    case "HIGH":
      return "warning";
    case "MEDIUM":
      return "info";
    default:
      return "neutral";
  }
}

export default function AdaptiveChunkingProductionReview() {
  const theme = useHostTheme();

  return (
    <Stack gap={20} style={{ padding: 20 }}>
      <Stack gap={8}>
        <H1>Adaptive chunking — production readiness</H1>
        <Text tone="secondary">
          Scope: uncommitted ADR-002 work — adaptive_chunking.py, ingest.py,
          document_processor.py, tasks.py, config, scripts, tests. Review date
          2026-07-25.
        </Text>
        <Row gap={8} wrap>
          <Pill active>Verdict: not prod-ready as default</Pill>
          <Pill>Flag-off OK for experiments</Pill>
          <Pill>18 findings</Pill>
        </Row>
      </Stack>

      <Callout tone="warning" title="Overall verdict">
        Safe to keep ADAPTIVE_CHUNKING_ENABLED=false for experimentation, but
        do not promote or treat ingestion as production-hardened yet. Dual
        pipelines disagree on idempotency/metadata, Celery partial-write retries
        can wedge, and content-hash skip is both fail-open and cross-document.
      </Callout>

      <Grid columns={5} gap={12}>
        <Stat
          label="CRITICAL"
          value={String(countBySeverity("CRITICAL"))}
          tone={toneForSeverity("CRITICAL")}
        />
        <Stat
          label="HIGH"
          value={String(countBySeverity("HIGH"))}
          tone={toneForSeverity("HIGH")}
        />
        <Stat
          label="MEDIUM"
          value={String(countBySeverity("MEDIUM"))}
          tone={toneForSeverity("MEDIUM")}
        />
        <Stat label="LOW" value={String(countBySeverity("LOW"))} />
        <Stat label="NIT" value={String(countBySeverity("NIT"))} />
      </Grid>

      <H2>Prioritized action list</H2>
      <TodoListCard
        defaultExpanded
        todos={[
          {
            id: "a1",
            content:
              "Fix Celery Chroma upsert + cross-store commit/compensate (F1)",
            status: "pending",
          },
          {
            id: "a2",
            content:
              "Fail-closed, document-scoped existence checks; batch Neo4j (F2, F8)",
            status: "pending",
          },
          {
            id: "a3",
            content: "Per-file document_id in ingest_directory (F3)",
            status: "pending",
          },
          {
            id: "a4",
            content:
              "Unify Celery ↔ IdempotentGraphIngester contract + chunk_hash (F4, F5)",
            status: "pending",
          },
          {
            id: "a5",
            content: "Oversized table/sentence split + overlap validator (F6, F7)",
            status: "pending",
          },
          {
            id: "a6",
            content:
              "Content-based document_id + structured ingest metrics (F9, F11)",
            status: "pending",
          },
          {
            id: "a7",
            content:
              "Tighten caption heuristic, JSON coordinates, encoding (F10, F13, F15)",
            status: "pending",
          },
          {
            id: "a8",
            content:
              "Add integration tests: retry/partial write, dual-path, huge table",
            status: "pending",
          },
        ]}
      />

      <H2>Findings by severity</H2>
      <Table
        headers={["Sev", "ID", "Area", "Finding", "Location", "Fix"]}
        columnAlign={["left", "left", "left", "left", "left", "left"]}
        rowTone={FINDINGS.map((f) => rowTone(f.severity))}
        rows={FINDINGS.map((f) => [
          f.severity,
          f.id,
          f.area,
          f.title,
          f.where,
          f.fix,
        ])}
        striped
        stickyHeader
      />

      <H2>Deep dives</H2>
      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader trailing={<Pill active>CRITICAL</Pill>}>
            Dual ingest pipelines
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                API → Celery process_document writes Chroma then Neo4j without
                chunk_hash idempotency or ingestion_validation.
              </Text>
              <Text>
                Scripts / IdempotentGraphIngester use PropertyGraphIndex +
                content-hash skip + validation report.
              </Text>
              <Text tone="secondary" size="small">
                ADR-002 Implementation bullets claim both paths share the
                contract; only the graph-ingester path does. Cross-path reingest
                and enrichment verification will disagree in production.
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill active>CRITICAL</Pill>}>
            Idempotency semantics
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                Skip key is SHA-256(text) only. Same boilerplate across docs →
                second doc silently drops chunks.
              </Text>
              <Text>
                On Neo4j check failure, code returns False and proceeds to
                write — classic fail-open duplicate amplifier during incidents.
              </Text>
              <Text tone="secondary" size="small">
                Prefer stable chunk_id existence (includes document_id + policy)
                or composite (document_id, chunk_hash), and abort/retry when the
                graph is unreachable.
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill>HIGH</Pill>}>
            Chunk boundary edge cases
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                Empty/whitespace docs return [] / skipped — OK.
              </Text>
              <Text>
                Tables never go through _split_oversized; semantic mode can
                emit pieces larger than chunk_size when a sentence is huge.
              </Text>
              <Text>
                Image next-line heuristic (&lt;240 chars) can absorb real
                paragraphs into image chunks.
              </Text>
              <Text tone="secondary" size="small">
                Code fences and lists are treated as structure triggers even
                without headings — good — but unstructured PDF plain text from
                PyMuPDF fallback stays recursive_only effectively.
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill>HIGH</Pill>}>
            Operability & rollout
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                Defaults off (ADAPTIVE_CHUNKING_ENABLED=false) — correct safe
                default. Semantic double-gated.
              </Text>
              <Text>
                Missing: overlap&lt;size validator, CHUNK_SIZE upper bound,
                max document bytes, per-policy version pin for migrations.
              </Text>
              <Text tone="secondary" size="small">
                Keep flag off until frozen-corpus acceptance (ADR) plus the
                dual-path contract fixes land. Then canary one corpus, compare
                Recall@K / storage / incomplete enrichment rate.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H2>Quick wins vs deeper work</H2>
      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader>Quick wins (&lt;1 day)</CardHeader>
          <CardBody>
            <Stack gap={6}>
              <Text>F3: per-file document_id in ingest_directory</Text>
              <Text>F5: stamp real policy_version on records</Text>
              <Text>F7: pydantic validator overlap &lt; size</Text>
              <Text>F10: caption prefix-only heuristic</Text>
              <Text>F13: json.dumps for coordinates</Text>
              <Text>F15: encoding errors=replace + per-file try/except</Text>
              <Text>F2 partial: fail closed when Neo4j check errors</Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Deeper work (multi-day)</CardHeader>
          <CardBody>
            <Stack gap={6}>
              <Text>F1/F4: unified durable ingest + upsert semantics</Text>
              <Text>F2/F8: batched, document-scoped idempotency</Text>
              <Text>F6: table/sentence oversized policy</Text>
              <Text>F9/F12: content doc IDs + embed batching/limits</Text>
              <Text>F11: metrics + tracing for ingest pipeline</Text>
              <Text>F14: hardened extraction prompt + schema validation</Text>
              <Text>Integration tests across Celery retry & dual path</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H2>Testing gaps</H2>
      <Table
        headers={["Gap", "Why it matters", "Suggested test"]}
        rows={[
          [
            "Celery partial failure",
            "Chroma OK / Neo4j fail wedges retries",
            "Mock add success then graph fail; assert upsert/idempotent second run",
          ],
          [
            "Fail-open Neo4j check",
            "Duplicates during outages",
            "is_chunk_ingested raises → ingest aborts or retries, no write",
          ],
          [
            "Cross-document same text",
            "Silent skip of valid evidence",
            "Two docs share paragraph; both persist under own document_id",
          ],
          [
            "Directory document_id override",
            "ID collisions across files",
            "ingest_directory with metadata.document_id → unique per file",
          ],
          [
            "Huge table / long sentence",
            "Embedding / context blowups",
            "chunk_size=200, 5k-char table → multiple table/recursive slices",
          ],
          [
            "overlap >= size",
            "Pathological splitter",
            "Settings validation rejects CHUNK_OVERLAP=1000, CHUNK_SIZE=1000",
          ],
          [
            "Policy metadata when flag off",
            "Migration/A-B confusion",
            "ADAPTIVE=false → metadata policy_version is legacy, not adaptive v1",
          ],
          [
            "Caption theft",
            "Lost prose after images",
            "Image followed by normal paragraph stays structural, not caption",
          ],
        ]}
        striped
      />

      <H2>Stack alignment notes</H2>
      <Stack gap={8}>
        <Text>
          FastAPI upload path hardening (UUID store names, basename validation)
          is solid and should stay. Adaptive chunking itself is clean 3.12
          (PEP 604 unions, frozen dataclasses, Literal types).
        </Text>
        <Text>
          Celery sync task + asyncio.run for persist matches existing project
          pattern; the production gap is store consistency, not the event-loop
          choice. Prefer aligning with existing ChromaService/Neo4jService
          rather than introducing a new orchestration framework.
        </Text>
        <Text tone="secondary" size="small">
          Source: git working tree diff for ADR-002 files · not a live runtime
          profile.
        </Text>
      </Stack>

      <Divider />
      <Text tone="tertiary" size="small" style={{ color: theme.text.tertiary }}>
        Severity tags: CRITICAL / HIGH / MEDIUM / LOW / NIT · Canvas for
        parent-agent relay · No production code modified.
      </Text>
    </Stack>
  );
}
