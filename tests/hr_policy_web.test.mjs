import assert from "node:assert/strict";
import test from "node:test";

import { errorView, queryView } from "../use-cases/hr-policy-assistant/web/app.mjs";


test("successful response exposes answer and evidence", () => {
  const view = queryView({
    effective_mode: "hybrid",
    generated_answer: "Both department heads approve.",
    execution_time_ms: 42,
    citation_validation: { valid: true },
    results: [{ source: "graph", metadata: { file_name: "handbook.pdf" } }],
  }, "hybrid");

  assert.equal(view.status, "success");
  assert.equal(view.answer, "Both department heads approve.");
  assert.deepEqual(view.evidence, { citation_validation: { valid: true } });
  assert.equal(view.sources[0].metadata.file_name, "handbook.pdf");
});

test("missing evidence and route downgrade require review", () => {
  const view = queryView({
    effective_mode: "vector",
    generated_answer: "Answer",
    citation_validation: null,
    grounded_claims: [],
    grounding_critique: {},
  }, "hybrid");

  assert.equal(view.status, "warning");
  assert.match(view.message, /API used vector/);
  assert.match(view.message, /did not include citation/);
});

test("validation diagnostics without cited claims require review", () => {
  const view = queryView({
    effective_mode: "hybrid",
    grounded_claims: [],
    citation_validation: { structured_output_valid: false },
  }, "hybrid");

  assert.equal(view.status, "warning");
  assert.match(view.message, /grounded claims with citations/);
});

test("API failures have an explicit error state", () => {
  assert.deepEqual(errorView(new Error("Service unavailable")), {
    status: "error",
    stateLabel: "Request failed",
    message: "Service unavailable",
  });
});
