import assert from "node:assert/strict";
import test from "node:test";

import { errorView, queryView } from "../use-cases/hr-policy-assistant/web/app.mjs";


test("successful response exposes answer and evidence", () => {
  const view = queryView({
    effective_mode: "hybrid",
    generated_answer: "Both department heads approve.",
    execution_time_ms: 42,
    citation_validation: { valid: true },
  }, "hybrid");

  assert.equal(view.status, "success");
  assert.equal(view.answer, "Both department heads approve.");
  assert.deepEqual(view.evidence, { citation_validation: { valid: true } });
});

test("missing evidence and route downgrade require review", () => {
  const view = queryView({ effective_mode: "vector", generated_answer: "Answer" }, "hybrid");

  assert.equal(view.status, "warning");
  assert.match(view.message, /API used vector/);
  assert.match(view.message, /did not include citation/);
});

test("API failures have an explicit error state", () => {
  assert.deepEqual(errorView(new Error("Service unavailable")), {
    status: "error",
    stateLabel: "Request failed",
    message: "Service unavailable",
  });
});
