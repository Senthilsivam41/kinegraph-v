const evidenceKeys = ["citation_validation", "grounded_claims", "grounding_critique"];

export function queryView(result, requestedMode) {
  const evidence = Object.fromEntries(
    evidenceKeys.filter((key) => result[key] != null).map((key) => [key, result[key]])
  );
  const warnings = [];
  if (result.effective_mode !== requestedMode) {
    warnings.push(`Requested ${requestedMode}, but the API used ${result.effective_mode ?? "an unknown route"}.`);
  }
  if (Object.keys(evidence).length === 0) {
    warnings.push("The response did not include citation or grounding evidence.");
  }
  const latency = result.execution_time_ms != null
    ? `${result.execution_time_ms} ms`
    : result.latency_breakdown ? JSON.stringify(result.latency_breakdown) : "Not returned";
  return {
    status: warnings.length ? "warning" : "success",
    stateLabel: warnings.length ? "Review needed" : "Evidence returned",
    message: warnings.join(" "),
    requestedMode,
    effectiveMode: result.effective_mode ?? "Not returned",
    latency,
    answer: result.generated_answer || "No generated answer returned.",
    evidence,
  };
}

export function errorView(error) {
  return {
    status: "error",
    stateLabel: "Request failed",
    message: error instanceof Error ? error.message : String(error),
  };
}

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

if (typeof document !== "undefined") {
  const byId = (id) => document.getElementById(id);
  let contract;

  function baseUrl() {
    const url = new URL(byId("apiUrl").value);
    if (!["http:", "https:"].includes(url.protocol)) throw new Error("API URL must use HTTP or HTTPS.");
    return url.href.replace(/\/$/, "");
  }

  async function api(path, options = {}) {
    const response = await fetch(`${baseUrl()}${path}`, options);
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try { detail = (await response.json()).detail || detail; } catch { /* keep HTTP status */ }
      throw new Error(detail);
    }
    byId("connectionState").textContent = "API connected";
    return response.json();
  }

  async function pollTask(taskId) {
    for (let attempt = 0; attempt < 120; attempt += 1) {
      const task = await api(`/api/v1/ingest/task/${encodeURIComponent(taskId)}`);
      byId("ingestStatus").textContent = `Task ${taskId}: ${task.status}`;
      if (task.status === "SUCCESS") return task;
      if (task.status === "FAILURE") throw new Error(task.error || "Ingestion failed.");
      await delay(1000);
    }
    throw new Error("Ingestion did not finish within two minutes.");
  }

  async function ingest() {
    const file = byId("pdfFile").files[0];
    if (!file || !file.name.toLowerCase().endsWith(".pdf")) throw new Error("Choose the synthetic handbook PDF.");
    const data = new FormData();
    data.append("file", file);
    data.append("metadata", JSON.stringify({ source_corpus_sha256: contract.corpus_sha256 }));
    byId("ingestStatus").textContent = "Uploading…";
    const task = await api("/api/v1/ingest/document", { method: "POST", body: data });
    await pollTask(task.task_id);
    byId("ingestStatus").textContent = "Handbook ingestion completed.";
  }

  function render(view) {
    byId("resultState").textContent = view.stateLabel;
    byId("resultState").className = `state state-${view.status}`;
    byId("resultMessage").textContent = view.message;
    byId("warningPanel").hidden = !view.message || view.status === "error";
    byId("warningPanel").textContent = view.status === "error" ? "" : view.message;
    const success = view.status !== "error";
    byId("resultMeta").hidden = !success;
    byId("answerPanel").hidden = !success;
    byId("evidencePanel").hidden = !success;
    if (success) {
      byId("requestedMode").textContent = view.requestedMode;
      byId("effectiveMode").textContent = view.effectiveMode;
      byId("latency").textContent = view.latency;
      byId("answer").textContent = view.answer;
      byId("evidence").textContent = JSON.stringify(view.evidence, null, 2);
    }
  }

  async function query() {
    const selected = contract.cases.find((item) => item.id === byId("promptSelect").value);
    const mode = document.querySelector('input[name="mode"]:checked').value;
    const result = await api("/api/v1/query/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: selected.question, mode, allow_mode_downgrade: false }),
    });
    render(queryView(result, mode));
  }

  async function run(button, action) {
    button.disabled = true;
    try { await action(); } catch (error) { render(errorView(error)); }
    finally { button.disabled = false; }
  }

  async function init() {
    contract = await fetch("../contract/prompts.json").then((response) => {
      if (!response.ok) throw new Error("Could not load the HR prompt contract.");
      return response.json();
    });
    for (const item of contract.cases) {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = `${item.id} — ${item.question}`;
      byId("promptSelect").append(option);
    }
    byId("ingestButton").addEventListener("click", () => run(byId("ingestButton"), ingest));
    byId("queryButton").addEventListener("click", () => run(byId("queryButton"), query));
  }

  init().catch((error) => render(errorView(error)));
}
