const evidenceKeys = ["citation_validation", "grounded_claims", "grounding_critique"];

export function queryView(result, requestedMode) {
  const evidence = Object.fromEntries(
    evidenceKeys.filter((key) => result[key] != null).map((key) => [key, result[key]])
  );
  const warnings = [];
  if (result.effective_mode !== requestedMode) {
    warnings.push(`Requested ${requestedMode}, but the API used ${result.effective_mode ?? "an unknown route"}.`);
  }
  if (!Object.values(evidence).some((value) => Object.keys(value).length > 0)) {
    warnings.push("The response did not include citation or grounding evidence.");
  } else if (Array.isArray(result.grounded_claims) && !result.grounded_claims.some((claim) => claim?.chunk_ids?.length)) {
    warnings.push("The response did not return any grounded claims with citations.");
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
    sources: (result.results ?? []).map(({ source, metadata, explanation }) => ({
      source, metadata, explanation,
    })),
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
    if (url.username || url.password || url.search || url.hash) {
      throw new Error("Enter the API base URL without credentials, query parameters, or a fragment.");
    }
    return url.href.replace(/\/$/, "");
  }

  async function api(url, path, options = {}) {
    const response = await fetch(`${url}${path}`, {
      signal: AbortSignal.timeout(120000),
      ...options,
    });
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const value = (await response.json()).detail;
        if (value) detail += ` — ${typeof value === "string" ? value : JSON.stringify(value)}`;
      } catch { /* keep HTTP status */ }
      throw new Error(detail);
    }
    const result = await response.json();
    byId("connectionState").textContent = "API connected";
    return result;
  }

  async function pollTask(url, taskId) {
    const deadline = Date.now() + 300000;
    while (Date.now() < deadline) {
      const task = await api(url, `/api/v1/ingest/task/${encodeURIComponent(taskId)}`, {
        signal: AbortSignal.timeout(Math.max(1, deadline - Date.now())),
      });
      byId("ingestStatus").textContent = `Task ${taskId}: ${task.status}`;
      if (task.status === "SUCCESS") return task;
      if (["FAILURE", "REVOKED"].includes(task.status)) throw new Error(task.error || `Ingestion ended with ${task.status}.`);
      await delay(1000);
    }
    throw new Error("Ingestion did not finish within five minutes.");
  }

  async function ingest() {
    const url = baseUrl();
    const asset = await fetch(contract.ingestion_pdf);
    if (!asset.ok) throw new Error("Could not load the bundled synthetic handbook PDF.");
    const bytes = await asset.arrayBuffer();
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const hash = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
    if (hash !== contract.ingestion_pdf_sha256) throw new Error("The handbook PDF hash differs from the prompt contract.");
    const data = new FormData();
    data.append("file", new Blob([bytes], { type: "application/pdf" }), contract.ingestion_pdf.split("/").pop());
    data.append("metadata", JSON.stringify({
      source_corpus_sha256: contract.corpus_sha256,
      demo_pdf_sha256: hash,
    }));
    byId("ingestStatus").textContent = "Uploading…";
    const task = await api(url, "/api/v1/ingest/document", { method: "POST", body: data });
    if (!task.task_id) throw new Error("The API did not return an ingestion task ID.");
    try {
      await pollTask(url, task.task_id);
    } catch (error) {
      throw new Error(`Task ${task.task_id} — ${error.message}`, { cause: error });
    }
    byId("ingestStatus").textContent = `Handbook ingestion completed. Task ${task.task_id}.`;
  }

  function render(view) {
    byId("resultState").textContent = view.stateLabel;
    byId("resultState").className = `state state-${view.status}`;
    byId("resultMessage").textContent = view.message;
    const warning = view.status === "warning";
    byId("resultMessage").hidden = warning || !view.message;
    byId("warningPanel").hidden = !warning;
    byId("warningPanel").textContent = warning ? view.message : "";
    const success = view.status === "success" || view.status === "warning";
    byId("resultMeta").hidden = !success;
    byId("answerPanel").hidden = !success;
    byId("evidencePanel").hidden = !success;
    if (success) {
      byId("requestedMode").textContent = view.requestedMode;
      byId("effectiveMode").textContent = view.effectiveMode;
      byId("latency").textContent = view.latency;
      byId("answer").textContent = view.answer;
      byId("evidence").textContent = JSON.stringify({ ...view.evidence, sources: view.sources }, null, 2);
    }
  }

  async function query() {
    const selected = contract.cases.find((item) => item.id === byId("promptSelect").value);
    const mode = document.querySelector('input[name="mode"]:checked').value;
    render({ status: "idle", stateLabel: "Running", message: "Waiting for the API response…" });
    const result = await api(baseUrl(), "/api/v1/query/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: selected.question, mode, allow_mode_downgrade: false,
        filters: { demo_pdf_sha256: contract.ingestion_pdf_sha256 },
      }),
    });
    render(queryView(result, mode));
  }

  async function run(button, action) {
    button.disabled = true;
    try { await action(); } catch (error) {
      byId("connectionState").textContent = "Check the latest request";
      if (button.id === "ingestButton") byId("ingestStatus").textContent = `Ingestion failed. ${error.message}`;
      else render(errorView(error));
    }
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
    byId("handbookLink").href = contract.ingestion_pdf;
    byId("handbookLink").hidden = false;
    byId("ingestButton").addEventListener("click", () => run(byId("ingestButton"), ingest));
    byId("queryButton").addEventListener("click", () => run(byId("queryButton"), query));
    byId("ingestButton").disabled = false;
    byId("queryButton").disabled = false;
  }

  init().catch((error) => render(errorView(error)));
}
