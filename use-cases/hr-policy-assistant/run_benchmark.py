"""Run the HR feature benchmark and record paired retrieval evidence."""

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

import client
from run_comparison import graph_checks


CONTRACT_PATH = client.ROOT / "contract" / "benchmark.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text())
HANDBOOK = (client.ROOT / "contract" / client.CONTRACT["corpus"]).resolve()
PDF = (client.ROOT / "contract" / client.CONTRACT["ingestion_pdf"]).resolve()
FIXED_OPTIONS = {
    "max_results": 3,
    "candidate_pool_size": 10,
    "max_hops": 2,
    "traversal_strategy": "bfs",
    "allow_mode_downgrade": False,
    "allow_vectorless_auto_route": False,
    "enable_adaptive_routing": False,
    "enable_conservative_routing": False,
    "enable_lexical_fusion": False,
    "enable_conditional_recovery": False,
    "enable_hyde_fallback": False,
    "enable_cross_encoder_reranking": False,
    "enable_retrieval_orchestration": False,
    "context_max_per_source": 2,
    "context_max_per_community": 0,
    "enable_corrective_rag": False,
    "enable_grounding_critique": True,
    "enable_verification_framework": False,
    "vector_fusion_weight": 1.0,
    "graph_fusion_weight": 1.0,
    "lexical_fusion_weight": 1.0,
}
# Each profile records its requested route, comparison baseline, and changed options.
PROFILES = {
    "vector": ("vector", None, {}),
    "hybrid": ("hybrid", "vector", {}),
    "graph": ("graph", "vector", {}),
    "graph_one_hop": ("graph", "graph", {"max_hops": 1}),
    "graph_dfs": ("graph", "graph", {"traversal_strategy": "dfs"}),
    "graph_community": ("graph", "graph", {"traversal_strategy": "community"}),
    "hybrid_lexical": ("hybrid", "hybrid", {"enable_lexical_fusion": True}),
    "vectorless": ("vectorless", "vector", {}),
    "vectorless_attachment": ("vectorless", None, {}),
    "adaptive": ("hybrid", "hybrid", {"enable_adaptive_routing": True, "allow_mode_downgrade": True}),
    "attachment_auto": ("hybrid", "vectorless_attachment", {"allow_vectorless_auto_route": True}),
    "recovery": ("hybrid", "hybrid", {"enable_conditional_recovery": True}),
    "hyde": ("hybrid", "recovery", {"enable_conditional_recovery": True, "enable_hyde_fallback": True}),
    "cross_encoder": ("hybrid", "hybrid", {"enable_cross_encoder_reranking": True}),
    "context_optimization": ("hybrid", "hybrid", {"enable_retrieval_orchestration": True}),
    "corrective": ("hybrid", "hybrid", {"enable_corrective_rag": True}),
    "verification": ("hybrid", "hybrid", {"enable_verification_framework": True}),
}


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def contains(normalized_text, phrase):
    return f" {normalize(phrase)} " in f" {normalized_text} "


def match_evidence(evidence, texts):
    """Return gold sections whose authored anchors occur in the available chunks."""
    return [
        section for section, alternatives in evidence.items()
        if any(all(any(contains(text, phrase) for text in texts) for phrase in group) for group in alternatives)
    ]


def load_corpus():
    markdown = HANDBOOK.read_bytes()
    for data, key in ((markdown, "corpus_sha256"), (PDF.read_bytes(), "ingestion_pdf_sha256")):
        if hashlib.sha256(data).hexdigest() != client.CONTRACT[key]:
            raise ValueError(f"handbook does not match {key} in the smoke contract")
    text = markdown.decode("utf-8")
    sections = {}
    section = None
    for line in text.splitlines():
        heading = re.match(r"^#{2,}\s+(\d+(?:\.\d+)*)\.?\s+", line)
        if heading:
            section = heading.group(1)
            sections[section] = []
        if section:
            sections[section].append(line)
    for case in CONTRACT["cases"]:
        for section, anchors in case["evidence"].items():
            source = normalize(" ".join(sections.get(section, [])))
            if not match_evidence({section: anchors}, [source]):
                raise ValueError(f"{case['id']} has no matching gold anchors in handbook section {section}")
    return text


def check_ingestion_entities(receipt, cases):
    """Check a saved task receipt for the named entities the benchmark needs."""
    task = receipt.get("task") or receipt
    task = task.get("body", task)
    entities = ((task.get("result") or {}).get("enrichment") or {}).get("linked_entity_ids")
    if not isinstance(entities, list):
        raise ValueError("ingestion receipt is missing result.enrichment.linked_entity_ids")
    present = {normalize(name) for name in entities}
    return {
        case["id"]: list(dict.fromkeys(
            node for path in case["paths"] for node in path if normalize(node) not in present
        ))
        for case in cases if case.get("paths")
    }


def matching_paths(expected_paths, chunks):
    observed = set()
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        path = (metadata.get("graph_paths") or {}).get("graph") or []
        if not path or any(not str(step.get("evidence_text") or "").strip() for step in path):
            continue
        edges = [
            (normalize(str(step.get("from_node_id") or "")), normalize(str(step.get("to_node_id") or "")))
            for step in path
        ]
        if any(not source or not target for source, target in edges):
            continue
        if all(left[1] == right[0] for left, right in zip(edges, edges[1:])):
            observed.add((edges[0][0], *(edge[1] for edge in edges)))

    matched = []
    for nodes in expected_paths:
        expected = tuple(normalize(node) for node in nodes)
        size = len(expected)
        if any(
            actual[start:start + size] in (expected, expected[::-1])
            for actual in observed for start in range(len(actual) - size + 1)
        ):
            matched.append(nodes)
    return matched


def score_response(case, profile, response):
    chunks = response.get("results") or []
    attachment = profile in {"vectorless_attachment", "attachment_auto"}
    claims = response.get("grounded_claims") or []
    cited_ids = {chunk_id for claim in claims for chunk_id in claim.get("chunk_ids", [])}
    evidence = case["evidence"]
    section_ranks = {}
    scoped_chunks, scoped_texts, cited_texts = [], [], []
    for rank, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        in_scope = (metadata.get("is_attachment") is True and metadata.get("file_name") == HANDBOOK.name
                    if attachment else metadata.get("demo_pdf_sha256") == client.PDF_SHA256)
        if not in_scope:
            continue
        scoped_chunks.append(chunk)
        text = normalize(chunk.get("content") or "")
        scoped_texts.append(text)
        if metadata.get("citation_id") in cited_ids:
            cited_texts.append(text)
        for section in match_evidence(evidence, scoped_texts):
            section_ranks.setdefault(section, rank)
    retrieved_sections = list(section_ranks)
    cited_sections = match_evidence(evidence, cited_texts)
    expected_paths = case.get("paths", [])
    paths = matching_paths(expected_paths, scoped_chunks)

    return {
        "out_of_scope_chunks": len(chunks) - len(scoped_chunks),
        "gold_sections_retrieved": retrieved_sections,
        "gold_sections_cited": cited_sections,
        "gold_section_coverage_ranks": section_ranks,
        "evidence_recall_at_1": sum(rank == 1 for rank in section_ranks.values()) / len(evidence) if evidence else None,
        "evidence_recall": len(retrieved_sections) / len(evidence) if evidence else None,
        "cited_evidence_recall": len(cited_sections) / len(evidence) if evidence else None,
        "expected_paths": expected_paths,
        "matched_paths": paths,
        "all_expected_paths_observed": len(paths) == len(expected_paths) if expected_paths else None,
        "cited_claim_count": sum(bool(claim.get("chunk_ids")) for claim in claims),
        "answer_review": "manual",
        **graph_checks(response),
    }


def feature_observation(case, profile, response, checks):
    mode = response.get("effective_mode")
    routing = response.get("routing_details") or {}
    plan = routing.get("execution_plan") or {}
    recovery = response.get("recovery_details") or {}
    escalation = recovery.get("route_escalation") or {}
    initial_counts = escalation.get("initial_channel_counts") or {}
    planned_mode = escalation.get("from_mode") or plan.get("effective_mode")
    results = response.get("results") or []
    verify = response.get("verification_outcome") or {}
    expected_verification = case.get("verification_status") if profile == "verification" else None
    positive_graph = (checks["initial_graph_candidate_count"] or 0) > 0
    observed = {
        "vector": mode == "vector" and (initial_counts.get("vector") or 0) > 0,
        "hybrid": mode == "hybrid" and positive_graph and (initial_counts.get("vector") or 0) > 0,
        "graph": mode == "graph" and positive_graph,
        "hybrid_lexical": (initial_counts.get("lexical") or 0) > 0,
        "vectorless": mode == "vectorless" and bool(results),
        "vectorless_attachment": mode == "vectorless" and bool(results),
        "adaptive": plan.get("policy") == "adaptive" and (
            case.get("adaptive_plan_mode") is None or planned_mode == case["adaptive_plan_mode"]
        ),
        "attachment_auto": mode == "vectorless" and bool(results),
        "recovery": recovery.get("structured_recovery_used") is True,
        "hyde": recovery.get("hyde_used") is True,
        "cross_encoder": any(
            (chunk.get("metadata") or {}).get("rerank_mode") in {"cross_encoder", "graph_aware_cross_encoder"}
            for chunk in results
        ),
        "context_optimization": ((response.get("retrieval_orchestration") or {}).get("context_optimization") or {}).get("enabled") is True,
        "corrective": (response.get("corrective_rag") or {}).get("enabled") is True,
        "verification": verify.get("status") is not None,
    }.get(profile)
    if profile in {"graph_one_hop", "graph_dfs", "graph_community"}:
        observed = None  # The public response does not expose pre-filter traversal diagnostics.
    return {
        "observed": observed,
        "relationship_intent": routing.get("relationship_signal"),
        "planned_mode": planned_mode,
        "route_escalated": escalation.get("triggered"),
        "expected_verification_status": expected_verification,
        "observed_verification_status": verify.get("status"),
        "verification_status_matches": verify.get("status") == expected_verification if expected_verification else None,
    }


def benchmark_plan(corpus, case_ids=(), profile_names=()):
    cases = [case for case in CONTRACT["cases"] if not case_ids or case["id"] in case_ids]
    for case in cases:
        for profile in ("vector", "hybrid", *case["profiles"]):
            if profile_names and profile not in profile_names:
                continue
            mode, baseline, overrides = PROFILES[profile]
            options = {**FIXED_OPTIONS, **overrides}
            if profile in {"vectorless_attachment", "attachment_auto"}:
                options.update(attachment_content=corpus, attachment_name=HANDBOOK.name)
            yield case, profile, mode, baseline, options


def build_benchmark(query, plan, code_revision, environment):
    rows = []
    for case, profile, mode, baseline, options in plan:
        recorded_options = {key: value for key, value in options.items() if key != "attachment_content"}
        row = {"case": case, "profile": profile, "baseline": baseline, "requested_mode": mode, "options": recorded_options}
        started = perf_counter()
        try:
            response = query(case, mode, options)
        except (OSError, ValueError) as exc:
            row.update(status="request_failed", error=str(exc))
        else:
            checks = score_response(case, profile, response)
            row.update(
                status="success",
                response=response,
                checks=checks,
                feature=feature_observation(case, profile, response, checks),
            )
        row["client_elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
        rows.append(row)

    indexed = {(row["case"]["id"], row["profile"]): row for row in rows}
    for row in rows:
        baseline = indexed.get((row["case"]["id"], row["baseline"]))
        row["delta_vs_baseline"] = None
        if row["profile"] == "hybrid" and row["case"].get("paths"):
            row["graph_comparison_ready"] = (
                baseline is not None and comparable_response(baseline) and comparable_response(row)
                and row["feature"]["observed"] and row["checks"]["all_expected_paths_observed"]
            )
        if not baseline or any(not comparable_response(item) for item in (baseline, row)):
            continue
        row["delta_vs_baseline"] = {}
        for key in ("evidence_recall_at_1", "evidence_recall", "cited_evidence_recall"):
            left, right = baseline["checks"][key], row["checks"][key]
            row["delta_vs_baseline"][key] = round(right - left, 4) if left is not None and right is not None else None
        left = baseline["response"].get("execution_time_ms")
        right = row["response"].get("execution_time_ms")
        row["delta_vs_baseline"]["api_ms"] = round(right - left, 2) if left is not None and right is not None else None

    by_profile = {}
    for profile in dict.fromkeys(row["profile"] for row in rows):
        samples = [row for row in rows if row["profile"] == profile]
        success = [row for row in samples if row["status"] == "success"]
        paired = [row["delta_vs_baseline"] for row in success if row["delta_vs_baseline"] is not None]
        latencies = [pair["api_ms"] for pair in paired if pair["api_ms"] is not None]
        recall = [pair["evidence_recall"] for pair in paired if pair["evidence_recall"] is not None]
        groundable_graph_rows = [row for row in samples if row["case"].get("paths")]
        by_profile[profile] = {
            "runs": len(samples),
            "successful_requests": len(success),
            "feature_observed_runs": sum(row["feature"]["observed"] is True for row in success),
            "all_expected_paths_observed_runs": sum(row["checks"]["all_expected_paths_observed"] is True for row in success),
            "runs_with_expected_paths": len(groundable_graph_rows),
            "paired_evidence_recall_deltas": {"better": sum(value > 0 for value in recall), "equal": recall.count(0), "worse": sum(value < 0 for value in recall)},
            "paired_median_api_delta_ms": round(median(latencies), 2) if latencies else None,
        }
    return {
        "schema_version": CONTRACT["schema_version"],
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "kinegraph_code_revision": code_revision,
        "environment": environment,
        "question_contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
        "source_corpus_sha256": client.CONTRACT["corpus_sha256"],
        "ingestion_pdf_sha256": client.PDF_SHA256,
        "graph_comparison_ready_case_ids": [row["case"]["id"] for row in rows if row.get("graph_comparison_ready")],
        "by_profile": by_profile,
        "results": rows,
    }


def comparable_response(row):
    if row["status"] != "success" or row["checks"]["out_of_scope_chunks"]:
        return False
    allowed_modes = {
        "adaptive": {"vector", "graph", "hybrid"},
        "attachment_auto": {"vectorless"},
    }.get(row["profile"], {row["requested_mode"]})
    return row["response"].get("effective_mode") in allowed_modes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("KINEGRAPH_API_URL", "http://localhost:8000"))
    parser.add_argument("--code-revision")
    parser.add_argument("--environment", help="Record chat and embedding models, chunking policy, and index setup")
    parser.add_argument("--case", action="append", choices=[case["id"] for case in CONTRACT["cases"]])
    parser.add_argument("--profile", action="append", choices=PROFILES)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ingest-receipt", type=Path, help="Saved ingestion task response or browser smoke artifact")
    parser.add_argument("--plan", action="store_true", help="Validate the contract against the handbook and list requests offline")
    args = parser.parse_args()
    try:
        plan = list(benchmark_plan(load_corpus(), args.case, args.profile))
        missing_entities = None
        if args.ingest_receipt:
            cases = {case["id"]: case for case, *_ in plan}
            missing_entities = check_ingestion_entities(json.loads(args.ingest_receipt.read_text()), cases.values())
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not plan:
        parser.error("no cases use the selected profiles")
    if args.plan:
        for case, profile, _, baseline, _ in plan:
            print(f"{case['id']}\t{case['slice']}\t{profile}\tbaseline={baseline or '-'}")
        print(f"{len(plan)} requests across {len({case['id'] for case, *_ in plan})} questions")
        if missing_entities is not None:
            print(json.dumps({"missing_entities_in_ingestion_receipt": missing_entities}, indent=2))
        return 0
    if not (args.output and args.code_revision and args.environment) or args.timeout < 1:
        parser.error("live runs require --output, --code-revision, --environment and a positive timeout")

    report = build_benchmark(
        lambda case, mode, options: client.query_case(args.base_url, case, mode, timeout=args.timeout, options=options),
        plan, args.code_revision, args.environment,
    )
    report["missing_entities_in_ingestion_receipt"] = missing_entities
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["by_profile"], indent=2))
    return int(any(row["status"] != "success" for row in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
