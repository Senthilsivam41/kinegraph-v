import importlib.util
import json
import sys
from pathlib import Path

from backend.core.adaptive_routing import build_execution_plan
from backend.core.intent_classifier import classify_intent


ROOT = Path(__file__).parents[1]
DEMO = ROOT / "use-cases/hr-policy-assistant"


def load_benchmark():
    spec = importlib.util.spec_from_file_location("hr_policy_benchmark", DEMO / "run_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(DEMO))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_benchmark_gold_anchors_and_routing_are_valid():
    benchmark = load_benchmark()
    corpus = benchmark.load_corpus()
    plan = list(benchmark.benchmark_plan(corpus))
    cases = benchmark.CONTRACT["cases"]
    assert len({case["id"] for case in cases}) == 14
    assert not any(classify_intent(case["question"])["relationship_signal"] for case in benchmark.client.CASES)
    for case in cases:
        intent = classify_intent(case["question"])
        if case.get("paths"):
            assert intent["relationship_signal"]
        if "adaptive_plan_mode" in case:
            route = build_execution_plan(
                intent_result=intent, requested_mode="hybrid", allow_mode_downgrade=True,
                adaptive_enabled=True, lexical_enabled=False, vectorless_eligible=False,
                vectorless_reason=None,
            )
            assert route.effective_mode == case["adaptive_plan_mode"], case["id"]
    planned = {(case["id"], name): options for case, name, _, _, options in plan}
    assert {name for _, name in planned} == set(benchmark.PROFILES)
    assert len(planned) == len(plan) == 63
    for case, _, _, baseline, _ in plan:
        assert baseline is None or (case["id"], baseline) in planned
    assert planned["HR-B11", "vectorless_attachment"]["attachment_content"] == corpus
    assert planned["HR-B11", "attachment_auto"]["attachment_content"] == corpus


def test_original_ingestion_receipt_lacks_entities_for_the_graph_questions():
    benchmark = load_benchmark()
    path = ROOT / "reports/test-runs/2026-11-11/hr-policy/browser-smoke.json"
    receipt = json.loads(path.read_text())

    assert benchmark.check_ingestion_entities(receipt, benchmark.CONTRACT["cases"]) == {
        "HR-B01": ["Customer Success"],
        "HR-B02": ["Data Science", "Engineering", "Customer Success", "Revenue"],
        "HR-B03": ["Sales"],
    }


def test_benchmark_scores_path_and_citation_evidence_separately_from_http_success():
    benchmark = load_benchmark()
    plan = list(benchmark.benchmark_plan(benchmark.load_corpus(), ["HR-B01"], ["vector", "hybrid"]))

    def query(case, mode, options):
        assert options["max_results"] == 3
        graph = mode == "hybrid"
        metadata = {"demo_pdf_sha256": benchmark.client.PDF_SHA256, "citation_id": "roster"}
        if graph:
            metadata["graph_paths"] = {"graph": [{
                "from_node_id": "Dana Whitfield", "to_node_id": "Customer Success",
                "relationship_type": "LEADS", "evidence_text": "Customer Success | Dana Whitfield | Revenue",
            }]}
        return {
            "effective_mode": mode,
            "execution_time_ms": 20 if graph else 10,
            "recovery_details": {"route_escalation": {"initial_channel_counts": {"vector": 2, "graph": int(graph)}}},
            "results": [{"content": "Customer Success | Dana Whitfield | Revenue" if graph else "PTO policy", "metadata": metadata}],
            "grounded_claims": [{"chunk_ids": ["roster"]}] if graph else [],
        }

    report = benchmark.build_benchmark(query, plan, "test-revision", "test")
    vector, hybrid = report["results"]
    assert vector["checks"]["evidence_recall"] == 0.0
    assert hybrid["feature"]["observed"] is True
    assert hybrid["checks"]["matched_paths"] == [["Customer Success", "Dana Whitfield"]]
    assert hybrid["checks"]["gold_sections_cited"] == ["2"]
    assert report["graph_comparison_ready_case_ids"] == ["HR-B01"]
    assert hybrid["delta_vs_baseline"] == {
        "evidence_recall_at_1": 1.0, "evidence_recall": 1.0, "cited_evidence_recall": 1.0, "api_ms": 10,
    }

    hybrid["response"]["results"][0]["metadata"]["demo_pdf_sha256"] = "different-document"
    checks = benchmark.score_response(hybrid["case"], "hybrid", hybrid["response"])
    assert checks["out_of_scope_chunks"] == 1
    assert checks["matched_paths"] == []
    assert checks["evidence_recall"] == 0.0


def test_graph_path_requires_named_hops_continuity_and_edge_evidence():
    benchmark = load_benchmark()
    expected = [["Sales", "Client-Facing Role", "Expense Approval"]]
    path = [
        {"from_node_id": "Sales", "to_node_id": "Client-Facing Role", "evidence_text": "The role definition."},
        {"from_node_id": "Client-Facing Role", "to_node_id": "Expense Approval", "evidence_text": "The expense rule."},
    ]
    chunks = [{"metadata": {"graph_paths": {"graph": path}}}]
    assert benchmark.matching_paths(expected, chunks) == expected
    path[1]["from_node_id"] = "Other Role"
    assert benchmark.matching_paths(expected, chunks) == []
    path[1].update(from_node_id="Client-Facing Role", evidence_text="")
    assert benchmark.matching_paths(expected, chunks) == []


def test_cross_encoder_observation_recognizes_the_graph_aware_mode():
    benchmark = load_benchmark()
    case = next(case for case in benchmark.CONTRACT["cases"] if case["id"] == "HR-B03")
    metadata = {"demo_pdf_sha256": benchmark.client.PDF_SHA256, "rerank_mode": "graph_aware_cross_encoder"}
    response = {"effective_mode": "hybrid", "results": [{"metadata": metadata}]}

    checks = benchmark.score_response(case, "cross_encoder", response)
    assert benchmark.feature_observation(case, "cross_encoder", response, checks)["observed"] is True
    metadata["rerank_mode"] = "graph_aware_keyword"
    assert benchmark.feature_observation(case, "cross_encoder", response, checks)["observed"] is False
