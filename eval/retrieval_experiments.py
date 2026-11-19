"""Paired promotion gates for Hybrid lexical fusion (#47) and reranking (#48)."""
from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from statistics import mean

import numpy as np

from eval.benchmark_profiles import get_profile
from eval.experiment_validation import (
    DEFAULT_METRIC_WEIGHTS,
    compare_manifests,
    validate_metric_values,
    weighted_composite,
)
from eval.provenance import validate_provenance_record

EXPERIMENTS = ("lexical-fusion", "cross-encoder")
EXACT_TOKEN_IDS = {f"KGV1-{index:03d}" for index in (10, 11, 13, 14, 15)}
METRICS = tuple(DEFAULT_METRIC_WEIGHTS)
EPSILON = 1e-12


def load_experiment_provenance(manifest):
    path = Path(manifest["artifacts"]["provenance_jsonl"])
    expected = manifest.get("artifact_sha256", {}).get("provenance_jsonl")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"provenance checksum missing or mismatched for {path}")
    return [json.loads(line) for line in payload.splitlines()]


def _index_samples(manifest, records, profile_name):
    samples = {}
    for record in records:
        validate_provenance_record(record)
        sample_id = record["sample_id"]
        profile = record["profile"]
        evaluation = record["evaluation"]
        retrieval = record["retrieval"]
        if sample_id in samples:
            raise ValueError(f"duplicate provenance for {sample_id}")
        if (
            profile["name"] != profile_name
            or profile["effective_mode"] != "hybrid"
            or profile["valid"] is not True
            or evaluation["ragas_failed"] is not False
            or evaluation["judge_model"] != manifest["models"]["judge"]
            or evaluation["embedding_model"] != manifest["models"]["embedding"]
            or retrieval["candidate_provenance_completeness"] != 1.0
            or retrieval["failures"]
        ):
            raise ValueError(f"{sample_id}: invalid Hybrid, RAGAS, or retrieval provenance")
        validate_metric_values(evaluation["metrics"], METRICS)
        samples[sample_id] = record
    if not samples or len(samples) != manifest["report"]["summary"]["total_samples"]:
        raise ValueError("provenance must cover every manifest sample")
    return samples


def _stage_latency(record, stage):
    value = record["latency_ms"]["stages"].get(stage)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{record['sample_id']}: missing or invalid {stage}")
    return value


def _metrics(record):
    return record["evaluation"]["metrics"]


def compare_retrieval_experiment(experiment, baseline, candidate, baseline_records, candidate_records):
    """Measure a fixed Hybrid pair and return evidence and a keep/revert/invalid decision."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown retrieval experiment: {experiment}")
    lexical = experiment == "lexical-fusion"
    stage, budget = ("lexical_retrieval_ms", 50) if lexical else ("rerank_ms", 250)
    section, flag = (
        ("retrieval", "enable_lexical_fusion") if lexical
        else ("reranking", "enable_cross_encoder_reranking")
    )
    candidate_profile = "hybrid_lexical" if lexical else "hybrid"
    normalized = []
    for manifest, profile_name, enabled in (
        (baseline, "hybrid", False),
        (candidate, candidate_profile, True),
    ):
        config = manifest["pipeline_config"]
        retrieval = config["retrieval"]
        profile = retrieval["profile"]
        if (
            profile != get_profile(profile_name).to_dict()
            or retrieval["mode"] != "hybrid"
            or profile["enable_lexical_fusion"] is not retrieval["enable_lexical_fusion"]
            or config[section][flag] is not enabled
        ):
            raise ValueError("expected a fixed Hybrid pair with the requested lever off and on")
        if lexical and enabled and retrieval["lexical_fusion_weight"] <= 0:
            raise ValueError("lexical fusion requires a positive lexical weight")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", config["benchmark"].get("corpus_sha256") or ""):
            raise ValueError("both arms must record the frozen corpus SHA-256")
        if manifest["provenance"].get("git_revision") in (None, "", "unknown"):
            raise ValueError("both arms must record the frozen code revision")
        copy = deepcopy(manifest)
        # Profile name and profile flag mirror the lexical lever in the runner.
        normalized_profile = copy["pipeline_config"]["retrieval"]["profile"]
        normalized_profile.pop("name")
        normalized_profile.pop("enable_lexical_fusion")
        normalized.append(copy)

    result = compare_manifests(*normalized)
    if result["changed_levers"] != [f"pipeline.{section}.{flag}"]:
        result["decision"] = "invalid"
        result["reasons"].append("the requested retrieval lever must be the only change")
    left = _index_samples(baseline, baseline_records, "hybrid")
    right = _index_samples(candidate, candidate_records, candidate_profile)
    if left.keys() != right.keys():
        raise ValueError("baseline and candidate must cover identical sample IDs")

    slices = {"exact_token"} if lexical else {"exact_token", "misspelled", "single_hop", "multi_hop"}
    sample_ids = {name: [] for name in sorted(slices)}
    per_query = {}
    before_ms, after_ms = [], []
    fallback_samples = {}
    model_revisions = set()
    for sample_id in sorted(left):
        before, after = left[sample_id], right[sample_id]
        if before["categories"] != after["categories"] or before["query"]["original"] != after["query"]["original"]:
            raise ValueError(f"{sample_id}: query or category labels changed")
        for name in slices.intersection(before["categories"]):
            sample_ids[name].append(sample_id)
        baseline_ms = 0 if lexical else _stage_latency(before, stage)
        candidate_ms = _stage_latency(after, stage)
        before_ms.append(baseline_ms)
        after_ms.append(candidate_ms)
        rerank = after["retrieval"]["reranking"]
        if not lexical:
            if before["retrieval"]["reranking"]["mode"] != "keyword":
                raise ValueError(f"{sample_id}: baseline did not use keyword reranking")
            if before["retrieval"]["fusion"]["candidates"] != after["retrieval"]["fusion"]["candidates"]:
                raise ValueError(f"{sample_id}: the reranker received different fused candidates")
            if rerank.get("fallback") or rerank.get("mode") != "cross_encoder":
                fallback_samples[sample_id] = rerank.get("fallback_reason") or "cross-encoder did not run"
            else:
                model_revisions.add(rerank.get("model_revision"))
        per_query[sample_id] = {
            "categories": before["categories"],
            "metric_deltas": {
                metric: round(float(_metrics(after)[metric]) - float(_metrics(before)[metric]), 6)
                for metric in METRICS
            },
            "stage_latency_ms": {"baseline": baseline_ms, "candidate": candidate_ms},
        }

    if not all(sample_ids.values()) or not EXACT_TOKEN_IDS.issubset(sample_ids["exact_token"]):
        raise ValueError("required query slices or exact-token queries are missing")

    def summarize(ids):
        metrics = {}
        for metric in METRICS:
            prior = mean(float(_metrics(left[key])[metric]) for key in ids)
            current = mean(float(_metrics(right[key])[metric]) for key in ids)
            metrics[metric] = {"baseline": prior, "candidate": current, "delta": current - prior}
        return metrics

    per_category = {
        name: {"sample_ids": ids, "metrics": summarize(ids)} for name, ids in sample_ids.items()
    }
    overall = summarize(left)
    precision = overall["context_precision"]
    composite_delta = weighted_composite({key: value["candidate"] for key, value in overall.items()})
    composite_delta -= weighted_composite({key: value["baseline"] for key, value in overall.items()})
    baseline_p95 = float(np.percentile(before_ms, 95))
    candidate_p95 = float(np.percentile(after_ms, 95))
    overhead = candidate_p95 - baseline_p95

    failures = []
    for metric, values in overall.items():
        if values["delta"] < -0.05 - EPSILON:
            failures.append(f"{metric} regressed by more than 0.05")
    if precision["candidate"] < 0.90 and (lexical or precision["delta"] <= EPSILON):
        failures.append("overall context precision did not meet the promotion requirement")
    if lexical:
        if per_category["exact_token"]["metrics"]["context_recall"]["delta"] < 0.05 - EPSILON:
            failures.append("exact-token context recall did not gain at least 0.05")
    else:
        if composite_delta <= 0.01 + EPSILON:
            failures.append("weighted composite gain did not exceed 0.01")
        if fallback_samples:
            failures.append("the cross-encoder arm did not use the cross-encoder for every query")
        if len(model_revisions) != 1 or None in model_revisions or "" in model_revisions:
            failures.append("the cross-encoder arm lacks one resolved model revision")
    if overhead > budget + EPSILON:
        failures.append(f"{stage} p95 overhead exceeded {budget} ms")
    if failures and result["decision"] == "keep":
        result["decision"] = "revert"
    result["reasons"].extend(failures)
    return {
        **result,
        "accepted": result["decision"] == "keep",
        "experiment": experiment,
        "per_category": per_category,
        "per_query": per_query,
        "stage_latency_ms": {
            "stage": stage, "baseline_p95": baseline_p95, "candidate_p95": candidate_p95,
            "p95_overhead": overhead, "budget": budget,
        },
        "fallback_samples": fallback_samples,
        "cross_encoder_model_revisions": sorted(revision for revision in model_revisions if revision),
    }
