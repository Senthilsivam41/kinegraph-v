import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.core.config import settings
from backend.core.grounding import (
    apply_critic_response,
    assign_citation_ids,
    validate_grounded_response,
)
from backend.core.langgraph_workflow import HybridRAGWorkflow


def test_assign_citation_ids_prefers_store_ids_and_makes_them_unique():
    chunks = assign_citation_ids([
        {"id": "chroma-1", "content": "A"},
        {"metadata": {"chunk_id": "neo4j-1"}, "content": "B"},
        {"id": "chroma-1", "content": "C"},
        {"content": "D"},
    ])

    assert [chunk["citation_id"] for chunk in chunks] == [
        "chroma-1", "neo4j-1", "chroma-1#2", "ctx-4"
    ]


def test_validation_rejects_missing_and_unknown_citations():
    content = json.dumps({
        "claims": [
            {"text": "Supported", "chunk_ids": ["chunk-1"]},
            {"text": "Missing citation", "chunk_ids": []},
            {"text": "Invented citation", "chunk_ids": ["chunk-404"]},
        ],
        "confidence": 0.9,
    })

    claims, confidence, details = validate_grounded_response(content, {"chunk-1"})

    assert [claim["text"] for claim in claims] == ["Supported"]
    assert confidence == 0.3
    assert details["accepted_claims"] == 1
    assert [item["reason"] for item in details["rejected_claims"]] == [
        "missing_citation", "invalid_citation"
    ]


def test_empty_structured_claim_list_is_valid_and_grounded():
    claims, confidence, details = validate_grounded_response(
        '{"claims": [], "confidence": 0}', {"chunk-1"}
    )

    assert claims == []
    assert confidence == 0.0
    assert details["structured_output_valid"] is True


def test_critic_can_only_filter_existing_claims():
    claims = [
        {"claim_id": "claim-1", "text": "A", "chunk_ids": ["a"]},
        {"claim_id": "claim-2", "text": "B", "chunk_ids": ["b"]},
    ]
    retained, details = apply_critic_response(
        claims,
        json.dumps({
            "supported_claim_ids": ["claim-2", "invented-claim"],
            "unsupported_reasons": {"claim-1": "not entailed"},
        }),
    )

    assert retained == [claims[1]]
    assert details["removed_claim_ids"] == ["claim-1"]


@pytest.mark.asyncio
async def test_generate_node_exposes_verified_citations_only():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    workflow.llm = object()
    workflow._invoke_prompt = AsyncMock(return_value=SimpleNamespace(content=json.dumps({
        "claims": [
            {"text": "Grounded answer", "chunk_ids": ["stored-1"]},
            {"text": "Ungrounded answer", "chunk_ids": ["missing"]},
        ],
        "confidence": 0.8,
    })))
    state = {
        "query": "question",
        "reranked_results": [{"id": "stored-1", "content": "Grounded answer"}],
        "latency_breakdown": {},
    }

    result = await workflow._generate_node(state)

    assert result["generated_answer"] == "Grounded answer [stored-1]"
    assert result["citation_validation"]["accepted_claims"] == 1
    assert result["citation_validation"]["rejected_claims"][0]["reason"] == "invalid_citation"


@pytest.mark.asyncio
async def test_critique_removes_unsupported_claim_without_rewriting():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    workflow.critic_llm = object()
    workflow._invoke_prompt = AsyncMock(return_value=SimpleNamespace(content=json.dumps({
        "supported_claim_ids": ["claim-1"],
        "unsupported_reasons": {"claim-2": "not directly supported"},
    })))
    claims = [
        {"claim_id": "claim-1", "text": "Keep", "chunk_ids": ["a"]},
        {"claim_id": "claim-2", "text": "Drop", "chunk_ids": ["b"]},
    ]
    state = {
        "enable_grounding_critique": True,
        "grounded_claims": claims,
        "citation_context": {"a": "Keep", "b": "unrelated"},
        "generated_answer": "",
        "answer_confidence": 0.8,
        "latency_breakdown": {},
    }

    result = await workflow._grounding_critique(state)

    assert result["generated_answer"] == "Keep [a]"
    assert result["answer_confidence"] == 0.4
    assert result["grounding_critique"]["removed_claim_ids"] == ["claim-2"]
    critic_payload = json.loads(workflow._invoke_prompt.await_args.args[2]["claims_json"])
    assert critic_payload[0]["cited_context"] == {"a": "Keep"}
    assert critic_payload[1]["cited_context"] == {"b": "unrelated"}


def test_synthesis_and_critic_temperatures_are_deterministic():
    assert 0.0 <= settings.GENERATION_TEMPERATURE <= 0.2
    assert settings.FAITHFULNESS_CRITIC_TEMPERATURE == 0.0
