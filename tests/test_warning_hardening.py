import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from neo4j import READ_ACCESS

import backend.core.config as config
from backend.app.api.routes.ingest import (
    _allocate_upload_path,
    _validate_upload_filename,
)
from backend.app.main import app, lifespan
from backend.core.langgraph_workflow import (
    HybridRAGWorkflow,
    _load_vectorless_document,
)
from backend.app.models import QueryMode
from backend.services.neo4j_service import GraphWriteResult, Neo4jService
from backend.workers.document_processor import generate_chunk_id, generate_document_id


REPO_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "filename",
    [
        "../secret.pdf",
        "..\\secret.pdf",
        "/tmp/secret.pdf",
        "folder/document.pdf",
        "folder\\document.pdf",
        "",
        None,
        "document.txt",
    ],
)
def test_upload_filename_rejects_paths_and_non_pdf_names(filename):
    with pytest.raises(HTTPException):
        _validate_upload_filename(filename)


def test_upload_path_is_server_generated_and_created_lazily(tmp_path):
    upload_dir = tmp_path / "not-created-at-import"
    assert not upload_dir.exists()

    first = _allocate_upload_path(upload_dir)
    second = _allocate_upload_path(upload_dir)

    assert upload_dir.is_dir()
    assert first.parent == upload_dir
    assert first.suffix == ".pdf"
    assert first.name != "document.pdf"
    assert first != second


def test_safe_pdf_filename_is_preserved_only_for_display():
    assert _validate_upload_filename("Quarterly Report.PDF") == "Quarterly Report.PDF"


def test_document_and_chunk_ids_use_full_sha256_digests():
    chunk_id = generate_chunk_id("same content", 3)
    document_id = generate_document_id("/safe/server/path.pdf")

    assert chunk_id.startswith("chunk_3_")
    assert len(chunk_id.removeprefix("chunk_3_")) == 64
    assert len(document_id.removeprefix("doc_")) == 64
    assert generate_chunk_id("same content", 3) == chunk_id


def test_settings_proxy_does_not_construct_settings_until_first_access():
    config.get_settings.cache_clear()
    fake = SimpleNamespace(APP_NAME="isolated-test")
    with patch.object(config, "Settings", return_value=fake) as settings_class:
        assert settings_class.call_count == 0
        assert config.settings.APP_NAME == "isolated-test"
        settings_class.assert_called_once_with()
        assert config.settings.APP_NAME == "isolated-test"
        settings_class.assert_called_once_with()
    config.get_settings.cache_clear()


def test_graph_write_result_is_request_scoped_and_boolean_compatible():
    first = GraphWriteResult(success=True, enrichment={"enriched": 1})
    second = GraphWriteResult(success=True, enrichment={"enriched": 2})

    assert first
    assert second
    assert first.enrichment != second.enrichment
    assert not hasattr(Neo4jService.__new__(Neo4jService), "last_enrichment_result")


def test_graph_search_consumes_cursor_after_early_limit():
    service = Neo4jService.__new__(Neo4jService)
    service.query_to_cypher = AsyncMock(return_value="MATCH (d:Document) RETURN d")
    service.driver = MagicMock()
    session = service.driver.session.return_value.__enter__.return_value
    cursor = MagicMock()
    cursor.__iter__.return_value = [
        {"d": {"content": "first", "id": "1"}},
        {"d": {"content": "second", "id": "2"}},
    ]
    session.run.return_value = cursor

    results = asyncio.run(service.graph_search("find documents", n_results=1))

    assert [item["content"] for item in results] == ["first"]
    service.driver.session.assert_called_once_with(default_access_mode=READ_ACCESS)
    cursor.consume.assert_called_once_with()


def test_intent_router_offloads_local_document_lookup():
    workflow = HybridRAGWorkflow.__new__(HybridRAGWorkflow)
    state = {
        "query": "Summarize document.pdf",
        "rewritten_query": "",
        "intent": "",
        "suggested_mode": "",
        "requested_mode": QueryMode.HYBRID,
        "mode": QueryMode.HYBRID,
        "allow_mode_downgrade": False,
        "enable_conservative_routing": False,
        "allow_vectorless_auto_route": True,
        "routing_details": {},
        "attachment_content": None,
        "filters": {"file_name": "document.pdf"},
        "latency_breakdown": {},
    }

    with patch(
        "backend.core.langgraph_workflow.asyncio.to_thread",
        new=AsyncMock(return_value="short local document"),
    ) as to_thread:
        routed = asyncio.run(workflow._intent_router(state))

    to_thread.assert_awaited_once_with(_load_vectorless_document, "document.pdf")
    assert routed["mode"] == QueryMode.VECTORLESS


def test_lifespan_always_closes_chroma_when_neo4j_shutdown_fails():
    chroma = MagicMock()
    neo4j = MagicMock()
    neo4j.close.side_effect = RuntimeError("shutdown failure")

    async def exercise():
        with (
            patch("backend.app.main.ChromaService", return_value=chroma),
            patch("backend.app.main.Neo4jService", return_value=neo4j),
        ):
            async with lifespan(app):
                pass

    with pytest.raises(RuntimeError, match="shutdown failure"):
        asyncio.run(exercise())

    neo4j.close.assert_called_once_with()
    chroma.close.assert_called_once_with()


def test_backend_has_no_print_calls_and_vectorless_import_is_explicit():
    backend_files = list((REPO_ROOT / "backend").rglob("*.py"))
    assert all("print(" not in path.read_text() for path in backend_files)

    tasks_source = (REPO_ROOT / "backend" / "workers" / "tasks.py").read_text()
    assert "from backend.services.vectorless_service import VectorlessService" in tasks_source
    assert "try:\n            from backend.services.vectorless_service" not in tasks_source


def test_container_runtime_hardening_is_persisted():
    compose = (REPO_ROOT / "infra" / "docker-compose.yml").read_text()
    dockerfile = (REPO_ROOT / "infra" / "Dockerfile").read_text()

    assert "liteparse-server:main" not in compose
    assert "liteparse-server@sha256:" in compose
    assert compose.count("restart: unless-stopped") >= 3
    assert "import requests" not in dockerfile
    assert "urllib.request.urlopen" in dockerfile
