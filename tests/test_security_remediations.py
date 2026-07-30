import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j import READ_ACCESS
from pydantic import ValidationError
from fastapi.middleware.cors import CORSMiddleware

from backend.app.main import app
from backend.core.config import Settings
from backend.services.chroma_service import ChromaService
from backend.services.neo4j_service import (
    Neo4jService,
    UnsafeCypherError,
    validate_read_only_cypher,
)
from backend.workers.tasks import _persist_document


REPO_ROOT = Path(__file__).parents[1]


def _task():
    task = MagicMock()
    task.request.id = "task-1"
    return task


def test_celery_ingestion_uses_one_event_loop_boundary():
    source = (REPO_ROOT / "backend" / "workers" / "tasks.py").read_text()
    assert source.count("asyncio.run(") == 1


def test_async_ingestion_closes_chroma_and_neo4j_clients():
    chroma = MagicMock()
    chroma.add_documents = AsyncMock(return_value=True)
    neo4j = MagicMock()
    neo4j.add_document_graph = AsyncMock(return_value=True)
    extractor = AsyncMock(return_value=([{"name": "A"}], []))

    with (
        patch("backend.workers.tasks.ChromaService", return_value=chroma),
        patch("backend.workers.tasks.Neo4jService", return_value=neo4j),
        patch("backend.workers.tasks.extract_entities_and_relationships", extractor),
        patch("backend.workers.tasks.VectorlessService"),
    ):
        entities, relationships = asyncio.run(_persist_document(
            _task(),
            doc_id="doc-1",
            file_name="document.pdf",
            text="A document",
            chunks=["A document"],
            chunk_metadata=[{"chunk_index": 0}],
            chunk_ids=["chunk-1"],
            metadata={},
        ))

    assert entities == [{"name": "A"}]
    assert relationships == []
    chroma.close.assert_called_once_with()
    neo4j.close.assert_called_once_with()


def test_async_ingestion_closes_clients_on_graph_failure():
    chroma = MagicMock()
    chroma.add_documents = AsyncMock(return_value=True)
    neo4j = MagicMock()
    neo4j.add_document_graph = AsyncMock(return_value=False)

    with (
        patch("backend.workers.tasks.ChromaService", return_value=chroma),
        patch("backend.workers.tasks.Neo4jService", return_value=neo4j),
        patch(
            "backend.workers.tasks.extract_entities_and_relationships",
            AsyncMock(return_value=([], [])),
        ),
        patch("backend.workers.tasks.VectorlessService"),
        pytest.raises(RuntimeError, match="Neo4j"),
    ):
        asyncio.run(_persist_document(
            _task(),
            doc_id="doc-1",
            file_name="document.pdf",
            text="document",
            chunks=["document"],
            chunk_metadata=[{}],
            chunk_ids=["chunk-1"],
            metadata={},
        ))

    chroma.close.assert_called_once_with()
    neo4j.close.assert_called_once_with()


def test_chroma_service_close_releases_underlying_client():
    service = ChromaService.__new__(ChromaService)
    service.client = MagicMock()
    service.close()
    service.client.close.assert_called_once_with()


def test_read_only_cypher_is_bounded_and_literals_do_not_trigger_false_positive():
    query = "MATCH (d:Document) WHERE d.content CONTAINS 'CREATE DELETE SET' RETURN d"
    safe = validate_read_only_cypher(query, 6)
    assert safe.endswith("LIMIT $result_limit")


def test_read_only_cypher_masks_escaped_literals_without_regex_backtracking():
    escaped = "\\\\" * 200
    query = f"MATCH (d:Document) WHERE d.content CONTAINS '{escaped}' RETURN d"

    safe = validate_read_only_cypher(query, 6)

    assert safe.endswith("LIMIT $result_limit")


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) CREATE (x) RETURN n",
        "MATCH (n) SET n.admin = true RETURN n",
        "MATCH (n) DETACH DELETE n RETURN n",
        "CALL db.labels() YIELD label RETURN label",
        "MATCH (n) RETURN n; MATCH (x) RETURN x",
        "MATCH (n) /* hidden */ DELETE n RETURN n",
        "MATCH (n) RETURN n LIMIT 1000",
        "SHOW DATABASES",
    ],
)
def test_generated_cypher_rejects_write_admin_and_multi_statement_queries(query):
    with pytest.raises(UnsafeCypherError):
        validate_read_only_cypher(query, 10)


def test_graph_search_never_opens_session_for_unsafe_generated_cypher():
    service = Neo4jService.__new__(Neo4jService)
    service.query_to_cypher = AsyncMock(return_value="MATCH (n) DELETE n RETURN n")
    service.driver = MagicMock()

    with pytest.raises(UnsafeCypherError):
        asyncio.run(service.graph_search("delete everything"))

    service.driver.session.assert_not_called()


def test_graph_search_uses_read_access_and_parameterized_limit():
    service = Neo4jService.__new__(Neo4jService)
    service.query_to_cypher = AsyncMock(return_value="MATCH (d:Document) RETURN d")
    service.driver = MagicMock()
    session = service.driver.session.return_value.__enter__.return_value
    session.run.return_value = []

    assert asyncio.run(service.graph_search("find documents", n_results=7)) == []

    service.driver.session.assert_called_once_with(default_access_mode=READ_ACCESS)
    session.run.assert_called_once_with(
        "MATCH (d:Document) RETURN d\nLIMIT $result_limit",
        result_limit=7,
    )


def test_security_settings_reject_credentialed_wildcard_cors():
    with pytest.raises(ValidationError, match="credentialed CORS"):
        Settings(
            _env_file=None,
            OPENAI_API_KEY="test-key",
            NEO4J_PASSWORD="a-secure-test-password",
            CORS_ALLOWED_ORIGINS="*",
            CORS_ALLOW_CREDENTIALS=True,
        )


def test_chunking_settings_are_declared_once_and_reject_invalid_overlap():
    source = (REPO_ROOT / "backend" / "core" / "config.py").read_text()
    assert source.count("ADAPTIVE_CHUNKING_ENABLED: bool") == 1
    assert source.count("CHUNK_SIZE: int") == 1
    assert source.count("CHUNK_OVERLAP: int") == 1

    configured = Settings(
        _env_file=None,
        OPENAI_API_KEY="test-key",
        NEO4J_PASSWORD="a-secure-test-password",
    )
    assert configured.CHUNK_OVERLAP < configured.CHUNK_SIZE

    with pytest.raises(ValidationError, match="CHUNK_OVERLAP must be smaller than CHUNK_SIZE"):
        Settings(
            _env_file=None,
            OPENAI_API_KEY="test-key",
            NEO4J_PASSWORD="a-secure-test-password",
            CHUNK_SIZE=200,
            CHUNK_OVERLAP=200,
        )


def test_default_cors_origins_are_explicit_and_neo4j_secret_is_externalized():
    configured = Settings(
        _env_file=None,
        OPENAI_API_KEY="test-key",
        NEO4J_PASSWORD="a-secure-test-password",
    )
    compose = (REPO_ROOT / "infra" / "docker-compose.yml").read_text()

    assert configured.cors_allowed_origins == [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]
    assert configured.CORS_ALLOW_CREDENTIALS is False
    assert "NEO4J_PASSWORD:?" in compose
    insecure_default = "kinetic_password" + "_change_in_production"
    assert insecure_default not in compose


def test_app_cors_middleware_uses_explicit_methods_headers_and_origins():
    middleware = next(item for item in app.user_middleware if item.cls is CORSMiddleware)

    assert middleware.kwargs["allow_origins"] != ["*"]
    assert middleware.kwargs["allow_methods"] == ["GET", "POST", "OPTIONS"]
    assert middleware.kwargs["allow_headers"] == ["Authorization", "Content-Type"]
