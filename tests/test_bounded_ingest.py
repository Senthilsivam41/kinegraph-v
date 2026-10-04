import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, Request, UploadFile

from backend.app.api.routes import ingest
from backend.services.chroma_service import ChromaService
from backend.ingestion.auth import require_workspace


def test_upload_rejects_oversize_and_cleans_up(tmp_path):
    file = UploadFile(filename="large.txt", file=BytesIO(b"abcde"))
    with (
        patch.object(ingest.settings, "UPLOAD_DIR", tmp_path),
        patch.object(ingest.settings, "MAX_UPLOAD_BYTES", 4),
        patch.object(ingest.process_document, "delay") as enqueue,
    ):
        with pytest.raises(HTTPException) as error:
            asyncio.run(ingest.ingest_document(Request({"type": "http", "headers": []}), file, metadata=None))
    assert error.value.status_code == 413
    assert list(tmp_path.iterdir()) == []
    enqueue.assert_not_called()


def test_text_upload_is_admitted_at_exact_limit(tmp_path):
    file = UploadFile(filename="note.txt", file=BytesIO(b"hello"))
    with (
        patch.object(ingest.settings, "UPLOAD_DIR", tmp_path),
        patch.object(ingest.settings, "MAX_UPLOAD_BYTES", 5),
        patch.object(ingest.process_document, "delay", return_value=SimpleNamespace(id="task-1")) as enqueue,
    ):
        response = asyncio.run(ingest.ingest_document(Request({"type": "http", "headers": []}), file, metadata=None))
    assert response.task_id == "task-1"
    saved_path = enqueue.call_args.args[0]
    assert saved_path.endswith(".txt")
    assert list(tmp_path.iterdir())[0].read_bytes() == b"hello"


def test_chroma_replay_upserts_in_bounded_batches():
    service = ChromaService.__new__(ChromaService)
    collection = MagicMock()
    service.get_or_create_collection = MagicMock(return_value=collection)
    service.embeddings = MagicMock()
    service.embeddings.embed_documents.side_effect = lambda texts: [[1.0] for _ in texts]
    with patch.object(ingest.settings, "CHROMA_UPSERT_BATCH_SIZE", 2):
        for _ in range(2):
            assert asyncio.run(service.add_documents(["a", "b", "c"], [{}, {}, {}], ["1", "2", "3"]))
    assert collection.upsert.call_count == 4
    collection.add.assert_not_called()
    assert collection.upsert.call_args.kwargs["ids"] == ["3"]


def test_chroma_rejects_duplicate_chunk_ids():
    service = ChromaService.__new__(ChromaService)
    with pytest.raises(ValueError, match="unique"):
        asyncio.run(service.add_documents(["a", "b"], [{}, {}], ["1", "1"]))


def test_durable_admission_uses_outbox_and_removes_local_source(tmp_path):
    file = UploadFile(filename="note.txt", file=BytesIO(b"hello"))
    request = Request({"type": "http", "headers": [(b"idempotency-key", b"request-1")]})
    with (
        patch.object(ingest.settings, "UPLOAD_DIR", tmp_path),
        patch.object(ingest.settings, "INGEST_DATABASE_URL", "postgresql+psycopg://example"),
        patch("backend.ingestion.auth.require_workspace", return_value="workspace"),
        patch("backend.ingestion.store.reserve_document", return_value=("00000000-0000-4000-8000-000000000001", True)) as reserve,
        patch("backend.ingestion.store.mark_accepted") as accepted,
        patch("backend.ingestion.objects.upload_source") as upload,
    ):
        response = asyncio.run(ingest.ingest_document(request, file, metadata=None))
    assert response.doc_id == "00000000-0000-4000-8000-000000000001"
    assert response.status_url.endswith(response.doc_id)
    assert reserve.call_args.kwargs["idempotency_key"] == "request-1"
    assert reserve.call_args.kwargs["digest"] == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert upload.call_count == 1
    accepted.assert_called_once_with(response.doc_id)
    assert list(tmp_path.iterdir()) == []


def test_proxy_group_header_is_required_for_workspace():
    request = Request({"type": "http", "headers": [
        (b"x-forwarded-user", b"person"), (b"x-forwarded-groups", b"readers,ingest"),
    ]})
    with (
        patch.object(ingest.settings, "OIDC_ISSUER_URL", "https://issuer.example"),
        patch.object(ingest.settings, "OIDC_AUDIENCE", "kinegraph"),
        patch.object(ingest.settings, "OIDC_ALLOWED_GROUP", "ingest"),
        patch.object(ingest.settings, "TRUST_PROXY_AUTH_HEADERS", True),
    ):
        assert require_workspace(request) == "workspace"
        with patch.object(ingest.settings, "OIDC_ALLOWED_GROUP", "admins"):
            with pytest.raises(HTTPException) as denied:
                require_workspace(request)
    assert denied.value.status_code == 403
