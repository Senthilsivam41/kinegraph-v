"""
Document Ingestion Endpoints
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from backend.app.models import IngestResponse, TaskStatus
from backend.core.config import settings
from backend.workers.celery_app import celery_app
from backend.workers.tasks import process_document
from typing import Optional
import json
import hashlib
import zipfile
from pathlib import Path
from uuid import uuid4

router = APIRouter()
SUPPORTED_SUFFIXES = {".pdf": "pdf", ".docx": "docx", ".txt": "text"}


def _validate_upload_filename(filename: Optional[str]) -> str:
    """Return a display-safe basename or reject path-like upload names."""
    if not filename or "\x00" in filename:
        raise HTTPException(status_code=400, detail="A valid document filename is required")

    portable_name = filename.replace("\\", "/")
    basename = Path(portable_name).name
    if basename != portable_name or basename in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="Unsafe upload filename")
    if Path(basename).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX and TXT files are supported")
    return basename


def _allocate_upload_path(upload_dir: Path, suffix: str = ".pdf") -> Path:
    """Allocate a server-controlled filename beneath the configured directory."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Unsupported upload suffix")
    return upload_dir / f"{uuid4().hex}{suffix}"


def _validate_upload_content(path: Path, kind: str) -> None:
    """Check a small, untrusted upload before it reaches a parser or queue."""
    if kind == "pdf":
        with path.open("rb") as source:
            header = source.read(5)
        if header != b"%PDF-":
            raise HTTPException(status_code=400, detail="Invalid PDF header")
    elif kind == "docx":
        if not zipfile.is_zipfile(path):
            raise HTTPException(status_code=400, detail="Invalid DOCX archive")
        with zipfile.ZipFile(path) as archive:
            if len(archive.infolist()) > 1000:
                raise HTTPException(status_code=413, detail="DOCX has too many archive members")
            if "[Content_Types].xml" not in archive.namelist() or "word/document.xml" not in archive.namelist():
                raise HTTPException(status_code=400, detail="Invalid DOCX document")
            if sum(item.file_size for item in archive.infolist()) > settings.MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="DOCX expands beyond the size limit")
    elif kind == "text":
        try:
            with path.open("r", encoding="utf-8-sig") as source:
                for _ in iter(lambda: source.read(1024 * 1024), ""):
                    pass
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="TXT must be UTF-8") from exc


@router.post("/document", response_model=IngestResponse, status_code=202)
@router.post("/documents", response_model=IngestResponse, status_code=202)
async def ingest_document(
    request: Request, file: UploadFile = File(...), metadata: Optional[str] = Form(None)
):
    """
    Ingest a document (PDF) and process it asynchronously

    The document will be:
    1. Split into chunks
    2. Embedded and stored in ChromaDB
    3. Entities extracted and stored in Neo4j
    """
    owner = None
    if settings.INGEST_DATABASE_URL:
        from backend.ingestion.auth import require_workspace

        owner = require_workspace(request)
    original_name = _validate_upload_filename(file.filename)

    # Parse metadata before creating a temporary file.
    metadata_dict = {}
    if metadata:
        if len(metadata) > 16_384:
            raise HTTPException(status_code=413, detail="Metadata exceeds the size limit")
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")
        if not isinstance(metadata_dict, dict):
            raise HTTPException(status_code=400, detail="Metadata must be a JSON object")
    metadata_dict["original_file_name"] = original_name

    # Store under a server-generated name; the client filename is metadata only.
    suffix = Path(original_name).suffix.lower()
    file_path = _allocate_upload_path(settings.UPLOAD_DIR, suffix)
    try:
        with file_path.open("wb") as buffer:
            size = 0
            digest = hashlib.sha256()
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Document exceeds the upload size limit")
                digest.update(chunk)
                buffer.write(chunk)
        if not size:
            raise HTTPException(status_code=400, detail="Document is empty")
        _validate_upload_content(file_path, SUPPORTED_SUFFIXES[suffix])
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

    if settings.INGEST_DATABASE_URL:
        from backend.ingestion import objects, store

        try:
            key = request.headers.get("idempotency-key")
            if key is not None and not (1 <= len(key) <= 128):
                raise HTTPException(status_code=400, detail="Invalid Idempotency-Key")
            try:
                doc_id, created = store.reserve_document(
                    owner_sub=owner, idempotency_key=key, digest=digest.hexdigest(),
                    file_name=original_name, kind=SUPPORTED_SUFFIXES[suffix], size_bytes=size,
                    metadata=metadata_dict,
                )
            except store.AdmissionConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except store.AdmissionCapacity as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from exc
            if created:
                try:
                    objects.upload_source(file_path, f"uploads/{doc_id}", size, SUPPORTED_SUFFIXES[suffix])
                    store.mark_accepted(doc_id)
                except Exception as exc:
                    store.fail_document(doc_id, "storage_or_admission_failed")
                    raise HTTPException(status_code=503, detail="Upload storage is unavailable") from exc
            existing_state = "accepted" if created else store.get_document(doc_id, owner)["state"]
            return IngestResponse(
                task_id=doc_id, doc_id=doc_id, status=existing_state,
                status_url=f"/api/v1/ingest/documents/{doc_id}",
                message=f"Document '{original_name}' accepted for processing",
            )
        finally:
            file_path.unlink(missing_ok=True)

    if settings.ENVIRONMENT == "production":
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="Durable ingestion is not configured")

    # Queue processing task
    try:
        task = process_document.delay(str(file_path), metadata_dict)
    except Exception:
        file_path.unlink(missing_ok=True)
        raise

    return IngestResponse(
        task_id=task.id,
        status="PENDING",
        message=f"Document '{original_name}' queued for processing",
    )


@router.get("/documents/{doc_id}")
async def get_document_status(doc_id: str, request: Request):
    from backend.ingestion.auth import require_workspace
    from backend.ingestion import store

    if not settings.INGEST_DATABASE_URL:
        raise HTTPException(status_code=503, detail="Durable ingestion is not configured")
    from uuid import UUID

    try:
        UUID(doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid document ID") from exc
    owner = require_workspace(request)
    try:
        document = store.get_document(doc_id, owner)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid document ID") from exc
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    document.pop("source_key", None)
    document.pop("owner_sub", None)
    return document


@router.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str, request: Request):
    """
    Check the status of a document processing task
    """
    if settings.INGEST_DATABASE_URL:
        document = await get_document_status(task_id, request)
        return TaskStatus(
            task_id=task_id,
            status={"done": "SUCCESS", "failed": "FAILURE"}.get(document["state"], "PROGRESS"),
            result={"document_id": task_id, "validation": document["validation"]} if document["state"] == "done" else None,
            error=document["error_code"] if document["state"] == "failed" else None,
        )
    if settings.ENVIRONMENT == "production":
        raise HTTPException(status_code=503, detail="Durable ingestion is not configured")
    task = celery_app.AsyncResult(task_id)

    response = TaskStatus(
        task_id=task_id,
        status=task.status,
        result=task.result if task.status == "SUCCESS" else None,
        error=str(task.result) if task.status == "FAILURE" else None,
    )

    return response
