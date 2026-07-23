"""
Document Ingestion Endpoints
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from backend.app.models import IngestResponse, TaskStatus
from backend.core.config import settings
from backend.workers.celery_app import celery_app
from backend.workers.tasks import process_document
from typing import Optional
import json
import shutil
from pathlib import Path
from uuid import uuid4

router = APIRouter()


def _validate_upload_filename(filename: Optional[str]) -> str:
    """Return a display-safe basename or reject path-like upload names."""
    if not filename or "\x00" in filename:
        raise HTTPException(status_code=400, detail="A valid PDF filename is required")

    portable_name = filename.replace("\\", "/")
    basename = Path(portable_name).name
    if basename != portable_name or basename in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="Unsafe upload filename")
    if Path(basename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    return basename


def _allocate_upload_path(upload_dir: Path) -> Path:
    """Allocate a server-controlled filename beneath the configured directory."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir / f"{uuid4().hex}.pdf"


@router.post("/document", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...), metadata: Optional[str] = Form(None)
):
    """
    Ingest a document (PDF) and process it asynchronously

    The document will be:
    1. Split into chunks
    2. Embedded and stored in ChromaDB
    3. Entities extracted and stored in Neo4j
    """
    original_name = _validate_upload_filename(file.filename)

    # Parse metadata before creating a temporary file.
    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")
        if not isinstance(metadata_dict, dict):
            raise HTTPException(status_code=400, detail="Metadata must be a JSON object")
    metadata_dict["original_file_name"] = original_name

    # Store under a server-generated name; the client filename is metadata only.
    file_path = _allocate_upload_path(settings.UPLOAD_DIR)
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

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


@router.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """
    Check the status of a document processing task
    """
    task = celery_app.AsyncResult(task_id)

    response = TaskStatus(
        task_id=task_id,
        status=task.status,
        result=task.result if task.status == "SUCCESS" else None,
        error=str(task.result) if task.status == "FAILURE" else None,
    )

    return response
