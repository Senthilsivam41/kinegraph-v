"""
Document Ingestion Endpoints
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from app.models import IngestResponse, TaskStatus
from workers.celery_app import celery_app
from workers.tasks import process_document
from typing import Optional
import json
import shutil
from pathlib import Path

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


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
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save file temporarily
    file_path = UPLOAD_DIR / file.filename
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

    # Parse metadata
    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    # Queue processing task
    task = process_document.delay(str(file_path), metadata_dict)

    return IngestResponse(
        task_id=task.id,
        status="PENDING",
        message=f"Document '{file.filename}' queued for processing",
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
