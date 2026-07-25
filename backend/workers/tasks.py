"""
Celery Tasks for Document Processing
"""
from celery import Task
from celery.utils.log import get_task_logger
from backend.workers.celery_app import celery_app
from backend.workers.document_processor import (
    extract_text_from_pdf,
    build_document_chunks,
    extract_entities_and_relationships,
    generate_document_id
)
from backend.graph_ingestion.adaptive_chunking import (
    CHUNK_POLICY_VERSION,
    build_ingestion_validation_report,
)
from backend.core.config import settings
from backend.services.chroma_service import ChromaService
from backend.services.neo4j_service import Neo4jService
from backend.services.vectorless_service import VectorlessService
from typing import Dict, Any
from pathlib import Path
import asyncio

logger = get_task_logger(__name__)


def _save_vectorless_document(**kwargs) -> bool:
    """Persist the vectorless cache outside the ingestion event loop."""
    return VectorlessService().save_document_chunks(**kwargs)


class CallbackTask(Task):
    """Base task with callbacks"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error("Task %s failed: %s", task_id, exc)
        super().on_failure(exc, task_id, args, kwargs, einfo)
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        logger.info("Task %s succeeded", task_id)
        super().on_success(retval, task_id, args, kwargs)


async def _persist_document(
    task: Task,
    *,
    doc_id: str,
    file_name: str,
    text: str,
    chunks: list[str],
    chunk_metadata: list[Dict[str, Any]],
    chunk_ids: list[str],
    metadata: Dict[str, Any],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Run all async ingestion work in one event loop and close every client."""
    chroma = ChromaService()
    neo4j = None
    try:
        task.update_state(state='PROGRESS', meta={'status': 'Storing in ChromaDB...'})
        success = await chroma.add_documents(
            texts=chunks,
            metadatas=chunk_metadata,
            ids=chunk_ids,
        )
        if not success:
            raise RuntimeError("Failed to store documents in ChromaDB")
        logger.info("[Task %s] Stored in ChromaDB", task.request.id)

        try:
            vectorless_saved = await asyncio.to_thread(
                _save_vectorless_document,
                doc_id=doc_id,
                file_name=file_name,
                chunks=chunks,
                metadatas=chunk_metadata,
                ids=chunk_ids,
            )
            if vectorless_saved:
                logger.info(
                    "[Task %s] Stored chunks and raw text for Vectorless RAG",
                    task.request.id,
                )
            else:
                logger.warning(
                    "[Task %s] Vectorless persistence returned failure",
                    task.request.id,
                )
        except Exception as exc:
            logger.exception(
                "[Task %s] Failed to save chunks for Vectorless RAG",
                task.request.id,
            )

        task.update_state(state='PROGRESS', meta={'status': 'Extracting entities...'})
        entities, relationships = await extract_entities_and_relationships(text[:10000])
        logger.info(
            "[Task %s] Extracted %d entities and %d relationships",
            task.request.id,
            len(entities),
            len(relationships),
        )

        task.update_state(state='PROGRESS', meta={'status': 'Storing in Neo4j...'})
        neo4j = Neo4jService()
        graph_write = await neo4j.add_document_graph(
            doc_id=doc_id,
            content=text[:5000],
            metadata={
                "file_name": file_name,
                "total_chunks": len(chunks),
                **metadata,
            },
            entities=entities,
            relationships=relationships,
            chunks=chunks,
            chunk_ids=chunk_ids,
        )
        if not graph_write:
            raise RuntimeError("Failed to store document in Neo4j")
        logger.info("[Task %s] Stored in Neo4j", task.request.id)
        return entities, relationships
    finally:
        if neo4j is not None:
            neo4j.close()
        chroma.close()


@celery_app.task(
    base=CallbackTask,
    bind=True,
    name="workers.tasks.process_document",
    max_retries=3,
    default_retry_delay=60
)
def process_document(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a document: extract text, chunk, embed, extract entities
    
    Args:
        file_path: Path to the PDF file
        metadata: Document metadata
        
    Returns:
        Processing results
    """
    try:
        logger.info("[Task %s] Processing document: %s", self.request.id, file_path)
        
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Extracting text...'})
        
        # Extract text from PDF
        text = extract_text_from_pdf(file_path)
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        # Update state
        self.update_state(state='PROGRESS', meta={'status': 'Chunking text...'})

        # Generate document ID before chunking so records carry provenance.
        doc_id = generate_document_id(file_path)
        file_name = str(metadata.get("original_file_name") or Path(file_path).name)

        records = build_document_chunks(
            text,
            document_id=doc_id,
            adaptive_enabled=bool(
                metadata.get("adaptive_chunking", settings.ADAPTIVE_CHUNKING_ENABLED)
            ),
        )
        chunks = [record.text for record in records]
        logger.info(
            "[Task %s] Created %d chunks (policy=%s adaptive=%s)",
            self.request.id,
            len(chunks),
            CHUNK_POLICY_VERSION,
            settings.ADAPTIVE_CHUNKING_ENABLED,
        )

        chunk_ids = [record.chunk_id for record in records]
        chunk_metadata = []
        for record in records:
            chunk_meta = {
                **record.to_metadata(),
                "file_name": file_name,
                "total_chunks": len(records),
                **metadata,
            }
            chunk_metadata.append(chunk_meta)
        
        entities, relationships = asyncio.run(_persist_document(
            self,
            doc_id=doc_id,
            file_name=file_name,
            text=text,
            chunks=chunks,
            chunk_metadata=chunk_metadata,
            chunk_ids=chunk_ids,
            metadata=metadata,
        ))

        # Clean up uploaded file
        try:
            Path(file_path).unlink()
            logger.info("[Task %s] Cleaned up file: %s", self.request.id, file_path)
        except Exception as e:
            logger.warning(
                "[Task %s] Could not delete file %s: %s",
                self.request.id,
                file_path,
                e,
            )
        
        validation = build_ingestion_validation_report(
            chunk_ids=chunk_ids,
            verified_chunk_ids=chunk_ids,
            enriched_entity_ids=[e.get("name") for e in entities if e.get("name")],
            incomplete_entity_ids=[],
        )

        # Return results
        return {
            "document_id": doc_id,
            "file_name": file_name,
            "total_chunks": len(chunks),
            "entities_count": len(entities),
            "relationships_count": len(relationships),
            "chunk_policy_version": CHUNK_POLICY_VERSION,
            "adaptive_chunking": bool(
                metadata.get("adaptive_chunking", settings.ADAPTIVE_CHUNKING_ENABLED)
            ),
            "validation": validation,
            "status": "success"
        }
        
    except Exception as e:
        logger.exception("[Task %s] Document processing failed", self.request.id)
        # Retry the task
        raise self.retry(exc=e)


@celery_app.task(name="workers.tasks.health_check")
def health_check() -> Dict[str, str]:
    """
    Simple health check task for monitoring worker status
    """
    return {"status": "healthy", "worker": "operational"}
