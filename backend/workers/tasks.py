"""
Celery Tasks for Document Processing
"""
from celery import Task
from celery.utils.log import get_task_logger
from backend.workers.celery_app import celery_app
from backend.workers.document_processor import (
    extract_document_text,
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
import tempfile

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
    stage_callback=None,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Any]:
    """Run all async ingestion work in one event loop and close every client."""
    chroma = ChromaService()
    neo4j = None
    try:
        if stage_callback:
            stage_callback("embedding")
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

        if stage_callback:
            stage_callback("extracting")
        task.update_state(state='PROGRESS', meta={'status': 'Extracting entities...'})
        entities, relationships = await extract_entities_and_relationships(text[:10000])
        logger.info(
            "[Task %s] Extracted %d entities and %d relationships",
            task.request.id,
            len(entities),
            len(relationships),
        )

        if stage_callback:
            stage_callback("graph")
        task.update_state(state='PROGRESS', meta={'status': 'Storing in Neo4j...'})
        neo4j = Neo4jService()
        neo4j.create_indexes()
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
            raise RuntimeError(
                f"Failed to store document in Neo4j: {getattr(graph_write, 'error', None)}"
            )
        logger.info("[Task %s] Stored in Neo4j", task.request.id)
        return entities, relationships, graph_write
    finally:
        if neo4j is not None:
            neo4j.close()
            chroma.close()


def _run_persist_document(*args, **kwargs):
    """Bridge the synchronous Celery task into exactly one async boundary."""
    return asyncio.run(_persist_document(*args, **kwargs))


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
        suffix = Path(file_path).suffix.lower()
        kind = {".pdf": "pdf", ".docx": "docx", ".txt": "text"}.get(suffix)
        text = extract_document_text(file_path, kind)
        
        if not text.strip():
            raise ValueError("No text could be extracted from the document")
        if len(text) > settings.MAX_EXTRACTED_CHARACTERS:
            raise ValueError("Extracted text exceeds the configured limit")
        
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
        if len(records) > settings.MAX_DOCUMENT_CHUNKS:
            raise ValueError("Document exceeds the configured chunk limit")
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
        
        entities, relationships, graph_write = _run_persist_document(
            self,
            doc_id=doc_id,
            file_name=file_name,
            text=text,
            chunks=chunks,
            chunk_metadata=chunk_metadata,
            chunk_ids=chunk_ids,
            metadata=metadata,
        )

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

        enrichment = getattr(graph_write, "enrichment", None) or {}
        incomplete_entity_ids = list(enrichment.get("incomplete_entity_ids") or [])
        linked_entity_ids = list(enrichment.get("linked_entity_ids") or [])
        missing_vector_links = int(enrichment.get("missing_vector_links", 0) or 0)
        verified_chunk_ids = chunk_ids if missing_vector_links == 0 else []
        if enrichment.get("skipped_without_verified_context"):
            skipped = int(enrichment["skipped_without_verified_context"])
            incomplete_entity_ids.extend(
                f"entity_without_verified_context:{idx}" for idx in range(skipped)
            )

        validation = build_ingestion_validation_report(
            chunk_ids=chunk_ids,
            verified_chunk_ids=verified_chunk_ids,
            enriched_entity_ids=linked_entity_ids,
            incomplete_entity_ids=incomplete_entity_ids,
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
            "enrichment": enrichment,
            "validation": validation,
            "status": "success" if validation.get("complete") else "incomplete",
        }
        
    except Exception as e:
        logger.exception("[Task %s] Document processing failed", self.request.id)
        # Retry the task
        raise self.retry(exc=e)


@celery_app.task(base=CallbackTask, bind=True, name="workers.tasks.process_stored_document",
                 max_retries=3, default_retry_delay=60)
def process_stored_document(self, doc_id: str) -> Dict[str, Any]:
    """Replay-safe durable job; only object keys and IDs cross the broker."""
    from backend.ingestion import objects, store

    document = store.claim_document(doc_id)
    if document is None:
        return {"document_id": doc_id, "status": "already_claimed_or_terminal"}
    suffix = {"pdf": ".pdf", "docx": ".docx", "text": ".txt"}[document["kind"]]
    with tempfile.NamedTemporaryFile(prefix="kinegraph-ingest-", suffix=suffix, delete=False) as temporary:
        file_path = Path(temporary.name)
    try:
        objects.download_source(document["source_key"], file_path, document["content_sha256"])
        store.set_stage(doc_id, "parsing")
        text = extract_document_text(str(file_path), document["kind"])
        if not text.strip():
            raise ValueError("Document contains no extractable text")
        if len(text) > settings.MAX_EXTRACTED_CHARACTERS:
            raise ValueError("Extracted text exceeds the configured limit")
        store.set_stage(doc_id, "chunking")
        records = build_document_chunks(text, document_id=doc_id)
        if len(records) > settings.MAX_DOCUMENT_CHUNKS:
            raise ValueError("Document exceeds the configured chunk limit")
        chunks = [record.text for record in records]
        chunk_ids = [record.chunk_id for record in records]
        metadata = {**document["metadata"], "original_file_name": document["file_name"]}
        chunk_metadata = [
            {**record.to_metadata(), "file_name": document["file_name"],
             "total_chunks": len(records), **metadata}
            for record in records
        ]
        entities, relationships, graph_write = _run_persist_document(
            self, doc_id=doc_id, file_name=document["file_name"], text=text,
            chunks=chunks, chunk_metadata=chunk_metadata, chunk_ids=chunk_ids,
            metadata=metadata, stage_callback=lambda stage: store.set_stage(doc_id, stage),
        )
        enrichment = getattr(graph_write, "enrichment", None) or {}
        incomplete = list(enrichment.get("incomplete_entity_ids") or [])
        linked = list(enrichment.get("linked_entity_ids") or [])
        missing = int(enrichment.get("missing_vector_links", 0) or 0)
        validation = build_ingestion_validation_report(
            chunk_ids=chunk_ids, verified_chunk_ids=chunk_ids if missing == 0 else [],
            enriched_entity_ids=linked, incomplete_entity_ids=incomplete,
        )
        complete = bool(validation.get("complete"))
        store.set_stage(doc_id, "finalizing")
        store.finish_document(doc_id, validation=validation, success=complete,
                              error_code=None if complete else "index_incomplete")
        if complete:
            try:
                objects.delete_source(document["source_key"])
                store.clear_source(doc_id)
            except Exception:
                logger.exception("Source cleanup pending for document %s", doc_id)
        return {"document_id": doc_id, "status": "done" if complete else "failed",
                "total_chunks": len(chunks), "entities_count": len(entities),
                "relationships_count": len(relationships), "validation": validation}
    except ValueError:
        logger.exception("Permanent ingestion failure for %s", doc_id)
        store.fail_document(doc_id, "invalid_document")
        raise
    except Exception as exc:
        logger.exception("Retryable ingestion failure for %s", doc_id)
        if self.request.retries >= self.max_retries:
            store.fail_document(doc_id, "processing_failed")
            raise
        store.release_retry(doc_id)
        raise self.retry(exc=exc)
    finally:
        file_path.unlink(missing_ok=True)


@celery_app.task(name="workers.tasks.dispatch_ingest_outbox")
def dispatch_ingest_outbox() -> int:
    from backend.ingestion import store

    dispatched = 0
    for event in store.pending_dispatches():
        try:
            celery_app.send_task("workers.tasks.process_stored_document",
                                 args=[str(event["doc_id"])], queue="ingest")
        except Exception:
            store.record_dispatch(str(event["event_id"]), False)
            logger.exception("Could not dispatch ingestion event %s", event["event_id"])
        else:
            store.record_dispatch(str(event["event_id"]), True)
            dispatched += 1
    return dispatched


@celery_app.task(name="workers.tasks.reconcile_ingest")
def reconcile_ingest() -> Dict[str, int]:
    from backend.ingestion import objects, store

    requeued = store.stale_processing()
    requeued.extend(store.requeue_unstarted())
    cleaned = 0
    for document in store.expired_sources():
        try:
            objects.delete_source(document["source_key"])
            store.clear_source(str(document["doc_id"]))
            if document.get("state") == "uploading":
                store.fail_document(str(document["doc_id"]), "upload_expired")
            cleaned += 1
        except Exception:
            logger.exception("Could not expire source for %s", document["doc_id"])
    return {"requeued": len(requeued), "cleaned": cleaned}


@celery_app.task(name="workers.tasks.health_check")
def health_check() -> Dict[str, str]:
    """
    Simple health check task for monitoring worker status
    """
    return {"status": "healthy", "worker": "operational"}
