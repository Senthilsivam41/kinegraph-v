"""
Celery Tasks for Document Processing
"""
from celery import Task
from backend.workers.celery_app import celery_app
from backend.workers.document_processor import (
    extract_text_from_pdf,
    chunk_text,
    extract_entities_and_relationships,
    generate_chunk_id,
    generate_document_id
)
from backend.services.chroma_service import ChromaService
from backend.services.neo4j_service import Neo4jService
from typing import Dict, Any
from pathlib import Path
import asyncio


class CallbackTask(Task):
    """Base task with callbacks"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        print(f"Task {task_id} failed: {exc}")
        super().on_failure(exc, task_id, args, kwargs, einfo)
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        print(f"Task {task_id} succeeded")
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
        print(f"[Task {task.request.id}] Stored in ChromaDB")

        try:
            from backend.services.vectorless_service import VectorlessService
            VectorlessService().save_document_chunks(
                doc_id=doc_id,
                file_name=file_name,
                chunks=chunks,
                metadatas=chunk_metadata,
                ids=chunk_ids,
            )
            print(f"[Task {task.request.id}] Stored chunks and raw text for Vectorless RAG")
        except Exception as exc:
            print(f"[Task {task.request.id}] Error saving chunks for Vectorless RAG: {exc}")

        task.update_state(state='PROGRESS', meta={'status': 'Extracting entities...'})
        entities, relationships = await extract_entities_and_relationships(text[:10000])
        print(
            f"[Task {task.request.id}] Extracted {len(entities)} entities "
            f"and {len(relationships)} relationships"
        )

        task.update_state(state='PROGRESS', meta={'status': 'Storing in Neo4j...'})
        neo4j = Neo4jService()
        success = await neo4j.add_document_graph(
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
        if not success:
            raise RuntimeError("Failed to store document in Neo4j")
        print(f"[Task {task.request.id}] Stored in Neo4j")
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
        print(f"[Task {self.request.id}] Processing document: {file_path}")
        
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Extracting text...'})
        
        # Extract text from PDF
        text = extract_text_from_pdf(file_path)
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        # Update state
        self.update_state(state='PROGRESS', meta={'status': 'Chunking text...'})
        
        # Chunk the text
        chunks = chunk_text(text)
        
        print(f"[Task {self.request.id}] Created {len(chunks)} chunks")
        
        # Generate document ID
        doc_id = generate_document_id(file_path)
        
        # Prepare metadata for chunks
        file_name = Path(file_path).name
        chunk_metadata = []
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = generate_chunk_id(chunk, i)
            chunk_ids.append(chunk_id)
            
            chunk_meta = {
                "document_id": doc_id,
                "chunk_index": i,
                "file_name": file_name,
                "total_chunks": len(chunks),
                **metadata
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
            print(f"[Task {self.request.id}] Cleaned up file: {file_path}")
        except Exception as e:
            print(f"[Task {self.request.id}] Could not delete file: {e}")
        
        # Return results
        return {
            "document_id": doc_id,
            "file_name": file_name,
            "total_chunks": len(chunks),
            "entities_count": len(entities),
            "relationships_count": len(relationships),
            "status": "success"
        }
        
    except Exception as e:
        print(f"[Task {self.request.id}] Error: {e}")
        # Retry the task
        raise self.retry(exc=e)


@celery_app.task(name="workers.tasks.health_check")
def health_check() -> Dict[str, str]:
    """
    Simple health check task for monitoring worker status
    """
    return {"status": "healthy", "worker": "operational"}
