"""
Celery Tasks for Document Processing
"""
from celery import Task
from workers.celery_app import celery_app
from workers.document_processor import (
    extract_text_from_pdf,
    chunk_text,
    extract_entities_and_relationships,
    generate_chunk_id,
    generate_document_id
)
from services.chroma_service import ChromaService
from services.neo4j_service import Neo4jService
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
        
        # Update state
        self.update_state(state='PROGRESS', meta={'status': 'Storing in ChromaDB...'})
        
        # Store in ChromaDB
        chroma = ChromaService()
        success = asyncio.run(
            chroma.add_documents(
                texts=chunks,
                metadatas=chunk_metadata,
                ids=chunk_ids
            )
        )
        
        if not success:
            raise Exception("Failed to store documents in ChromaDB")
        
        print(f"[Task {self.request.id}] Stored in ChromaDB")
        
        # Update state
        self.update_state(state='PROGRESS', meta={'status': 'Extracting entities...'})
        
        # Extract entities from full text (or from a summary for large documents)
        text_sample = text[:10000]  # Use first 10k chars for entity extraction
        entities, relationships = asyncio.run(
            extract_entities_and_relationships(text_sample)
        )
        
        print(f"[Task {self.request.id}] Extracted {len(entities)} entities and {len(relationships)} relationships")
        
        # Update state
        self.update_state(state='PROGRESS', meta={'status': 'Storing in Neo4j...'})
        
        # Store in Neo4j
        neo4j = Neo4jService()
        try:
            success = asyncio.run(
                neo4j.add_document_graph(
                    doc_id=doc_id,
                    content=text[:5000],  # Store summary in graph
                    metadata={
                        "file_name": file_name,
                        "total_chunks": len(chunks),
                        **metadata
                    },
                    entities=entities,
                    relationships=relationships
                )
            )
            
            if not success:
                raise Exception("Failed to store document in Neo4j")
            
            print(f"[Task {self.request.id}] Stored in Neo4j")
        finally:
            neo4j.close()
        
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
