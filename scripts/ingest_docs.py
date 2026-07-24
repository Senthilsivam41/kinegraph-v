import os
import sys
import asyncio
from dotenv import load_dotenv

# Load env variables
load_dotenv(os.path.abspath('.env'))

sys.path.insert(0, os.path.abspath('.'))

from backend.core.config import settings
from backend.services.chroma_service import ChromaService
from backend.services.neo4j_service import Neo4jService
from backend.workers.document_processor import (
    build_document_chunks,
    extract_entities_and_relationships,
)

DOCS_DIR = "docs"

async def ingest_file(filename: str):
    filepath = os.path.join(DOCS_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"\nProcessing '{filename}' ({len(content)} chars)...")
    doc_id = f"doc_{filename.replace('.', '_')}"
    
    # 1. Chunk document (ADR-002 contract; adaptive when enabled)
    records = build_document_chunks(content, document_id=doc_id)
    chunks = [record.text for record in records]
    print(
        f"  Created {len(chunks)} chunks "
        f"(adaptive={settings.ADAPTIVE_CHUNKING_ENABLED})."
    )
    
    chunk_metadata = []
    chunk_ids = []
    for record in records:
        chunk_ids.append(record.chunk_id)
        chunk_meta = {
            **record.to_metadata(),
            "file_name": filename,
            "total_chunks": len(records),
        }
        chunk_metadata.append(chunk_meta)
        
    # 2. Store in ChromaDB
    print("  Storing in ChromaDB...")
    chroma = ChromaService()
    success = await chroma.add_documents(
        texts=chunks,
        metadatas=chunk_metadata,
        ids=chunk_ids
    )
    if success:
        print("  Successfully stored in ChromaDB.")
    else:
        print("  Failed to store in ChromaDB.")
        
    # 3. Store in Vectorless Cache
    try:
        from backend.services.vectorless_service import VectorlessService
        vectorless = VectorlessService()
        vectorless.save_document_chunks(
            doc_id=doc_id,
            file_name=filename,
            chunks=chunks,
            metadatas=chunk_metadata,
            ids=chunk_ids
        )
        print("  Stored in Local Lexical Cache (Vectorless RAG).")
    except Exception as ve:
        print(f"  Warning: Vectorless storage failed: {ve}")

    # 4. Extract entities and relationships
    print("  Extracting entities & relationships...")
    text_sample = content[:8000] # Use first 8k chars
    try:
        entities, relationships = await extract_entities_and_relationships(text_sample)
        print(f"  Extracted {len(entities)} entities and {len(relationships)} relationships.")
        
        # 5. Store in Neo4j
        print("  Storing in Neo4j...")
        neo4j = Neo4jService()
        success = await neo4j.add_document_graph(
            doc_id=doc_id,
            content=content[:3000],
            metadata={
                "file_name": filename,
                "total_chunks": len(chunks)
            },
            entities=entities,
            relationships=relationships,
            chunks=chunks,
            chunk_ids=chunk_ids,
        )
        if success:
            print("  Successfully stored in Neo4j.")
        else:
            print("  Failed to store in Neo4j.")
        neo4j.close()
    except Exception as ne:
        print(f"  Error extracting or storing graph entities: {ne}")

async def main():
    if not os.path.exists(DOCS_DIR):
        print(f"Error: Directory '{DOCS_DIR}' not found.")
        return
        
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".md")]
    print(f"Found {len(files)} markdown documents to ingest.")
    
    for filename in files:
        await ingest_file(filename)
        
    print("\nIngestion complete!")

if __name__ == "__main__":
    asyncio.run(main())
