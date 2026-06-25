"""
Document Processing Utilities
"""
from typing import List, Dict, Any, Tuple
from billiard.pool import Pool
import fitz  # PyMuPDF
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from backend.core.config import settings
import json
import hashlib


def _extract_page_range(args: Tuple[str, int, int]) -> Tuple[int, str]:
    """Extract text from a range of pages in a PDF document (opened inside the worker process)."""
    pdf_path, start_page, end_page = args
    try:
        text_parts = []
        with fitz.open(pdf_path) as doc:
            for idx in range(start_page, end_page):
                text_parts.append(doc.load_page(idx).get_text())
        return start_page, "\n".join(text_parts)
    except Exception as e:
        print(f"Error extracting page range {start_page}-{end_page}: {e}")
        return start_page, ""


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file in parallel using billiard process Pool and PyMuPDF.
    Splits the PDF into batches equal to CPU core count to utilize multiple cores safely.
    """
    try:
        # Get total page count
        with fitz.open(pdf_path) as doc:
            num_pages = len(doc)
            
        if num_pages == 0:
            return ""

        # If it's a very small document, just parse sequentially to avoid process overhead
        if num_pages <= 10:
            with fitz.open(pdf_path) as doc:
                return "\n".join(page.get_text() for page in doc)

        # Split page range into CPU core count batches
        cpu_count = os.cpu_count() or 4
        batch_size = (num_pages + cpu_count - 1) // cpu_count
        
        batches = []
        for i in range(cpu_count):
            start = i * batch_size
            end = min(start + batch_size, num_pages)
            if start < end:
                batches.append((pdf_path, start, end))

        # Run extraction in parallel using process pool
        with Pool(processes=len(batches)) as pool:
            results = pool.map(_extract_page_range, batches)
            
        # Sort results to ensure pages remain in order
        results.sort(key=lambda x: x[0])
        
        return "\n".join(r[1] for r in results)

    except Exception as e:
        print(f"Error during parallel PDF text extraction: {e}")
        # Fallback to sequential extraction
        try:
            with fitz.open(pdf_path) as doc:
                return "\n".join(page.get_text() for page in doc)
        except Exception as e_fallback:
            print(f"Fallback PDF extraction failed: {e_fallback}")
            return ""


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Split text into chunks
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    return chunks


async def extract_entities_and_relationships(
    text: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract entities and relationships from text using LLM
    
    Args:
        text: Text to analyze
        
    Returns:
        Tuple of (entities, relationships)
    """
    llm = ChatOpenAI(
        model="gpt-4",
        openai_api_key=settings.OPENAI_API_KEY,
        temperature=0
    )
    
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
Extract entities and relationships from the following text.

Text: {text}

Return a JSON object with two arrays:
1. "entities": Array of objects with "name" and "type" (e.g., Person, Organization, Location, Concept)
2. "relationships": Array of objects with "source", "target", and "type" (describing the relationship)

Example:
{{
  "entities": [
    {{"name": "Albert Einstein", "type": "Person"}},
    {{"name": "Theory of Relativity", "type": "Concept"}}
  ],
  "relationships": [
    {{"source": "Albert Einstein", "target": "Theory of Relativity", "type": "DEVELOPED"}}
  ]
}}

Return ONLY the JSON object, no additional text.

JSON:
"""
    )
    
    try:
        response = await llm.ainvoke(prompt.format(text=text[:4000]))  # Limit text length
        result = json.loads(response.content)
        
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])
        
        return entities, relationships
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return [], []


def generate_chunk_id(content: str, index: int) -> str:
    """
    Generate a unique ID for a chunk
    
    Args:
        content: Chunk content
        index: Chunk index
        
    Returns:
        Unique ID
    """
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return f"chunk_{index}_{content_hash[:8]}"


def generate_document_id(file_path: str) -> str:
    """
    Generate a unique ID for a document
    
    Args:
        file_path: Path to the document
        
    Returns:
        Unique ID
    """
    path_hash = hashlib.md5(file_path.encode()).hexdigest()
    return f"doc_{path_hash}"
