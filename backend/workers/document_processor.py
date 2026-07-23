"""
Document Processing Utilities
"""
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from backend.core.config import settings
from backend.graph_ingestion.lite_parser import LiteParseClient, LiteParseUnavailableError
import json
import hashlib
import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.

    Tries the self-hosted LiteParse service first, which produces
    layout-preserved, table-aware Markdown suitable for graph ingestion.
    Falls back to PyMuPDF sequentially if LiteParse is unavailable.
    """
    client = LiteParseClient()
    try:
        return client.extract_to_markdown(pdf_path)
    except LiteParseUnavailableError:
        logger.warning(
            "LiteParse unavailable — falling back to PyMuPDF for '%s'", pdf_path
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "LiteParse extraction failed (%s) — falling back to PyMuPDF for '%s'",
            exc,
            pdf_path,
        )

    # PyMuPDF fallback — sequential page extraction
    try:
        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        logger.error("PyMuPDF also failed for '%s': %s", pdf_path, e)
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
    openai_key = settings.OPENAI_API_KEY
    kw = {
        "model": settings.LLM_MODEL,
        "openai_api_key": openai_key,
        "temperature": 0
    }
    if openai_key and (openai_key.startswith("sk-or-") or "openrouter" in openai_key):
        kw["base_url"] = "https://openrouter.ai/api/v1"
    llm = ChatOpenAI(**kw)
    
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
        logger.exception("Entity and relationship extraction failed")
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
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"chunk_{index}_{content_hash}"


def generate_document_id(file_path: str) -> str:
    """
    Generate a unique ID for a document
    
    Args:
        file_path: Path to the document
        
    Returns:
        Unique ID
    """
    path_hash = hashlib.sha256(file_path.encode("utf-8")).hexdigest()
    return f"doc_{path_hash}"
