"""
Document Processing Utilities
"""
from pypdf import PdfReader
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from core.config import settings
import json
import hashlib


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text
    """
    reader = PdfReader(pdf_path)
    text = ""
    
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    return text


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
