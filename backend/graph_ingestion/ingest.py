import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from neo4j import GraphDatabase

from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.schema import TransformComponent

from backend.core.config import settings
from backend.services.chroma_service import ChromaService
from backend.graph_ingestion.embedding_wrapper import LangChainEmbeddingWrapper
from backend.graph_ingestion.stores import get_neo4j_graph_store, get_chroma_vector_store, get_llm
from backend.graph_ingestion.schema import OntologySchema
from backend.graph_ingestion.extractors import get_extractor_stack
from backend.graph_ingestion.dedup import EntityResolver

logger = logging.getLogger(__name__)

class IdempotentGraphIngester:
    """
    Ingests files/folders into PropertyGraphIndex idempotently.
    Uses content hashing to skip already-ingested chunks and applies entity deduplication.
    """
    def __init__(self):
        self.chroma_service = ChromaService()
        self.embed_model = LangChainEmbeddingWrapper(self.chroma_service.embeddings)
        self.graph_store = get_neo4j_graph_store()
        self.vector_store = get_chroma_vector_store()
        self.schema = OntologySchema("config/ontology_schema.yaml")
        self.llm = get_llm()
        self.resolver = EntityResolver(self.embed_model)
        
        # Initialize Neo4j driver directly for quick chunk existence checks
        self.neo4j_driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

    def close(self):
        if self.neo4j_driver:
            self.neo4j_driver.close()

    def compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of a string."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def is_chunk_ingested(self, chunk_hash: str) -> bool:
        """Check if a chunk with the given hash is already in the graph database."""
        try:
            with self.neo4j_driver.session() as session:
                query = "MATCH (n:`__Chunk__`) WHERE n.chunk_hash = $hash RETURN count(n) as count"
                res = session.run(query, hash=chunk_hash)
                record = res.single()
                return record["count"] > 0 if record else False
        except Exception as e:
            logger.error("Error checking chunk ingestion status in Neo4j: %s", e)
            return False

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks using standard splitter."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)

    def ingest_file(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Ingests a single file (PDF or Markdown/text) into the PropertyGraphIndex.
        Returns a summary dictionary of what was ingested.
        """
        if metadata is None:
            metadata = {}
            
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # 1. Read file content
        content = ""
        if file_path_obj.suffix.lower() == ".pdf":
            from backend.workers.document_processor import extract_text_from_pdf
            content = extract_text_from_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
        if not content.strip():
            logger.warning("Empty file skipped: %s", file_path)
            return {"status": "skipped", "reason": "empty content"}

        # 2. Chunk text and filter by content hash
        chunks = self.chunk_text(content)
        nodes_to_ingest = []
        skipped_chunks = 0

        for idx, chunk_text_content in enumerate(chunks):
            chunk_hash = self.compute_hash(chunk_text_content)
            if self.is_chunk_ingested(chunk_hash):
                skipped_chunks += 1
                continue
                
            # Create LlamaIndex TextNode
            node_meta = {
                "file_name": file_path_obj.name,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "chunk_hash": chunk_hash,
                **metadata
            }
            node = TextNode(text=chunk_text_content, metadata=node_meta)
            nodes_to_ingest.append(node)

        if not nodes_to_ingest:
            logger.info("All %d chunks for file '%s' already ingested.", len(chunks), file_path_obj.name)
            return {
                "file_name": file_path_obj.name,
                "status": "skipped",
                "total_chunks": len(chunks),
                "skipped_chunks": skipped_chunks,
                "ingested_chunks": 0
            }

        logger.info(
            "Ingesting %d / %d chunks for file '%s'...",
            len(nodes_to_ingest), len(chunks), file_path_obj.name
        )

        # 3. Extract entities and relationships using stacked extractors
        extractors = get_extractor_stack(self.schema, self.llm)
        processed_nodes = list(nodes_to_ingest)
        for extractor in extractors:
            processed_nodes = extractor(processed_nodes)

        # 4. Resolve entity aliases (Deduplication) in metadata
        entity_names = set()
        for node in processed_nodes:
            if "nodes" in node.metadata:
                for ent in node.metadata["nodes"]:
                    entity_names.add(ent.name)
                    
        resolution_map = self.resolver.resolve_entities(list(entity_names))
        
        # Apply name mapping to entities and relations
        for node in processed_nodes:
            if "nodes" in node.metadata:
                for ent in node.metadata["nodes"]:
                    ent.name = resolution_map.get(ent.name, ent.name)
            if "relations" in node.metadata:
                for rel in node.metadata["relations"]:
                    rel.source_id = resolution_map.get(rel.source_id, rel.source_id)
                    rel.target_id = resolution_map.get(rel.target_id, rel.target_id)

        # 5. Build/Update PropertyGraphIndex
        # LlamaIndex writes nodes to the graph store and vector store
        PropertyGraphIndex(
            nodes=processed_nodes,
            property_graph_store=self.graph_store,
            vector_store=self.vector_store,
            embed_model=self.embed_model,
            show_progress=True
        )

        return {
            "file_name": file_path_obj.name,
            "status": "success",
            "total_chunks": len(chunks),
            "skipped_chunks": skipped_chunks,
            "ingested_chunks": len(processed_nodes)
        }

    def ingest_directory(self, dir_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Ingests all supported files from a directory."""
        dir_path_obj = Path(dir_path)
        if not dir_path_obj.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
            
        results = []
        for file_path in dir_path_obj.iterdir():
            if file_path.suffix.lower() in [".pdf", ".md", ".txt"]:
                logger.info("Processing directory file: %s", file_path.name)
                try:
                    res = self.ingest_file(str(file_path), metadata)
                    results.append(res)
                except Exception as e:
                    logger.error("Failed to ingest file %s: %s", file_path.name, e)
                    results.append({"file_name": file_path.name, "status": "failed", "error": str(e)})
        return results
