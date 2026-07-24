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
from backend.graph_ingestion.enrichment import NodeEnricher
from backend.graph_ingestion.adaptive_chunking import (
    CHUNK_POLICY_VERSION,
    ChunkRecord,
    build_ingestion_validation_report,
    chunk_document,
)
from backend.workers.document_processor import generate_document_id

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

    def _enrich_ingested_nodes(self, nodes: List[TextNode]) -> Dict[str, Any]:
        """Enrich only entities touched by this ingestion and verify Chroma links."""
        chunk_ids = [node.node_id for node in nodes]
        try:
            result = NodeEnricher(
                driver=self.neo4j_driver,
                chroma_client=self.chroma_service.client,
            ).enrich(chunk_ids=chunk_ids)
            result["status"] = "success" if result["complete"] else "incomplete"
            incomplete = []
            skipped = int(result.get("skipped_without_verified_context", 0) or 0)
            if skipped:
                incomplete = [f"entity_without_verified_context:{idx}" for idx in range(skipped)]
            missing_count = int(result.get("missing_vector_links", 0) or 0)
            verified_ids = chunk_ids if missing_count == 0 else []
            result["validation"] = build_ingestion_validation_report(
                chunk_ids=chunk_ids,
                verified_chunk_ids=verified_ids,
                incomplete_entity_ids=incomplete,
            )
            return result
        except Exception as exc:
            logger.exception("Automatic v3 node enrichment failed")
            return {
                "enriched": 0,
                "verified_vector_links": 0,
                "missing_vector_links": len(chunk_ids),
                "complete": False,
                "status": "failed",
                "error": str(exc),
                "validation": build_ingestion_validation_report(
                    chunk_ids=chunk_ids,
                    verified_chunk_ids=[],
                    incomplete_entity_ids=["enrichment_failed"],
                ),
            }

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
                # LlamaIndex versions have used both labels; accept either so
                # re-ingestion cannot silently duplicate a known chunk.
                query = """
                    MATCH (n) WHERE (n:Chunk OR n:`__Chunk__`) AND n.chunk_hash = $hash
                    RETURN count(n) as count
                """
                res = session.run(query, hash=chunk_hash)
                record = res.single()
                return record["count"] > 0 if record else False
        except Exception as e:
            logger.error("Error checking chunk ingestion status in Neo4j: %s", e)
            return False

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks using the recursive fallback policy."""
        records = chunk_document(
            text,
            document_id="doc_legacy",
            adaptive_enabled=False,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return [record.text for record in records]

    def build_chunks(
        self,
        text: str,
        *,
        document_id: str,
        adaptive_enabled: bool | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> List[ChunkRecord]:
        """Build ADR-002 chunk records (adaptive when enabled)."""
        enabled = (
            settings.ADAPTIVE_CHUNKING_ENABLED
            if adaptive_enabled is None
            else adaptive_enabled
        )
        if not enabled:
            # Preserve testability: callers/tests may patch chunk_text.
            texts = self.chunk_text(
                text,
                chunk_size=settings.CHUNK_SIZE if chunk_size is None else chunk_size,
                chunk_overlap=settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap,
            )
            return [
                ChunkRecord(
                    text=part,
                    chunk_type="recursive",
                    chunk_id=f"chunk_{idx}_{self.compute_hash(part)}",
                    ordinal=idx,
                    document_id=document_id,
                    policy_version=CHUNK_POLICY_VERSION,
                    tokenizer_version="recursive-character-v1",
                    overlap=settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap,
                    boundary_reason="unstructured_text",
                )
                for idx, part in enumerate(texts)
            ]
        return chunk_document(
            text,
            document_id=document_id,
            adaptive_enabled=True,
            chunk_size=settings.CHUNK_SIZE if chunk_size is None else chunk_size,
            chunk_overlap=settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap,
        )

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

        doc_id = generate_document_id(file_path)
        records = self.build_chunks(content, document_id=doc_id)
        nodes_to_ingest = []
        skipped_chunks = 0

        for record in records:
            chunk_hash = record.content_hash()
            if self.is_chunk_ingested(chunk_hash):
                skipped_chunks += 1
                continue
                
            node_meta = {
                "file_name": file_path_obj.name,
                "total_chunks": len(records),
                **record.to_metadata(),
                **metadata,
            }
            node = TextNode(text=record.text, metadata=node_meta, id_=record.chunk_id)
            nodes_to_ingest.append(node)

        if not nodes_to_ingest:
            logger.info("All %d chunks for file '%s' already ingested.", len(records), file_path_obj.name)
            return {
                "file_name": file_path_obj.name,
                "status": "skipped",
                "total_chunks": len(records),
                "skipped_chunks": skipped_chunks,
                "ingested_chunks": 0,
                "chunk_policy_version": CHUNK_POLICY_VERSION,
            }

        logger.info(
            "Ingesting %d / %d chunks for file '%s'...",
            len(nodes_to_ingest), len(records), file_path_obj.name
        )

        # 3. Extract and resolve entities using PropertyGraphIndex pipeline
        extractors = get_extractor_stack(self.schema, self.llm, self.resolver)

        # 4. Build/Update PropertyGraphIndex
        # LlamaIndex writes nodes to the graph store and vector store
        PropertyGraphIndex(
            nodes=nodes_to_ingest,
            property_graph_store=self.graph_store,
            vector_store=self.vector_store,
            embed_model=self.embed_model,
            llm=self.llm,
            kg_extractors=extractors,
            show_progress=True
        )

        enrichment = self._enrich_ingested_nodes(nodes_to_ingest)

        return {
            "file_name": file_path_obj.name,
            "status": "success",
            "total_chunks": len(records),
            "skipped_chunks": skipped_chunks,
            "ingested_chunks": len(nodes_to_ingest),
            "chunk_policy_version": CHUNK_POLICY_VERSION,
            "adaptive_chunking": settings.ADAPTIVE_CHUNKING_ENABLED,
            "enrichment": enrichment,
        }

    def ingest_directory(self, dir_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Ingests all supported files from a directory in a single batch to avoid
        event loop deadlocks and enable global entity resolution.
        """
        dir_path_obj = Path(dir_path)
        if not dir_path_obj.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
            
        if metadata is None:
            metadata = {}

        all_nodes_to_ingest = []
        skipped_chunks = 0
        total_chunks = 0
        file_summaries = []

        # 1. Collect and chunk all files, checking content hashes
        for file_path in dir_path_obj.iterdir():
            if file_path.suffix.lower() not in [".pdf", ".md", ".txt"]:
                continue
                
            content = ""
            if file_path.suffix.lower() == ".pdf":
                from backend.workers.document_processor import extract_text_from_pdf
                content = extract_text_from_pdf(str(file_path))
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
            if not content.strip():
                continue
                
            doc_id = generate_document_id(str(file_path))
            records = self.build_chunks(content, document_id=doc_id)
            total_chunks += len(records)
            file_nodes = []
            
            for record in records:
                chunk_hash = record.content_hash()
                if self.is_chunk_ingested(chunk_hash):
                    skipped_chunks += 1
                    continue
                    
                node_meta = {
                    "file_name": file_path.name,
                    "total_chunks": len(records),
                    **record.to_metadata(),
                    **metadata,
                }
                node = TextNode(text=record.text, metadata=node_meta, id_=record.chunk_id)
                file_nodes.append(node)
                
            if file_nodes:
                all_nodes_to_ingest.extend(file_nodes)
                file_summaries.append({
                    "file_name": file_path.name,
                    "status": "pending_ingestion",
                    "total_chunks": len(records),
                    "skipped_chunks": skipped_chunks,
                    "ingested_chunks": len(file_nodes),
                    "chunk_policy_version": CHUNK_POLICY_VERSION,
                })
            else:
                file_summaries.append({
                    "file_name": file_path.name,
                    "status": "skipped",
                    "total_chunks": len(records),
                    "skipped_chunks": len(records),
                    "ingested_chunks": 0,
                    "chunk_policy_version": CHUNK_POLICY_VERSION,
                })

        if not all_nodes_to_ingest:
            logger.info("All files in directory already ingested.")
            return file_summaries

        logger.info(
            "Batch ingesting %d / %d chunks from %d files...",
            len(all_nodes_to_ingest), total_chunks, len(file_summaries)
        )

        # 2. Extract entities and relationships
        # 2. Extract and resolve entities using PropertyGraphIndex pipeline
        extractors = get_extractor_stack(self.schema, self.llm, self.resolver)

        # 3. Write to index (ONE SINGLE CALL)
        PropertyGraphIndex(
            nodes=all_nodes_to_ingest,
            property_graph_store=self.graph_store,
            vector_store=self.vector_store,
            embed_model=self.embed_model,
            llm=self.llm,
            kg_extractors=extractors,
            show_progress=True
        )

        enrichment = self._enrich_ingested_nodes(all_nodes_to_ingest)

        # Update statuses in summary
        for f in file_summaries:
            if f["status"] == "pending_ingestion":
                f["status"] = "success"
                f["enrichment"] = enrichment

        return file_summaries
