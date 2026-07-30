"""Idempotent PropertyGraph ingestion with ADR-002 chunk contracts."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.schema import TextNode
from neo4j import GraphDatabase

from backend.core.config import settings
from backend.graph_ingestion.adaptive_chunking import (
    CHUNK_POLICY_VERSION,
    ChunkRecord,
    build_ingestion_validation_report,
    chunk_document,
    content_hash,
    stable_chunk_id,
)
from backend.graph_ingestion.dedup import EntityResolver
from backend.graph_ingestion.embedding_wrapper import LangChainEmbeddingWrapper
from backend.graph_ingestion.enrichment import NodeEnricher
from backend.graph_ingestion.extractors import get_extractor_stack
from backend.graph_ingestion.schema import OntologySchema
from backend.graph_ingestion.stores import get_chroma_vector_store, get_llm, get_neo4j_graph_store
from backend.services.chroma_service import ChromaService
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

        self.neo4j_driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
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
            incomplete: list[str] = []
            skipped = int(result.get("skipped_without_verified_context", 0) or 0)
            if skipped:
                incomplete = [
                    f"entity_without_verified_context:{idx}" for idx in range(skipped)
                ]
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
        return content_hash(content)

    def is_chunk_ingested(self, chunk_hash: str) -> bool:
        """Check if a chunk with the given hash is already in the graph database."""
        try:
            with self.neo4j_driver.session() as session:
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

    def chunk_text(
        self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
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
        size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        if not enabled:
            # Preserve testability: callers/tests may patch chunk_text.
            texts = self.chunk_text(text, chunk_size=size, chunk_overlap=overlap)
            return [
                ChunkRecord(
                    text=part,
                    chunk_type="recursive",
                    chunk_id=stable_chunk_id(part, idx, document_id),
                    ordinal=idx,
                    document_id=document_id,
                    policy_version=CHUNK_POLICY_VERSION,
                    tokenizer_version="recursive-character-v1",
                    overlap=overlap,
                    boundary_reason="unstructured_text",
                )
                for idx, part in enumerate(texts)
            ]
        return chunk_document(
            text,
            document_id=document_id,
            adaptive_enabled=True,
            chunk_size=size,
            chunk_overlap=overlap,
        )

    def chunk_records(
        self,
        text: str,
        *,
        document_id: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[ChunkRecord]:
        """Alias for build_chunks (structural-first when adaptive flag is on)."""
        return self.build_chunks(
            text,
            document_id=document_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _nodes_from_records(
        self,
        records: List[ChunkRecord],
        *,
        file_name: str,
        metadata: Dict[str, Any],
    ) -> tuple[List[TextNode], int, List[ChunkRecord]]:
        """Filter already-ingested chunks and build LlamaIndex nodes with stable IDs."""
        nodes_to_ingest: List[TextNode] = []
        accepted: List[ChunkRecord] = []
        skipped_chunks = 0
        total = len(records)
        for record in records:
            chunk_digest = record.content_hash()
            if self.is_chunk_ingested(chunk_digest):
                skipped_chunks += 1
                continue
            node_meta = record.to_metadata(
                file_name=file_name,
                total_chunks=total,
                **{
                    key: value
                    for key, value in metadata.items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                },
            )
            node = TextNode(
                text=record.text,
                id_=record.chunk_id,
                metadata=node_meta,
            )
            nodes_to_ingest.append(node)
            accepted.append(record)
        return nodes_to_ingest, skipped_chunks, accepted

    def ingest_file(
        self, file_path: str, metadata: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Ingests a single file (PDF or Markdown/text) into the PropertyGraphIndex.
        Returns a summary dictionary of what was ingested.
        """
        if metadata is None:
            metadata = {}

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = ""
        if file_path_obj.suffix.lower() == ".pdf":
            from backend.workers.document_processor import extract_text_from_pdf

            content = extract_text_from_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as handle:
                content = handle.read()

        if not content.strip():
            logger.warning("Empty file skipped: %s", file_path)
            return {"status": "skipped", "reason": "empty content"}

        doc_id = generate_document_id(file_path)
        records = self.build_chunks(content, document_id=doc_id)
        nodes_to_ingest, skipped_chunks, accepted = self._nodes_from_records(
            records,
            file_name=file_path_obj.name,
            metadata=metadata,
        )

        if not nodes_to_ingest:
            logger.info(
                "All %d chunks for file '%s' already ingested.",
                len(records),
                file_path_obj.name,
            )
            return {
                "file_name": file_path_obj.name,
                "document_id": doc_id,
                "status": "skipped",
                "total_chunks": len(records),
                "skipped_chunks": skipped_chunks,
                "ingested_chunks": 0,
                "chunk_policy_version": CHUNK_POLICY_VERSION,
            }

        logger.info(
            "Ingesting %d / %d chunks for file '%s'...",
            len(nodes_to_ingest),
            len(records),
            file_path_obj.name,
        )

        extractors = get_extractor_stack(self.schema, self.llm, self.resolver)
        PropertyGraphIndex(
            nodes=nodes_to_ingest,
            property_graph_store=self.graph_store,
            vector_store=self.vector_store,
            embed_model=self.embed_model,
            llm=self.llm,
            kg_extractors=extractors,
            show_progress=True,
        )

        enrichment = self._enrich_ingested_nodes(nodes_to_ingest)
        validation = enrichment.get("validation") or build_ingestion_validation_report(
            chunk_ids=[node.node_id for node in nodes_to_ingest],
            verified_chunk_ids=[],
            incomplete_entity_ids=["validation_missing"],
        )

        return {
            "file_name": file_path_obj.name,
            "document_id": doc_id,
            "status": "success" if validation.get("complete") else "incomplete",
            "total_chunks": len(records),
            "skipped_chunks": skipped_chunks,
            "ingested_chunks": len(accepted),
            "chunk_policy_version": CHUNK_POLICY_VERSION,
            "adaptive_chunking": settings.ADAPTIVE_CHUNKING_ENABLED,
            "enrichment": enrichment,
            "ingestion_validation": validation,
        }

    def ingest_directory(
        self, dir_path: str, metadata: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Ingests all supported files from a directory in a single batch to avoid
        event loop deadlocks and enable global entity resolution.
        """
        dir_path_obj = Path(dir_path)
        if not dir_path_obj.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        if metadata is None:
            metadata = {}

        all_nodes_to_ingest: List[TextNode] = []
        all_accepted: List[ChunkRecord] = []
        skipped_chunks = 0
        total_chunks = 0
        file_summaries: List[Dict[str, Any]] = []

        for file_path in dir_path_obj.iterdir():
            if file_path.suffix.lower() not in [".pdf", ".md", ".txt"]:
                continue

            content = ""
            if file_path.suffix.lower() == ".pdf":
                from backend.workers.document_processor import extract_text_from_pdf

                content = extract_text_from_pdf(str(file_path))
            else:
                with open(file_path, "r", encoding="utf-8") as handle:
                    content = handle.read()

            if not content.strip():
                continue

            doc_id = generate_document_id(str(file_path))
            records = self.build_chunks(content, document_id=doc_id)
            total_chunks += len(records)
            file_nodes, file_skipped, accepted = self._nodes_from_records(
                records,
                file_name=file_path.name,
                metadata=metadata,
            )
            skipped_chunks += file_skipped

            if file_nodes:
                all_nodes_to_ingest.extend(file_nodes)
                all_accepted.extend(accepted)
                file_summaries.append(
                    {
                        "file_name": file_path.name,
                        "document_id": doc_id,
                        "status": "pending_ingestion",
                        "total_chunks": len(records),
                        "skipped_chunks": file_skipped,
                        "ingested_chunks": len(file_nodes),
                        "chunk_policy_version": CHUNK_POLICY_VERSION,
                    }
                )
            else:
                file_summaries.append(
                    {
                        "file_name": file_path.name,
                        "document_id": doc_id,
                        "status": "skipped",
                        "total_chunks": len(records),
                        "skipped_chunks": len(records),
                        "ingested_chunks": 0,
                        "chunk_policy_version": CHUNK_POLICY_VERSION,
                    }
                )

        if not all_nodes_to_ingest:
            logger.info("All files in directory already ingested.")
            return file_summaries

        logger.info(
            "Batch ingesting %d / %d chunks from %d files...",
            len(all_nodes_to_ingest),
            total_chunks,
            len(file_summaries),
        )

        extractors = get_extractor_stack(self.schema, self.llm, self.resolver)
        PropertyGraphIndex(
            nodes=all_nodes_to_ingest,
            property_graph_store=self.graph_store,
            vector_store=self.vector_store,
            embed_model=self.embed_model,
            llm=self.llm,
            kg_extractors=extractors,
            show_progress=True,
        )

        enrichment = self._enrich_ingested_nodes(all_nodes_to_ingest)
        validation = enrichment.get("validation") or build_ingestion_validation_report(
            chunk_ids=[node.node_id for node in all_nodes_to_ingest],
            verified_chunk_ids=[],
            incomplete_entity_ids=["validation_missing"],
        )

        for summary in file_summaries:
            if summary["status"] == "pending_ingestion":
                summary["status"] = (
                    "success" if validation.get("complete") else "incomplete"
                )
                summary["enrichment"] = enrichment
                summary["ingestion_validation"] = validation
                summary["chunk_policy_version"] = CHUNK_POLICY_VERSION

        return file_summaries
