"""
Vectorless RAG Service — KineticGraph-Vectra
Implements a pure-Python BM25 search engine for document chunks and attachments.
Provides sub-millisecond retrieval without vector embeddings or database queries.
"""
from __future__ import annotations

import glob
import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BM25 Retriever Implementation (Pure Python)
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Standard Okapi BM25 implementation.
    Reference: https://en.wikipedia.org/wiki/Okapi_BM25
    """

    def __init__(
        self,
        corpus: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """
        Args:
            corpus: List of chunk dicts. Each must have "content" and "metadata".
            k1: BM25 scaling parameter. Controls term frequency saturation.
            b: BM25 scaling parameter. Controls document length normalization.
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus)
        self.doc_lengths: List[int] = []
        self.avg_doc_len = 0.0
        self.doc_term_freqs: List[Counter[str]] = []
        self.idf: Dict[str, float] = {}
        self._initialize()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer to split text into words, removing punctuation."""
        return re.findall(r"\w+", text.lower())

    def _initialize(self) -> None:
        if not self.corpus:
            return

        total_length = 0
        df: Dict[str, int] = {}

        for doc in self.corpus:
            tokens = self._tokenize(doc.get("content", ""))
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)

            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)

            # Record document frequency (presence of word in document)
            for word in tf:
                df[word] = df.get(word, 0) + 1

        self.avg_doc_len = total_length / self.doc_count if self.doc_count > 0 else 0.0

        # Calculate Inverse Document Frequency (IDF)
        for word, freq in df.items():
            # Standard BM25 IDF formula with smoothing to avoid negative IDFs
            numerator = self.doc_count - freq + 0.5
            denominator = freq + 0.5
            self.idf[word] = math.log(max(numerator / denominator, 0.0001) + 1.0)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the corpus and return the top-k matched chunks."""
        if not self.corpus:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            # If query is empty or only punctuation, return the first few chunks
            return [
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": 0.0,
                    "source": "vectorless",
                }
                for doc in self.corpus[:top_k]
            ]

        scores: List[Tuple[int, float]] = []

        for idx, doc in enumerate(self.corpus):
            score = 0.0
            tf = self.doc_term_freqs[idx]
            doc_len = self.doc_lengths[idx]

            for token in query_tokens:
                if token in tf:
                    token_tf = tf[token]
                    token_idf = self.idf.get(token, 0.0)

                    # BM25 term weighting formula
                    numerator = token_tf * (self.k1 + 1)
                    denominator = token_tf + self.k1 * (
                        1 - self.b + self.b * (doc_len / self.avg_doc_len)
                    )
                    score += token_idf * (numerator / denominator)

            scores.append((idx, score))

        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            doc = self.corpus[idx]
            results.append(
                {
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": round(score, 4),
                    "source": "vectorless",
                }
            )
        return results


# ---------------------------------------------------------------------------
# Vectorless RAG Service
# ---------------------------------------------------------------------------

class VectorlessService:
    """Service handling disk-based chunk caches, BM25 searching, and attachment processing."""

    def __init__(self) -> None:
        self.chunks_dir = Path("data/chunks")
        self.docs_dir = Path("data/documents")
        
        # Ensure directories exist
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    def save_document_chunks(
        self,
        doc_id: str,
        file_name: str,
        chunks: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> bool:
        """Save chunk structures to a JSON cache and raw text to a text file."""
        try:
            # 1. Save chunks JSON
            chunk_list = []
            full_text_parts = []
            for i, chunk in enumerate(chunks):
                chunk_list.append(
                    {
                        "id": ids[i],
                        "content": chunk,
                        "metadata": metadatas[i],
                    }
                )
                full_text_parts.append(chunk)

            json_path = self.chunks_dir / f"{doc_id}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(chunk_list, f, ensure_ascii=False, indent=2)

            # 2. Save full text file
            txt_path = self.docs_dir / f"{doc_id}.txt"
            full_text = "\n\n".join(full_text_parts)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            # 3. Create a symlink or secondary text file named after file_name for easy direct lookups
            # (Sanitize filename to prevent directory traversal issues)
            safe_file_name = re.sub(r"[^a-zA-Z0-9_\.-]", "_", file_name)
            name_txt_path = self.docs_dir / f"{safe_file_name}.txt"
            with open(name_txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            logger.info("Saved %d chunks to disk for doc %s", len(chunks), doc_id)
            return True
        except Exception as exc:
            logger.error("Error saving document chunks to disk: %s", exc)
            return False

    def get_local_document_text(self, file_name: str) -> Optional[str]:
        """Load the raw text of a document from disk using its filename."""
        safe_file_name = re.sub(r"[^a-zA-Z0-9_\.-]", "_", file_name)
        txt_path = self.docs_dir / f"{safe_file_name}.txt"
        if txt_path.exists():
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.error("Failed to read text file %s: %s", txt_path, e)
        return None

    def search_attachment(
        self,
        query: str,
        attachment_content: str,
        attachment_name: Optional[str] = None,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Process and query an attachment.
        If it's small, returns the whole thing. If large, chunks on the fly and retrieves.
        """
        att_name = attachment_name or "attachment.txt"
        content_len = len(attachment_content)

        # Threshold: if text is small (fits easily in context), return the whole file as a single chunk.
        if content_len < 15000:
            return [
                {
                    "content": attachment_content,
                    "metadata": {
                        "file_name": att_name,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "is_attachment": True,
                    },
                    "score": 1.0,
                    "source": "vectorless",
                }
            ]

        # Otherwise, chunk on-the-fly and search
        # Split by paragraph
        paragraphs = [p.strip() for p in attachment_content.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in attachment_content.split("\n") if p.strip()]

        corpus = []
        for idx, para in enumerate(paragraphs):
            corpus.append(
                {
                    "content": para,
                    "metadata": {
                        "file_name": att_name,
                        "chunk_index": idx,
                        "total_chunks": len(paragraphs),
                        "is_attachment": True,
                    },
                }
            )

        retriever = BM25Retriever(corpus)
        return retriever.search(query, top_k=max_results)

    def search_chunks(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Load cached JSON chunk files, filter them, and run BM25 search."""
        corpus: List[Dict[str, Any]] = []
        
        # Load all JSON files in the chunks directory
        json_pattern = str(self.chunks_dir / "*.json")
        for file_path in glob.glob(json_pattern):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                    corpus.extend(chunks)
            except Exception as e:
                logger.error("Failed to load chunk file %s: %s", file_path, e)

        # Apply metadata filters if provided
        if filters:
            filtered_corpus = []
            for chunk in corpus:
                match = True
                meta = chunk.get("metadata", {})
                for k, v in filters.items():
                    if meta.get(k) != v:
                        match = False
                        break
                if match:
                    filtered_corpus.append(chunk)
            corpus = filtered_corpus

        if not corpus:
            logger.warning("Vectorless search: empty corpus loaded.")
            return []

        # Run BM25
        retriever = BM25Retriever(corpus)
        return retriever.search(query, top_k=top_k)
