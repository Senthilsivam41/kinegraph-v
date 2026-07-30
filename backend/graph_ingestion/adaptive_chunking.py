"""Versioned structural-first chunking contract for ADR-002."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - older langchain layouts
    from langchain.text_splitter import RecursiveCharacterTextSplitter


CHUNK_POLICY_VERSION = "kinegraph.adaptive-chunking.v1"
TOKENIZER_VERSION = "recursive-character-v1"
CHUNK_TYPES = ("structural", "semantic", "recursive", "table", "image")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def content_hash(text: str) -> str:
    """SHA-256 digest of chunk text (used for idempotency checks)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChunkRecord:
    """Source-faithful chunk with provenance and policy metadata."""

    text: str
    chunk_type: str
    chunk_id: str
    ordinal: int
    document_id: str
    policy_version: str = CHUNK_POLICY_VERSION
    section_path: str = ""
    page_start: int | None = None
    page_end: int | None = None
    parent_structural_id: str | None = None
    boundary_reason: str | None = None
    table_id: str | None = None
    headers: str | None = None
    caption: str | None = None
    image_id: str | None = None
    extraction_method: str | None = None
    confidence: float | None = None
    tokenizer_version: str | None = None
    overlap: int | None = None

    def content_hash(self) -> str:
        return content_hash(self.text)

    def to_metadata(self, **extra: Any) -> dict[str, Any]:
        """Flat metadata suitable for Chroma / Neo4j property bags."""
        payload: dict[str, Any] = {
            "document_id": self.document_id,
            "chunk_index": self.ordinal,
            "chunk_type": self.chunk_type,
            "chunk_policy_version": self.policy_version,
            "section_path": self.section_path or "",
            "page_start": self.page_start if self.page_start is not None else -1,
            "page_end": self.page_end if self.page_end is not None else -1,
            "parent_structural_id": self.parent_structural_id or "",
            "boundary_reason": self.boundary_reason or "",
            "table_id": self.table_id or "",
            "headers": self.headers or "",
            "caption": self.caption or "",
            "image_id": self.image_id or "",
            "extraction_method": self.extraction_method or "",
            "confidence": self.confidence if self.confidence is not None else -1.0,
            "tokenizer_version": self.tokenizer_version or "",
            "overlap": self.overlap if self.overlap is not None else -1,
            "chunk_hash": self.content_hash(),
        }
        for key, value in extra.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = value
        return payload

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_chunk_id(
    text: str,
    ordinal: int,
    document_id: str,
    policy_version: str = CHUNK_POLICY_VERSION,
) -> str:
    """SHA-256-derived stable chunk ID scoped to document + policy."""
    payload = f"{document_id}|{policy_version}|{ordinal}|{text}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"chunk_{ordinal}_{digest}"


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return [part for part in splitter.split_text(text) if part.strip()]


def _make_record(
    *,
    text: str,
    chunk_type: str,
    ordinal: int,
    document_id: str,
    section_path: str = "",
    parent_structural_id: str | None = None,
    boundary_reason: str | None = None,
    table_id: str | None = None,
    headers: str | None = None,
    caption: str | None = None,
    image_id: str | None = None,
    extraction_method: str | None = None,
    confidence: float | None = None,
    tokenizer_version: str | None = None,
    overlap: int | None = None,
) -> ChunkRecord:
    return ChunkRecord(
        text=text,
        chunk_type=chunk_type,
        chunk_id=stable_chunk_id(text, ordinal, document_id),
        ordinal=ordinal,
        document_id=document_id,
        policy_version=CHUNK_POLICY_VERSION,
        section_path=section_path,
        parent_structural_id=parent_structural_id,
        boundary_reason=boundary_reason,
        table_id=table_id,
        headers=headers,
        caption=caption,
        image_id=image_id,
        extraction_method=extraction_method,
        confidence=confidence,
        tokenizer_version=tokenizer_version,
        overlap=overlap,
    )


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2


def _parse_blocks(text: str) -> list[dict[str, Any]]:
    """Parse markdown into typed blocks without inventing missing structure."""
    lines = text.splitlines()
    blocks: list[dict[str, Any]] = []
    i = 0
    prose_buf: list[str] = []

    def flush_prose() -> None:
        nonlocal prose_buf
        body = "\n".join(prose_buf).strip()
        if body:
            blocks.append({"type": "prose", "text": body})
        prose_buf = []

    while i < len(lines):
        line = lines[i]
        heading = _HEADING_RE.match(line)
        if heading:
            flush_prose()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            blocks.append({"type": "heading", "level": level, "title": title, "text": line.strip()})
            i += 1
            continue

        if _is_table_row(line):
            flush_prose()
            table_lines = [line]
            i += 1
            while i < len(lines) and _is_table_row(lines[i]):
                table_lines.append(lines[i])
                i += 1
            headers = ""
            separator = ""
            body_rows = table_lines
            if len(table_lines) >= 2 and _TABLE_SEPARATOR_RE.match(table_lines[1].strip()):
                headers = table_lines[0].strip()
                separator = table_lines[1].strip()
                body_rows = table_lines[2:]
            blocks.append(
                {
                    "type": "table",
                    "text": "\n".join(table_lines).strip(),
                    "headers": headers,
                    "separator": separator,
                    "body_rows": body_rows,
                }
            )
            continue

        image_match = _IMAGE_RE.search(line)
        if image_match and line.strip().startswith("!["):
            flush_prose()
            caption = image_match.group(1).strip()
            target = image_match.group(2).strip()
            blocks.append(
                {
                    "type": "image",
                    "text": caption or target,
                    "caption": caption,
                    "target": target,
                }
            )
            i += 1
            continue

        prose_buf.append(line)
        i += 1

    flush_prose()
    return blocks


def _section_path(stack: list[str]) -> str:
    return " > ".join(stack)


def _chunk_recursive_records(
    text: str,
    *,
    document_id: str,
    start_ordinal: int,
    section_path: str = "",
    parent_structural_id: str | None = None,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    parts = _recursive_split(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    records: list[ChunkRecord] = []
    for offset, part in enumerate(parts):
        records.append(
            _make_record(
                text=part,
                chunk_type="recursive",
                ordinal=start_ordinal + offset,
                document_id=document_id,
                section_path=section_path,
                parent_structural_id=parent_structural_id,
                boundary_reason="recursive_fallback" if parent_structural_id else "unstructured_text",
                tokenizer_version=TOKENIZER_VERSION,
                overlap=chunk_overlap,
            )
        )
    return records


def _split_table_slices(
    *,
    headers: str,
    separator: str,
    body_rows: list[str],
    full_text: str,
    chunk_size: int,
) -> list[str]:
    """Slice oversized tables, repeating header/separator in each slice."""
    if len(full_text) <= chunk_size:
        return [full_text]

    preface_parts = [part for part in (headers, separator) if part]
    preface = "\n".join(preface_parts)
    preface_len = len(preface) + (1 if preface else 0)

    if not body_rows:
        # No parseable body — fall back to recursive character split of full table.
        return _recursive_split(full_text, chunk_size=chunk_size, chunk_overlap=0)

    # If a single row plus header exceeds the budget, still emit it (cannot invent
    # cell splits without losing table structure) but keep header repeated.
    slices: list[str] = []
    current_rows: list[str] = []
    current_len = preface_len

    def flush() -> None:
        nonlocal current_rows, current_len
        if not current_rows:
            return
        body = "\n".join(current_rows)
        slices.append(f"{preface}\n{body}" if preface else body)
        current_rows = []
        current_len = preface_len

    for row in body_rows:
        row_cost = len(row) + (1 if current_rows else 0)
        if current_rows and current_len + row_cost > chunk_size:
            flush()
            row_cost = len(row)
        current_rows.append(row)
        current_len += row_cost
        # Extremely long single row: flush immediately so later rows can start fresh.
        if current_len > chunk_size and len(current_rows) == 1:
            flush()

    flush()
    return slices or [full_text]


def _adaptive_chunk(
    text: str,
    *,
    document_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    blocks = _parse_blocks(text)
    if not blocks:
        return []

    heading_stack: list[tuple[int, str]] = []
    records: list[ChunkRecord] = []
    ordinal = 0
    pending_heading: dict[str, Any] | None = None
    section_parts: list[str] = []

    def current_path() -> str:
        return _section_path([title for _, title in heading_stack])

    def flush_section() -> None:
        nonlocal ordinal, pending_heading, section_parts
        body = "\n\n".join(part for part in section_parts if part.strip()).strip()
        section_parts = []
        if pending_heading is None and not body:
            return

        path = current_path()
        if pending_heading is not None:
            # Keep heading+short body together when it fits; otherwise anchor + recurse.
            combined = pending_heading["text"]
            if body:
                combined = f"{combined}\n\n{body}"
            if len(combined) <= chunk_size:
                records.append(
                    _make_record(
                        text=combined,
                        chunk_type="structural",
                        ordinal=ordinal,
                        document_id=document_id,
                        section_path=path,
                        boundary_reason="heading_section",
                    )
                )
                ordinal += 1
            else:
                anchor = _make_record(
                    text=pending_heading["text"],
                    chunk_type="structural",
                    ordinal=ordinal,
                    document_id=document_id,
                    section_path=path,
                    boundary_reason="heading_anchor",
                )
                records.append(anchor)
                ordinal += 1
                if body:
                    recursive_parts = _chunk_recursive_records(
                        body,
                        document_id=document_id,
                        start_ordinal=ordinal,
                        section_path=path,
                        parent_structural_id=anchor.chunk_id,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    records.extend(recursive_parts)
                    ordinal += len(recursive_parts)
            pending_heading = None
            return

        if body:
            recursive_parts = _chunk_recursive_records(
                body,
                document_id=document_id,
                start_ordinal=ordinal,
                section_path=path,
                parent_structural_id=None,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            records.extend(recursive_parts)
            ordinal += len(recursive_parts)

    for block in blocks:
        btype = block["type"]
        if btype == "heading":
            flush_section()
            level = int(block["level"])
            title = str(block["title"])
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            pending_heading = block
            continue

        if btype == "table":
            flush_section()
            path = current_path()
            table_text = str(block["text"])
            table_hash = hashlib.sha256(table_text.encode("utf-8")).hexdigest()[:16]
            table_id = f"table_{table_hash}"
            headers = str(block.get("headers") or "")
            slices = _split_table_slices(
                headers=headers,
                separator=str(block.get("separator") or ""),
                body_rows=list(block.get("body_rows") or []),
                full_text=table_text,
                chunk_size=chunk_size,
            )
            for slice_text in slices:
                records.append(
                    _make_record(
                        text=slice_text,
                        chunk_type="table",
                        ordinal=ordinal,
                        document_id=document_id,
                        section_path=path,
                        table_id=table_id,
                        headers=headers,
                        caption="",
                        boundary_reason=(
                            "table_block_sliced" if len(slices) > 1 else "table_block"
                        ),
                    )
                )
                ordinal += 1
            continue

        if btype == "image":
            flush_section()
            path = current_path()
            caption = str(block.get("caption") or "")
            target = str(block.get("target") or "")
            image_text = caption if caption else target
            image_hash = hashlib.sha256(f"{caption}|{target}".encode("utf-8")).hexdigest()[:16]
            records.append(
                _make_record(
                    text=image_text,
                    chunk_type="image",
                    ordinal=ordinal,
                    document_id=document_id,
                    section_path=path,
                    image_id=f"image_{image_hash}",
                    caption=caption,
                    extraction_method="markdown_alt_text",
                    confidence=1.0 if caption else 0.0,
                    boundary_reason="image_block",
                )
            )
            ordinal += 1
            continue

        # prose
        section_parts.append(str(block["text"]))

    flush_section()
    return records


def chunk_document(
    text: str,
    *,
    document_id: str,
    adaptive_enabled: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[ChunkRecord]:
    """
    Produce versioned chunks.

    Default (adaptive_enabled=False) preserves recursive-only behavior.
    Adaptive mode prefers structural/table/image chunks and recurses only as needed.
    Semantic refinement remains experimental and is not applied by default.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    if not adaptive_enabled:
        return _chunk_recursive_records(
            cleaned,
            document_id=document_id,
            start_ordinal=0,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    return _adaptive_chunk(
        cleaned,
        document_id=document_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def build_ingestion_validation_report(
    *,
    chunk_ids: Iterable[str],
    verified_chunk_ids: Iterable[str],
    enriched_entity_ids: Iterable[str] | None = None,
    incomplete_entity_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Additive, idempotent validation summary — never invents missing links."""
    all_ids = list(chunk_ids)
    verified = set(verified_chunk_ids)
    missing = [chunk_id for chunk_id in all_ids if chunk_id not in verified]
    incomplete = list(incomplete_entity_ids or [])
    enriched = list(enriched_entity_ids or [])
    return {
        "policy_version": CHUNK_POLICY_VERSION,
        "total_chunks": len(all_ids),
        "verified_chunk_links": len(all_ids) - len(missing),
        "missing_chunk_links": missing,
        "enriched_entities": enriched,
        "incomplete_enrichment": incomplete,
        "complete": not missing and not incomplete,
    }
