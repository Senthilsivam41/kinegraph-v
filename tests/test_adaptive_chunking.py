"""ADR-002 adaptive chunking contract tests."""
from backend.graph_ingestion.adaptive_chunking import (
    CHUNK_POLICY_VERSION,
    ChunkRecord,
    build_ingestion_validation_report,
    chunk_document,
)


SAMPLE_MARKDOWN = """# Overview

Kinetic-V stores evidence in vector and graph indexes.

## Retrieval

Hybrid retrieval fuses candidates with RRF.

| Channel | Role |
| --- | --- |
| Vector | Semantic recall |
| Graph | Topology |

![Architecture diagram](architecture.png)

Plain trailing notes without a heading stay recursive when needed.
"""


def test_disabled_policy_uses_recursive_chunks_with_stable_ids():
    chunks = chunk_document(
        "Alpha paragraph.\n\nBeta paragraph that continues the idea.",
        document_id="doc_test",
        adaptive_enabled=False,
        chunk_size=40,
        chunk_overlap=5,
    )

    assert chunks
    assert all(isinstance(chunk, ChunkRecord) for chunk in chunks)
    assert all(chunk.chunk_type == "recursive" for chunk in chunks)
    assert all(chunk.policy_version == CHUNK_POLICY_VERSION for chunk in chunks)
    assert all(chunk.document_id == "doc_test" for chunk in chunks)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.chunk_id.startswith("chunk_") for chunk in chunks)
    # Stable SHA-256-derived IDs: same input yields same IDs.
    again = chunk_document(
        "Alpha paragraph.\n\nBeta paragraph that continues the idea.",
        document_id="doc_test",
        adaptive_enabled=False,
        chunk_size=40,
        chunk_overlap=5,
    )
    assert [c.chunk_id for c in again] == [c.chunk_id for c in chunks]


def test_adaptive_policy_emits_structural_table_and_image_chunks():
    chunks = chunk_document(
        SAMPLE_MARKDOWN,
        document_id="doc_md",
        adaptive_enabled=True,
        chunk_size=1000,
        chunk_overlap=50,
    )

    types = {chunk.chunk_type for chunk in chunks}
    assert "structural" in types
    assert "table" in types
    assert "image" in types
    assert all(chunk.policy_version == CHUNK_POLICY_VERSION for chunk in chunks)

    structural = [c for c in chunks if c.chunk_type == "structural"]
    assert any(c.section_path == "Overview" for c in structural)
    assert any(c.section_path == "Overview > Retrieval" for c in structural)

    table = next(c for c in chunks if c.chunk_type == "table")
    assert table.table_id
    assert "Channel" in (table.headers or "")
    assert "Vector" in table.text

    image = next(c for c in chunks if c.chunk_type == "image")
    assert image.image_id
    assert image.caption == "Architecture diagram"
    assert image.extraction_method == "markdown_alt_text"
    # Never invent description beyond caption/alt.
    assert "Architecture diagram" in image.text
    assert "unknown" not in image.text.lower()


def test_oversized_section_falls_back_to_recursive_with_parent_link():
    long_body = "Sentence. " * 80
    text = f"# Section\n\n{long_body}"
    chunks = chunk_document(
        text,
        document_id="doc_long",
        adaptive_enabled=True,
        chunk_size=120,
        chunk_overlap=20,
    )

    structural = [c for c in chunks if c.chunk_type == "structural"]
    recursive = [c for c in chunks if c.chunk_type == "recursive"]
    assert structural, "section header should remain a structural anchor"
    assert recursive, "oversized body must recurse"
    parent_ids = {c.chunk_id for c in structural}
    assert all(c.parent_structural_id in parent_ids for c in recursive)
    assert all(c.section_path == "Section" for c in recursive)
    assert all(c.overlap == 20 for c in recursive)
    assert all(c.tokenizer_version for c in recursive)


def test_chunk_metadata_is_chroma_flat_and_includes_provenance():
    chunks = chunk_document(
        SAMPLE_MARKDOWN,
        document_id="doc_meta",
        adaptive_enabled=True,
    )
    meta = chunks[0].to_metadata()
    assert meta["document_id"] == "doc_meta"
    assert meta["chunk_policy_version"] == CHUNK_POLICY_VERSION
    assert meta["chunk_type"] in {"structural", "table", "image", "recursive", "semantic"}
    assert "chunk_index" in meta
    # Flat primitives only — no nested dict/list values.
    assert all(isinstance(v, (str, int, float, bool)) or v is None for v in meta.values())


def test_validation_report_is_additive_and_reports_incomplete_links():
    report = build_ingestion_validation_report(
        chunk_ids=["c1", "c2", "c3"],
        verified_chunk_ids=["c1", "c3"],
        enriched_entity_ids=["e1"],
        incomplete_entity_ids=["e2"],
    )
    assert report["complete"] is False
    assert report["total_chunks"] == 3
    assert report["verified_chunk_links"] == 2
    assert report["missing_chunk_links"] == ["c2"]
    assert report["incomplete_enrichment"] == ["e2"]
    assert report["policy_version"] == CHUNK_POLICY_VERSION
    # Additive: callers can merge without inventing context.
    assert "invented" not in str(report).lower()


def test_chunk_ids_differ_across_documents_with_identical_text():
    text = "Shared paragraph about Kinetic-V retrieval."
    a = chunk_document(text, document_id="doc-a", adaptive_enabled=False, chunk_size=200)
    b = chunk_document(text, document_id="doc-b", adaptive_enabled=False, chunk_size=200)
    assert a and b
    assert [c.text for c in a] == [c.text for c in b]
    assert [c.chunk_id for c in a] != [c.chunk_id for c in b]
    assert all(c.document_id == "doc-a" for c in a)
    assert all(c.document_id == "doc-b" for c in b)


def test_oversized_table_is_sliced_with_repeated_headers():
    header = "| ColA | ColB |"
    separator = "| --- | --- |"
    rows = [f"| value-{i} | detail-{i}-{'x' * 20} |" for i in range(40)]
    table = "\n".join([header, separator, *rows])
    chunks = chunk_document(
        f"# Tables\n\n{table}",
        document_id="doc_table",
        adaptive_enabled=True,
        chunk_size=120,
        chunk_overlap=0,
    )
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) > 1, "oversized table must be sliced"
    assert all(len(c.text) <= 120 for c in tables)
    assert all(header in c.text for c in tables)
    assert all(c.headers and "ColA" in c.headers for c in tables)
    assert len({c.table_id for c in tables}) == 1
