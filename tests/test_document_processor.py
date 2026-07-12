"""
Unit tests for backend.workers.document_processor — focused on
the LiteParse-first / PyMuPDF-fallback behaviour of extract_text_from_pdf.
"""
import pytest
from unittest.mock import MagicMock, patch, call

import requests

from backend.graph_ingestion.lite_parser import LiteParseUnavailableError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fitz_doc(text_per_page):
    """Build a minimal mock fitz document that yields text_per_page."""
    pages = []
    for text in text_per_page:
        page = MagicMock()
        page.get_text.return_value = text
        pages.append(page)

    doc_ctx = MagicMock()
    doc_ctx.__enter__ = MagicMock(return_value=iter(pages))
    doc_ctx.__exit__ = MagicMock(return_value=False)
    return doc_ctx


# ---------------------------------------------------------------------------
# LiteParse succeeds
# ---------------------------------------------------------------------------

def test_extract_uses_liteparse_when_available(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")

    mock_client = MagicMock()
    mock_client.extract_to_markdown.return_value = "# LiteParse Output\n\n| A | B |\n"

    with patch(
        "backend.workers.document_processor.LiteParseClient",
        return_value=mock_client,
    ):
        from backend.workers.document_processor import extract_text_from_pdf
        result = extract_text_from_pdf(str(pdf))

    assert result == "# LiteParse Output\n\n| A | B |\n"
    mock_client.extract_to_markdown.assert_called_once_with(str(pdf))


# ---------------------------------------------------------------------------
# LiteParse unavailable → PyMuPDF fallback
# ---------------------------------------------------------------------------

def test_extract_falls_back_to_pymupdf_on_unavailable(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")

    mock_client = MagicMock()
    mock_client.extract_to_markdown.side_effect = LiteParseUnavailableError("down")

    fitz_doc = _make_fitz_doc(["Page 1 text", "Page 2 text"])

    with patch(
        "backend.workers.document_processor.LiteParseClient",
        return_value=mock_client,
    ), patch("backend.workers.document_processor.fitz.open", return_value=fitz_doc):
        from backend.workers.document_processor import extract_text_from_pdf
        result = extract_text_from_pdf(str(pdf))

    assert result == "Page 1 text\nPage 2 text"


def test_extract_falls_back_to_pymupdf_on_generic_liteparse_error(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")

    mock_client = MagicMock()
    mock_client.extract_to_markdown.side_effect = requests.exceptions.HTTPError("500")

    fitz_doc = _make_fitz_doc(["Fallback content"])

    with patch(
        "backend.workers.document_processor.LiteParseClient",
        return_value=mock_client,
    ), patch("backend.workers.document_processor.fitz.open", return_value=fitz_doc):
        from backend.workers.document_processor import extract_text_from_pdf
        result = extract_text_from_pdf(str(pdf))

    assert result == "Fallback content"


# ---------------------------------------------------------------------------
# Both parsers fail
# ---------------------------------------------------------------------------

def test_extract_returns_empty_string_when_both_parsers_fail(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")

    mock_client = MagicMock()
    mock_client.extract_to_markdown.side_effect = LiteParseUnavailableError("down")

    fitz_doc = MagicMock()
    fitz_doc.__enter__ = MagicMock(side_effect=RuntimeError("fitz broken"))
    fitz_doc.__exit__ = MagicMock(return_value=False)

    with patch(
        "backend.workers.document_processor.LiteParseClient",
        return_value=mock_client,
    ), patch("backend.workers.document_processor.fitz.open", return_value=fitz_doc):
        from backend.workers.document_processor import extract_text_from_pdf
        result = extract_text_from_pdf(str(pdf))

    assert result == ""
