"""
Unit tests for backend.graph_ingestion.lite_parser
"""
import pytest
import requests
from unittest.mock import MagicMock, patch, mock_open

from backend.graph_ingestion.lite_parser import LiteParseClient, LiteParseUnavailableError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """LiteParseClient pointed at a dummy URL so no real network calls occur."""
    return LiteParseClient(base_url="http://localhost:5000", timeout=10)


# ---------------------------------------------------------------------------
# is_available()
# ---------------------------------------------------------------------------

def test_is_available_returns_true_on_200(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    with patch("requests.get", return_value=mock_response):
        assert client.is_available() is True


def test_is_available_returns_false_on_non_200(client):
    mock_response = MagicMock()
    mock_response.status_code = 503
    with patch("requests.get", return_value=mock_response):
        assert client.is_available() is False


def test_is_available_returns_false_on_connection_error(client):
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError):
        assert client.is_available() is False


# ---------------------------------------------------------------------------
# extract_to_markdown() — success paths
# ---------------------------------------------------------------------------

def test_extract_to_markdown_returns_markdown(client, tmp_path):
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF dummy content")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"markdown": "# Heading\n\n| A | B |\n|---|---|\n"}
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response):
        result = client.extract_to_markdown(str(pdf_file))

    assert "# Heading" in result
    assert "| A | B |" in result


def test_extract_to_markdown_returns_empty_string_when_no_markdown_key(client, tmp_path):
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF dummy content")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response):
        result = client.extract_to_markdown(str(pdf_file))

    assert result == ""


# ---------------------------------------------------------------------------
# extract_to_markdown() — error paths
# ---------------------------------------------------------------------------

def test_extract_to_markdown_raises_file_not_found(client):
    with pytest.raises(FileNotFoundError):
        client.extract_to_markdown("/nonexistent/path/document.pdf")


def test_extract_to_markdown_raises_unavailable_on_connection_error(client, tmp_path):
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF dummy")

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("refused")):
        with pytest.raises(LiteParseUnavailableError):
            client.extract_to_markdown(str(pdf_file))


def test_extract_to_markdown_raises_unavailable_on_timeout(client, tmp_path):
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF dummy")

    with patch("requests.post", side_effect=requests.exceptions.Timeout):
        with pytest.raises(LiteParseUnavailableError):
            client.extract_to_markdown(str(pdf_file))


def test_extract_to_markdown_propagates_http_error(client, tmp_path):
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF dummy")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
    with patch("requests.post", return_value=mock_response):
        with pytest.raises(requests.exceptions.HTTPError):
            client.extract_to_markdown(str(pdf_file))


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------

def test_client_uses_settings_defaults():
    """Verify client falls back to settings.PARSER_URL / PARSER_TIMEOUT_SECONDS."""
    from backend.core.config import settings
    c = LiteParseClient()
    assert c.base_url == settings.PARSER_URL
    assert c.timeout == settings.PARSER_TIMEOUT_SECONDS
    assert c.parse_endpoint == f"{settings.PARSER_URL}/api/v1/parse"
