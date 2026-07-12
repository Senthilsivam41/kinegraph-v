"""
LiteParse HTTP client.

Streams local documents to the self-hosted LiteParse container and returns
layout-preserved, structural Markdown.  Callers should catch
``LiteParseUnavailableError`` to detect network-level failures and apply
a fallback strategy (e.g. PyMuPDF) without swallowing hard parse errors.
"""
import os
import logging
from typing import Optional, Dict, Any

import requests

from backend.core.config import settings

logger = logging.getLogger(__name__)


class LiteParseUnavailableError(RuntimeError):
    """Raised when the LiteParse container cannot be reached or times out."""


class LiteParseClient:
    """Thin HTTP wrapper around the local LiteParse parsing service."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.base_url = base_url or settings.PARSER_URL
        self.timeout = timeout if timeout is not None else settings.PARSER_TIMEOUT_SECONDS
        self.parse_endpoint = f"{self.base_url}/parse"

    def is_available(self) -> bool:
        """Perform a lightweight liveness probe.

        POST /parse with an empty body returns HTTP 400 (Bad Request) when the
        server is up — the image has no dedicated /health endpoint.

        Returns:
            True if the server is reachable and returns 400, False otherwise.
        """
        try:
            requests.post(self.parse_endpoint, data={}, timeout=5)
            return True  # unexpected 2xx — still counts as alive
        except requests.exceptions.HTTPError as exc:
            return exc.response is not None and exc.response.status_code == 400
        except requests.exceptions.RequestException:
            return False

    def extract_to_markdown(
        self,
        file_path: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Stream a local document to LiteParse and return structured Markdown.

        Args:
            file_path: Absolute path to the document to parse.
            options: Optional dict of LiteParse extraction options.  Defaults
                to ``{"preserve_tables": True, "ocr_enabled": True}``.

        Returns:
            Layout-preserved Markdown string produced by LiteParse.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist on disk.
            LiteParseUnavailableError: If the container is unreachable or the
                request times out.
            requests.HTTPError: If the server returns a non-2xx status code.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target document not found at: {file_path}")

        payload_options = options or {
            "preserve_tables": True,
            "ocr_enabled": True,
        }

        try:
            with open(file_path, "rb") as doc_file:
                files = {
                    "file": (os.path.basename(file_path), doc_file, "application/pdf")
                }
                data = {"options": str(payload_options)}

                logger.info(
                    "Dispatching extraction payload for '%s' to LiteParse at %s",
                    file_path,
                    self.parse_endpoint,
                )
                response = requests.post(
                    self.parse_endpoint,
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json().get("markdown", "")

        except requests.exceptions.ConnectionError as exc:
            logger.warning("LiteParse container unreachable at %s: %s", self.base_url, exc)
            raise LiteParseUnavailableError(
                f"Could not connect to LiteParse at {self.base_url}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            logger.error(
                "LiteParse timed out after %ds processing '%s'",
                self.timeout,
                file_path,
            )
            raise LiteParseUnavailableError(
                f"LiteParse timed out after {self.timeout}s for '{file_path}'"
            ) from exc
        except requests.exceptions.RequestException:
            # Re-raise HTTP errors and other transport errors as-is so the
            # caller can decide whether to retry or surface the error.
            raise
