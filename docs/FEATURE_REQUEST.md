# Feature Request: Local Document Parsing Service via LiteParse Integration

## 1. Overview & Objective
To support fully offline, private, and high-performance document ingestion within `kinegraph-v`, this feature request proposes replacing or augmenting the existing traditional Python PDF extraction libraries with a self-hosted, local **LiteParse Server** container instance. 

Knowledge Graph (GraphRAG) creation relies heavily on structural semantic accuracy. Traditional parsers frequently scramble multi-column text and strip table formats, resulting in corrupted graph entities and malformed relationships. Integrating LiteParse allows the ingestion pipeline to parse complex documents natively into clean, structural **Markdown** locally without data leaving the host boundary or incurring cloud API costs.

---

## 2. Architecture & Component Interaction

The proposed architecture runs the parsing engine as a localized microservice alongside the main application pipeline:

```
[ Local File System ] ──> ( PDF Ingestion Engine )
                                 │
                                 ▼ (Multipart Form POST)
                        [ LiteParse Container ]
                        ( Port 5000: PDFium + OCR )
                                 │
                                 ▼ (Returns Structured Markdown)
                        ( Extract Markdown Structure )
                                 │
                                 ▼
                     [ Graph Construction Layer ]
```

### Key Components:
1. **LiteParse Server Container:** A Dockerized standalone environment executing an internal layout-aware extraction engine based on PDFium coordinates and localized Tesseract engines.
2. **Document Extraction Driver (`lite_parser.py`):** A lightweight client driver implemented within the repository utilities to manage HTTP payloads, handle connection timeouts, and stream multipart document data to the local container endpoint.

---

## 3. Implementation Blueprint

### A. Infrastructure Configuration (`docker-compose.yml`)
To orchestrate the ingestion service alongside existing application infrastructure, the `liteparse-server` will be defined as an independent, bridge-networked service.

```yaml
version: '3.8'

services:
  # Main application logic container
  app:
    build: .
    volumes:
      - .:/app
    environment:
      - PARSER_URL=http://liteparse:5000
    ports:
      - "8000:8000"
    depends_on:
      - liteparse

  # Self-hosted, local document parsing engine
  liteparse:
    image: ghcr.io/run-llama/liteparse-server:main
    container_name: kinegraph_liteparse_engine
    ports:
      - "5000:5000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
```

### B. Ingestion Driver Implementation (`utils/lite_parser.py`)
This utility replaces standard text extraction routines by converting raw file inputs into multi-part streams and executing requests against the localized text extraction endpoints.

```python
import os
import requests
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LiteParseClient:
    def __init__(self, base_url: Optional[str] = None):
        # Fallback to local default container endpoint if environment variable is unset
        self.base_url = base_url or os.getenv("PARSER_URL", "http://localhost:5000")
        self.parse_endpoint = f"{self.base_url}/api/v1/parse"
        
    def extract_to_markdown(self, file_path: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Streams a local document to the self-hosted LiteParse server and returns 
        layout-preserved structural Markdown text.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target document not found at: {file_path}")
            
        payload_options = options or {
            "preserve_tables": True,
            "ocr_enabled": True
        }
        
        try:
            with open(file_path, 'rb') as doc_file:
                files = {'file': (os.path.basename(file_path), doc_file, 'application/pdf')}
                data = {'options': str(payload_options)}
                
                logger.info(f"Dispatching extraction payload for {file_path} to local LiteParse container...")
                response = requests.post(
                    self.parse_endpoint, 
                    files=files, 
                    data=data,
                    timeout=120  # Allocate generous latency threshold for large, image-dense multi-page PDFs
                )
                
                response.raise_for_status()
                result_json = response.json()
                
                # Extract structured markdown output strings from the response scheme
                return result_json.get("markdown", "")
                
        except requests.exceptions.Timeout:
            logger.error("The local parsing container timed out processing the document payload.")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to communicate with local LiteParse container microservice: {str(e)}")
            raise
```

---

## 4. Why This Architecture Upgrades Knowledge Graph Extraction

Knowledge extraction loops require predictable markup blocks to properly map entities and predicates. The shift to a layout-aware engine brings strict structural stability over raw text blocks:

1. **Table Context Isolation:** Tables are isolated and output as clean standard Markdown syntax (`| Entity A | Relation | Entity B |`). When chunks are parsed by downstream agents, column values map directly back to their header semantic bounds rather than becoming space-separated anomalies.
2. **Column Serialization Rectification:** Multipage columns are traced using absolute bounding box arrays via PDFium rather than sequential rendering layout text streams. This eliminates horizontal reading flaws that bind unrelated paragraphs together.
3. **Data Security & Zero Operational Cost:** Because processing runs locally within the enterprise boundary, corporate files never leave the system. API usage charges are completely eliminated, removing throughput bottlenecks during large-scale document backfills.

---

## 5. Verification & Test Strategy

To seamlessly swap this feature in without breaking existing extraction logic, deployment verification can be verified via the following checks:

* **Container Healthcheck validation:** Verify the engine responds appropriately on standard validation sweeps:
  ```bash
  curl -I http://localhost:5000/health
  ```
* **Structural Table Ingestion Test:** Run ingestion on a standard sample document containing multi-column blocks and embedded financial text tables. Validate that the string output includes cleanly serialized Markdown grid outlines (`---`) rather than single-line text arrays.