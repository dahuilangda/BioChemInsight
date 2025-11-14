# DOCKER_PADDLE_OCR Service Setup Guide

Standalone OCR inference service powered by the latest `PaddleOCR` (PPStructureV3). The container exposes HTTP endpoints that return Markdown, making it easy for the main application or any other project to consume the results.

## 1. Prerequisites

1. **NVIDIA Docker** (GPU-enabled host required)
2. **Docker Compose v2**
3. Optional: create `cache/paddlex` in advance if you want to persist the model cache

> **Note**: The first startup automatically downloads the official models. Expect the download to take a while and occupy roughly 8–10 GB of disk space.

## 2. Build and Start

```bash
# Run from the repository root
cd DOCKER_PADDLE_OCR

# Build the image (first run or after updates)
docker compose build

# Start the service
docker compose up -d
```

The service listens on port `8010` by default, and the container is named `paddle-ocr-server`.

Tail logs:

```bash
docker compose logs -f
```

Stop the stack:

```bash
docker compose down
```

## 3. API Overview

Two primary HTTP endpoints are available:

### Endpoint 1: PDF to Markdown

- **URL**: `POST http://<host>:8010/v1/pdf-to-markdown`
- **Description**: Converts a page range of a PDF into Markdown text.
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` *(required)*: PDF file payload.
  - `page_start` *(optional)*: Start page (default `1`, 1-based index).
  - `page_end` *(optional)*: End page (default `-1`, meaning until the last page).
  - `return_raw` *(optional)*: Return the raw JSON structure instead of Markdown, default `false`.

**Sample Response (PDF)**:

```json
{
  "markdown": "# Page 1\n...\n\n-#-#-#-#-\n\n# Page 2\n...",
  "page_numbers": [1, 2],
  "page_count": 2
}
```

### Endpoint 2: Image to Markdown

- **URL**: `POST http://<host>:8010/v1/image-to-markdown`
- **Description**: Converts a single image (PNG, JPG, etc.) into Markdown text.
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` *(required)*: Image file; supports `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, and similar formats.
  - `return_raw` *(optional)*: Return the raw JSON structure instead of Markdown, default `false`.

**Sample Response (Image)**:

```json
{
  "markdown": "...",
  "page_numbers": [1],
  "page_count": 1
}
```

## 4. Python Usage Examples

### Example 1: Process a PDF

```python
import requests

url = "http://localhost:8010/v1/pdf-to-markdown"
files = {"file": ("demo.pdf", open("data/demo.pdf", "rb"), "application/pdf")}
form = {"page_start": "1", "page_end": "3"}

resp = requests.post(url, files=files, data=form, timeout=600)
resp.raise_for_status()
markdown = resp.json()["markdown"]
print("--- PDF OCR Result ---")
print(markdown[:500])
```

### Example 2: Process an Image

```python
import requests

# Prepare an image file for testing
# with open("my_image.png", "wb") as f:
#     f.write(...)

url = "http://localhost:8010/v1/image-to-markdown"
files = {"file": ("test.png", open("path/to/your/image.png", "rb"), "image/png")}
form = {"return_raw": "false"}

resp = requests.post(url, files=files, data=form, timeout=300)
resp.raise_for_status()
markdown = resp.json()["markdown"]
print("\n--- Image OCR Result ---")
print(markdown[:500])
```

## 5. Integrating with the Main Project

1. In `constants.py`, add or update:
   ```python
   PADDLEOCR_SERVER_URL = "http://localhost:8010"
   ```
2. When running the CLI or frontend, pass `--ocr-engine paddleocr` to route OCR requests to this container.

## 6. FAQ

- **Model downloads repeatedly**:
  - Ensure the `./cache/paddlex` volume defined in `docker-compose.yml` exists so the cache is reused.
- **High GPU memory usage**:
  - The service renders pages at 2× scale for higher accuracy; adjust the scaling matrix in `app/server.py` if needed.
- **Request timeouts**:
  - Increase the client-side `timeout` for large documents or submit smaller page ranges in batches.
