from __future__ import annotations

import fcntl
import hashlib
import html
import json
import logging
import math
import os
import re
import tempfile
from contextlib import nullcontext
from pathlib import Path

import fitz

from utils.paddleocr_client import request_pdf_to_markdown

logger = logging.getLogger(__name__)


class SearchBusy(Exception):
    pass


def _rect(bounds, width, height):
    x0, y0, x1, y1 = bounds
    if (
        width <= 0
        or height <= 0
        or not all(math.isfinite(value) for value in (*bounds, width, height))
    ):
        raise ValueError("Invalid text coordinates")
    x0, y0 = max(0, min(1, x0 / width)), max(0, min(1, y0 / height))
    x1, y1 = max(x0, min(1, x1 / width)), max(y0, min(1, y1 / height))
    return {
        "x": round(x0, 6),
        "y": round(y0, 6),
        "w": round(x1 - x0, 6),
        "h": round(y1 - y0, 6),
    }


def _ocr_items(payload):
    predictions = payload.get("raw_predictions") or []
    if len(predictions) != 1:
        raise ValueError("OCR did not return one page with text coordinates")
    prediction = predictions[0]
    width, height = prediction["width"], prediction["height"]
    items = []
    for block in prediction.get("parsing_res_list", []):
        text = html.unescape(
            re.sub(r"<[^>]+>", " ", str(block.get("block_content") or ""))
        ).strip()
        bounds = block.get("block_bbox")
        if text and bounds and len(bounds) == 4:
            items.append({"text": text, "rect": _rect(bounds, width, height)})
    if not items and any(
        str(value).strip() for value in payload.get("page_markdowns", [])
    ):
        raise ValueError("OCR text coordinates are missing")
    return items


def page_text(
    path: Path,
    number: int,
    ocr: bool,
    server_url: str,
    lang: str,
    *,
    opened_document=None,
):
    stat = path.stat()
    signature = hashlib.sha256(
        f"{stat.st_size}:{stat.st_mtime_ns}:v3:{server_url}:{lang}".encode()
    ).hexdigest()[:24]
    directory = path.parent / "search" / signature
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / f"{number}.json"
    cached = _read(cache)
    if cached is not None:
        return cached
    native_cache = directory / f"{number}.text.json"
    native = _read(native_cache)
    if native is not None and not ocr:
        return native
    with (
        nullcontext(opened_document)
        if opened_document is not None
        else fitz.open(path) as document
    ):
        if not 1 <= number <= len(document):
            raise IndexError(number)
        page = document[number - 1]
        words = page.get_text("words", sort=True)
        width, height = page.rect.width, page.rect.height
        items = [
            {
                "text": word[4],
                "rect": _rect(
                    fitz.Rect(word[:4]) * page.rotation_matrix, width, height
                ),
            }
            for word in words
        ]
        images = page.get_image_info()
        image_bounds = max(
            (fitz.Rect(image["bbox"]) for image in images),
            key=lambda rect: rect.get_area(),
            default=fitz.Rect(),
        )
        image_area = image_bounds.get_area()
        image_text = sum(
            len(word[4]) for word in words if image_bounds.contains(fitz.Rect(word[:4]))
        )
        needs_ocr = (image_area > width * height * 0.7 and image_text < 100) or (
            image_area > width * height * 0.3
            and sum(len(word[4]) for word in words) < 100
        )
        result = {
            "page": number,
            "source": "text",
            "needs_ocr": needs_ocr,
            "items": items,
        }
        if not needs_ocr or not ocr:
            _write(native_cache if needs_ocr else cache, result)
            return result
        if not server_url:
            raise RuntimeError("OCR service is not configured")
        with (path.parent.parent / "search-ocr.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SearchBusy() from None
            cached = _read(cache)
            if cached is not None:
                return cached
            with tempfile.TemporaryDirectory(prefix="pdf-search-") as temporary:
                single = Path(temporary) / "page.pdf"
                with fitz.open() as output:
                    output.insert_pdf(
                        document, from_page=number - 1, to_page=number - 1
                    )
                    output.save(single)
                payload = request_pdf_to_markdown(
                    single,
                    1,
                    1,
                    lang,
                    True,
                    server_url,
                    document_key=f"search-{path.parent.name}-{signature}-{number}",
                    timeout_seconds=90,
                    submit_timeout=15,
                    max_submit_retries=0,
                )
            result = {
                "page": number,
                "source": "ocr",
                "needs_ocr": False,
                "items": _ocr_items(payload),
            }
            _write(cache, result)
            native_cache.unlink(missing_ok=True)
            return result


def pages_text(path: Path, numbers: list[int], server_url: str, lang: str):
    result = {"pages": [], "failed": []}
    with fitz.open(path) as document:
        for number in numbers:
            try:
                result["pages"].append(
                    page_text(
                        path, number, False, server_url, lang, opened_document=document
                    )
                )
            except Exception:
                logger.exception(
                    "PDF text indexing failed: %s page %s", path.parent.name, number
                )
                result["failed"].append(number)
    return result


def _read(path):
    try:
        data = json.loads(path.read_text())
        if (
            isinstance(data, dict)
            and isinstance(data.get("items"), list)
            and data.get("source") in ("text", "ocr")
            and isinstance(data.get("needs_ocr"), bool)
            and isinstance(data.get("page"), int)
            and data["page"] > 0
        ):
            return data
    except (OSError, ValueError):
        pass
    return None


def _write(path, data):
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(
                data, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
