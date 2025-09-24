import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PPStructureV3  # type: ignore

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

LOGGER = logging.getLogger("paddle_ocr_server")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

app = FastAPI(
    title="PaddleOCR PPStructureV3 Service",
    description="Lightweight service that exposes PaddleOCR PPStructureV3 for PDF to Markdown conversion.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PaddleOCRService:
    """Singleton-like wrapper that keeps PPStructureV3 in memory."""

    def __init__(self) -> None:
        LOGGER.info("Initialising PPStructureV3 pipeline …")
        self.pipeline = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        LOGGER.info("PPStructureV3 initialised.")

    def _render_page(self, document: fitz.Document, page_index: int) -> Path:
        """Render a PDF page to an image and return the path."""
        page = document.load_page(page_index)
        matrix = fitz.Matrix(2, 2)  # upscale for better OCR fidelity
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"paddleocr_page_{page_index + 1}_"))
        image_path = temp_dir / f"page_{page_index + 1}.png"
        pix.save(image_path)  # type: ignore[arg-type]
        return image_path

    def _predict_to_markdown(self, image_path: Path, return_raw: bool) -> Tuple[str, List[dict]]:
        """Generate markdown (and optionally raw predictions) for an image."""
        workspace = image_path.parent
        predictions = self.pipeline.predict(input=str(image_path))

        markdown_parts: List[str] = []
        raw_items: List[dict] = []

        for prediction in predictions:
            prediction.save_to_markdown(save_path=str(workspace))
            if return_raw:
                prediction.save_to_json(save_path=str(workspace))

        markdown_files = sorted(workspace.glob("*.md"))
        for md_file in markdown_files:
            markdown_parts.append(md_file.read_text(encoding="utf-8"))

        if return_raw:
            json_files = sorted(workspace.glob("*.json"))
            for json_file in json_files:
                with json_file.open("r", encoding="utf-8") as fh:
                    raw_items.append(json.load(fh))

        return "\n".join(markdown_parts), raw_items

    def process_pdf(self, pdf_path: Path, page_start: int, page_end: int, return_raw: bool) -> Tuple[str, List[List[dict]], List[int]]:
        """Run OCR on the requested page range and return markdown plus optional raw predictions."""
        document = fitz.open(pdf_path)
        total_pages = len(document)

        if page_end < 0 or page_end > total_pages:
            page_end = total_pages
        if page_start < 1 or page_start > total_pages:
            raise HTTPException(status_code=400, detail="page_start is out of range")
        if page_start > page_end:
            raise HTTPException(status_code=400, detail="page_start must be <= page_end")

        separator = "\n\n-#-#-#-#-\n\n"
        page_markdowns: List[str] = []
        raw_predictions: List[List[dict]] = []
        processed_pages: List[int] = []

        try:
            for page_index in range(page_start - 1, page_end):
                LOGGER.info("Processing page %s/%s", page_index + 1, total_pages)
                image_path = self._render_page(document, page_index)

                page_markdown, page_raw_items = self._predict_to_markdown(image_path, return_raw)

                page_markdowns.append(page_markdown)
                if return_raw:
                    raw_predictions.append(page_raw_items)
                processed_pages.append(page_index + 1)

                # Clean temporary directory for this page
                shutil.rmtree(image_path.parent, ignore_errors=True)

        finally:
            document.close()

        joined_markdown = separator.join(page_markdowns)
        return joined_markdown, raw_predictions, processed_pages

    def process_image(self, contents: bytes, suffix: str, return_raw: bool) -> Tuple[str, List[dict]]:
        """Run OCR on a single image represented by bytes."""
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        temp_dir = Path(tempfile.mkdtemp(prefix="paddleocr_image_"))
        image_suffix = suffix if suffix else ".png"
        image_path = temp_dir / f"uploaded{image_suffix.lower()}"

        try:
            image_path.write_bytes(contents)
            markdown, raw_predictions = self._predict_to_markdown(image_path, return_raw)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return markdown, raw_predictions


service = PaddleOCRService()


@app.post("/v1/pdf-to-markdown")
async def pdf_to_markdown_endpoint(
    file: UploadFile = File(..., description="PDF document to analyse"),
    page_start: int = Form(1, description="1-based index of the first page to analyse"),
    page_end: int = Form(-1, description="1-based index of the last page to analyse. Use -1 to process all remaining pages."),
    lang: str = Form('en', description="Language hint (reserved for future use)"),
    return_raw: bool = Form(False, description="Return raw JSON predictions alongside markdown output"),
) -> dict:
    """Convert a PDF into markdown text using PaddleOCR."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    LOGGER.info(
        "Received OCR request for %s (pages %s-%s, lang=%s, return_raw=%s)",
        file.filename,
        page_start,
        page_end,
        lang,
        return_raw,
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(contents)
        tmp_pdf_path = Path(tmp_pdf.name)

    try:
        markdown, raw_predictions, processed_pages = service.process_pdf(
            pdf_path=tmp_pdf_path,
            page_start=page_start,
            page_end=page_end,
            return_raw=return_raw,
        )
    finally:
        try:
            tmp_pdf_path.unlink(missing_ok=True)
        except OSError:
            LOGGER.warning("Failed to delete temporary file %s", tmp_pdf_path)

    response: dict = {
        "markdown": markdown,
        "page_numbers": processed_pages,
        "page_count": len(processed_pages),
    }

    if return_raw:
        response["raw_predictions"] = raw_predictions

    return response


@app.post("/v1/image-to-markdown")
async def image_to_markdown_endpoint(
    file: UploadFile = File(..., description="Image to analyse"),
    lang: str = Form("en", description="Language hint (reserved for future use)"),
    return_raw: bool = Form(False, description="Return raw JSON predictions alongside markdown output"),
) -> dict:
    """Convert a single page image into markdown text using PaddleOCR."""
    filename = file.filename or "uploaded.png"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    LOGGER.info(
        "Received image OCR request for %s (lang=%s, return_raw=%s)",
        filename,
        lang,
        return_raw,
    )

    markdown, raw_predictions = service.process_image(contents=contents, suffix=suffix, return_raw=return_raw)

    response: dict = {
        "markdown": markdown,
        "page_numbers": [1],
        "page_count": 1,
    }

    if return_raw:
        response["raw_predictions"] = raw_predictions

    return response
