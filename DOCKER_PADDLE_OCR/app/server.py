import json
import importlib.metadata as package_metadata
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from paddleocr import PPStructureV3  # type: ignore
from starlette.background import BackgroundTask

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

LOGGER = logging.getLogger("paddle_ocr_server")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
PADDLEOCR_DEVICE = (os.getenv("PADDLEOCR_DEVICE", "gpu") or "gpu").strip()
PADDLEOCR_RENDER_SCALE = float(os.getenv("PADDLEOCR_RENDER_SCALE", "1.3") or "1.3")
PADDLEOCR_USE_DOC_ORIENTATION = (os.getenv("PADDLEOCR_USE_DOC_ORIENTATION", "false") or "false").strip().lower() in {"1", "true", "yes", "on"}
PADDLEOCR_USE_DOC_UNWARPING = (os.getenv("PADDLEOCR_USE_DOC_UNWARPING", "false") or "false").strip().lower() in {"1", "true", "yes", "on"}
PADDLEOCR_LANG = (os.getenv("PADDLEOCR_LANG", "auto") or "auto").strip()

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
        self._pipelines: Dict[Optional[str], PPStructureV3] = {}
        self.default_lang = self._normalize_lang(PADDLEOCR_LANG)
        self._get_pipeline(self.default_lang)

    @staticmethod
    def _normalize_lang(lang: str | None) -> Optional[str]:
        normalized = (lang or "").strip().lower()
        if normalized in {"", "auto", "default", "none"}:
            return None
        return normalized

    def _get_pipeline(self, lang: Optional[str]) -> PPStructureV3:
        if lang in self._pipelines:
            return self._pipelines[lang]
        LOGGER.info(
            "Initialising PPStructureV3 pipeline (device=%s, lang=%s, render_scale=%s, orientation=%s, unwarp=%s) …",
            PADDLEOCR_DEVICE,
            lang or "auto",
            PADDLEOCR_RENDER_SCALE,
            PADDLEOCR_USE_DOC_ORIENTATION,
            PADDLEOCR_USE_DOC_UNWARPING,
        )
        pipeline = PPStructureV3(
            device=PADDLEOCR_DEVICE,
            lang=lang,
            use_doc_orientation_classify=PADDLEOCR_USE_DOC_ORIENTATION,
            use_doc_unwarping=PADDLEOCR_USE_DOC_UNWARPING,
        )
        self._pipelines[lang] = pipeline
        LOGGER.info("PPStructureV3 initialised (lang=%s).", lang or "auto")
        return pipeline

    def _render_page(self, document: fitz.Document, page_index: int) -> Path:
        """Render a PDF page to an image and return the path."""
        page = document.load_page(page_index)
        matrix = fitz.Matrix(PADDLEOCR_RENDER_SCALE, PADDLEOCR_RENDER_SCALE)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"paddleocr_page_{page_index + 1}_"))
        image_path = temp_dir / f"page_{page_index + 1}.png"
        pix.save(image_path)  # type: ignore[arg-type]
        pix = None
        page = None
        return image_path

    def _predict_to_markdown(self, image_path: Path, return_raw: bool, lang: Optional[str]) -> Tuple[str, List[dict]]:
        """Generate markdown (and optionally raw predictions) for an image."""
        workspace = image_path.parent
        predictions = self._get_pipeline(lang).predict(input=str(image_path))

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

    def _predict_to_word(self, input_path: Path, lang: Optional[str]) -> Path:
        """Generate a DOCX or ZIP archive with DOCX files for the input document."""
        workspace = input_path.parent
        predictions = self._get_pipeline(lang).predict(input=str(input_path))

        for prediction in predictions:
            if not hasattr(prediction, "save_to_word"):
                raise HTTPException(
                    status_code=501,
                    detail="Installed PaddleOCR does not support Word export. Upgrade to paddleocr>=3.5.0.",
                )
            prediction.save_to_word(save_path=str(workspace))

        docx_files = sorted(workspace.glob("*.docx"))
        if not docx_files:
            raise HTTPException(status_code=500, detail="PaddleOCR did not generate a Word document.")
        if len(docx_files) == 1:
            return docx_files[0]

        archive_path = workspace / "paddleocr-word-output.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for docx_file in docx_files:
                archive.write(docx_file, arcname=docx_file.name)
        return archive_path

    def process_pdf(self, pdf_path: Path, page_start: int, page_end: int, lang: str | None, return_raw: bool) -> Tuple[List[str], List[List[dict]], List[int]]:
        """Run OCR on the requested page range and return per-page markdown plus optional raw predictions."""
        document = fitz.open(pdf_path)
        total_pages = len(document)
        normalized_lang = self._normalize_lang(lang)

        if page_end < 0 or page_end > total_pages:
            page_end = total_pages
        if page_start < 1 or page_start > total_pages:
            raise HTTPException(status_code=400, detail="page_start is out of range")
        if page_start > page_end:
            raise HTTPException(status_code=400, detail="page_start must be <= page_end")

        page_markdowns: List[str] = []
        raw_predictions: List[List[dict]] = []
        processed_pages: List[int] = []

        try:
            for page_index in range(page_start - 1, page_end):
                LOGGER.info("Processing page %s/%s", page_index + 1, total_pages)
                image_path = self._render_page(document, page_index)

                page_markdown, page_raw_items = self._predict_to_markdown(image_path, return_raw, normalized_lang)

                page_markdowns.append(page_markdown)
                if return_raw:
                    raw_predictions.append(page_raw_items)
                processed_pages.append(page_index + 1)

                # Clean temporary directory for this page
                shutil.rmtree(image_path.parent, ignore_errors=True)

        finally:
            document.close()

        return page_markdowns, raw_predictions, processed_pages

    def process_image(self, contents: bytes, suffix: str, lang: str | None, return_raw: bool) -> Tuple[str, List[dict]]:
        """Run OCR on a single image represented by bytes."""
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        temp_dir = Path(tempfile.mkdtemp(prefix="paddleocr_image_"))
        image_suffix = suffix if suffix else ".png"
        image_path = temp_dir / f"uploaded{image_suffix.lower()}"

        try:
            image_path.write_bytes(contents)
            markdown, raw_predictions = self._predict_to_markdown(image_path, return_raw, self._normalize_lang(lang))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return markdown, raw_predictions

    def process_pdf_to_word(self, contents: bytes, page_start: int, page_end: int, lang: str | None) -> Path:
        """Run OCR on a PDF or page subset and return a generated Word document path."""
        temp_dir = Path(tempfile.mkdtemp(prefix="paddleocr_word_pdf_"))
        input_pdf_path = temp_dir / "input.pdf"
        input_pdf_path.write_bytes(contents)
        document = None

        try:
            document = fitz.open(input_pdf_path)
            total_pages = len(document)
            if page_end < 0 or page_end > total_pages:
                page_end = total_pages
            if page_start < 1 or page_start > total_pages:
                raise HTTPException(status_code=400, detail="page_start is out of range")
            if page_start > page_end:
                raise HTTPException(status_code=400, detail="page_start must be <= page_end")

            if page_start == 1 and page_end == total_pages:
                source_path = input_pdf_path
            else:
                subset_path = temp_dir / f"pages_{page_start}_{page_end}.pdf"
                with fitz.open() as subset:
                    subset.insert_pdf(document, from_page=page_start - 1, to_page=page_end - 1)
                    subset.save(subset_path)
                source_path = subset_path
        except HTTPException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=f"Unable to read PDF: {exc}") from exc
        finally:
            if document is not None:
                document.close()

        try:
            return self._predict_to_word(source_path, self._normalize_lang(lang))
        except HTTPException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        except ModuleNotFoundError as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(
                status_code=500,
                detail=f"Word export dependency is missing: {exc.name}. Rebuild the image with python-docx.",
            ) from exc
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"Unable to export Word document: {exc}") from exc

    def process_image_to_word(self, contents: bytes, suffix: str, lang: str | None) -> Path:
        """Run OCR on a single image and return a generated Word document path."""
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        temp_dir = Path(tempfile.mkdtemp(prefix="paddleocr_word_image_"))
        image_suffix = suffix if suffix else ".png"
        image_path = temp_dir / f"uploaded{image_suffix.lower()}"
        image_path.write_bytes(contents)
        return self._predict_to_word(image_path, self._normalize_lang(lang))


service = PaddleOCRService()


@app.get("/healthz")
async def healthz() -> dict:
    versions = {}
    for package_name in ("paddleocr", "paddlex", "paddlepaddle-gpu", "paddlepaddle"):
        try:
            versions[package_name] = package_metadata.version(package_name)
        except package_metadata.PackageNotFoundError:
            versions[package_name] = None

    return {
        "status": "ok",
        "device": PADDLEOCR_DEVICE,
        "lang": service.default_lang or "auto",
        "render_scale": PADDLEOCR_RENDER_SCALE,
        "versions": versions,
    }


@app.post("/v1/pdf-to-markdown")
async def pdf_to_markdown_endpoint(
    file: UploadFile = File(..., description="PDF document to analyse"),
    page_start: int = Form(1, description="1-based index of the first page to analyse"),
    page_end: int = Form(-1, description="1-based index of the last page to analyse. Use -1 to process all remaining pages."),
    lang: str = Form("auto", description="Language hint for PaddleOCR. Use auto/default for PaddleOCR defaults."),
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
        page_markdowns, raw_predictions, processed_pages = service.process_pdf(
            pdf_path=tmp_pdf_path,
            page_start=page_start,
            page_end=page_end,
            lang=lang,
            return_raw=return_raw,
        )
    finally:
        try:
            tmp_pdf_path.unlink(missing_ok=True)
        except OSError:
            LOGGER.warning("Failed to delete temporary file %s", tmp_pdf_path)

    response: dict = {
        "page_markdowns": page_markdowns,
        "pages": [
            {
                "page_number": page_number,
                "markdown": page_markdown,
            }
            for page_number, page_markdown in zip(processed_pages, page_markdowns)
        ],
        "page_numbers": processed_pages,
        "page_count": len(processed_pages),
    }

    if return_raw:
        response["raw_predictions"] = raw_predictions

    return response


def _download_generated_document(path: Path, filename: str) -> FileResponse:
    media_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if path.suffix.lower() == ".docx"
        else "application/zip"
    )
    return FileResponse(
        path=path,
        media_type=media_type,
        filename=filename,
        background=BackgroundTask(shutil.rmtree, path.parent, ignore_errors=True),
    )


@app.post("/v1/pdf-to-word")
async def pdf_to_word_endpoint(
    file: UploadFile = File(..., description="PDF document to convert into Word"),
    page_start: int = Form(1, description="1-based index of the first page to analyse"),
    page_end: int = Form(-1, description="1-based index of the last page to analyse. Use -1 to process all remaining pages."),
    lang: str = Form("auto", description="Language hint for PaddleOCR. Use auto/default for PaddleOCR defaults."),
) -> FileResponse:
    """Convert a PDF into a Word document using PaddleOCR 3.5+."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    LOGGER.info(
        "Received Word export request for %s (pages %s-%s, lang=%s)",
        file.filename,
        page_start,
        page_end,
        lang,
    )

    output_path = service.process_pdf_to_word(contents=contents, page_start=page_start, page_end=page_end, lang=lang)
    download_name = f"{Path(file.filename).stem}_paddleocr_word{output_path.suffix}"
    return _download_generated_document(output_path, download_name)


@app.post("/v1/image-to-word")
async def image_to_word_endpoint(
    file: UploadFile = File(..., description="Image to convert into Word"),
    lang: str = Form("auto", description="Language hint for PaddleOCR. Use auto/default for PaddleOCR defaults."),
) -> FileResponse:
    """Convert a page image into a Word document using PaddleOCR 3.5+."""
    filename = file.filename or "uploaded.png"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    LOGGER.info("Received image Word export request for %s (lang=%s)", filename, lang)
    output_path = service.process_image_to_word(contents=contents, suffix=suffix, lang=lang)
    download_name = f"{Path(filename).stem}_paddleocr_word{output_path.suffix}"
    return _download_generated_document(output_path, download_name)


@app.post("/v1/image-to-markdown")
async def image_to_markdown_endpoint(
    file: UploadFile = File(..., description="Image to analyse"),
    lang: str = Form("auto", description="Language hint for PaddleOCR. Use auto/default for PaddleOCR defaults."),
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

    markdown, raw_predictions = service.process_image(contents=contents, suffix=suffix, lang=lang, return_raw=return_raw)

    response: dict = {
        "page_markdowns": [markdown],
        "pages": [{"page_number": 1, "markdown": markdown}],
        "page_numbers": [1],
        "page_count": 1,
    }

    if return_raw:
        response["raw_predictions"] = raw_predictions

    return response
