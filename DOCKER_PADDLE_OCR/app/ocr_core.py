from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
from paddleocr import PPStructureV3  # type: ignore

LOGGER = logging.getLogger("paddle_ocr_core")

PADDLEOCR_DEVICE = (os.getenv("PADDLEOCR_DEVICE", "gpu") or "gpu").strip()
PADDLEOCR_RENDER_SCALE = float(os.getenv("PADDLEOCR_RENDER_SCALE", "1.3") or "1.3")
PADDLEOCR_USE_DOC_ORIENTATION = (os.getenv("PADDLEOCR_USE_DOC_ORIENTATION", "false") or "false").strip().lower() in {"1", "true", "yes", "on"}
PADDLEOCR_USE_DOC_UNWARPING = (os.getenv("PADDLEOCR_USE_DOC_UNWARPING", "false") or "false").strip().lower() in {"1", "true", "yes", "on"}
PADDLEOCR_LANG = (os.getenv("PADDLEOCR_LANG", "auto") or "auto").strip()

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class PaddleOCRCore:
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

    def render_page(self, document: fitz.Document, page_index: int) -> Path:
        page = document.load_page(page_index)
        matrix = fitz.Matrix(PADDLEOCR_RENDER_SCALE, PADDLEOCR_RENDER_SCALE)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"paddleocr_page_{page_index + 1}_"))
        image_path = temp_dir / f"page_{page_index + 1}.png"
        pix.save(image_path)  # type: ignore[arg-type]
        pix = None
        page = None
        return image_path

    def predict_to_markdown(self, image_path: Path, return_raw: bool, lang: Optional[str]) -> Tuple[str, List[dict]]:
        workspace = image_path.parent
        predictions = self._get_pipeline(lang).predict(input=str(image_path))
        markdown_parts: List[str] = []
        raw_items: List[dict] = []

        for prediction in predictions:
            prediction.save_to_markdown(save_path=str(workspace))
            if return_raw:
                prediction.save_to_json(save_path=str(workspace))

        for md_file in sorted(workspace.glob("*.md")):
            markdown_parts.append(md_file.read_text(encoding="utf-8"))

        if return_raw:
            for json_file in sorted(workspace.glob("*.json")):
                with json_file.open("r", encoding="utf-8") as fh:
                    raw_items.append(json.load(fh))

        return "\n".join(markdown_parts), raw_items

    def predict_to_word(self, input_path: Path, lang: Optional[str]) -> Path:
        workspace = input_path.parent
        predictions = self._get_pipeline(lang).predict(input=str(input_path))
        for prediction in predictions:
            if not hasattr(prediction, "save_to_word"):
                raise RuntimeError("Installed PaddleOCR does not support Word export.")
            prediction.save_to_word(save_path=str(workspace))
        docx_files = sorted(workspace.glob("*.docx"))
        if not docx_files:
            raise RuntimeError("PaddleOCR did not generate a Word document.")
        if len(docx_files) == 1:
            return docx_files[0]
        archive_path = workspace / "paddleocr-word-output.zip"
        import zipfile

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for docx_file in docx_files:
                archive.write(docx_file, arcname=docx_file.name)
        return archive_path

    def process_pdf(self, pdf_path: Path, page_start: int, page_end: int, lang: str | None, return_raw: bool) -> Tuple[List[str], List[List[dict]], List[int]]:
        document = fitz.open(pdf_path)
        total_pages = len(document)
        normalized_lang = self._normalize_lang(lang)

        if page_end < 0 or page_end > total_pages:
            page_end = total_pages
        if page_start < 1 or page_start > total_pages:
            raise ValueError("page_start is out of range")
        if page_start > page_end:
            raise ValueError("page_start must be <= page_end")

        page_markdowns: List[str] = []
        raw_predictions: List[List[dict]] = []
        processed_pages: List[int] = []

        try:
            for page_index in range(page_start - 1, page_end):
                image_path = self.render_page(document, page_index)
                page_markdown, page_raw_items = self.predict_to_markdown(image_path, return_raw, normalized_lang)
                page_markdowns.append(page_markdown)
                if return_raw:
                    raw_predictions.append(page_raw_items)
                processed_pages.append(page_index + 1)
                shutil.rmtree(image_path.parent, ignore_errors=True)
        finally:
            document.close()

        return page_markdowns, raw_predictions, processed_pages

    def process_image(self, contents: bytes, suffix: str, lang: str | None, return_raw: bool) -> Tuple[str, List[dict]]:
        if not contents:
            raise ValueError("Uploaded file is empty.")
        temp_dir = Path(tempfile.mkdtemp(prefix="paddleocr_image_"))
        image_suffix = suffix if suffix else ".png"
        image_path = temp_dir / f"uploaded{image_suffix.lower()}"
        try:
            image_path.write_bytes(contents)
            markdown, raw_predictions = self.predict_to_markdown(image_path, return_raw, self._normalize_lang(lang))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return markdown, raw_predictions

    def process_pdf_to_word(self, contents: bytes, page_start: int, page_end: int, lang: str | None) -> Path:
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
                raise ValueError("page_start is out of range")
            if page_start > page_end:
                raise ValueError("page_start must be <= page_end")
            if page_start == 1 and page_end == total_pages:
                source_path = input_pdf_path
            else:
                subset_path = temp_dir / f"pages_{page_start}_{page_end}.pdf"
                with fitz.open() as subset:
                    subset.insert_pdf(document, from_page=page_start - 1, to_page=page_end - 1)
                    subset.save(subset_path)
                source_path = subset_path
        finally:
            if document is not None:
                document.close()
        return self.predict_to_word(source_path, self._normalize_lang(lang))

    def process_image_to_word(self, contents: bytes, suffix: str, lang: str | None) -> Path:
        if not contents:
            raise ValueError("Uploaded file is empty.")
        temp_dir = Path(tempfile.mkdtemp(prefix="paddleocr_word_image_"))
        image_suffix = suffix if suffix else ".png"
        image_path = temp_dir / f"uploaded{image_suffix.lower()}"
        image_path.write_bytes(contents)
        return self.predict_to_word(image_path, self._normalize_lang(lang))


def safe_rmtree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
