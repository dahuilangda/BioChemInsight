"""PaddleOCR document parsing pipeline (PaddleOCR-VL-1.6)."""
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from paddleocr import PaddleOCRVL  # type: ignore

# Reuse environment variables set in the Dockerfile
PADDLEOCR_DEVICE = __import__("os").environ.get("PADDLEOCR_DEVICE", "gpu")
PADDLEOCR_LANG = __import__("os").environ.get("PADDLEOCR_LANG", "auto")
PADDLEOCR_RENDER_SCALE = float(__import__("os").environ.get("PADDLEOCR_RENDER_SCALE", "1.3"))
PADDLEOCR_USE_DOC_ORIENTATION = __import__("os").environ.get("PADDLEOCR_USE_DOC_ORIENTATION", "false").lower() == "true"
PADDLEOCR_USE_DOC_UNWARPING = __import__("os").environ.get("PADDLEOCR_USE_DOC_UNWARPING", "false").lower() == "true"
PADDLEOCR_MAX_NEW_TOKENS = int(__import__("os").environ.get("PADDLEOCR_MAX_NEW_TOKENS", "4096"))
# decoder repetition-loop guard; the model's generate() drops per-call
# values, so this lands in the cached generation_config instead
PADDLEOCR_NO_REPEAT_NGRAM_SIZE = int(__import__("os").environ.get("PADDLEOCR_NO_REPEAT_NGRAM_SIZE", "20"))

LOGGER = logging.getLogger(__name__)


class JobCanceledError(RuntimeError):
    pass

SUPPORTED_IMAGE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


class OcrCore:
    """PaddleOCR-VL-1.6 document parsing: markdown and Word outputs."""

    def __init__(self, default_lang: Optional[str] = None):
        self.default_lang = default_lang
        self._pipelines: Dict[Optional[str], PaddleOCRVL] = {}
        # Fail fast: a dedicated worker must surface model/GPU errors at boot,
        # not on the first queued job.
        self._get_pipeline(self.default_lang)

    @staticmethod
    def _patch_generation_config() -> None:
        """The VL decoder loops on repetitive SMILES motifs (arginine/
        aspartate runs) and its generate() only forwards max_new_tokens/
        use_cache — any per-call no_repeat_ngram_size is dropped. The
        durable switch is the model's own generation_config, patched in
        the local model cache (idempotent)."""
        import json

        root = Path("/root/.paddlex/official_models/PaddleOCR-VL-1.6")
        cfg_path = root / "generation_config.json"
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if cfg.get("no_repeat_ngram_size") == PADDLEOCR_NO_REPEAT_NGRAM_SIZE:
            return
        cfg["no_repeat_ngram_size"] = PADDLEOCR_NO_REPEAT_NGRAM_SIZE
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        LOGGER.info("generation_config: no_repeat_ngram_size=%s",
                    PADDLEOCR_NO_REPEAT_NGRAM_SIZE)

    def _get_pipeline(self, lang: Optional[str]) -> PaddleOCRVL:
        if lang in self._pipelines:
            return self._pipelines[lang]
        self._patch_generation_config()
        LOGGER.info(
            "Initialising PaddleOCR-VL-1.6 pipeline (device=%s, lang=%s) …",
            PADDLEOCR_DEVICE, lang or "auto",
        )
        pipeline = PaddleOCRVL(
            pipeline_version="v1.6",
            device=PADDLEOCR_DEVICE,
            use_doc_orientation_classify=PADDLEOCR_USE_DOC_ORIENTATION,
            use_doc_unwarping=PADDLEOCR_USE_DOC_UNWARPING,
        )
        self._pipelines[lang] = pipeline
        LOGGER.info("PaddleOCR-VL-1.6 initialised (lang=%s).", lang or "auto")
        return pipeline

    def _predict_file(self, path: Path, lang: Optional[str]) -> list:
        # Cap per-region VL decoding: a degenerate block can otherwise loop
        # for tens of minutes and wedge the single-worker queue.
        return list(self._get_pipeline(lang or self.default_lang).predict(
            str(path), max_new_tokens=PADDLEOCR_MAX_NEW_TOKENS))

    @staticmethod
    def _page_number(result: Any) -> int:
        return int(result["page_index"]) + 1

    def _predict_page_range(
        self,
        pdf_path: Path,
        page_start: int,
        page_end: Optional[int],
        lang: Optional[str],
        cancel_check=None,
    ) -> Tuple[list, int]:
        """Return one result per page and the first 1-based source page."""
        import fitz

        with fitz.open(pdf_path) as doc:
            last = doc.page_count if page_end is None or page_end <= 0 else min(page_end, doc.page_count)
            if page_start < 1 or page_start > last:
                return [], 0

        results: List[Any] = []
        for page_number in range(page_start, last + 1):
            if cancel_check is not None and cancel_check():
                raise JobCanceledError(f"job canceled at page {page_number}")
            with fitz.open(pdf_path) as doc:
                slim = fitz.open()
                slim.insert_pdf(doc, from_page=page_number - 1, to_page=page_number - 1)
                slim_path = pdf_path.with_name(f"page_{page_number}.pdf")
                slim.save(slim_path)
                slim.close()
            try:
                page_results = self._predict_file(slim_path, lang)
            finally:
                slim_path.unlink(missing_ok=True)
            if len(page_results) != 1:
                raise RuntimeError(
                    f"OCR returned {len(page_results)} results for PDF page {page_number}; expected one"
                )
            results.append(page_results[0])
        return results, page_start

    @staticmethod
    def _raw_predictions(results: list) -> List[Dict[str, Any]]:
        raws = []
        for result in results:
            raw = result.json
            raws.append(raw[next(iter(raw))] if isinstance(raw, dict) and len(raw) == 1 else raw)
        return raws

    def process_pdf(
        self,
        pdf_path: Path,
        page_start: int = 1,
        page_end: Optional[int] = None,
        lang: Optional[str] = None,
        return_raw: bool = False,
        cancel_check=None,
    ) -> Tuple[List[str], List[Dict[str, Any]], List[int]]:
        results, offset = self._predict_page_range(pdf_path, int(page_start), page_end, lang, cancel_check)
        if not results:
            raise RuntimeError(f"no pages in range [{page_start}, {page_end}] of {pdf_path.name}")
        page_markdowns = [result.markdown["markdown_texts"] for result in results]
        processed_pages = list(range(offset, offset + len(results)))
        raw_predictions = self._raw_predictions(results) if return_raw else []
        return page_markdowns, raw_predictions, processed_pages

    def process_image(
        self,
        contents: bytes,
        suffix: str = ".png",
        lang: Optional[str] = None,
        return_raw: bool = False,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / f"input{suffix}"
            source.write_bytes(contents)
            results = self._predict_file(source, lang)
        if not results:
            raise RuntimeError(f"pipeline produced no result for {suffix} image")
        markdown = results[0].markdown["markdown_texts"]
        raw_predictions = self._raw_predictions(results) if return_raw else []
        return markdown, raw_predictions

    def process_pdf_to_word(
        self,
        contents: bytes,
        page_start: int = 1,
        page_end: Optional[int] = None,
        lang: Optional[str] = None,
    ) -> Path:
        """Caller owns the returned path and copies it before use."""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.pdf"
            source.write_bytes(contents)
            results, _ = self._predict_page_range(source, int(page_start), page_end, lang)
            if not results:
                raise RuntimeError(f"no pages in range [{page_start}, {page_end}]")
            output = Path(tempfile.mkdtemp(prefix="paddleocr-docx-")) / "document.docx"
            self._as_document(results).save_to_word(output)
        return output

    def process_image_to_word(
        self,
        contents: bytes,
        suffix: str = ".png",
        lang: Optional[str] = None,
    ) -> Path:
        """Caller owns the returned path and copies it before use."""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / f"input{suffix}"
            source.write_bytes(contents)
            results = self._predict_file(source, lang)
            if not results:
                raise RuntimeError(f"pipeline produced no result for {suffix} image")
            output = Path(tempfile.mkdtemp(prefix="paddleocr-docx-")) / "document.docx"
            self._as_document(results).save_to_word(output)
        return output

    def _as_document(self, results: list):
        """Merge page results into one document result when several pages."""
        if len(results) == 1:
            return results[0]
        pipeline = next(iter(self._pipelines.values()))
        return pipeline.restructure_pages(results, concatenate_pages=True)
