import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
import sys
import argparse
import json
import gc
import hashlib
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import PyPDF2
import re
import fitz
import numpy as np
import tempfile
import requests
from utils.compound_id_utils import build_compound_id_alias_map, resolve_compound_id_alias, remap_assay_dict_to_official_ids, normalize_compound_id_text, canonicalize_record_compound_ids, resolve_compound_id_with_trace
from utils.llm_utils import resolve_compound_id_alias as resolve_compound_id_alias_with_llm
from utils.paddleocr_client import request_pdf_to_markdown
from utils.skill_prompt_loader import render_skill_reference
from utils.model_harness import parse_validated_json_object, require_decision_contract, run_json_task

try:  # noqa: SIM105
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - dependency is expected in Docker image
    Image = None
    ImageDraw = None

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stderr = open(os.devnull, 'w')

# Suppress other warnings
import warnings
warnings.filterwarnings("ignore")

try:  # noqa: SIM105
    import constants as _constants
except ImportError:  # pragma: no cover - optional user configuration
    _constants = None

STRUCTURE_AUTO_DETECT_VISION_BATCH_SIZE = int(getattr(_constants, 'STRUCTURE_AUTO_DETECT_VISION_BATCH_SIZE', 12)) if _constants else 12
STRUCTURE_AUTO_DETECT_VISION_COLUMNS = int(getattr(_constants, 'STRUCTURE_AUTO_DETECT_VISION_COLUMNS', 3)) if _constants else 3
STRUCTURE_AUTO_DETECT_VISION_RENDER_SCALE = float(getattr(_constants, 'STRUCTURE_AUTO_DETECT_VISION_RENDER_SCALE', 0.7)) if _constants else 0.7
STRUCTURE_AUTO_DETECT_VISION_THUMB_WIDTH = int(getattr(_constants, 'STRUCTURE_AUTO_DETECT_VISION_THUMB_WIDTH', 420)) if _constants else 420
STRUCTURE_AUTO_DETECT_VISION_MAX_RETRIES = int(getattr(_constants, 'STRUCTURE_AUTO_DETECT_VISION_MAX_RETRIES', 1)) if _constants else 1
STRUCTURE_AUTO_DETECT_VISION_REVIEW_ENABLED = bool(getattr(_constants, 'STRUCTURE_AUTO_DETECT_VISION_REVIEW_ENABLED', True)) if _constants else True
STRUCTURE_AUTO_DETECT_VISION_REVIEW_BATCH_SIZE = int(getattr(_constants, 'STRUCTURE_AUTO_DETECT_VISION_REVIEW_BATCH_SIZE', 4)) if _constants else 4
STRUCTURE_AUTO_DETECT_VISION_REVIEW_COLUMNS = int(getattr(_constants, 'STRUCTURE_AUTO_DETECT_VISION_REVIEW_COLUMNS', 2)) if _constants else 2
STRUCTURE_AUTO_DETECT_VISION_REVIEW_RENDER_SCALE = float(getattr(_constants, 'STRUCTURE_AUTO_DETECT_VISION_REVIEW_RENDER_SCALE', 1.0)) if _constants else 1.0
STRUCTURE_AUTO_DETECT_VISION_REVIEW_THUMB_WIDTH = int(getattr(_constants, 'STRUCTURE_AUTO_DETECT_VISION_REVIEW_THUMB_WIDTH', 700)) if _constants else 700
ASSAY_AUTO_DETECT_LLM_BATCH_SIZE = int(getattr(_constants, 'ASSAY_AUTO_DETECT_LLM_BATCH_SIZE', 6)) if _constants else 6
ASSAY_AUTO_DETECT_LLM_MAX_PAGE_CHARS = int(getattr(_constants, 'ASSAY_AUTO_DETECT_LLM_MAX_PAGE_CHARS', 5000)) if _constants else 5000
ASSAY_AUTO_DETECT_LLM_MAX_RETRIES = int(getattr(_constants, 'ASSAY_AUTO_DETECT_LLM_MAX_RETRIES', 1)) if _constants else 1
ASSAY_AUTO_DETECT_OCR_BATCH_SIZE = int(getattr(_constants, 'ASSAY_AUTO_DETECT_OCR_BATCH_SIZE', 3)) if _constants else 3
ASSAY_AUTO_DETECT_OCR_CONCURRENCY = int(getattr(_constants, 'ASSAY_AUTO_DETECT_OCR_CONCURRENCY', 1)) if _constants else 1
ASSAY_AUTO_DETECT_OCR_SPLIT_PDF = bool(getattr(_constants, 'ASSAY_AUTO_DETECT_OCR_SPLIT_PDF', True)) if _constants else True
ASSAY_AUTO_DETECT_OCR_TIMEOUT_SECONDS = int(getattr(_constants, 'ASSAY_AUTO_DETECT_OCR_TIMEOUT_SECONDS', 360)) if _constants else 360
ASSAY_AUTO_DETECT_OCR_SPLIT_RETRY_ENABLED = bool(getattr(_constants, 'ASSAY_AUTO_DETECT_OCR_SPLIT_RETRY_ENABLED', True)) if _constants else True
ASSAY_AUTO_DETECT_LLM_TIMEOUT_SECONDS = int(getattr(_constants, 'ASSAY_AUTO_DETECT_LLM_TIMEOUT_SECONDS', 120)) if _constants else 120
DEFAULT_OCR_LANG = str(getattr(_constants, 'PADDLEOCR_LANG', 'auto') or 'auto') if _constants else 'auto'
DOCUMENT_AUTO_DETECT_CACHE_ENABLED = bool(getattr(_constants, 'DOCUMENT_AUTO_DETECT_CACHE_ENABLED', True)) if _constants else True
DOCUMENT_AUTO_DETECT_CACHE_DIR = str(getattr(_constants, 'DOCUMENT_AUTO_DETECT_CACHE_DIR', '') or '').strip() if _constants else ''
DOCUMENT_AUTO_DETECT_CACHE_VERSION = 8


def _normalize_assay_name_for_dedupe(name):
    text = str(name or '').strip().lower()
    return re.sub(r'\s+', ' ', text).strip()


def normalize_detected_assay_names(assay_names):
    """Only remove exact duplicate assay names; semantic reconciliation is model-owned."""
    ordered = []
    seen = set()
    for raw_name in assay_names or []:
        name = str(raw_name or '').strip()
        key = _normalize_assay_name_for_dedupe(name)
        if name and key and key not in seen:
            seen.add(key)
            ordered.append(name)
    return ordered


def _build_assay_name_reconciliation_context(page_markdowns, max_chars=12000):
    parts = []
    remaining = max(0, int(max_chars or 0))
    for page_num in sorted((page_markdowns or {}).keys()):
        text = str((page_markdowns or {}).get(page_num) or '').strip()
        if not text or remaining <= 0:
            continue
        if len(text) > remaining:
            text = text[:remaining] + "\n...[truncated]"
        parts.append(f"[Page {page_num}]\n{text}")
        remaining -= len(text)
    return "\n\n".join(parts)


def _normalize_assay_reconciliation_evidence_pages(raw_pages, assay_name):
    pages = []
    for page_num in raw_pages:
        if page_num < 1:
            raise ValueError(
                f"reconcile_detected_assay_names evidence_pages for {assay_name!r} contains invalid page {page_num}"
            )
        pages.append(page_num)
    return pages


def _page_decision_supports_assay(page_decision, assay_name):
    if not isinstance(page_decision, dict) or page_decision.get('has_assay_data') is not True:
        return False
    target_key = _normalize_assay_name_for_dedupe(assay_name)
    for raw_name in page_decision.get('assay_names') or []:
        if _normalize_assay_name_for_dedupe(raw_name) == target_key:
            return True
    return False


def reconcile_detected_assay_names_with_model(
    assay_names,
    decisions_by_page=None,
    page_markdowns=None,
    audit_path=None,
    metadata=None,
):
    assay_names = normalize_detected_assay_names(assay_names)
    if len(assay_names) <= 1:
        return assay_names

    from utils.llm_utils import (
        TEXT_MODEL_OUTPUT_SCHEMAS,
        build_reconcile_detected_assay_names_prompt,
        run_text_json_task,
    )

    original_set = set(assay_names)
    ocr_context = _build_assay_name_reconciliation_context(page_markdowns or {})
    prompt = build_reconcile_detected_assay_names_prompt(
        assay_names,
        page_decisions=decisions_by_page or {},
        ocr_context=ocr_context,
    )

    def _parser(response_text):
        schema = TEXT_MODEL_OUTPUT_SCHEMAS.get('reconcile_detected_assay_names', {})
        payload = parse_validated_json_object(
            response_text,
            schema,
            'reconcile_detected_assay_names',
        )
        raw_kept = payload.get('assay_names')
        if not isinstance(raw_kept, list):
            raise ValueError("reconcile_detected_assay_names assay_names must be a list")
        kept = normalize_detected_assay_names(raw_kept)
        if not kept:
            raise ValueError("reconcile_detected_assay_names returned no assay names")
        outside = [name for name in kept if name not in original_set]
        if outside:
            raise ValueError(
                "reconcile_detected_assay_names returned names outside candidates: "
                + ", ".join(outside[:10])
            )

        decisions = payload.get('decisions')
        if not isinstance(decisions, dict):
            raise ValueError("reconcile_detected_assay_names decisions must be an object")
        missing = [name for name in assay_names if name not in decisions]
        if missing:
            raise ValueError(
                "reconcile_detected_assay_names decisions missing candidates: "
                + ", ".join(missing[:10])
            )

        kept_set = set(kept)
        for name in assay_names:
            decision = decisions.get(name)
            require_decision_contract(
                decision,
                schema,
                'reconcile_detected_assay_names',
                decision_key=name,
            )
            if not isinstance(decision.get('keep'), bool):
                raise ValueError(f"reconcile_detected_assay_names decision for {name!r} has non-boolean keep")
            canonical = str(decision.get('canonical_assay_name') or '').strip()
            if decision.get('keep') is True:
                if canonical != name or name not in kept_set:
                    raise ValueError(f"reconcile_detected_assay_names keep=true mismatch for {name!r}")
                evidence_pages = _normalize_assay_reconciliation_evidence_pages(
                    decision.get('evidence_pages'),
                    name,
                )
                unsupported_pages = [
                    page
                    for page in evidence_pages
                    if not _page_decision_supports_assay((decisions_by_page or {}).get(int(page)), name)
                ]
                if unsupported_pages:
                    raise ValueError(
                        f"reconcile_detected_assay_names keep=true evidence_pages unsupported for {name!r}: "
                        + ", ".join(str(page) for page in unsupported_pages[:10])
                    )
            else:
                if canonical != 'None' and canonical not in kept_set:
                    raise ValueError(f"reconcile_detected_assay_names invalid canonical for {name!r}: {canonical!r}")
        return [name for name in assay_names if name in kept_set]

    return run_text_json_task(
        task_name='reconcile_detected_assay_names',
        prompt=prompt,
        parser=_parser,
        retry=max(1, int(ASSAY_AUTO_DETECT_LLM_MAX_RETRIES or 1) + 1),
        audit_path=audit_path,
        timeout_seconds=max(15, int(ASSAY_AUTO_DETECT_LLM_TIMEOUT_SECONDS or 120)),
        metadata={
            'candidate_count': len(assay_names),
            'page_decision_count': len(decisions_by_page or {}),
            'ocr_context_chars': len(ocr_context),
            **(metadata or {}),
        },
    )


def get_total_pages(pdf_file):
    """
    获取 PDF 文件的总页数。
    """
    with open(pdf_file, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        return len(pdf_reader.pages)


def _normalize_page_text(text):
    return re.sub(r'\s+', ' ', text or '').strip()


def _load_pdf_page_texts(pdf_file):
    doc = fitz.open(pdf_file)
    try:
        return [(idx + 1, _normalize_page_text(doc[idx].get_text("text"))) for idx in range(doc.page_count)]
    finally:
        doc.close()


def _get_document_auto_detect_cache_dir():
    cache_dir = DOCUMENT_AUTO_DETECT_CACHE_DIR or os.path.join('/tmp', 'biocheminsight_auto_detect_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_document_auto_detect_cache_path(pdf_file, assay_names=None):
    assay_names = [name.strip() for name in (assay_names or []) if name and name.strip()]
    stat = os.stat(pdf_file)
    cache_payload = {
        'pdf_path': os.path.abspath(pdf_file),
        'size': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
        'assay_names': sorted(assay_names),
        'cache_version': DOCUMENT_AUTO_DETECT_CACHE_VERSION,
    }
    cache_key = hashlib.sha256(json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
    return os.path.join(_get_document_auto_detect_cache_dir(), f'{cache_key}.json')


def _load_document_auto_detect_cache(pdf_file, assay_names=None):
    if not DOCUMENT_AUTO_DETECT_CACHE_ENABLED:
        return None
    cache_path = _get_document_auto_detect_cache_path(pdf_file, assay_names=assay_names)
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        if not isinstance(cached, dict):
            return None
        return cached
    except Exception:
        return None


def _save_document_auto_detect_cache(pdf_file, assay_names, plan):
    if not DOCUMENT_AUTO_DETECT_CACHE_ENABLED:
        return None
    cache_path = _get_document_auto_detect_cache_path(pdf_file, assay_names=assay_names)
    payload = {
        'pdf_file': os.path.abspath(pdf_file),
        'cache_version': DOCUMENT_AUTO_DETECT_CACHE_VERSION,
        'assay_names_input': [name.strip() for name in (assay_names or []) if name and name.strip()],
        'plan': plan,
    }
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return cache_path
    except Exception:
        return None


def _coerce_page_texts(page_texts):
    if page_texts is None:
        return None
    if isinstance(page_texts, dict):
        return sorted(page_texts.items())
    return list(page_texts)


def _render_pdf_page_to_array(doc, page_index, scale=1.0):
    page = doc[page_index]
    matrix = fitz.Matrix(float(scale), float(scale))
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        image = image[:, :, :3]
    elif pix.n == 1:
        image = np.repeat(image[:, :, None], 3, axis=2)
    return image


def _chunked(items, chunk_size):
    chunk_size = max(1, int(chunk_size or 1))
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def _render_pdf_page_thumbnail(doc, page_num, thumb_width, scale=None):
    image = _render_pdf_page_to_array(
        doc,
        page_num - 1,
        scale=STRUCTURE_AUTO_DETECT_VISION_RENDER_SCALE if scale is None else scale,
    )
    pil_image = Image.fromarray(image)
    width, height = pil_image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Rendered page {page_num} has invalid dimensions")
    target_width = max(120, int(thumb_width or STRUCTURE_AUTO_DETECT_VISION_THUMB_WIDTH))
    ratio = target_width / float(width)
    target_height = max(1, int(height * ratio))
    return pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _build_structure_page_contact_sheet(
    pdf_file,
    page_numbers,
    output_path,
    *,
    columns=None,
    thumb_width=None,
    render_scale=None,
):
    if Image is None or ImageDraw is None:
        raise ImportError("Pillow is required for vision contact-sheet structure page detection")

    page_numbers = [int(page) for page in page_numbers]
    columns = max(1, int(STRUCTURE_AUTO_DETECT_VISION_COLUMNS if columns is None else columns))
    thumb_width = max(120, int(STRUCTURE_AUTO_DETECT_VISION_THUMB_WIDTH if thumb_width is None else thumb_width))
    label_height = 34
    padding = 12
    border = 2

    thumbnails = []
    doc = fitz.open(pdf_file)
    try:
        for page_num in page_numbers:
            thumb = _render_pdf_page_thumbnail(doc, page_num, thumb_width, scale=render_scale)
            thumbnails.append((page_num, thumb))
    finally:
        doc.close()

    if not thumbnails:
        raise ValueError("No pages provided for contact-sheet rendering")

    cell_width = thumb_width + padding * 2
    cell_height = max(thumb.height for _, thumb in thumbnails) + label_height + padding * 2
    rows = int(math.ceil(len(thumbnails) / float(columns)))
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)

    for index, (page_num, thumb) in enumerate(thumbnails):
        row = index // columns
        col = index % columns
        x = col * cell_width
        y = row * cell_height
        label = f"PAGE {page_num}"
        draw.rectangle([x, y, x + cell_width - 1, y + cell_height - 1], outline=(40, 55, 90), width=border)
        draw.rectangle([x, y, x + cell_width - 1, y + label_height], fill=(30, 41, 59))
        draw.text((x + padding, y + 9), label, fill=(255, 255, 255))
        image_x = x + padding
        image_y = y + label_height + padding
        sheet.paste(thumb, (image_x, image_y))
        draw.rectangle(
            [image_x, image_y, image_x + thumb.width - 1, image_y + thumb.height - 1],
            outline=(203, 213, 225),
            width=1,
        )

    sheet.save(output_path, format="PNG", optimize=True)
    thumbnails.clear()
    gc.collect()


def _build_structure_page_detection_prompt(page_numbers):
    return render_skill_reference(
        'biocheminsight-vision-models',
        'references/detect_structure_pages_prompt.md',
        {'PAGES_JSON': json.dumps([int(page) for page in page_numbers], ensure_ascii=False)},
    )


def _build_structure_page_review_prompt(page_numbers):
    return render_skill_reference(
        'biocheminsight-vision-models',
        'references/review_structure_pages_prompt.md',
        {'PAGES_JSON': json.dumps([int(page) for page in page_numbers], ensure_ascii=False)},
    )


def _parse_structure_page_detection_response(response_text, page_numbers):
    from utils.llm_utils import VISION_MODEL_OUTPUT_SCHEMAS

    task_name = 'structure_page_detection'
    schema = VISION_MODEL_OUTPUT_SCHEMAS.get(task_name, {})
    payload = parse_validated_json_object(response_text, schema, task_name)
    allowed_pages = [int(page) for page in page_numbers]
    allowed_page_set = set(allowed_pages)

    raw_decisions = payload.get('decisions')
    if not isinstance(raw_decisions, list):
        raise ValueError(f"{task_name} payload decisions must be a list")
    if len(raw_decisions) != len(allowed_pages):
        raise ValueError(
            f"{task_name} payload must include one decision per page "
            f"({len(raw_decisions)} decisions for {len(allowed_pages)} pages)"
        )

    decisions_by_page = {}
    for item in raw_decisions:
        require_decision_contract(item, schema, task_name)
        page = item.get('page')
        if page not in allowed_page_set:
            raise ValueError(f"{task_name} decision contains out-of-batch page: {page}")
        if page in decisions_by_page:
            raise ValueError(f"{task_name} decision duplicates page: {page}")
        has_structure = item.get('has_structure')
        decisions_by_page[page] = {
            'has_structure': has_structure,
            'confidence': str(item.get('confidence') or '').strip().lower(),
            'reason': str(item.get('reason') or '').strip(),
        }

    missing_pages = allowed_page_set - set(decisions_by_page)
    if missing_pages:
        raise ValueError(f"{task_name} payload missing decisions for pages: {sorted(missing_pages)}")

    detected_pages = {page for page, item in decisions_by_page.items() if item.get('has_structure')}
    return sorted(detected_pages), decisions_by_page


def detect_structure_pages_with_vision_contact_sheets(pdf_file, page_numbers, audit_path=None):
    from utils.llm_utils import call_visual_model

    page_numbers = [int(page) for page in page_numbers]
    detected_pages = set()
    decisions_by_page = {}

    with tempfile.TemporaryDirectory(prefix='biocheminsight_structure_detect_') as tmp_dir:
        for batch_index, batch_pages in enumerate(_chunked(page_numbers, STRUCTURE_AUTO_DETECT_VISION_BATCH_SIZE), start=1):
            contact_sheet = os.path.join(tmp_dir, f'structure_pages_batch_{batch_index}.png')
            _build_structure_page_contact_sheet(pdf_file, batch_pages, contact_sheet)
            prompt = _build_structure_page_detection_prompt(batch_pages)

            attempts = max(1, int(STRUCTURE_AUTO_DETECT_VISION_MAX_RETRIES or 1) + 1)
            try:
                batch_detected, batch_decisions = run_json_task(
                    task_name='structure_page_detection',
                    channel='vision',
                    operation=lambda: call_visual_model(contact_sheet, prompt, retries=1),
                    parser=lambda text: _parse_structure_page_detection_response(text, batch_pages),
                    retry=attempts,
                    audit_path=audit_path,
                    metadata={
                        'scope': 'structure_page_detection',
                        'batch_index': batch_index,
                        'pages': batch_pages,
                    },
                )
                detected_pages.update(batch_detected)
                decisions_by_page.update(batch_decisions)
            except Exception as exc:
                raise RuntimeError(
                    f"Vision structure-page detection failed for pages {batch_pages}: {exc}"
                ) from exc

            try:
                os.remove(contact_sheet)
            except OSError:
                pass
            gc.collect()

    return sorted(detected_pages), decisions_by_page


def review_structure_pages_with_vision_contact_sheets(pdf_file, candidate_pages, audit_path=None):
    from utils.llm_utils import call_visual_model

    candidate_pages = sorted({int(page) for page in candidate_pages})
    if not candidate_pages:
        return [], {}

    reviewed_pages = set()
    decisions_by_page = {}
    batch_size = max(1, int(STRUCTURE_AUTO_DETECT_VISION_REVIEW_BATCH_SIZE or 1))

    with tempfile.TemporaryDirectory(prefix='biocheminsight_structure_review_') as tmp_dir:
        for batch_index, batch_pages in enumerate(_chunked(candidate_pages, batch_size), start=1):
            contact_sheet = os.path.join(tmp_dir, f'structure_pages_review_{batch_index}.png')
            _build_structure_page_contact_sheet(
                pdf_file,
                batch_pages,
                contact_sheet,
                columns=STRUCTURE_AUTO_DETECT_VISION_REVIEW_COLUMNS,
                thumb_width=STRUCTURE_AUTO_DETECT_VISION_REVIEW_THUMB_WIDTH,
                render_scale=STRUCTURE_AUTO_DETECT_VISION_REVIEW_RENDER_SCALE,
            )
            prompt = _build_structure_page_review_prompt(batch_pages)

            attempts = max(1, int(STRUCTURE_AUTO_DETECT_VISION_MAX_RETRIES or 1) + 1)
            try:
                batch_reviewed, batch_decisions = run_json_task(
                    task_name='structure_page_review',
                    channel='vision',
                    operation=lambda: call_visual_model(contact_sheet, prompt, retries=1),
                    parser=lambda text: _parse_structure_page_detection_response(text, batch_pages),
                    retry=attempts,
                    audit_path=audit_path,
                    metadata={
                        'scope': 'structure_page_review',
                        'batch_index': batch_index,
                        'pages': batch_pages,
                    },
                )
                reviewed_pages.update(batch_reviewed)
                decisions_by_page.update(batch_decisions)
            except Exception as exc:
                raise RuntimeError(
                    f"Strict vision structure-page review failed for pages {batch_pages}: {exc}"
                ) from exc

            try:
                os.remove(contact_sheet)
            except OSError:
                pass
            gc.collect()

    return sorted(reviewed_pages), decisions_by_page


def auto_detect_structure_pages(pdf_file, audit_path=None):
    return auto_detect_structure_pages_from_texts(pdf_file, _load_pdf_page_texts(pdf_file), audit_path=audit_path)


def auto_detect_structure_pages_from_texts(pdf_file, page_texts, audit_path=None):
    page_texts = _coerce_page_texts(page_texts) or []

    if page_texts:
        page_numbers = [int(page_num) for page_num, _ in page_texts]
    else:
        page_numbers = list(range(1, get_total_pages(pdf_file) + 1))
    diagnostics = [{'page': page_num} for page_num in page_numbers]

    initial_pages, initial_decisions = detect_structure_pages_with_vision_contact_sheets(pdf_file, page_numbers, audit_path=audit_path)
    if STRUCTURE_AUTO_DETECT_VISION_REVIEW_ENABLED:
        detected_pages, review_decisions = review_structure_pages_with_vision_contact_sheets(pdf_file, initial_pages, audit_path=audit_path)
        auto_detect_source = 'vision_contact_sheet_strict_review'
    else:
        detected_pages, review_decisions = initial_pages, {}
        auto_detect_source = 'vision_contact_sheet'

    initial_set = set(initial_pages)
    detected_set = set(detected_pages)
    for diag in diagnostics:
        page_num = int(diag.get('page', 0) or 0)
        diag['vision_initial_detection'] = initial_decisions.get(page_num, {})
        diag['vision_initial_keep'] = page_num in initial_set
        if STRUCTURE_AUTO_DETECT_VISION_REVIEW_ENABLED and page_num in initial_set:
            diag['vision_review_detection'] = review_decisions.get(page_num, {})
            diag['vision_review_keep'] = page_num in detected_set
        diag['include'] = page_num in detected_set
        diag['auto_detect_source'] = auto_detect_source

    return sorted(detected_set), diagnostics


def _extract_payload_page_markdowns(payload):
    if not isinstance(payload, dict):
        return []
    direct = payload.get('page_markdowns')
    if isinstance(direct, list):
        return [str(item or '').strip() for item in direct]
    pages = payload.get('pages')
    if isinstance(pages, list):
        page_markdowns = []
        for item in pages:
            if isinstance(item, dict):
                page_markdowns.append(str(item.get('markdown') or '').strip())
        if page_markdowns:
            return page_markdowns
    return []


def _emit_auto_detect_progress(progress_callback, stage, current, total, message, **extra):
    if not progress_callback:
        return
    try:
        event = {
            'stage': stage,
            'current': int(current or 0),
            'total': int(total or 0),
            'message': str(message or ''),
        }
        event.update(extra)
        progress_callback(event)
    except Exception as exc:
        print(f"Warning: auto-detect progress callback failed: {exc}")


def _write_pdf_page_subset(source_pdf, page_numbers, output_pdf):
    page_numbers = [int(page) for page in page_numbers]
    with open(source_pdf, 'rb') as src_stream:
        reader = PyPDF2.PdfReader(src_stream)
        writer = PyPDF2.PdfWriter()
        total_pages = len(reader.pages)
        for page_num in page_numbers:
            if page_num < 1 or page_num > total_pages:
                raise ValueError(f"Page {page_num} is outside PDF page range 1-{total_pages}.")
            writer.add_page(reader.pages[page_num - 1])
        with open(output_pdf, 'wb') as out_stream:
            writer.write(out_stream)


def _build_ocr_document_key(pdf_file):
    stat = os.stat(pdf_file)
    return hashlib.sha256(
        json.dumps(
            {
                'path': os.path.abspath(pdf_file),
                'size': int(stat.st_size),
                'mtime_ns': int(stat.st_mtime_ns),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode('utf-8')
    ).hexdigest()


def _is_timeout_exception(exc):
    timeout_types = tuple(
        exc_type
        for exc_type in (
            getattr(requests, 'Timeout', None),
            getattr(getattr(requests, 'exceptions', None), 'Timeout', None),
            getattr(getattr(requests, 'exceptions', None), 'ReadTimeout', None),
            getattr(getattr(requests, 'exceptions', None), 'ConnectTimeout', None),
        )
        if isinstance(exc_type, type)
    )
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        if timeout_types and isinstance(current, timeout_types):
            return True
        seen.add(id(current))
        current = getattr(current, '__cause__', None) or getattr(current, '__context__', None)
    return False


def load_auto_detect_page_markdowns(pdf_file, page_numbers, lang=DEFAULT_OCR_LANG, progress_callback=None):
    page_numbers = sorted({int(page) for page in page_numbers})
    if not page_numbers:
        return {}

    server_url = str(getattr(_constants, 'PADDLEOCR_SERVER_URL', '') or '').strip() if _constants else ''
    if not server_url:
        raise RuntimeError("PADDLEOCR_SERVER_URL is required for LLM-based assay page auto-detection.")

    document_key = _build_ocr_document_key(pdf_file)
    page_markdowns = {}
    ocr_batch_size = max(1, int(ASSAY_AUTO_DETECT_OCR_BATCH_SIZE or ASSAY_AUTO_DETECT_LLM_BATCH_SIZE or 1))
    ocr_concurrency = max(1, int(ASSAY_AUTO_DETECT_OCR_CONCURRENCY or 1))
    groups = list(_chunked(page_numbers, ocr_batch_size))
    total_groups = len(groups)
    ocr_timeout = max(15, int(ASSAY_AUTO_DETECT_OCR_TIMEOUT_SECONDS or 180))
    _emit_auto_detect_progress(
        progress_callback,
        'ocr',
        0,
        total_groups,
        f"Starting OCR for bioactivity page detection ({len(page_numbers)} page{'s' if len(page_numbers) != 1 else ''})",
    )

    def fetch_group(group):
        expected_pages = list(group)
        page_start = min(expected_pages)
        page_end = max(expected_pages)

        def request_ocr_pages(batch_pages):
            batch_pages = list(batch_pages)
            batch_page_start = batch_pages[0]
            batch_page_end = batch_pages[-1]
            upload_pdf = pdf_file
            upload_name = os.path.basename(pdf_file) or 'document.pdf'
            request_page_start = batch_page_start
            request_page_end = batch_page_end
            page_number_offset = 0
            temp_pdf_path = None
            try:
                if ASSAY_AUTO_DETECT_OCR_SPLIT_PDF:
                    with tempfile.NamedTemporaryFile(
                        prefix=f"biocheminsight_ocr_{request_page_start}_{request_page_end}_",
                        suffix=".pdf",
                        delete=False,
                    ) as tmp:
                        temp_pdf_path = tmp.name
                    _write_pdf_page_subset(pdf_file, batch_pages, temp_pdf_path)
                    upload_pdf = temp_pdf_path
                    upload_name = f"pages_{request_page_start}_{request_page_end}.pdf"
                    page_number_offset = batch_page_start - 1
                    request_page_start = 1
                    request_page_end = len(batch_pages)

                payload = request_pdf_to_markdown(
                    upload_pdf,
                    request_page_start,
                    request_page_end,
                    lang,
                    False,
                    server_url,
                    document_key=document_key,
                    page_number_offset=page_number_offset,
                    timeout_seconds=ocr_timeout,
                )
            except requests.RequestException as exc:
                if ASSAY_AUTO_DETECT_OCR_SPLIT_RETRY_ENABLED and _is_timeout_exception(exc) and len(batch_pages) > 1:
                    midpoint = max(1, len(batch_pages) // 2)
                    left_pages = batch_pages[:midpoint]
                    right_pages = batch_pages[midpoint:]
                    print(
                        f"Warning: PaddleOCR timed out for pages {request_page_start}-{request_page_end} "
                        f"after {ocr_timeout}s; retrying as {left_pages[0]}-{left_pages[-1]} and "
                        f"{right_pages[0]}-{right_pages[-1]}."
                    )
                    left_result = request_ocr_pages(left_pages)
                    right_result = request_ocr_pages(right_pages)
                    merged = dict(left_result)
                    merged.update(right_result)
                    return merged
                if len(batch_pages) == 1:
                    print(
                        f"Warning: PaddleOCR failed for page {batch_page_start}; continuing with blank markdown."
                    )
                    return {batch_page_start: ''}
                raise RuntimeError(
                    f"PaddleOCR request failed for pages {batch_page_start}-{batch_page_end} within {ocr_timeout}s."
                ) from exc
            except (OSError, ValueError) as exc:
                if len(batch_pages) == 1:
                    print(
                        f"Warning: PaddleOCR failed for page {batch_page_start}; continuing with blank markdown."
                    )
                    return {batch_page_start: ''}
                raise RuntimeError(
                    f"PaddleOCR request failed for pages {batch_page_start}-{batch_page_end} within {ocr_timeout}s."
                ) from exc
            finally:
                if temp_pdf_path:
                    try:
                        os.remove(temp_pdf_path)
                    except OSError:
                        pass

            content_list = _extract_payload_page_markdowns(payload)
            if len(content_list) != len(batch_pages):
                if len(batch_pages) == 1:
                    print(
                        f"Warning: PaddleOCR returned no usable markdown for page {batch_page_start}; "
                        "continuing with blank markdown."
                    )
                    return {batch_page_start: ''}
                raise RuntimeError(
                    f"PaddleOCR page split mismatch for pages {request_page_start}-{request_page_end}: "
                    f"expected {len(batch_pages)}, got {len(content_list)}."
                )
            return dict(zip(batch_pages, content_list))

        return request_ocr_pages(expected_pages)

    if ocr_concurrency <= 1 or len(groups) <= 1:
        for group_index, group in enumerate(groups, start=1):
            page_start = min(group)
            page_end = max(group)
            _emit_auto_detect_progress(
                progress_callback,
                'ocr',
                group_index - 1,
                total_groups,
                f"OCR bioactivity pages {page_start}-{page_end} ({group_index}/{total_groups})",
                page_start=page_start,
                page_end=page_end,
            )
            page_markdowns.update(fetch_group(group))
            _emit_auto_detect_progress(
                progress_callback,
                'ocr',
                group_index,
                total_groups,
                f"OCR completed for pages {page_start}-{page_end} ({group_index}/{total_groups})",
                page_start=page_start,
                page_end=page_end,
            )
    else:
        max_workers = min(ocr_concurrency, len(groups))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_group = {executor.submit(fetch_group, group): group for group in groups}
            completed = 0
            for future in as_completed(future_to_group):
                group = future_to_group[future]
                page_start = min(group)
                page_end = max(group)
                page_markdowns.update(future.result())
                completed += 1
                _emit_auto_detect_progress(
                    progress_callback,
                    'ocr',
                    completed,
                    total_groups,
                    f"OCR completed for pages {page_start}-{page_end} ({completed}/{total_groups})",
                    page_start=page_start,
                    page_end=page_end,
                )

    return page_markdowns


def _build_assay_page_detection_prompt(batch_pages, assay_names=None):
    assay_names = [str(name).strip() for name in (assay_names or []) if str(name).strip()]
    pages_payload = []
    max_chars = max(500, int(ASSAY_AUTO_DETECT_LLM_MAX_PAGE_CHARS or 5000))
    for page_num, markdown in batch_pages:
        text = str(markdown or '').strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        pages_payload.append({'page': int(page_num), 'markdown': text})
    return render_skill_reference(
        'biocheminsight-text-models',
        'references/detect_assay_pages_prompt.md',
        {
            'ASSAY_NAMES_JSON': json.dumps(assay_names, ensure_ascii=False),
            'PAGES_JSON': json.dumps(pages_payload, ensure_ascii=False),
        },
    )


def _parse_assay_page_detection_response(response_text, page_numbers):
    from utils.llm_utils import TEXT_MODEL_OUTPUT_SCHEMAS

    task_name = 'detect_assay_pages'
    payload = parse_validated_json_object(
        response_text,
        TEXT_MODEL_OUTPUT_SCHEMAS.get(task_name, {}),
        task_name,
    )
    allowed_pages = {int(page) for page in page_numbers}
    allowed_page_count = len(allowed_pages)

    detected = set()
    for item in payload.get('assay_pages', []) if isinstance(payload.get('assay_pages', []), list) else []:
        try:
            page = int(item)
        except Exception:
            continue
        if page in allowed_pages:
            detected.add(page)

    assay_names = []
    seen_names = set()
    for item in payload.get('assay_names', []) if isinstance(payload.get('assay_names', []), list) else []:
        name = str(item or '').strip()
        key = name.lower()
        if name and key not in seen_names:
            seen_names.add(key)
            assay_names.append(name)

    decisions_by_page = {}
    raw_decisions = payload.get('decisions', [])
    if not isinstance(raw_decisions, list):
        raise ValueError(f"{task_name} payload decisions must be a list")
    if len(raw_decisions) != allowed_page_count:
        raise ValueError(
            f"{task_name} payload must include one decision per page "
            f"({len(raw_decisions)} decisions for {allowed_page_count} pages)"
        )
    schema = TEXT_MODEL_OUTPUT_SCHEMAS.get(task_name, {})
    for item in raw_decisions:
        require_decision_contract(item, schema, task_name)
        page = item.get('page')
        if page not in allowed_pages:
            raise ValueError(f"{task_name} decision contains out-of-batch page: {page}")
        if page in decisions_by_page:
            raise ValueError(f"{task_name} decision duplicates page: {page}")
        has_assay_data = item.get('has_assay_data')
        if has_assay_data:
            detected.add(page)
        page_names = []
        raw_page_names = item.get('assay_names')
        for name_item in raw_page_names:
            name = str(name_item or '').strip()
            if name:
                page_names.append(name)
                key = name.lower()
                if key not in seen_names:
                    seen_names.add(key)
                    assay_names.append(name)
        decisions_by_page[page] = {
            'has_assay_data': has_assay_data,
            'confidence': str(item.get('confidence') or '').strip().lower(),
            'assay_names': page_names,
            'reason': str(item.get('reason') or '').strip(),
        }

    missing_pages = allowed_pages - set(decisions_by_page)
    if missing_pages:
        raise ValueError(f"{task_name} payload missing decisions for pages: {sorted(missing_pages)}")

    assay_names = normalize_detected_assay_names(assay_names)
    for decision in decisions_by_page.values():
        decision['assay_names'] = normalize_detected_assay_names(decision.get('assay_names') or [])

    return sorted(detected), assay_names, decisions_by_page


def detect_assay_pages_with_ocr_llm(pdf_file, page_numbers, assay_names=None, page_markdowns=None, progress_callback=None, audit_path=None):
    from utils.llm_utils import (
        LLM_TEXT_MODEL_KEY,
        LLM_TEXT_MODEL_NAME,
        LLM_TEXT_MODEL_URL,
        TEXT_MODEL_RUNTIME,
        get_retry_delays,
        get_system_prompt,
        get_task_temperature,
        require_openai,
        sanitize_model_response_text,
    )

    page_numbers = sorted({int(page) for page in page_numbers})
    if page_markdowns is None:
        page_markdowns = load_auto_detect_page_markdowns(pdf_file, page_numbers, progress_callback=progress_callback)

    detected_pages = set()
    detected_names = []
    seen_names = set()
    decisions_by_page = {}
    json_system_prompt = get_system_prompt(
        TEXT_MODEL_RUNTIME,
        'text',
        'json_extraction',
        "You are a precise information extraction assistant. Return JSON only.",
    )
    temperature = get_task_temperature(TEXT_MODEL_RUNTIME, 'detect_assay_pages', channel='text', default=0.0)
    attempts = max(1, int(ASSAY_AUTO_DETECT_LLM_MAX_RETRIES or 1) + 1)
    llm_timeout = max(15, int(ASSAY_AUTO_DETECT_LLM_TIMEOUT_SECONDS or 120))

    if not LLM_TEXT_MODEL_KEY or not LLM_TEXT_MODEL_URL or not LLM_TEXT_MODEL_NAME:
        raise ValueError("OpenAI-compatible text model is not configured for assay page detection.")

    llm_groups = list(_chunked(page_numbers, ASSAY_AUTO_DETECT_LLM_BATCH_SIZE))
    total_llm_groups = len(llm_groups)
    _emit_auto_detect_progress(
        progress_callback,
        'llm',
        0,
        total_llm_groups,
        f"Analyzing OCR text for bioactivity pages ({total_llm_groups} batch{'es' if total_llm_groups != 1 else ''})",
    )

    for batch_index, batch_pages in enumerate(llm_groups, start=1):
        batch_payload = [(page, page_markdowns.get(page, '')) for page in batch_pages]
        prompt = _build_assay_page_detection_prompt(batch_payload, assay_names=assay_names)
        _emit_auto_detect_progress(
            progress_callback,
            'llm',
            batch_index - 1,
            total_llm_groups,
            (
                f"Analyzing bioactivity OCR batch {batch_index}/{total_llm_groups} "
                f"(pages {min(batch_pages)}-{max(batch_pages)})"
            ),
            page_start=min(batch_pages),
            page_end=max(batch_pages),
            attempts=attempts,
        )
        client = require_openai()(
            api_key=LLM_TEXT_MODEL_KEY,
            base_url=LLM_TEXT_MODEL_URL,
            timeout=llm_timeout,
        )

        def _operation():
            response = client.chat.completions.create(
                model=LLM_TEXT_MODEL_NAME,
                messages=[
                    {"role": "system", "content": json_system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return sanitize_model_response_text(response.choices[0].message.content or '')

        try:
            batch_detected, batch_names, batch_decisions = run_json_task(
                task_name='detect_assay_pages',
                channel='text',
                operation=_operation,
                parser=lambda text: _parse_assay_page_detection_response(text, batch_pages),
                retry=attempts,
                retry_delays=get_retry_delays(TEXT_MODEL_RUNTIME, 'detect_assay_pages', channel='text'),
                audit_path=audit_path,
                metadata={
                    'scope': 'assay_page_detection',
                    'batch_index': batch_index,
                    'pages': batch_pages,
                    'page_count': len(batch_pages),
                },
            )
        except Exception as exc:
            raise RuntimeError(f"LLM assay-page detection failed for pages {batch_pages}: {exc}") from exc

        detected_pages.update(batch_detected)
        decisions_by_page.update(batch_decisions)
        for name in batch_names:
            key = _normalize_assay_name_for_dedupe(name)
            if key not in seen_names:
                seen_names.add(key)
                detected_names.append(name)
        _emit_auto_detect_progress(
            progress_callback,
            'llm',
            batch_index,
            total_llm_groups,
            (
                f"Analyzed bioactivity OCR batch {batch_index}/{total_llm_groups} "
                f"(found {len(batch_detected)} page{'s' if len(batch_detected) != 1 else ''})"
            ),
            page_start=min(batch_pages),
            page_end=max(batch_pages),
        )

    detected_names = reconcile_detected_assay_names_with_model(
        detected_names,
        decisions_by_page=decisions_by_page,
        page_markdowns=page_markdowns,
        audit_path=audit_path,
        metadata={'scope': 'assay_page_detection_name_reconciliation'},
    )
    return sorted(detected_pages), detected_names, decisions_by_page, page_markdowns


def auto_detect_assay_pages(pdf_file, assay_names=None, page_texts=None, progress_callback=None, audit_path=None):
    assay_names = [name.strip() for name in (assay_names or []) if name and name.strip()]
    if page_texts:
        page_numbers = [int(page_num) for page_num, _ in _coerce_page_texts(page_texts)]
    else:
        page_numbers = list(range(1, get_total_pages(pdf_file) + 1))

    detected_pages, detected_names, decisions, page_markdowns = detect_assay_pages_with_ocr_llm(
        pdf_file,
        page_numbers,
        assay_names=assay_names,
        progress_callback=progress_callback,
        audit_path=audit_path,
    )
    detected_set = set(detected_pages)
    diagnostics = []
    for page_num in page_numbers:
        page_decision = dict(decisions.get(int(page_num), {}) or {})
        diagnostics.append({
            'page': int(page_num),
            'include': int(page_num) in detected_set,
            'auto_detect_source': 'ocr_llm_skill',
            'llm_page_detection': page_decision,
            'ocr_markdown_chars': len(page_markdowns.get(int(page_num), '') or ''),
        })
    if detected_names:
        detected_names = normalize_detected_assay_names(detected_names)
        diagnostics.append({
            'page': None,
            'auto_detect_source': 'ocr_llm_skill',
            'detected_assay_names': detected_names,
        })
    return sorted(detected_set), diagnostics


def auto_detect_assay_names(pdf_file, assay_pages=None, page_texts=None):
    if assay_pages is None:
        assay_pages, diagnostics = auto_detect_assay_pages(pdf_file, assay_names=None, page_texts=page_texts)
        for item in diagnostics:
            names = item.get('detected_assay_names') if isinstance(item, dict) else None
            if isinstance(names, list):
                return normalize_detected_assay_names([str(name).strip() for name in names if str(name).strip()])

    assay_pages = sorted({int(page) for page in (assay_pages or [])})
    if not assay_pages:
        return []

    _, detected_names, _, _ = detect_assay_pages_with_ocr_llm(
        pdf_file,
        assay_pages,
        assay_names=None,
    )
    return detected_names


def verify_assay_names_for_pages(pdf_file, assay_pages, assay_names, output_dir=None, lang=DEFAULT_OCR_LANG, audit_path=None):
    assay_names = normalize_detected_assay_names(assay_names)
    if len(assay_names) <= 1:
        return assay_names

    if isinstance(assay_pages, (int, tuple)):
        if isinstance(assay_pages, tuple) and len(assay_pages) == 2:
            page_list = list(range(int(assay_pages[0]), int(assay_pages[1]) + 1))
        else:
            page_list = [int(assay_pages)] if isinstance(assay_pages, int) else list(assay_pages)
    elif isinstance(assay_pages, list):
        page_list = [int(page) for page in assay_pages]
    else:
        page_list = []

    if not page_list:
        raise ValueError("assay-name verifier requires assay pages for multi-candidate reconciliation")

    from activity_parser import load_assay_page_contents

    scratch_dir = output_dir or tempfile.mkdtemp(prefix='biocheminsight_assay_name_verify_')
    content_list = load_assay_page_contents(
        pdf_file=pdf_file,
        assay_page_start=min(page_list),
        assay_page_end=max(page_list),
        output_dir=scratch_dir,
        lang=lang,
    )

    page_markdowns = {
        page: content_list[page - min(page_list)]
        for page in page_list
        if 0 <= page - min(page_list) < len(content_list)
    }
    page_decisions = {
        page: {
            'has_assay_data': True,
            'assay_names': assay_names,
            'reason': 'assay names supplied for extraction on this page',
        }
        for page in page_list
    }
    verified = reconcile_detected_assay_names_with_model(
        assay_names,
        decisions_by_page=page_decisions,
        page_markdowns=page_markdowns,
        audit_path=audit_path,
        metadata={'scope': 'assay_name_verifier_for_extraction', 'page_count': len(page_list)},
    )
    return verified


def build_document_auto_plan(pdf_file, assay_names=None, use_cache=True):
    if use_cache:
        cached = _load_document_auto_detect_cache(pdf_file, assay_names=assay_names)
        if cached and isinstance(cached.get('plan'), dict):
            plan = cached['plan']
            plan['cache_hit'] = True
            plan['cache_path'] = _get_document_auto_detect_cache_path(pdf_file, assay_names=assay_names)
            return plan

    page_texts = _load_pdf_page_texts(pdf_file)
    detected_structure_pages, structure_diagnostics = auto_detect_structure_pages_from_texts(pdf_file, page_texts)
    detected_assay_pages, assay_diagnostics = auto_detect_assay_pages(pdf_file, assay_names=assay_names, page_texts=page_texts)
    detected_assay_names = auto_detect_assay_names(pdf_file, assay_pages=detected_assay_pages, page_texts=page_texts)
    plan = {
        'structure_pages': detected_structure_pages,
        'structure_diagnostics': structure_diagnostics,
        'assay_pages': detected_assay_pages,
        'assay_diagnostics': assay_diagnostics,
        'assay_names': detected_assay_names,
        'cache_hit': False,
    }
    cache_path = _save_document_auto_detect_cache(pdf_file, assay_names, plan)
    if cache_path:
        plan['cache_path'] = cache_path
    return plan


def _build_structure_alias_context(record, stage='structure'):
    if not isinstance(record, dict):
        return ''
    return (
        f"Stage: {stage}\n"
        f"COMPOUND_ID: {record.get('COMPOUND_ID', '')}\n"
        f"PAGE_NUM: {record.get('PAGE_NUM', '')}\n"
        f"SMILES: {record.get('SMILES', '')}\n"
        f"STRUCTURE_TYPE: {record.get('STRUCTURE_TYPE', '')}\n"
        f"STRUCTURE_FILTER_REASON: {record.get('STRUCTURE_FILTER_REASON', '')}\n"
        f"SOURCE_PAGES: {record.get('source_pages', [])}\n"
        f"GROUP_ID: {record.get('group_id', '')}\n"
        f"SEGMENT_FILE: {record.get('SEGMENT_FILE', '')}"
    )


def _safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _truncate_text(value, max_chars):
    text = str(value or '').strip()
    max_chars = max(0, int(max_chars or 0))
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n...[truncated]"
    return text


def _preserve_molblock(value):
    if value is None:
        return ''
    text = str(value)
    return text.rstrip('\r\n')


def _read_box_coords(record):
    path = str(record.get('BOX_COORDS_FILE') or '').strip()
    if path.startswith('/app/'):
        local_path = os.path.join(os.getcwd(), path[len('/app/'):])
    else:
        local_path = path
    if not local_path or not os.path.exists(local_path):
        return None
    try:
        with open(local_path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        return payload.get('box')
    except (OSError, ValueError, TypeError):
        return None


def _build_markush_structure_candidates(records, max_items=80):
    candidates = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        structure_type = str(record.get('STRUCTURE_TYPE') or '').strip()
        if structure_type not in {'markush', 'fragment', 'text_substituent'}:
            continue
        page = _safe_int(record.get('PAGE_NUM'))
        if page is None:
            continue
        segment_file = str(record.get('SEGMENT_FILE') or '')
        ref = f"page_{page}:{os.path.basename(segment_file) or len(candidates) + 1}"
        markush_cell_payload = {
            'visual_role': str(record.get('MARKUSH_CELL_VISUAL_ROLE') or '').strip(),
            'compound_id': str(record.get('MARKUSH_CELL_COMPOUND_ID') or '').strip(),
            'compound_id_source': str(record.get('MARKUSH_CELL_COMPOUND_ID_SOURCE') or '').strip(),
            'variable_position': str(record.get('MARKUSH_CELL_VARIABLE_POSITION') or '').strip(),
            'substituent_text': _truncate_text(record.get('MARKUSH_CELL_SUBSTITUENT_TEXT'), 200),
            'has_visual_structure': bool(record.get('MARKUSH_CELL_HAS_VISUAL_STRUCTURE')),
            'has_attachment_evidence': bool(record.get('MARKUSH_CELL_HAS_ATTACHMENT_EVIDENCE')),
            'confidence': str(record.get('MARKUSH_CELL_CONFIDENCE') or '').strip(),
        }
        has_markush_cell_payload = any(
            value not in {'', False, None}
            for value in markush_cell_payload.values()
        )
        candidates.append({
            'ref': ref,
            'page': page,
            'structure_type': structure_type,
            'smiles': str(record.get('FRAGMENT_SMILES') or record.get('SMILES') or '').strip(),
            'molblock': _truncate_text(record.get('MOLBLOCK'), 1200),
            'molblock_full': _preserve_molblock(record.get('MOLBLOCK')),
            'bbox': _read_box_coords(record),
            'filter_reason': _truncate_text(record.get('STRUCTURE_FILTER_REASON'), 320),
            'image_file': str(record.get('IMAGE_FILE') or '').strip(),
            'segment_file': segment_file,
            'box_coords_file': str(record.get('BOX_COORDS_FILE') or '').strip(),
            'page_image_file': str(record.get('PAGE_IMAGE_FILE') or '').strip(),
            'candidate_source': str(record.get('CANDIDATE_SOURCE') or '').strip(),
            'markush_cell': markush_cell_payload if has_markush_cell_payload else {},
        })
        if len(candidates) >= max_items:
            break
    return candidates


def review_markush_fragment_candidates(markush_candidates, page_contexts, audit_path=None, max_items=24, review_pages=None):
    from utils.llm_utils import review_markush_fragment_candidate

    review_page_set = {
        _safe_int(page)
        for page in (review_pages or [])
        if _safe_int(page) is not None
    }
    contexts_by_page = {
        _safe_int(context.get('page')): context
        for context in page_contexts or []
        if isinstance(context, dict) and _safe_int(context.get('page')) is not None
    }
    reviewed = 0
    for candidate in markush_candidates or []:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get('structure_type') or '').strip() != 'fragment':
            continue
        if review_page_set and _safe_int(candidate.get('page')) not in review_page_set:
            continue
        if not str(candidate.get('molblock_full') or candidate.get('molblock') or '').strip():
            candidate['fragment_visual_review'] = {
                'model_call_ok': False,
                'visual_role': 'unknown',
                'compound_id': 'None',
                'variable_position': '',
                'molnextr_consistent': False,
                'has_attachment_evidence': False,
                'molnextr_has_attachment_atom': False,
                'attachment_site_consistent': False,
                'confidence': 'low',
                'evidence': 'fragment has no MolNexTR MOLBLOCK',
            }
            continue
        image_file = _resolve_app_path(candidate.get('image_file'))
        if not image_file or not os.path.exists(image_file):
            candidate['fragment_visual_review'] = {
                'model_call_ok': False,
                'visual_role': 'unknown',
                'compound_id': 'None',
                'variable_position': '',
                'molnextr_consistent': False,
                'has_attachment_evidence': False,
                'molnextr_has_attachment_atom': False,
                'attachment_site_consistent': False,
                'confidence': 'low',
                'evidence': 'fragment red-box image not found',
            }
            continue
        if reviewed >= max_items:
            candidate['fragment_visual_review'] = {
                'model_call_ok': False,
                'visual_role': 'unknown',
                'compound_id': 'None',
                'variable_position': '',
                'molnextr_consistent': False,
                'has_attachment_evidence': False,
                'molnextr_has_attachment_atom': False,
                'attachment_site_consistent': False,
                'confidence': 'low',
                'evidence': 'bounded fragment visual review limit reached',
            }
            continue
        review = review_markush_fragment_candidate(
            image_file,
            candidate,
            contexts_by_page.get(_safe_int(candidate.get('page')), {}),
            audit_path=audit_path,
            metadata={'fragment_ref': candidate.get('ref'), 'page': candidate.get('page')},
        )
        reviewed += 1
        candidate['fragment_visual_review'] = review
    return markush_candidates


def _build_markush_page_contexts(page_numbers, page_markdowns=None, candidates=None, max_chars_per_page=1800):
    candidates_by_page = {}
    for candidate in candidates or []:
        page = _safe_int(candidate.get('page'))
        if page is not None:
            candidates_by_page.setdefault(page, []).append(candidate.get('ref'))
    contexts = []
    for page in sorted({_safe_int(page) for page in page_numbers if _safe_int(page) is not None}):
        markdown = ''
        if isinstance(page_markdowns, dict):
            markdown = page_markdowns.get(page, '')
        contexts.append({
            'page': page,
            'candidate_refs': [ref for ref in candidates_by_page.get(page, []) if ref][:20],
            'ocr_or_markdown_context': _truncate_text(markdown, max_chars_per_page),
        })
    return contexts


def _extract_markush_variable_positions(candidate):
    values = []
    if not isinstance(candidate, dict):
        return values
    cell = candidate.get('markush_cell') if isinstance(candidate.get('markush_cell'), dict) else {}
    variable = str(cell.get('variable_position') or '').strip()
    if variable:
        values.append(variable)
    for field in ('smiles', 'molblock', 'molblock_full', 'filter_reason'):
        text = str(candidate.get(field) or '')
        for match in re.findall(r'\bR\s*(\d+)\b|\[(\d+)\*\]', text):
            number = next((item for item in match if item), '')
            if number:
                values.append(f'R{number}')
    return list(dict.fromkeys(values))


_MARKUSH_VARIABLE_TOKEN_RE = re.compile(r'\b(?:R\s*\d+|X\s*\d*|Y\s*\d*|Z\s*\d*|Ar|Het)\b', flags=re.IGNORECASE)
_MARKUSH_ROW_ID_HEADER_RE = re.compile(
    r'\b(?:Ex(?:ample)?\.?|No\.?|Compound|Cmpd|Entry|Formula|ID)\b',
    flags=re.IGNORECASE,
)
_MARKUSH_TABLE_HEADER_RE = re.compile(
    r'\b(?:R\s*\d+|R-group|substituent|fragment|Markush|scaffold|core)\b',
    flags=re.IGNORECASE,
)


def _extract_markush_context_tokens(text, limit=12):
    if not text:
        return []
    tokens = []
    for raw in _MARKUSH_VARIABLE_TOKEN_RE.findall(str(text)):
        token = re.sub(r'\s+', '', str(raw).strip())
        if token:
            tokens.append(token.upper() if len(token) <= 2 else token[0].upper() + token[1:])
    return list(dict.fromkeys(tokens))[:limit]


def _extract_markush_compound_id_header(text):
    if not text:
        return ''
    match = _MARKUSH_ROW_ID_HEADER_RE.search(str(text))
    return match.group(0).strip() if match else ''


def _markush_page_scope_features(context, page_candidates):
    markdown = str((context or {}).get('ocr_or_markdown_context') or '')
    candidate_types = {
        str(candidate.get('structure_type') or '').strip()
        for candidate in page_candidates or []
        if isinstance(candidate, dict)
    }
    candidate_variables = []
    row_ids = []
    for candidate in page_candidates or []:
        if not isinstance(candidate, dict):
            continue
        candidate_variables.extend(_extract_markush_variable_positions(candidate))
        cell = candidate.get('markush_cell') if isinstance(candidate.get('markush_cell'), dict) else {}
        row_id = str(cell.get('compound_id') or '').strip()
        if row_id and row_id.lower() != 'none':
            row_ids.append(row_id)

    markdown_variables = _extract_markush_context_tokens(markdown)
    variables = list(dict.fromkeys([*candidate_variables, *markdown_variables]))
    compound_id_header = _extract_markush_compound_id_header(markdown)
    has_table_header = bool(_MARKUSH_TABLE_HEADER_RE.search(markdown) and compound_id_header)
    has_markush_text = bool(_MARKUSH_TABLE_HEADER_RE.search(markdown) or variables)
    has_markush_evidence = bool(
        candidate_types.intersection({'markush', 'text_substituent'})
        or (candidate_types.intersection({'fragment'}) and has_markush_text)
        or has_table_header
    )
    return {
        'variables': variables,
        'compound_id_header': compound_id_header,
        'has_table_header': has_table_header,
        'has_markush_text': has_markush_text,
        'has_markush_evidence': has_markush_evidence,
        'row_ids': list(dict.fromkeys(row_ids))[:12],
    }


def attach_markush_table_memory(page_contexts, candidates):
    contexts = [dict(context) for context in page_contexts or [] if isinstance(context, dict)]
    candidates_by_page = {}
    for candidate in candidates or []:
        page = _safe_int(candidate.get('page')) if isinstance(candidate, dict) else None
        if page is not None:
            candidates_by_page.setdefault(page, []).append(candidate)

    active_scaffold_ref = ''
    active_scaffold_page = None
    active_variable_positions = []
    active_variable_source_page = None
    active_compound_id_header = ''
    active_compound_id_header_page = None
    active_scope_id = None
    active_scope_pages = []
    last_scope_page = None
    scope_counter = 0
    for context in sorted(contexts, key=lambda item: _safe_int(item.get('page'), 0) or 0):
        page = _safe_int(context.get('page'))
        page_candidates = candidates_by_page.get(page, [])
        features = _markush_page_scope_features(context, page_candidates)
        scaffold = next(
            (
                candidate for candidate in page_candidates
                if str(candidate.get('structure_type') or '').strip() == 'markush'
            ),
            None,
        )
        starts_new_scope = False
        reset_reason = ''
        if scaffold:
            starts_new_scope = True
            reset_reason = 'new_markush_scaffold'
        elif not features['has_markush_evidence']:
            active_scaffold_ref = ''
            active_scaffold_page = None
            active_variable_positions = []
            active_variable_source_page = None
            active_compound_id_header = ''
            active_compound_id_header_page = None
            active_scope_id = None
            active_scope_pages = []
            last_scope_page = None
            reset_reason = 'no_markush_table_evidence'
        elif active_scope_id is None:
            starts_new_scope = True
            reset_reason = 'new_markush_table_scope'
        elif features['has_table_header'] and features['variables'] and active_variable_positions:
            overlap = set(features['variables']).intersection(set(active_variable_positions))
            if not overlap and not scaffold:
                starts_new_scope = True
                reset_reason = 'new_table_header_with_different_variables'
        elif last_scope_page is not None and page is not None and page > last_scope_page + 1:
            starts_new_scope = True
            reset_reason = 'non_contiguous_markush_scope'

        if starts_new_scope:
            scope_counter += 1
            active_scope_id = f"markush_table_scope_{scope_counter}"
            active_scope_pages = []
            if not scaffold:
                active_scaffold_ref = ''
                active_scaffold_page = None
                active_variable_positions = []
                active_variable_source_page = None
                active_compound_id_header = ''
                active_compound_id_header_page = None

        if scaffold:
            active_scaffold_ref = str(scaffold.get('ref') or '').strip()
            active_scaffold_page = page
            scaffold_vars = _extract_markush_variable_positions(scaffold)
            if scaffold_vars:
                active_variable_positions = scaffold_vars
                active_variable_source_page = page

        page_variables = []
        for candidate in page_candidates:
            if str(candidate.get('structure_type') or '').strip() != 'text_substituent':
                continue
            page_variables.extend(_extract_markush_variable_positions(candidate))
        if not page_variables and (
            scaffold
            or any(str(candidate.get('structure_type') or '').strip() == 'text_substituent' for candidate in page_candidates)
        ):
            page_variables = features['variables']
        page_variables = list(dict.fromkeys([item for item in page_variables if item]))
        if page_variables:
            active_variable_positions = page_variables
            active_variable_source_page = page
        if features['compound_id_header']:
            active_compound_id_header = features['compound_id_header']
            active_compound_id_header_page = page

        if active_scope_id and page is not None and features['has_markush_evidence']:
            active_scope_pages = [*active_scope_pages, page]
            active_scope_pages = list(dict.fromkeys(active_scope_pages))
            last_scope_page = page

        context['inherited_markush_context'] = {
            'active_scope_id': active_scope_id,
            'scope_source_pages': active_scope_pages,
            'scope_last_page': last_scope_page,
            'active_scaffold_ref': active_scaffold_ref or None,
            'active_scaffold_source_page': active_scaffold_page,
            'active_variable_positions': active_variable_positions,
            'variable_header_source_page': active_variable_source_page,
            'active_compound_id_header': active_compound_id_header or None,
            'compound_id_header_source_page': active_compound_id_header_page,
            'is_inherited': bool(
                active_scope_id
                and page is not None
                and (
                    (active_scaffold_page is not None and active_scaffold_page < page)
                    or (active_variable_source_page is not None and active_variable_source_page < page)
                    or (active_compound_id_header_page is not None and active_compound_id_header_page < page)
                )
            ),
            'scope_reset_reason': reset_reason,
            'current_page_has_markush_table_evidence': features['has_markush_evidence'],
            'current_page_row_ids': features['row_ids'],
        }
    return contexts


def _resolve_app_path(path):
    text = str(path or '').strip()
    if text.startswith('/app/'):
        return os.path.join(os.getcwd(), text[len('/app/'):])
    return text


def _relationship_needs_markush_visual_review(relationship):
    if not isinstance(relationship, dict):
        return False
    if str(relationship.get('assembly_status') or '').strip().lower() != 'ready':
        return False
    if str(relationship.get('pose_consistency') or '').strip().lower() != 'consistent':
        return True
    return True


def _downgrade_markush_relationship(relationship, status, pose, confidence, reason):
    relationship['assembly_status'] = status
    relationship['pose_consistency'] = pose
    relationship['confidence'] = confidence
    existing = str(relationship.get('reason') or '').strip()
    relationship['reason'] = f"{existing}; {reason}" if existing else reason
    return relationship


def review_markush_relationships_with_visual_evidence(
    plan,
    page_contexts,
    markush_candidates,
    audit_path=None,
    max_relationships=None,
):
    if not isinstance(plan, dict):
        return plan
    relationships = plan.get('relationships')
    if not isinstance(relationships, list) or not relationships:
        return plan

    from utils.llm_utils import MARKUSH_VISUAL_REVIEW_MAX_RELATIONSHIPS, review_markush_fragment_pose

    max_reviews = MARKUSH_VISUAL_REVIEW_MAX_RELATIONSHIPS if max_relationships is None else int(max_relationships)
    if max_reviews <= 0:
        for relationship in relationships:
            if _relationship_needs_markush_visual_review(relationship):
                _downgrade_markush_relationship(
                    relationship,
                    'needs_context',
                    'unknown',
                    'low',
                    'downgraded because Markush visual review is disabled',
                )
        return plan

    candidates_by_ref = {
        str(candidate.get('ref') or '').strip(): candidate
        for candidate in markush_candidates or []
        if isinstance(candidate, dict) and str(candidate.get('ref') or '').strip()
    }
    contexts_by_page = {
        _safe_int(context.get('page')): context
        for context in page_contexts or []
        if isinstance(context, dict) and _safe_int(context.get('page')) is not None
    }

    reviewed = 0
    for relationship in relationships:
        if not _relationship_needs_markush_visual_review(relationship):
            continue
        fragment_refs = [str(ref or '').strip() for ref in relationship.get('fragment_refs') or [] if str(ref or '').strip()]
        fragment_candidate = next((candidates_by_ref.get(ref) for ref in fragment_refs if candidates_by_ref.get(ref)), None)
        scaffold_ref = str(relationship.get('scaffold_ref') or '').strip()
        scaffold_candidate = candidates_by_ref.get(scaffold_ref) if scaffold_ref else None
        if not scaffold_candidate:
            _downgrade_markush_relationship(
                relationship,
                'needs_context',
                'unknown',
                'low',
                'downgraded because no red-box scaffold candidate matches the relationship',
            )
            relationship['visual_review'] = {
                'model_call_ok': False,
                'evidence': 'no matching scaffold candidate',
            }
            continue
        if str(scaffold_candidate.get('structure_type') or '').strip() != 'markush':
            _downgrade_markush_relationship(
                relationship,
                'uncertain',
                'unknown',
                'low',
                'downgraded because scaffold_ref does not point to a Markush scaffold candidate',
            )
            relationship['visual_review'] = {
                'model_call_ok': False,
                'evidence': 'scaffold candidate is not classified as markush',
            }
            continue
        if not fragment_candidate:
            _downgrade_markush_relationship(
                relationship,
                'needs_context',
                'unknown',
                'low',
                'downgraded because no MolNexTR fragment candidate matches the relationship',
            )
            relationship['visual_review'] = {
                'model_call_ok': False,
                'evidence': 'no matching fragment candidate',
            }
            continue
        if str(fragment_candidate.get('structure_type') or '').strip() == 'text_substituent':
            _downgrade_markush_relationship(
                relationship,
                'needs_context',
                'unknown',
                'medium',
                'downgraded because red-box substituent cell is text evidence without MolNexTR MOLBLOCK; no structure fallback is allowed',
            )
            relationship['visual_review'] = {
                'model_call_ok': False,
                'evidence': 'text substituent cell requires explicit structure evidence before pose-sensitive assembly',
                'markush_cell': fragment_candidate.get('markush_cell') or {},
            }
            continue
        if not str(fragment_candidate.get('molblock_full') or fragment_candidate.get('molblock') or '').strip():
            _downgrade_markush_relationship(
                relationship,
                'needs_context',
                'unknown',
                'low',
                'downgraded because fragment candidate has no MolNexTR MOLBLOCK; no SMILES fallback is allowed',
            )
            relationship['visual_review'] = {
                'model_call_ok': False,
                'evidence': 'fragment MOLBLOCK missing',
            }
            continue
        fragment_review = fragment_candidate.get('fragment_visual_review') if isinstance(fragment_candidate.get('fragment_visual_review'), dict) else {}
        fragment_review_id = str(fragment_review.get('compound_id') or '').strip()
        relationship_id = str(relationship.get('compound_id') or '').strip()
        relationship_variables = [str(item or '').strip() for item in relationship.get('variable_positions') or [] if str(item or '').strip()]
        fragment_review_variable = str(fragment_review.get('variable_position') or '').strip()
        if (
            not fragment_review
            or fragment_review.get('visual_role') not in {'fragment', 'substituent'}
            or fragment_review_id.lower() == 'none'
            or (relationship_id and fragment_review_id != relationship_id)
            or (relationship_variables and fragment_review_variable not in relationship_variables)
            or not fragment_review.get('molnextr_consistent')
            or not fragment_review.get('has_attachment_evidence')
            or not fragment_review.get('molnextr_has_attachment_atom')
            or not fragment_review.get('attachment_site_consistent')
        ):
            _downgrade_markush_relationship(
                relationship,
                'needs_context',
                'unknown',
                fragment_review.get('confidence') or 'low',
                'downgraded because fragment red-box visual review does not uniquely match compound_id, variable position, MolNexTR, and explicit attachment site evidence',
            )
            relationship['visual_review'] = {
                'model_call_ok': False,
                'evidence': 'fragment candidate visual review did not satisfy unique mapping gate',
                'fragment_visual_review': fragment_review,
            }
            continue
        fragment_page_context = contexts_by_page.get(_safe_int(fragment_candidate.get('page')), {})
        inherited_context = (
            fragment_page_context.get('inherited_markush_context')
            if isinstance(fragment_page_context.get('inherited_markush_context'), dict)
            else {}
        )
        inherited_scaffold_ref = str(inherited_context.get('active_scaffold_ref') or '').strip()
        inherited_variables = [
            str(item or '').strip()
            for item in inherited_context.get('active_variable_positions') or []
            if str(item or '').strip()
        ]
        inherited_variable_source_page = _safe_int(inherited_context.get('variable_header_source_page'))
        inherited_scaffold_source_page = _safe_int(inherited_context.get('active_scaffold_source_page'))
        fragment_context_has_table_evidence = bool(
            inherited_context.get('current_page_row_ids')
            or inherited_context.get('active_compound_id_header')
        )
        variable_scope_is_enforced = bool(
            inherited_variables
            and inherited_variable_source_page is not None
            and (
                inherited_variable_source_page == inherited_scaffold_source_page
                or (
                    inherited_variable_source_page == _safe_int(fragment_candidate.get('page'))
                    and fragment_context_has_table_evidence
                )
            )
        )
        if inherited_scaffold_ref and scaffold_ref and inherited_scaffold_ref != scaffold_ref:
            _downgrade_markush_relationship(
                relationship,
                'needs_context',
                'unknown',
                'low',
                'downgraded because fragment page inherited a different Markush table scope scaffold',
            )
            relationship['visual_review'] = {
                'model_call_ok': False,
                'evidence': 'fragment inherited scaffold scope did not match relationship scaffold_ref',
                'inherited_markush_context': inherited_context,
            }
            continue
        if variable_scope_is_enforced and relationship_variables:
            missing_variables = [item for item in relationship_variables if item not in inherited_variables]
            if missing_variables:
                _downgrade_markush_relationship(
                    relationship,
                    'needs_context',
                    'unknown',
                    'low',
                    'downgraded because relationship variables are outside the inherited Markush table scope',
                )
                relationship['visual_review'] = {
                    'model_call_ok': False,
                    'evidence': 'relationship variable positions did not match inherited table headers',
                    'inherited_markush_context': inherited_context,
                }
                continue
        image_file = _resolve_app_path(fragment_candidate.get('image_file'))
        if not image_file or not os.path.exists(image_file):
            _downgrade_markush_relationship(
                relationship,
                'needs_context',
                'unknown',
                'low',
                'downgraded because no red-box image is available for visual review',
            )
            relationship['visual_review'] = {
                'model_call_ok': False,
                'evidence': 'red-box image not found',
            }
            continue
        if reviewed >= max_reviews:
            _downgrade_markush_relationship(
                relationship,
                'needs_context',
                'unknown',
                'low',
                'downgraded because bounded Markush visual review limit was reached',
            )
            relationship['visual_review'] = {
                'model_call_ok': False,
                'evidence': 'visual review limit reached',
            }
            continue

        source_pages = relationship.get('source_pages') or []
        page_context = fragment_page_context or contexts_by_page.get(_safe_int(source_pages[0] if source_pages else None), {})
        scaffold_image_file = _resolve_app_path(scaffold_candidate.get('image_file'))
        if not scaffold_image_file or not os.path.exists(scaffold_image_file):
            _downgrade_markush_relationship(
                relationship,
                'needs_context',
                'unknown',
                'low',
                'downgraded because no red-box scaffold image is available for visual review',
            )
            relationship['visual_review'] = {
                'model_call_ok': False,
                'evidence': 'scaffold red-box image not found',
            }
            continue
        review = review_markush_fragment_pose(
            image_file,
            relationship,
            fragment_candidate,
            page_context,
            scaffold_candidate=scaffold_candidate,
            audit_path=audit_path,
            metadata={
                'relationship_id': relationship.get('record_id'),
                'fragment_ref': fragment_candidate.get('ref'),
                'scaffold_ref': scaffold_candidate.get('ref'),
                'source_pages': source_pages,
            },
        )
        reviewed += 1
        relationship['visual_review'] = review
        relationship['molnextr_fragment'] = {
            'ref': fragment_candidate.get('ref'),
            'smiles': fragment_candidate.get('smiles'),
            'molblock': fragment_candidate.get('molblock'),
            'image_file': fragment_candidate.get('image_file'),
            'bbox': fragment_candidate.get('bbox'),
        }
        relationship['molnextr_scaffold'] = {
            'ref': scaffold_candidate.get('ref'),
            'smiles': scaffold_candidate.get('smiles'),
            'molblock': scaffold_candidate.get('molblock'),
            'image_file': scaffold_candidate.get('image_file'),
            'bbox': scaffold_candidate.get('bbox'),
        }
        if review.get('assembly_status') != 'ready':
            _downgrade_markush_relationship(
                relationship,
                review.get('assembly_status') or 'uncertain',
                review.get('pose_consistency') or 'unknown',
                review.get('confidence') or 'low',
                f"visual review: {review.get('evidence') or 'not ready'}",
            )
        else:
            relationship['assembly_status'] = 'ready'
            relationship['pose_consistency'] = 'consistent'
            relationship['confidence'] = review.get('confidence') or relationship.get('confidence') or 'medium'
            relationship['reason'] = (
                f"{str(relationship.get('reason') or '').strip()}; "
                f"visual review: {review.get('evidence')}"
            ).strip('; ')

    return plan


def _build_markush_assembled_structure_records(markush_plan_payload):
    records = []
    if not isinstance(markush_plan_payload, dict):
        return records
    candidates = markush_plan_payload.get('structure_candidates') or []
    candidates_by_ref = {
        str(candidate.get('ref') or '').strip(): candidate
        for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get('ref') or '').strip()
    }
    plan = markush_plan_payload.get('plan') if isinstance(markush_plan_payload.get('plan'), dict) else {}
    for assembly in plan.get('assembly_candidates') or []:
        if not isinstance(assembly, dict):
            continue
        if str(assembly.get('assembly_status') or '').strip() != 'assembled':
            continue
        compound_id = str(assembly.get('compound_id') or '').strip()
        smiles = str(assembly.get('assembled_smiles') or '').strip()
        molblock = _preserve_molblock(assembly.get('assembled_molblock'))
        if not compound_id or compound_id.lower() == 'none' or not smiles:
            continue
        source_pages = [page for page in (assembly.get('source_pages') or []) if _safe_int(page) is not None]
        fragment_refs = [str(ref or '').strip() for ref in assembly.get('fragment_refs') or [] if str(ref or '').strip()]
        scaffold_ref = str(assembly.get('scaffold_ref') or '').strip()
        source_candidate = next(
            (candidates_by_ref.get(ref) for ref in fragment_refs if candidates_by_ref.get(ref)),
            None,
        ) or candidates_by_ref.get(scaffold_ref) or {}
        record = {
            'COMPOUND_ID': compound_id,
            'SMILES': smiles,
            'PAGE_NUM': source_pages[0] if source_pages else source_candidate.get('page', ''),
            'MOLBLOCK': molblock,
            'STRUCTURE_TYPE': 'markush_assembled',
            'IS_COMPLETE_COMPOUND': True,
            'FILTERED_OUT': False,
            'IMAGE_FILE': source_candidate.get('image_file', ''),
            'SEGMENT_FILE': source_candidate.get('segment_file', ''),
            'BOX_COORDS_FILE': source_candidate.get('box_coords_file', ''),
            'PAGE_IMAGE_FILE': source_candidate.get('page_image_file', ''),
            'source_pages': source_pages,
            'MARKUSH_ASSEMBLY_STATUS': 'assembled',
            'MARKUSH_SCAFFOLD_REF': scaffold_ref,
            'MARKUSH_FRAGMENT_REFS': ','.join(fragment_refs),
            'MARKUSH_VARIABLE_POSITIONS': ','.join(
                str(item or '').strip()
                for item in assembly.get('variable_positions') or []
                if str(item or '').strip()
            ),
            'MARKUSH_ASSEMBLY_METHOD': assembly.get('method', ''),
            'MARKUSH_NORMALIZATION_NOTES': ';'.join(
                str(item or '').strip()
                for item in assembly.get('normalization_notes') or []
                if str(item or '').strip()
            ),
        }
        records.append(record)
    return records


_INTERNAL_STRUCTURE_OUTPUT_COLUMNS = {
    'CANDIDATE_SOURCE',
}


def _strip_internal_structure_output_columns(record):
    if not isinstance(record, dict):
        return record
    return {
        key: value
        for key, value in record.items()
        if key not in _INTERNAL_STRUCTURE_OUTPUT_COLUMNS
    }


def plan_markush_relationships_for_group(
    pdf_file,
    group_pages,
    structures,
    filtered_structures,
    output_dir,
    audit_path=None,
    lang=DEFAULT_OCR_LANG,
    prior_markush_candidates=None,
    prior_page_contexts=None,
):
    from utils.llm_utils import plan_markush_structure_context

    candidate_records = list(filtered_structures or [])
    current_markush_candidates = _build_markush_structure_candidates(candidate_records)
    prior_markush_candidates = [
        candidate for candidate in (prior_markush_candidates or [])
        if isinstance(candidate, dict) and str(candidate.get('structure_type') or '').strip() == 'markush'
    ]
    prior_page_contexts = [context for context in (prior_page_contexts or []) if isinstance(context, dict)]
    markush_candidates = [*prior_markush_candidates, *current_markush_candidates]
    if not current_markush_candidates and not prior_markush_candidates:
        return None

    page_markdowns = {}
    context_pages = list(dict.fromkeys([
        *[
            _safe_int(context.get('page'))
            for context in prior_page_contexts
            if _safe_int(context.get('page')) is not None
        ],
        *[
            _safe_int(candidate.get('page'))
            for candidate in prior_markush_candidates
            if _safe_int(candidate.get('page')) is not None
        ],
        *group_pages,
    ]))
    try:
        page_markdowns = load_auto_detect_page_markdowns(pdf_file, context_pages, lang=lang)
    except Exception as exc:
        print(f"Warning: Markush context OCR failed for pages {group_pages}: {exc}")
        page_markdowns = {}

    page_contexts = _build_markush_page_contexts(
        context_pages,
        page_markdowns=page_markdowns,
        candidates=markush_candidates,
    )
    page_contexts = attach_markush_table_memory(page_contexts, markush_candidates)
    try:
        markush_candidates = review_markush_fragment_candidates(
            markush_candidates,
            page_contexts,
            audit_path=audit_path or os.path.join(output_dir, 'model_calls.jsonl'),
            review_pages=group_pages,
        )
        page_contexts = attach_markush_table_memory(page_contexts, markush_candidates)
    except Exception as exc:
        print(f"Warning: Markush fragment candidate visual review failed for pages {group_pages}: {exc}")
    try:
        plan = plan_markush_structure_context(
            page_contexts,
            markush_candidates,
            retry=2,
            audit_path=audit_path or os.path.join(output_dir, 'model_calls.jsonl'),
            metadata={'source': 'extract_structures', 'group_pages': list(group_pages)},
        )
    except Exception as exc:
        print(f"Warning: Markush relationship planner failed for pages {group_pages}: {exc}")
        plan = {
            'pages': [],
            'relationships': [],
            'error': str(exc),
        }
    else:
        try:
            plan = review_markush_relationships_with_visual_evidence(
                plan,
                page_contexts,
                markush_candidates,
                audit_path=audit_path or os.path.join(output_dir, 'model_calls.jsonl'),
            )
        except Exception as exc:
            print(f"Warning: Markush visual relationship review failed for pages {group_pages}: {exc}")
            plan['visual_review_error'] = str(exc)
    try:
        from utils.markush_assembly import build_markush_assembly_candidates
        plan['assembly_candidates'] = build_markush_assembly_candidates(plan, markush_candidates)
    except Exception as exc:
        print(f"Warning: Markush assembly planning failed for pages {group_pages}: {exc}")
        plan['assembly_error'] = str(exc)

    plan_payload = {
        'source_pages': list(group_pages),
        'page_contexts': page_contexts,
        'structure_candidates': markush_candidates,
        'plan': plan,
    }
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'markush_relationships.json')
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(plan_payload, handle, ensure_ascii=False, indent=2)
    return plan_payload


def extract_structures(
    pdf_file,
    structure_pages,
    output_dir,
    batch_size=4,
    page_workers=None,
    id_batch_size=None,
    progress_callback=None,
    structure_filter_strictness='strict',
    audit_path=None,
):
    """
    从 PDF 文件中提取化学结构并保存为 CSV 文件。
    支持不连续页面的解析。
    
    Args:
        pdf_file: PDF文件路径
        structure_pages: 页面列表，支持以下格式：
            - 单个页面: 5
            - 页面列表: [1, 3, 5, 7]
            - 页面范围: (start, end)
        output_dir: 输出目录
        batch_size: 并行处理的批处理大小，默认为4
        progress_callback: 进度回调函数
    """
    # 处理不同的输入格式
    if isinstance(structure_pages, (int, tuple)):
        if isinstance(structure_pages, tuple) and len(structure_pages) == 2:
            start, end = structure_pages
            page_list = list(range(start, end + 1))
        else:
            page_list = [structure_pages] if isinstance(structure_pages, int) else list(structure_pages)
    elif isinstance(structure_pages, list):
        page_list = structure_pages
    else:
        raise ValueError("structure_pages must be int, list, or tuple")
    
    print(f"Extracting structures from pages: {page_list}")
    
    all_structures = []
    all_filtered_structures = []
    all_markush_relationships = []
    
    def group_consecutive_pages(pages):
        pages = sorted(list(set(pages)))
        if not pages: return []
        groups = []
        current_group = [pages[0]]
        for i in range(1, len(pages)):
            if pages[i] == pages[i-1] + 1:
                current_group.append(pages[i])
            else:
                groups.append(current_group)
                current_group = [pages[i]]
        groups.append(current_group)
        return groups
    
    page_groups = group_consecutive_pages(page_list)
    print(f"Page groups for processing: {page_groups}")
    from structure_parser import extract_structures_from_pdf
    
    total_pages_in_task = len(page_list)
    pages_completed_so_far = 0
    prior_markush_context_candidates = []
    prior_markush_contexts = []

    for group_idx, group in enumerate(page_groups):
        start_page = min(group)
        end_page = max(group)
        
        print(f"Processing group {group_idx + 1}: pages {start_page}-{end_page}")
        
        group_output_dir = os.path.join(output_dir, f"structures_group_{group_idx}")
        os.makedirs(group_output_dir, exist_ok=True)
        
        def group_progress_callback(page_idx_in_group, total_pages_in_group, message):
            if progress_callback:
                global_pages_processed = pages_completed_so_far + page_idx_in_group
                progress_callback(global_pages_processed, total_pages_in_task, message)

        structures, filtered_structures = extract_structures_from_pdf(
            pdf_file=pdf_file,
            page_start=start_page,
            page_end=end_page,
            output=group_output_dir,
            structure_filter_strictness=structure_filter_strictness,
            batch_size=batch_size,
            page_workers=page_workers,
            id_batch_size=id_batch_size,
            progress_callback=group_progress_callback if progress_callback else None,
            audit_path=audit_path or os.path.join(output_dir, 'model_calls.jsonl'),
        )
        
        pages_completed_so_far += len(group)

        if structures:
            for structure in structures:
                if isinstance(structure, dict):
                    structure['source_pages'] = list(group)
                    structure['group_id'] = group_idx
                all_structures.extend([structure] if not isinstance(structure, list) else structure)
        if filtered_structures:
            for structure in filtered_structures:
                if isinstance(structure, dict):
                    structure['source_pages'] = list(group)
                    structure['group_id'] = group_idx
                all_filtered_structures.extend([structure] if not isinstance(structure, list) else structure)

        markush_plan = plan_markush_relationships_for_group(
            pdf_file,
            group,
            structures,
            filtered_structures,
            group_output_dir,
            audit_path=audit_path or os.path.join(output_dir, 'model_calls.jsonl'),
            prior_markush_candidates=prior_markush_context_candidates[-8:],
            prior_page_contexts=prior_markush_contexts[-8:],
        )
        if markush_plan:
            all_markush_relationships.append({
                'group_id': group_idx,
                **markush_plan,
            })
            for candidate in markush_plan.get('structure_candidates') or []:
                if (
                    isinstance(candidate, dict)
                    and str(candidate.get('structure_type') or '').strip() == 'markush'
                    and _safe_int(candidate.get('page')) is not None
                    and _safe_int(candidate.get('page')) <= max(group)
                ):
                    ref = str(candidate.get('ref') or '').strip()
                    if ref and not any(str(item.get('ref') or '').strip() == ref for item in prior_markush_context_candidates):
                        prior_markush_context_candidates.append(candidate)
            for context in markush_plan.get('page_contexts') or []:
                if isinstance(context, dict) and _safe_int(context.get('page')) is not None:
                    page = _safe_int(context.get('page'))
                    if page <= max(group) and not any(_safe_int(item.get('page')) == page for item in prior_markush_contexts):
                        prior_markush_contexts.append(context)
            markush_records = _build_markush_assembled_structure_records(markush_plan)
            if markush_records:
                print(f"Adding {len(markush_records)} assembled Markush structure(s) to review results")
                all_structures.extend(markush_records)
    
    if all_structures:
        canonicalize_record_compound_ids(
            all_structures,
            resolver_fn=resolve_compound_id_alias_with_llm,
            context_builder=lambda record: _build_structure_alias_context(record, stage='structure_group_aggregation'),
            overwrite_compound_id=False,
        )
        seen_combinations = set()
        unique_structures = []
        for structure in all_structures:
            if isinstance(structure, dict):
                compound_id = structure.get('COMPOUND_ID', '')
                smiles = structure.get('SMILES', '')
                combination_key = f"{compound_id}_{smiles}"
                if combination_key not in seen_combinations:
                    seen_combinations.add(combination_key)
                    unique_structures.append(_strip_internal_structure_output_columns(structure))
            else:
                structure_str = str(structure)
                if structure_str not in seen_combinations:
                    seen_combinations.add(structure_str)
                    unique_structures.append(structure)
        
        if unique_structures:
            structures_df = pd.DataFrame(unique_structures)
            structure_csv = os.path.join(output_dir, 'structures.csv')
            structures_df.to_csv(structure_csv, index=False, encoding='utf-8-sig')
            print(f"Chemical structures saved to {structure_csv} ({len(structures_df)} unique structures)")
            if all_filtered_structures:
                filtered_df = pd.DataFrame([
                    _strip_internal_structure_output_columns(record)
                    for record in all_filtered_structures
                ])
                filtered_csv = os.path.join(output_dir, 'filtered_structures.csv')
                filtered_df.to_csv(filtered_csv, index=False, encoding='utf-8-sig')
                print(f"Filtered structures saved to {filtered_csv} ({len(filtered_df)} filtered structures)")
            if all_markush_relationships:
                relationships_path = os.path.join(output_dir, 'markush_relationships.json')
                with open(relationships_path, 'w', encoding='utf-8') as handle:
                    json.dump(all_markush_relationships, handle, ensure_ascii=False, indent=2)
                print(f"Markush relationships saved to {relationships_path}")
            return structures_df
    elif all_filtered_structures:
        filtered_df = pd.DataFrame([
            _strip_internal_structure_output_columns(record)
            for record in all_filtered_structures
        ])
        filtered_csv = os.path.join(output_dir, 'filtered_structures.csv')
        filtered_df.to_csv(filtered_csv, index=False, encoding='utf-8-sig')
        print(f"Filtered structures saved to {filtered_csv} ({len(filtered_df)} filtered structures)")
        if all_markush_relationships:
            relationships_path = os.path.join(output_dir, 'markush_relationships.json')
            with open(relationships_path, 'w', encoding='utf-8') as handle:
                json.dump(all_markush_relationships, handle, ensure_ascii=False, indent=2)
            print(f"Markush relationships saved to {relationships_path}")

    print("No structures were extracted")
    return None


def extract_assay(pdf_file, assay_pages, assay_name, compound_id_list, output_dir, lang=DEFAULT_OCR_LANG, progress_callback=None):
    """
    提取指定活性数据，并保存为 JSON 文件。
    支持不连续页面的解析。
    
    Args:
        pdf_file: PDF文件路径
        assay_pages: 页面列表，支持以下格式：
            - 单个页面: 5
            - 页面列表: [1, 3, 5, 7]  
            - 页面范围: (start, end)
        assay_name: 活性测试名称
        compound_id_list: 化合物ID列表
        output_dir: 输出目录
        lang: 语言
        progress_callback: 进度回调函数
    """
    # 处理不同的输入格式
    if isinstance(assay_pages, (int, tuple)):
        if isinstance(assay_pages, tuple) and len(assay_pages) == 2:
            start, end = assay_pages
            page_list = list(range(start, end + 1))
        else:
            page_list = [assay_pages] if isinstance(assay_pages, int) else list(assay_pages)
    elif isinstance(assay_pages, list):
        page_list = assay_pages
    else:
        raise ValueError("assay_pages must be int, list, or tuple")
    
    print(f"Extracting assay '{assay_name}' from pages: {page_list}")
    
    all_assay_data = {}
    
    def group_consecutive_pages(pages):
        pages = sorted(list(set(pages)))
        if not pages: return []
        groups = []
        current_group = [pages[0]]
        for i in range(1, len(pages)):
            if pages[i] == pages[i-1] + 1:
                current_group.append(pages[i])
            else:
                groups.append(current_group)
                current_group = [pages[i]]
        groups.append(current_group)
        return groups
    
    page_groups = group_consecutive_pages(page_list)
    print(f"Assay page groups for processing: {page_groups}")
    from activity_parser import extract_activity_data
    
    total_pages_in_assay = len(page_list)
    pages_completed_so_far = 0

    for group_idx, group in enumerate(page_groups):
        start_page = min(group)
        end_page = max(group)
        
        print(f"Processing assay group {group_idx + 1}: pages {start_page}-{end_page}")
        
        def group_progress_callback(page_idx_in_group, total_pages_in_group, message):
            if progress_callback:
                global_pages_processed = pages_completed_so_far + page_idx_in_group
                progress_callback(global_pages_processed, total_pages_in_assay, message)

        assay_dict = extract_activity_data(
            pdf_file=pdf_file,
            assay_page_start=start_page,
            assay_page_end=end_page,
            assay_name=f"{assay_name}_Group_{group_idx}",
            pages_per_chunk=3,
            compound_id_list=compound_id_list,
            output_dir=output_dir,
            lang=lang,
            progress_callback=group_progress_callback if progress_callback else None
        )
        
        pages_completed_so_far += len(group)

        if assay_dict:
            all_assay_data.update(assay_dict)
    
    print(f"Assay data extracted for {assay_name}: {len(all_assay_data)} compounds")

    assay_name_clean = assay_name.replace(' ', '_').replace('/', '_')
    assay_json = os.path.join(output_dir, f"{assay_name_clean}_assay_data.json")
    import json
    with open(assay_json, 'w', encoding='utf-8') as f:
        json.dump(all_assay_data, f, ensure_ascii=False, indent=2)
    print(f"Assay data saved to {assay_json}")
    
    return all_assay_data


def extract_assays(
    pdf_file,
    assay_pages,
    assay_names,
    compound_id_list,
    output_dir,
    lang=DEFAULT_OCR_LANG,
    progress_callback=None,
    structure_records=None,
):
    """
    批量提取多个活性字段，共享同一份 OCR 页面内容与 chunk 解析，减少重复调用。
    """
    assay_names = [name.strip() for name in (assay_names or []) if name and name.strip()]
    if not assay_names:
        return {}

    if isinstance(assay_pages, (int, tuple)):
        if isinstance(assay_pages, tuple) and len(assay_pages) == 2:
            start, end = assay_pages
            page_list = list(range(start, end + 1))
        else:
            page_list = [assay_pages] if isinstance(assay_pages, int) else list(assay_pages)
    elif isinstance(assay_pages, list):
        page_list = assay_pages
    else:
        raise ValueError("assay_pages must be int, list, or tuple")

    print(f"Extracting assays {assay_names} from pages: {page_list}")
    all_assay_data_dicts = {assay_name: {} for assay_name in assay_names}

    def group_consecutive_pages(pages):
        pages = sorted(list(set(pages)))
        if not pages:
            return []
        groups = []
        current_group = [pages[0]]
        for i in range(1, len(pages)):
            if pages[i] == pages[i - 1] + 1:
                current_group.append(pages[i])
            else:
                groups.append(current_group)
                current_group = [pages[i]]
        groups.append(current_group)
        return groups

    page_groups = group_consecutive_pages(page_list)
    print(f"Assay page groups for shared processing: {page_groups}")
    from activity_parser import extract_activity_data_multi

    total_pages_in_assay = len(page_list)
    pages_completed_so_far = 0

    for group_idx, group in enumerate(page_groups):
        start_page = min(group)
        end_page = max(group)
        print(f"Processing shared assay group {group_idx + 1}: pages {start_page}-{end_page}")

        def group_progress_callback(page_idx_in_group, total_pages_in_group, message):
            if progress_callback:
                global_pages_processed = pages_completed_so_far + page_idx_in_group
                progress_callback(global_pages_processed, total_pages_in_assay, message)

        group_results = extract_activity_data_multi(
            pdf_file=pdf_file,
            assay_page_start=start_page,
            assay_page_end=end_page,
            assay_names=assay_names,
            pages_per_chunk=3,
            compound_id_list=compound_id_list,
            output_dir=output_dir,
            lang=lang,
            progress_callback=group_progress_callback if progress_callback else None,
            structure_records=structure_records,
        )

        pages_completed_so_far += len(group)
        for assay_name, assay_dict in (group_results or {}).items():
            if assay_name in all_assay_data_dicts and assay_dict:
                all_assay_data_dicts[assay_name].update(assay_dict)

    for assay_name, assay_dict in all_assay_data_dicts.items():
        assay_name_clean = assay_name.replace(' ', '_').replace('/', '_')
        assay_json = os.path.join(output_dir, f"{assay_name_clean}_assay_data.json")
        import json
        with open(assay_json, 'w', encoding='utf-8') as f:
            json.dump(assay_dict, f, ensure_ascii=False, indent=2)
        print(f"Assay data saved to {assay_json}")

    print(f"Shared assay extraction finished for {len(assay_names)} assays.")
    return all_assay_data_dicts


def merge_data(structures_df, assay_data_dicts, output_dir):
    """
    将提取的结构和活性数据合并成一个 CSV 文件。
    """
    structure_records = structures_df.to_dict(orient='records')
    canonicalize_record_compound_ids(
        structure_records,
        resolver_fn=resolve_compound_id_alias_with_llm,
        context_builder=lambda record: _build_structure_alias_context(record, stage='merge'),
        overwrite_compound_id=False,
    )
    structures_df = pd.DataFrame(structure_records)
    # COMPOUND_ID 转为字符串，防止匹配错误
    structures_df['COMPOUND_ID'] = structures_df['COMPOUND_ID'].astype(str)
    official_ids = [normalize_compound_id_text(value) for value in structures_df['COMPOUND_ID'].tolist()]
    alias_map, ambiguous_aliases = build_compound_id_alias_map(official_ids)
    canonical_values = []
    alias_sources = []
    raw_values = []
    for _, row in structures_df.iterrows():
        row_dict = row.to_dict()
        trace = resolve_compound_id_with_trace(
            row_dict.get('COMPOUND_ID', ''),
            official_ids,
            resolver_fn=resolve_compound_id_alias_with_llm,
            context=_build_structure_alias_context(row_dict, stage='merge_trace'),
        )
        canonical = (
            trace.get('canonical')
            or resolve_compound_id_alias(row_dict.get('COMPOUND_ID', ''), alias_map, ambiguous_aliases=ambiguous_aliases)
            or normalize_compound_id_text(row_dict.get('COMPOUND_ID', ''))
        )
        canonical_values.append(canonical)
        prior_source = row_dict.get('ALIAS_RESOLUTION_SOURCE', '')
        trace_source = trace.get('source') or ''
        if prior_source and trace_source == 'exact_match' and row_dict.get('RAW_COMPOUND_ID') != canonical:
            alias_sources.append(prior_source)
        else:
            alias_sources.append(trace_source or prior_source)
        raw_values.append(row_dict.get('RAW_COMPOUND_ID') or row_dict.get('COMPOUND_ID', ''))

    structures_df['RAW_COMPOUND_ID'] = raw_values
    structures_df['CANONICAL_COMPOUND_ID'] = canonical_values
    structures_df['ALIAS_RESOLUTION_SOURCE'] = alias_sources
    structures_df['_COMPOUND_ID_CANONICAL'] = canonical_values
    for assay_name, assay_dict in assay_data_dicts.items():
        resolved_assay_dict = remap_assay_dict_to_official_ids(
            assay_dict,
            official_ids,
            resolver_fn=resolve_compound_id_alias_with_llm,
            context_by_key={key: f"Assay name: {assay_name}" for key in assay_dict.keys()},
        )
        structures_df[assay_name] = structures_df['_COMPOUND_ID_CANONICAL'].map(resolved_assay_dict)

    print(
        "Assay merge inputs:",
        {assay_name: len(assay_dict or {}) for assay_name, assay_dict in assay_data_dicts.items()},
    )

    merged_csv = os.path.join(output_dir, 'merged.csv')
    structures_df = structures_df.drop(columns=['_COMPOUND_ID_CANONICAL'], errors='ignore')
    structures_df.to_csv(merged_csv, index=False)
    print(f"Merged data saved to {merged_csv}")
    return merged_csv


def load_structures(output_dir):
    """
    如果存在 structures.csv，则加载它。
    """
    structure_csv = os.path.join(output_dir, 'structures.csv')
    if os.path.exists(structure_csv):
        print(f"Loading existing structures from {structure_csv}")
        return pd.read_csv(structure_csv)
    else:
        print("No existing structures.csv found.")
        return None


def parse_pages_argument(pages_str):
    """
    解析页面参数字符串，支持以下格式：
    - "1-5": 页面范围
    - "1,3,5": 页面列表
    - "1-3,5,7-9": 混合格式
    """
    if not pages_str:
        return None
    
    pages = []
    parts = pages_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part and not part.startswith('-'):
            # 处理范围，如 "1-5"
            try:
                start, end = map(int, part.split('-'))
                pages.extend(range(start, end + 1))
            except ValueError:
                raise ValueError(f"Invalid page range format: {part}")
        else:
            # 处理单个页面
            try:
                pages.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid page number: {part}")
    
    return sorted(list(set(pages)))  # 去重并排序


def main():
    parser = argparse.ArgumentParser(description='Extract chemical structures and assay data from PDF files.')
    parser.add_argument('pdf_file', type=str, help='PDF file to process')
    parser.add_argument('--structure-pages', type=str, help='Pages for structures (e.g., "1-5" or "1,3,5" or "1-3,5,7-9")', default=None)
    parser.add_argument('--assay-pages', type=str, help='Pages for assays (e.g., "1-5" or "1,3,5" or "1-3,5,7-9")', default=None)
    parser.add_argument('--assay-names', type=str, help='Assay names to extract (comma-separated)', default='')
    parser.add_argument('--auto-structure-pages', action='store_true', help='Automatically detect likely structure pages')
    parser.add_argument('--auto-assay-pages', action='store_true', help='Automatically detect likely assay pages')
    parser.add_argument('--auto-assay-names', action='store_true', help='Automatically detect assay names when not provided')
    parser.add_argument('--batch-size', type=int, help='Batch size for parallel processing', default=4)
    parser.add_argument('--page-workers', type=int, help='Page-level concurrent workers for structure extraction', default=None)
    parser.add_argument('--id-batch-size', type=int, help='Concurrent workers for structure ID extraction', default=None)
    parser.add_argument('--output', type=str, help='Output directory', default='output')
    parser.add_argument('--lang', type=str, help='Language for text extraction', default=DEFAULT_OCR_LANG)
    
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    # 获取 PDF 总页数
    total_pages = get_total_pages(args.pdf_file)
    print(f"PDF has {total_pages} pages total")

    structures_df = None
    structure_requested = bool(args.structure_pages or args.auto_structure_pages)
    assay_requested = bool(args.assay_pages or args.auto_assay_pages or args.assay_names or args.auto_assay_names)
    default_full_extraction = not structure_requested and not assay_requested

    if default_full_extraction:
        args.auto_structure_pages = True
        args.auto_assay_pages = True
        args.auto_assay_names = True
        print("No structure pages provided; enabling automatic structure-page detection.")
        print("No assay pages provided; enabling automatic assay-page detection.")
        print("No assay names provided; enabling automatic assay-name detection.")
    elif not structure_requested:
        args.auto_structure_pages = True
        print("No structure pages provided; enabling automatic structure-page detection.")

    if assay_requested and not args.assay_pages and not args.auto_assay_pages:
        args.auto_assay_pages = True
        print("No assay pages provided; enabling automatic assay-page detection.")

    if assay_requested and not args.assay_names and not args.auto_assay_names:
        args.auto_assay_names = True
        print("No assay names provided; enabling automatic assay-name detection.")

    detection_plan = None
    if args.auto_structure_pages or args.auto_assay_pages or args.auto_assay_names:
        print("Building document auto-detection plan...")
        detection_plan = build_document_auto_plan(args.pdf_file)
        if detection_plan.get('cache_hit'):
            print(f"Auto-detection cache hit: {detection_plan.get('cache_path', '')}")
        elif detection_plan.get('cache_path'):
            print(f"Auto-detection cache saved: {detection_plan.get('cache_path')}")
        detection_report_path = os.path.join(args.output, 'auto_detection_report.json')
        with open(detection_report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'cache_hit': bool(detection_plan.get('cache_hit')),
                'cache_path': detection_plan.get('cache_path', ''),
                'structure_pages': detection_plan['structure_pages'],
                'structure_diagnostics_preview': detection_plan['structure_diagnostics'][:50],
                'assay_pages': detection_plan['assay_pages'],
                'assay_diagnostics_preview': detection_plan['assay_diagnostics'][:50],
                'assay_names': detection_plan['assay_names'],
            }, f, ensure_ascii=False, indent=2)
        print(f"Auto-detection report saved to {detection_report_path}")

    # 处理结构提取
    if args.auto_structure_pages:
        structure_pages = detection_plan['structure_pages'] if detection_plan else []
        diagnostics = detection_plan['structure_diagnostics'] if detection_plan else []
        print(f"Auto-detected structure pages: {structure_pages[:50]}{'...' if len(structure_pages) > 50 else ''}")
        print(f"Structure detection preview: {diagnostics[:10]}")
        structures_df = extract_structures(
            pdf_file=args.pdf_file,
            structure_pages=structure_pages,
            output_dir=args.output,
            batch_size=args.batch_size,
            page_workers=args.page_workers,
            id_batch_size=args.id_batch_size,
        )
    elif args.structure_pages:
        # 使用新的页面格式
        structure_pages = parse_pages_argument(args.structure_pages)
        print(f"Extracting structures from pages: {structure_pages}")
        structures_df = extract_structures(
            pdf_file=args.pdf_file,
            structure_pages=structure_pages,
            output_dir=args.output,
            batch_size=args.batch_size,
            page_workers=args.page_workers,
            id_batch_size=args.id_batch_size,
        )

    assay_data_dicts = {}

    # 处理活性数据提取
    assay_names = [name.strip() for name in args.assay_names.split(',') if name.strip()] if args.assay_names else []
    if args.auto_assay_names and not assay_names:
        assay_pages_for_names = detection_plan['assay_pages'] if (detection_plan and args.auto_assay_pages) else None
        if assay_pages_for_names is not None:
            diagnostics = detection_plan['assay_diagnostics']
            print(f"Auto-detected assay pages: {assay_pages_for_names[:50]}{'...' if len(assay_pages_for_names) > 50 else ''}")
            print(f"Assay detection preview: {diagnostics[:10]}")
        assay_names = detection_plan['assay_names'] if detection_plan else auto_detect_assay_names(args.pdf_file, assay_pages=assay_pages_for_names)
        print(f"Auto-detected assay names: {assay_names}")

    if assay_names:
        print(f"Assay names to extract: {assay_names}")
        
        # 尝试读取当前目录下的 structures.csv 以获取化合物ID
        if structures_df is None:
            structures_df = load_structures(args.output)

        compound_id_list = structures_df['COMPOUND_ID'].tolist() if structures_df is not None else None
        if compound_id_list:
            print(f"Found {len(compound_id_list)} compound IDs for assay extraction")
        else:
            print("No compound IDs available - extracting assay data without structure matching")

        if args.auto_assay_pages:
            assay_pages = detection_plan['assay_pages'] if detection_plan else []
            diagnostics = detection_plan['assay_diagnostics'] if detection_plan else []
            print(f"Auto-detected assay pages: {assay_pages[:50]}{'...' if len(assay_pages) > 50 else ''}")
        elif args.assay_pages:
            assay_pages = parse_pages_argument(args.assay_pages)
            print(f"Extracting assays from pages: {assay_pages}")
        else:
            print("No assay pages specified, using all pages")
            assay_pages = list(range(1, total_pages + 1))

        if len(assay_names) > 1:
            raw_assay_names = list(assay_names)
            assay_names = verify_assay_names_for_pages(
                args.pdf_file,
                assay_pages,
                raw_assay_names,
                output_dir=args.output,
                lang=args.lang,
                audit_path=os.path.join(args.output, 'model_calls.jsonl'),
            )
            if assay_names != raw_assay_names:
                print(f"Verified assay names: {assay_names} (from {raw_assay_names})")

        assay_data_dicts = extract_assays(
            pdf_file=args.pdf_file,
            assay_pages=assay_pages,
            assay_names=assay_names,
            compound_id_list=compound_id_list,
            output_dir=args.output,
            lang=args.lang,
            structure_records=structures_df.to_dict(orient='records') if structures_df is not None else None,
        )

    # 如果同时提取了结构和 assay 数据，则合并数据
    if structures_df is not None and assay_data_dicts:
        merge_data(structures_df, assay_data_dicts, args.output)
    elif assay_data_dicts:
        print("Assay data extracted but no structures available to merge.")
    elif structures_df is not None:
        print("Structures extracted successfully.")
    else:
        print("No data extracted. Please check your input parameters.")


if __name__ == '__main__':
    main()
