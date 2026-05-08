import os
import sys
import argparse
import json
import gc
import hashlib
import math
import pandas as pd
import PyPDF2
import re
import fitz
import numpy as np
import tempfile
import requests
from utils.compound_id_utils import build_compound_id_alias_map, resolve_compound_id_alias, remap_assay_dict_to_official_ids, normalize_compound_id_text, canonicalize_record_compound_ids, resolve_compound_id_with_trace
from utils.llm_utils import resolve_compound_id_alias as resolve_compound_id_alias_with_llm
from utils.skill_prompt_loader import render_skill_reference

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
DOCUMENT_AUTO_DETECT_CACHE_ENABLED = bool(getattr(_constants, 'DOCUMENT_AUTO_DETECT_CACHE_ENABLED', True)) if _constants else True
DOCUMENT_AUTO_DETECT_CACHE_DIR = str(getattr(_constants, 'DOCUMENT_AUTO_DETECT_CACHE_DIR', '') or '').strip() if _constants else ''
DOCUMENT_AUTO_DETECT_CACHE_VERSION = 7


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
    from utils.llm_utils import extract_json_content

    json_content = extract_json_content(response_text) or response_text
    payload = json.loads(json_content)
    allowed_pages = {int(page) for page in page_numbers}

    raw_pages = payload.get('structure_pages', [])
    detected = set()
    if isinstance(raw_pages, list):
        for item in raw_pages:
            try:
                page = int(item)
            except Exception:
                continue
            if page in allowed_pages:
                detected.add(page)

    decisions_by_page = {}
    raw_decisions = payload.get('decisions', [])
    if isinstance(raw_decisions, list):
        for item in raw_decisions:
            if not isinstance(item, dict):
                continue
            try:
                page = int(item.get('page'))
            except Exception:
                continue
            if page not in allowed_pages:
                continue
            has_structure = item.get('has_structure')
            if isinstance(has_structure, str):
                has_structure = has_structure.strip().lower() in {'true', 'yes', '1', 'structure', 'has_structure'}
            has_structure = bool(has_structure)
            if has_structure:
                detected.add(page)
            decisions_by_page[page] = {
                'has_structure': has_structure,
                'confidence': str(item.get('confidence', '') or '').strip(),
                'reason': str(item.get('reason', '') or '').strip(),
            }

    return sorted(detected), decisions_by_page


def detect_structure_pages_with_vision_contact_sheets(pdf_file, page_numbers):
    from utils.llm_utils import call_visual_model

    page_numbers = [int(page) for page in page_numbers]
    detected_pages = set()
    decisions_by_page = {}

    with tempfile.TemporaryDirectory(prefix='biocheminsight_structure_detect_') as tmp_dir:
        for batch_index, batch_pages in enumerate(_chunked(page_numbers, STRUCTURE_AUTO_DETECT_VISION_BATCH_SIZE), start=1):
            contact_sheet = os.path.join(tmp_dir, f'structure_pages_batch_{batch_index}.png')
            _build_structure_page_contact_sheet(pdf_file, batch_pages, contact_sheet)
            prompt = _build_structure_page_detection_prompt(batch_pages)

            last_error = None
            attempts = max(1, int(STRUCTURE_AUTO_DETECT_VISION_MAX_RETRIES or 1) + 1)
            for attempt in range(1, attempts + 1):
                try:
                    response_text = call_visual_model(contact_sheet, prompt)
                    batch_detected, batch_decisions = _parse_structure_page_detection_response(response_text, batch_pages)
                    detected_pages.update(batch_detected)
                    decisions_by_page.update(batch_decisions)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    print(
                        f"Warning: structure-page vision detection batch {batch_index} "
                        f"attempt {attempt}/{attempts} failed: {exc}"
                    )
            if last_error is not None:
                raise RuntimeError(
                    f"Vision structure-page detection failed for pages {batch_pages}: {last_error}"
                ) from last_error

            try:
                os.remove(contact_sheet)
            except OSError:
                pass
            gc.collect()

    return sorted(detected_pages), decisions_by_page


def review_structure_pages_with_vision_contact_sheets(pdf_file, candidate_pages):
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

            last_error = None
            attempts = max(1, int(STRUCTURE_AUTO_DETECT_VISION_MAX_RETRIES or 1) + 1)
            for attempt in range(1, attempts + 1):
                try:
                    response_text = call_visual_model(contact_sheet, prompt)
                    batch_reviewed, batch_decisions = _parse_structure_page_detection_response(response_text, batch_pages)
                    reviewed_pages.update(batch_reviewed)
                    decisions_by_page.update(batch_decisions)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    print(
                        f"Warning: structure-page strict vision review batch {batch_index} "
                        f"attempt {attempt}/{attempts} failed: {exc}"
                    )
            if last_error is not None:
                raise RuntimeError(
                    f"Strict vision structure-page review failed for pages {batch_pages}: {last_error}"
                ) from last_error

            try:
                os.remove(contact_sheet)
            except OSError:
                pass
            gc.collect()

    return sorted(reviewed_pages), decisions_by_page


def auto_detect_structure_pages(pdf_file):
    return auto_detect_structure_pages_from_texts(pdf_file, _load_pdf_page_texts(pdf_file))


def auto_detect_structure_pages_from_texts(pdf_file, page_texts):
    page_texts = _coerce_page_texts(page_texts) or []

    if page_texts:
        page_numbers = [int(page_num) for page_num, _ in page_texts]
    else:
        page_numbers = list(range(1, get_total_pages(pdf_file) + 1))
    diagnostics = [{'page': page_num} for page_num in page_numbers]

    initial_pages, initial_decisions = detect_structure_pages_with_vision_contact_sheets(pdf_file, page_numbers)
    if STRUCTURE_AUTO_DETECT_VISION_REVIEW_ENABLED:
        detected_pages, review_decisions = review_structure_pages_with_vision_contact_sheets(pdf_file, initial_pages)
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


def load_auto_detect_page_markdowns(pdf_file, page_numbers, lang='en'):
    page_numbers = sorted({int(page) for page in page_numbers})
    if not page_numbers:
        return {}

    server_url = str(getattr(_constants, 'PADDLEOCR_SERVER_URL', '') or '').strip() if _constants else ''
    if not server_url:
        raise RuntimeError("PADDLEOCR_SERVER_URL is required for LLM-based assay page auto-detection.")

    endpoint = f"{server_url.rstrip('/')}/v1/pdf-to-markdown"
    page_markdowns = {}

    for group in _chunked(page_numbers, ASSAY_AUTO_DETECT_LLM_BATCH_SIZE):
        page_start = min(group)
        page_end = max(group)
        expected_pages = list(range(page_start, page_end + 1))
        try:
            with open(pdf_file, 'rb') as pdf_stream:
                response = requests.post(
                    endpoint,
                    files={'file': (os.path.basename(pdf_file) or 'document.pdf', pdf_stream, 'application/pdf')},
                    data={
                        'page_start': str(page_start),
                        'page_end': str(page_end),
                        'lang': lang,
                        'return_raw': 'false',
                    },
                    timeout=600,
                )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, OSError, ValueError) as exc:
            raise RuntimeError(f"PaddleOCR request failed for pages {page_start}-{page_end}.") from exc

        content_list = _extract_payload_page_markdowns(payload)
        if len(content_list) != len(expected_pages):
            raise RuntimeError(
                f"PaddleOCR page split mismatch for pages {page_start}-{page_end}: "
                f"expected {len(expected_pages)}, got {len(content_list)}."
            )
        for page_num, markdown in zip(expected_pages, content_list):
            if page_num in group:
                page_markdowns[page_num] = markdown

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
    from utils.llm_utils import extract_json_content

    json_content = extract_json_content(response_text) or response_text
    payload = json.loads(json_content)
    allowed_pages = {int(page) for page in page_numbers}

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
    if isinstance(raw_decisions, list):
        for item in raw_decisions:
            if not isinstance(item, dict):
                continue
            try:
                page = int(item.get('page'))
            except Exception:
                continue
            if page not in allowed_pages:
                continue
            has_assay_data = item.get('has_assay_data')
            if isinstance(has_assay_data, str):
                has_assay_data = has_assay_data.strip().lower() in {'true', 'yes', '1', 'assay', 'has_assay_data'}
            has_assay_data = bool(has_assay_data)
            if has_assay_data:
                detected.add(page)
            page_names = []
            for name_item in item.get('assay_names', []) if isinstance(item.get('assay_names', []), list) else []:
                name = str(name_item or '').strip()
                if name:
                    page_names.append(name)
                    key = name.lower()
                    if key not in seen_names:
                        seen_names.add(key)
                        assay_names.append(name)
            decisions_by_page[page] = {
                'has_assay_data': has_assay_data,
                'confidence': str(item.get('confidence', '') or '').strip(),
                'assay_names': page_names,
                'reason': str(item.get('reason', '') or '').strip(),
            }

    return sorted(detected), assay_names, decisions_by_page


def detect_assay_pages_with_ocr_llm(pdf_file, page_numbers, assay_names=None, page_markdowns=None):
    from utils.llm_utils import (
        LLM_MODEL_TYPE,
        LLM_TEXT_MODEL_KEY,
        LLM_TEXT_MODEL_NAME,
        LLM_TEXT_MODEL_URL,
        DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT,
        GEMINI_API_KEY_FOR_GEMINI_MODELS,
        TEXT_MODEL_RUNTIME,
        configure_genai,
        get_system_prompt,
        get_task_temperature,
        require_genai,
        require_openai,
        sanitize_model_response_text,
    )

    page_numbers = sorted({int(page) for page in page_numbers})
    if page_markdowns is None:
        page_markdowns = load_auto_detect_page_markdowns(pdf_file, page_numbers)

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

    model_type = LLM_MODEL_TYPE
    if not LLM_TEXT_MODEL_KEY or not LLM_TEXT_MODEL_URL or not LLM_TEXT_MODEL_NAME:
        model_type = 'gemini'

    for batch_index, batch_pages in enumerate(_chunked(page_numbers, ASSAY_AUTO_DETECT_LLM_BATCH_SIZE), start=1):
        batch_payload = [(page, page_markdowns.get(page, '')) for page in batch_pages]
        prompt = _build_assay_page_detection_prompt(batch_payload, assay_names=assay_names)
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                if model_type == 'gemini':
                    configure_genai(GEMINI_API_KEY_FOR_GEMINI_MODELS)
                    model = require_genai().GenerativeModel(DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT)
                    response = model.generate_content(prompt)
                    if not response.candidates or not response.candidates[0].content.parts:
                        raise ValueError("Gemini returned no content for assay page detection.")
                    response_text = response.text
                else:
                    client = require_openai()(api_key=LLM_TEXT_MODEL_KEY, base_url=LLM_TEXT_MODEL_URL)
                    response = client.chat.completions.create(
                        model=LLM_TEXT_MODEL_NAME,
                        messages=[
                            {"role": "system", "content": json_system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                    )
                    response_text = sanitize_model_response_text(response.choices[0].message.content or '')
                batch_detected, batch_names, batch_decisions = _parse_assay_page_detection_response(response_text, batch_pages)
                detected_pages.update(batch_detected)
                decisions_by_page.update(batch_decisions)
                for name in batch_names:
                    key = name.lower()
                    if key not in seen_names:
                        seen_names.add(key)
                        detected_names.append(name)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print(
                    f"Warning: assay-page LLM detection batch {batch_index} "
                    f"attempt {attempt}/{attempts} failed: {exc}"
                )
        if last_error is not None:
            raise RuntimeError(f"LLM assay-page detection failed for pages {batch_pages}: {last_error}") from last_error

    return sorted(detected_pages), detected_names, decisions_by_page, page_markdowns


def auto_detect_assay_pages(pdf_file, assay_names=None, page_texts=None):
    assay_names = [name.strip() for name in (assay_names or []) if name and name.strip()]
    if page_texts:
        page_numbers = [int(page_num) for page_num, _ in _coerce_page_texts(page_texts)]
    else:
        page_numbers = list(range(1, get_total_pages(pdf_file) + 1))

    detected_pages, detected_names, decisions, page_markdowns = detect_assay_pages_with_ocr_llm(
        pdf_file,
        page_numbers,
        assay_names=assay_names,
    )
    detected_set = set(detected_pages)
    diagnostics = []
    for page_num in page_numbers:
        diagnostics.append({
            'page': int(page_num),
            'include': int(page_num) in detected_set,
            'auto_detect_source': 'ocr_llm_skill',
            'llm_page_detection': decisions.get(int(page_num), {}),
            'ocr_markdown_chars': len(page_markdowns.get(int(page_num), '') or ''),
        })
    if detected_names:
        diagnostics.append({
            'page': None,
            'auto_detect_source': 'ocr_llm_skill',
            'detected_assay_names': detected_names,
        })
    return sorted(detected_set), diagnostics


def _normalize_assay_name_candidate(candidate):
    candidate = re.sub(r'^(?:Example|Enzyme|Table\s+\d+\.?|Assay data with exemplary compounds\.?)\s*', '', candidate, flags=re.IGNORECASE)
    candidate = re.sub(r'Cyclin\s+El\b', 'Cyclin E1', candidate, flags=re.IGNORECASE)
    candidate = re.sub(r'Cyclin\s+Bl\b', 'Cyclin B1', candidate, flags=re.IGNORECASE)
    candidate = candidate.replace('CDKl', 'CDK1')
    candidate = re.sub(r'^.*?(NanoBRET|CDK\d+/?cyclin|IC50|EC50|Ki|Kd)', r'\1', candidate, flags=re.IGNORECASE)
    candidate = re.sub(r'\s+', ' ', candidate).strip(' ,;:-')
    return candidate


def auto_detect_assay_names(pdf_file, assay_pages=None, page_texts=None):
    if assay_pages is None:
        assay_pages, diagnostics = auto_detect_assay_pages(pdf_file, assay_names=None, page_texts=page_texts)
        for item in diagnostics:
            names = item.get('detected_assay_names') if isinstance(item, dict) else None
            if isinstance(names, list):
                return [str(name).strip() for name in names if str(name).strip()]

    assay_pages = sorted({int(page) for page in (assay_pages or [])})
    if not assay_pages:
        return []

    _, detected_names, _, _ = detect_assay_pages_with_ocr_llm(
        pdf_file,
        assay_pages,
        assay_names=None,
    )
    return detected_names


def _legacy_auto_detect_assay_names_from_text(pdf_file, assay_pages=None, page_texts=None):
    page_text_map = dict(_coerce_page_texts(page_texts) or _load_pdf_page_texts(pdf_file))
    discovered = []
    discovered_scores = {}
    explicit_header_pattern = re.compile(
        r'((?:NanoBRET|Enzyme)\s+[A-Za-z0-9/\-+ ]{0,80}?\b(?:IC50|EC50|Ki|Kd)\b\s*(?:\([^)]+\))?)',
        re.IGNORECASE,
    )
    secondary_pattern = re.compile(
        r'([A-Za-z0-9/\-+ ]{0,60}?\b(?:IC50|EC50|Ki|Kd)\b\s*(?:\([^)]+\))?)',
        re.IGNORECASE,
    )
    seen_lower = set()

    for page_num in assay_pages:
        text = page_text_map.get(page_num, '')
        normalized = _normalize_page_text(text)
        candidates = []
        for match in explicit_header_pattern.finditer(normalized):
            candidates.append(match.group(1))
        for match in secondary_pattern.finditer(normalized):
            candidates.append(match.group(1))

        for candidate in candidates:
            candidate = _normalize_assay_name_candidate(candidate)
            if not candidate:
                continue
            metric = re.search(r'\b(IC50|EC50|Ki|Kd)\b', candidate, flags=re.IGNORECASE)
            if not metric:
                continue
            if len(candidate) < len(metric.group(1)) or len(candidate) > 45:
                continue
            if not any(token in candidate.lower() for token in ['ic50', 'ec50', 'ki', 'kd']):
                continue
            if candidate.lower() in {'ki', 'kd', 'ic50', 'ec50'}:
                continue
            if not (
                'nanobret' in candidate.lower()
                or 'cdk' in candidate.lower()
                or '/' in candidate
                or 'cyclin' in candidate.lower()
            ):
                continue
            lowered = candidate.lower()
            specificity = 0
            specificity += 3 if 'nanobret' in lowered else 0
            specificity += 2 if 'cdk' in lowered else 0
            specificity += 1 if 'cyclin' in lowered else 0
            specificity += 1 if '(' in candidate and ')' in candidate else 0
            previous_score = discovered_scores.get(lowered, -1)
            if lowered not in seen_lower:
                seen_lower.add(lowered)
                discovered.append(candidate)
                discovered_scores[lowered] = specificity
            elif specificity > previous_score:
                discovered_scores[lowered] = specificity
                for idx, item in enumerate(discovered):
                    if item.lower() == lowered:
                        discovered[idx] = candidate
                        break

    return discovered


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


def extract_structures(
    pdf_file,
    structure_pages,
    output_dir,
    engine='molnextr',
    batch_size=4,
    page_workers=None,
    id_batch_size=None,
    progress_callback=None,
    structure_filter_strictness='strict',
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
        engine: 结构识别引擎
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
            engine=engine,
            structure_filter_strictness=structure_filter_strictness,
            batch_size=batch_size,
            page_workers=page_workers,
            id_batch_size=id_batch_size,
            progress_callback=group_progress_callback if progress_callback else None
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
    
    if all_structures:
        canonicalize_record_compound_ids(
            all_structures,
            resolver_fn=resolve_compound_id_alias_with_llm,
            context_builder=lambda record: _build_structure_alias_context(record, stage='structure_group_aggregation'),
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
                    unique_structures.append(structure)
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
                filtered_df = pd.DataFrame(all_filtered_structures)
                filtered_csv = os.path.join(output_dir, 'filtered_structures.csv')
                filtered_df.to_csv(filtered_csv, index=False, encoding='utf-8-sig')
                print(f"Filtered structures saved to {filtered_csv} ({len(filtered_df)} filtered structures)")
            return structures_df
    elif all_filtered_structures:
        filtered_df = pd.DataFrame(all_filtered_structures)
        filtered_csv = os.path.join(output_dir, 'filtered_structures.csv')
        filtered_df.to_csv(filtered_csv, index=False, encoding='utf-8-sig')
        print(f"Filtered structures saved to {filtered_csv} ({len(filtered_df)} filtered structures)")

    print("No structures were extracted")
    return None


def extract_assay(pdf_file, assay_pages, assay_name, compound_id_list, output_dir, lang='en', progress_callback=None):
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


def extract_assays(pdf_file, assay_pages, assay_names, compound_id_list, output_dir, lang='en', progress_callback=None):
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
    parser.add_argument('--engine', type=str, help='Engine for structure extraction (molscribe, molnextr, molvec)', default='molnextr')
    parser.add_argument('--batch-size', type=int, help='Batch size for parallel processing', default=4)
    parser.add_argument('--page-workers', type=int, help='Page-level concurrent workers for structure extraction', default=None)
    parser.add_argument('--id-batch-size', type=int, help='Concurrent workers for structure ID extraction', default=None)
    parser.add_argument('--output', type=str, help='Output directory', default='output')
    parser.add_argument('--lang', type=str, help='Language for text extraction', default='en')
    
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
            engine=args.engine,
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
            engine=args.engine,
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

        assay_data_dicts = extract_assays(
            pdf_file=args.pdf_file,
            assay_pages=assay_pages,
            assay_names=assay_names,
            compound_id_list=compound_id_list,
            output_dir=args.output,
            lang=args.lang,
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
