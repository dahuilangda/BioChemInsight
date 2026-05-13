import os
import sys
import json
import re
import tempfile
from collections import OrderedDict
from html.parser import HTMLParser
from typing import Dict, Optional

import requests

from utils.llm_utils import (
    build_review_assay_values_prompt,
    call_visual_model,
    content_to_multi_assay_dict,
    extract_json_content,
    identify_assay_visual_review_requests,
    reconcile_assay_values_with_visual_report,
    resolve_compound_id_alias,
)
from utils.file_utils import write_json_file
from utils.compound_id_utils import remap_assay_dict_to_official_ids
# from constants import GEMINI_API_KEY, GEMINI_MODEL_NAME

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(SCRIPT_DIR) != '' and os.path.exists(os.path.join(SCRIPT_DIR, '..', 'constants.py')):
         sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
         import constants
         sys.path.pop(0)
    else:
        import constants
except ImportError:
    print("Error: constants.py not found. Please ensure it's in the same directory, parent directory, or your PYTHONPATH.")
    sys.exit(1)

PADDLEOCR_SERVER_URL: Optional[str] = getattr(constants, 'PADDLEOCR_SERVER_URL', None)
DEFAULT_OCR_LANG = str(getattr(constants, 'PADDLEOCR_LANG', 'auto') or 'auto')
ASSAY_PAGE_TEXT_CACHE_ENABLED = bool(getattr(constants, 'ASSAY_PAGE_TEXT_CACHE_ENABLED', True))
ASSAY_PAGE_TEXT_CACHE_MAX_ENTRIES = max(1, int(getattr(constants, 'ASSAY_PAGE_TEXT_CACHE_MAX_ENTRIES', 4) or 4))
ASSAY_PAGE_TEXT_CACHE_MAX_PAGES = max(1, int(getattr(constants, 'ASSAY_PAGE_TEXT_CACHE_MAX_PAGES', 64) or 64))
ASSAY_VISUAL_VALUE_REVIEW_ENABLED = bool(getattr(constants, 'ASSAY_VISUAL_VALUE_REVIEW_ENABLED', True))
ASSAY_VISUAL_VALUE_REVIEW_MAX_PAGES = max(1, int(getattr(constants, 'ASSAY_VISUAL_VALUE_REVIEW_MAX_PAGES', 3) or 3))
ASSAY_VISUAL_VALUE_REVIEW_RENDER_SCALE = float(getattr(constants, 'ASSAY_VISUAL_VALUE_REVIEW_RENDER_SCALE', 2.0) or 2.0)
ASSAY_VISUAL_VALUE_REVIEW_MAX_WIDTH = max(800, int(getattr(constants, 'ASSAY_VISUAL_VALUE_REVIEW_MAX_WIDTH', 1400) or 1400))


_ASSAY_PAGE_CONTENT_CACHE = OrderedDict()


class _SimpleHtmlTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self._table = None
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == 'table':
            self._table = []
        elif tag == 'tr' and self._table is not None:
            self._row = []
        elif tag in {'td', 'th'} and self._row is not None:
            attr_map = {str(key).lower(): value for key, value in attrs}
            self._cell = {
                'text_parts': [],
                'rowspan': _safe_positive_int(attr_map.get('rowspan'), 1),
                'colspan': _safe_positive_int(attr_map.get('colspan'), 1),
            }

    def handle_data(self, data):
        if self._cell is not None:
            self._cell['text_parts'].append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in {'td', 'th'} and self._cell is not None and self._row is not None:
            text = re.sub(r'\s+', ' ', ''.join(self._cell.get('text_parts') or [])).strip()
            self._row.append({
                'text': text,
                'rowspan': self._cell.get('rowspan') or 1,
                'colspan': self._cell.get('colspan') or 1,
            })
            self._cell = None
        elif tag == 'tr' and self._row is not None and self._table is not None:
            if any(cell for cell in self._row):
                self._table.append(self._row)
            self._row = None
        elif tag == 'table' and self._table is not None:
            if self._table:
                self.tables.append(_expand_html_table_grid(self._table))
            self._table = None


def _safe_positive_int(value, default=1):
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _expand_html_table_grid(rows):
    grid = []
    active_rowspans = {}
    max_cols = 0
    for raw_row in rows:
        row = []
        col_idx = 0
        cell_iter = iter(raw_row)
        next_cell = next(cell_iter, None)
        while next_cell is not None or col_idx in active_rowspans:
            if col_idx in active_rowspans:
                text, remaining = active_rowspans[col_idx]
                row.append(text)
                if remaining > 1:
                    active_rowspans[col_idx] = (text, remaining - 1)
                else:
                    active_rowspans.pop(col_idx, None)
                col_idx += 1
                continue

            cell = next_cell
            next_cell = next(cell_iter, None)
            text = str(cell.get('text') or '').strip()
            rowspan = _safe_positive_int(cell.get('rowspan'), 1)
            colspan = _safe_positive_int(cell.get('colspan'), 1)
            for _ in range(colspan):
                row.append(text)
                if rowspan > 1:
                    active_rowspans[col_idx] = (text, rowspan - 1)
                col_idx += 1
        max_cols = max(max_cols, len(row))
        grid.append(row)

    if max_cols:
        grid = [row + [''] * (max_cols - len(row)) for row in grid]
    return grid


def _extract_ocr_tables(markdown_text):
    tables = []
    parser = _SimpleHtmlTableParser()
    try:
        parser.feed(str(markdown_text or ''))
        tables.extend(parser.tables)
    except Exception:
        pass

    pipe_rows = []
    for line in str(markdown_text or '').splitlines():
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            cells = [cell.strip() for cell in stripped.strip('|').split('|')]
            if cells and not all(re.fullmatch(r':?-{2,}:?', cell or '') for cell in cells):
                pipe_rows.append(cells)
        elif pipe_rows:
            if len(pipe_rows) >= 2:
                tables.append(pipe_rows)
            pipe_rows = []
    if len(pipe_rows) >= 2:
        tables.append(pipe_rows)
    return tables


def _render_assay_review_contact_sheet(pdf_file, page_numbers, output_path):
    try:
        import fitz
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("PyMuPDF and Pillow are required for assay visual value review") from exc

    rendered_pages = []
    doc = fitz.open(pdf_file)
    try:
        for page_num in page_numbers:
            page = doc.load_page(int(page_num) - 1)
            matrix = fitz.Matrix(ASSAY_VISUAL_VALUE_REVIEW_RENDER_SCALE, ASSAY_VISUAL_VALUE_REVIEW_RENDER_SCALE)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
            if image.width > ASSAY_VISUAL_VALUE_REVIEW_MAX_WIDTH:
                ratio = ASSAY_VISUAL_VALUE_REVIEW_MAX_WIDTH / float(image.width)
                image = image.resize(
                    (ASSAY_VISUAL_VALUE_REVIEW_MAX_WIDTH, max(1, int(image.height * ratio))),
                    Image.Resampling.LANCZOS,
                )
            rendered_pages.append((int(page_num), image))
    finally:
        doc.close()

    if not rendered_pages:
        raise RuntimeError("No pages rendered for assay visual value review")

    label_height = 34
    padding = 16
    width = max(image.width for _, image in rendered_pages) + padding * 2
    height = padding + sum(label_height + image.height + padding for _, image in rendered_pages)
    sheet = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(sheet)
    y = padding
    for page_num, image in rendered_pages:
        draw.text((padding, y), f"PDF page {page_num}", fill='black')
        y += label_height
        sheet.paste(image, (padding, y))
        y += image.height + padding
    sheet.save(output_path)
    return output_path


def _review_assay_values_with_vision(pdf_file, page_numbers, review_payload, assay_dicts):
    if not review_payload or len(page_numbers) > ASSAY_VISUAL_VALUE_REVIEW_MAX_PAGES:
        return {}

    with tempfile.TemporaryDirectory(prefix='biocheminsight_assay_visual_review_') as tmp_dir:
        image_path = os.path.join(tmp_dir, 'assay_pages.png')
        _render_assay_review_contact_sheet(pdf_file, page_numbers, image_path)
        prompt = build_review_assay_values_prompt(assay_dicts, review_payload)
        response_text = call_visual_model(image_path, prompt)
    json_text = extract_json_content(response_text) or str(response_text or '').strip()
    try:
        parsed = json.loads(json_text)
    except Exception:
        print(f"Warning: assay visual value review returned non-JSON response: {str(response_text)[:300]}")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _apply_visual_value_review(pdf_file, page_numbers, assay_dicts, ocr_context):
    if not ASSAY_VISUAL_VALUE_REVIEW_ENABLED:
        return assay_dicts
    review_payload = identify_assay_visual_review_requests(
        ocr_context=ocr_context,
        assay_dicts=assay_dicts,
        parsed_tables=_extract_ocr_tables(ocr_context),
    )
    if not review_payload:
        return assay_dicts

    try:
        visual_report = _review_assay_values_with_vision(pdf_file, page_numbers, review_payload, assay_dicts)
    except Exception as exc:
        print(f"Warning: assay visual value review failed for pages {page_numbers}: {exc}")
        return assay_dicts

    reconciled = reconcile_assay_values_with_visual_report(
        ocr_context=ocr_context,
        assay_dicts=assay_dicts,
        visual_report=visual_report,
    )
    return reconciled if isinstance(reconciled, dict) and reconciled else assay_dicts


def build_alias_resolution_context(chunk, assay_name, raw_key, raw_value):
    lines = [line.strip() for line in str(chunk or '').splitlines() if line.strip()]
    key_text = str(raw_key or '').strip()
    value_text = str(raw_value or '').strip()
    matched_indices = []

    for idx, line in enumerate(lines):
        line_low = line.lower()
        key_hit = False
        value_hit = False
        if key_text:
            if len(key_text) <= 4:
                key_hit = (
                    f'| {key_text} |' in f' {line} '
                    or f'({key_text})' in line
                    or line.startswith(f'{key_text} ')
                    or line.endswith(f' {key_text}')
                )
            else:
                key_hit = key_text.lower() in line_low
        if value_text:
            value_hit = value_text.lower() in line_low
        if key_hit or value_hit:
            matched_indices.append(idx)
            if len(matched_indices) >= 2:
                break

    context_lines = []
    if matched_indices:
        seen = set()
        for idx in matched_indices:
            for offset in (-1, 0, 1):
                line_idx = idx + offset
                if 0 <= line_idx < len(lines) and line_idx not in seen:
                    seen.add(line_idx)
                    context_lines.append(lines[line_idx])
    else:
        context_lines = lines[:8]

    context_text = '\n'.join(context_lines[:8])
    return (
        f"Assay: {assay_name}\n"
        f"Observed raw key: {key_text}\n"
        f"Observed value: {value_text}\n"
        f"Local chunk context:\n{context_text}"
    )


def _build_assay_page_cache_key(pdf_file, assay_page_start, assay_page_end, lang):
    stat = os.stat(pdf_file)
    return (
        os.path.abspath(pdf_file),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(assay_page_start),
        int(assay_page_end),
        str(lang or '').strip().lower(),
    )


def _get_cached_assay_page_contents(cache_key):
    if not ASSAY_PAGE_TEXT_CACHE_ENABLED:
        return None
    cached = _ASSAY_PAGE_CONTENT_CACHE.get(cache_key)
    if cached is None:
        return None
    _ASSAY_PAGE_CONTENT_CACHE.move_to_end(cache_key)
    return list(cached)


def _set_cached_assay_page_contents(cache_key, content_list):
    if not ASSAY_PAGE_TEXT_CACHE_ENABLED:
        return
    _ASSAY_PAGE_CONTENT_CACHE[cache_key] = list(content_list)
    _ASSAY_PAGE_CONTENT_CACHE.move_to_end(cache_key)
    while len(_ASSAY_PAGE_CONTENT_CACHE) > ASSAY_PAGE_TEXT_CACHE_MAX_ENTRIES:
        _ASSAY_PAGE_CONTENT_CACHE.popitem(last=False)


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
            if not isinstance(item, dict):
                continue
            page_markdowns.append(str(item.get('markdown') or '').strip())
        if page_markdowns:
            return page_markdowns
    return []


def _normalize_multi_assay_payload(payload, assay_names):
    normalized = {assay_name: {} for assay_name in assay_names}
    if not isinstance(payload, dict):
        return normalized
    for assay_name in assay_names:
        assay_payload = payload.get(assay_name, {})
        normalized[assay_name] = assay_payload if isinstance(assay_payload, dict) else {}
    return normalized


def load_assay_page_contents(
    pdf_file,
    assay_page_start,
    assay_page_end,
    output_dir,
    lang=DEFAULT_OCR_LANG,
    progress_callback=None,
):
    total_pages = assay_page_end - assay_page_start + 1

    def report_progress(current: int, total: int, message: str) -> None:
        if progress_callback:
            try:
                progress_callback(current, total, message)
            except TypeError:
                progress_callback(message)  # type: ignore[call-arg]

    cache_key = None
    if total_pages <= ASSAY_PAGE_TEXT_CACHE_MAX_PAGES:
        try:
            cache_key = _build_assay_page_cache_key(pdf_file, assay_page_start, assay_page_end, lang)
            cached = _get_cached_assay_page_contents(cache_key)
            if cached is not None:
                report_progress(0, total_pages, f"♻️ Reusing cached page text for pages {assay_page_start}-{assay_page_end}")
                return cached
        except OSError:
            cache_key = None

    if not PADDLEOCR_SERVER_URL:
        raise RuntimeError("PaddleOCR server URL not configured. Set PADDLEOCR_SERVER_URL in constants.py.")

    endpoint = f"{PADDLEOCR_SERVER_URL.rstrip('/')}/v1/pdf-to-markdown"
    report_progress(0, total_pages, f"📖 Calling PaddleOCR service for pages {assay_page_start}-{assay_page_end}")
    try:
        with open(pdf_file, 'rb') as pdf_stream:
            response = requests.post(
                endpoint,
                files={'file': (os.path.basename(pdf_file) or 'document.pdf', pdf_stream, 'application/pdf')},
                data={
                    'page_start': str(assay_page_start),
                    'page_end': str(assay_page_end),
                    'lang': lang,
                    'return_raw': 'false',
                },
                timeout=600,
            )
        response.raise_for_status()
        payload = response.json()
        content_list = _extract_payload_page_markdowns(payload)
        if not content_list:
            raise RuntimeError(
                f"PaddleOCR returned empty markdown for pages {assay_page_start}-{assay_page_end}. "
                "The OCR service did not return usable page content."
            )
        if len(content_list) != total_pages:
            raise RuntimeError(
                f"PaddleOCR page split mismatch for pages {assay_page_start}-{assay_page_end}: "
                f"expected {total_pages}, got {len(content_list)}."
            )
    except (requests.RequestException, OSError, ValueError) as exc:  # pragma: no cover - network dependant
        raise RuntimeError(
            f"PaddleOCR request failed for pages {assay_page_start}-{assay_page_end}. "
            "Unable to obtain page-level OCR output from the service."
        ) from exc

    if cache_key is not None and len(content_list) <= ASSAY_PAGE_TEXT_CACHE_MAX_PAGES:
        _set_cached_assay_page_contents(cache_key, content_list)
    return content_list


def extract_activity_data_multi(
    pdf_file,
    assay_page_start,
    assay_page_end,
    assay_names,
    compound_id_list,
    output_dir,
    pages_per_chunk=3,
    lang=DEFAULT_OCR_LANG,
    progress_callback=None,
):
    assay_names = [str(name).strip() for name in (assay_names or []) if str(name).strip()]
    if not assay_names:
        return {}

    total_pages = assay_page_end - assay_page_start + 1
    assay_dicts: Dict[str, Dict[str, str]] = {assay_name: {} for assay_name in assay_names}

    def report_progress(current: int, total: int, message: str) -> None:
        if progress_callback:
            try:
                progress_callback(current, total, message)
            except TypeError:
                progress_callback(message)  # type: ignore[call-arg]

    report_progress(0, total_pages, f"🧪 Starting multi-assay extraction for pages {assay_page_start}-{assay_page_end} ({total_pages} pages)")

    content_list = load_assay_page_contents(
        pdf_file=pdf_file,
        assay_page_start=assay_page_start,
        assay_page_end=assay_page_end,
        output_dir=output_dir,
        lang=lang,
        progress_callback=progress_callback,
    )

    chunk_count = (len(content_list) + pages_per_chunk - 1) // max(1, pages_per_chunk)
    report_progress(0, total_pages, f"📊 Dividing {total_pages} pages into {chunk_count} shared processing chunks")
    print(f"Total {chunk_count} shared chunks to process for {len(assay_names)} assays.")

    for idx, start in enumerate(range(0, len(content_list), pages_per_chunk), 1):
        chunk = "\n\n".join(content_list[start:start + pages_per_chunk])
        processed_pages = min(total_pages, idx * pages_per_chunk)
        report_progress(processed_pages, total_pages, f"🔍 Analyzing shared chunk {idx} of {chunk_count}")
        print(f"Processing shared chunk {idx}/{chunk_count}...")
        print('Shared chunk content preview:', chunk[:1000])
        chunk_multi_assay_dict = content_to_multi_assay_dict(chunk, assay_names, compound_id_list=compound_id_list)
        normalized_chunk = _normalize_multi_assay_payload(chunk_multi_assay_dict, assay_names)
        chunk_page_numbers = list(range(assay_page_start + start, assay_page_start + min(len(content_list), start + pages_per_chunk)))
        normalized_chunk = _apply_visual_value_review(
            pdf_file=pdf_file,
            page_numbers=chunk_page_numbers,
            assay_dicts=normalized_chunk,
            ocr_context=chunk,
        )

        for assay_name in assay_names:
            chunk_assay_dict = normalized_chunk.get(assay_name, {}) or {}
            if not chunk_assay_dict:
                continue
            context_by_key = {
                raw_key: build_alias_resolution_context(chunk, assay_name, raw_key, raw_value)
                for raw_key, raw_value in chunk_assay_dict.items()
            }
            chunk_assay_dict = remap_assay_dict_to_official_ids(
                chunk_assay_dict,
                compound_id_list,
                resolver_fn=resolve_compound_id_alias,
                context_by_key=context_by_key,
            )
            assay_dicts[assay_name].update(chunk_assay_dict)

    for assay_name, assay_dict in assay_dicts.items():
        assay_name_clean = assay_name.replace(' ', '_').replace('/', '_')
        assay_dir = os.path.join(output_dir, assay_name_clean)
        os.makedirs(assay_dir, exist_ok=True)
        assay_json = os.path.join(assay_dir, 'assay_data.json')
        assay_json_named = os.path.join(assay_dir, f"{assay_name_clean}_assay_data.json")
        write_json_file(assay_json, assay_dict)
        write_json_file(assay_json_named, assay_dict)
        print(f"Saved shared multi-assay result for {assay_name} to {assay_json}")

    report_progress(total_pages, total_pages, "✅ Finished multi-assay extraction")
    return assay_dicts

def extract_activity_data(
    pdf_file,
    assay_page_start,
    assay_page_end,
    assay_name,
    compound_id_list,
    output_dir,
    pages_per_chunk=3,
    lang=DEFAULT_OCR_LANG,
    progress_callback=None,
):
    """
    根据PDF指定页码范围解析数据：
    
    1. 将指定页码范围上传到配置好的 PaddleOCR 服务，并获取 Markdown 结果。
    2. 根据参数 pages_per_chunk，将多个连续页面的 Markdown 内容组合为一个 chunk，
       每个 chunk 内部的内容通过页码信息分隔，保持原有页面结构。
    3. 针对每个 chunk 调用共享的 multi-assay 提取逻辑，并返回当前 assay 的结果。
    4. 最后将合并后的结果保存为 JSON 文件，并返回 assay_dict。
    
    参数:
      pdf_file (str): PDF 文件路径。
      assay_page_start (int): 起始页码。
      assay_page_end (int): 结束页码。
      assay_name (str): 测定名称。
      compound_id_list (list): 化合物ID列表，用于提示。
      output_dir (str): 输出目录。
      pages_per_chunk (int): 每个 chunk 包含的页数。
      lang (str): PDF转换时使用的语言，默认为自动。
      progress_callback (function): 进度回调函数，接收 (current, total, message)。
    """
    multi_result = extract_activity_data_multi(
        pdf_file=pdf_file,
        assay_page_start=assay_page_start,
        assay_page_end=assay_page_end,
        assay_names=[assay_name],
        compound_id_list=compound_id_list,
        output_dir=output_dir,
        pages_per_chunk=pages_per_chunk,
        lang=lang,
        progress_callback=progress_callback,
    )
    return multi_result.get(assay_name, {})
