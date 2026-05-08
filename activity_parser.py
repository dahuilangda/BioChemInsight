import os
import sys
from collections import OrderedDict
from typing import Dict, Optional

import requests

from utils.llm_utils import content_to_multi_assay_dict, resolve_compound_id_alias
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
ASSAY_PAGE_TEXT_CACHE_ENABLED = bool(getattr(constants, 'ASSAY_PAGE_TEXT_CACHE_ENABLED', True))
ASSAY_PAGE_TEXT_CACHE_MAX_ENTRIES = max(1, int(getattr(constants, 'ASSAY_PAGE_TEXT_CACHE_MAX_ENTRIES', 4) or 4))
ASSAY_PAGE_TEXT_CACHE_MAX_PAGES = max(1, int(getattr(constants, 'ASSAY_PAGE_TEXT_CACHE_MAX_PAGES', 64) or 64))


_ASSAY_PAGE_CONTENT_CACHE = OrderedDict()


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
    lang='en',
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
    lang='en',
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
    lang='en',
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
      lang (str): PDF转换时使用的语言，默认为英文。
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
