import os
import sys
import json
import re
import hashlib
import tempfile
from collections import OrderedDict
from html.parser import HTMLParser
from typing import Dict, Optional

import requests

from utils.paddleocr_client import request_pdf_to_markdown
from utils.llm_utils import (
    build_review_assay_values_prompt,
    content_to_dict,
    content_to_multi_assay_dict,
    identify_assay_visual_review_requests,
    parse_review_assay_values_payload,
    reconcile_assay_values_with_visual_report,
    resolve_compound_id_alias,
    route_assays_for_content,
    run_vision_json_task,
)
from utils.model_harness import classify_exception
from utils.file_utils import write_json_file
from utils.compound_id_utils import (
    parse_compound_id_parts,
    remap_assay_dict_to_official_ids,
)

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
ASSAY_VISUAL_VALUE_REVIEW_MAX_ITEMS_PER_CALL = max(
    1,
    int(getattr(constants, 'ASSAY_VISUAL_VALUE_REVIEW_MAX_ITEMS_PER_CALL', 20) or 20),
)
ASSAY_EXTRACTION_LLM_MAX_RETRIES = max(1, int(getattr(constants, 'ASSAY_EXTRACTION_LLM_MAX_RETRIES', 1) or 1))
ASSAY_EXTRACTION_LLM_TIMEOUT_SECONDS = max(30, int(getattr(constants, 'ASSAY_EXTRACTION_LLM_TIMEOUT_SECONDS', 120) or 120))
ASSAY_EXTRACTION_MODE = str(getattr(constants, 'ASSAY_EXTRACTION_MODE', 'per_assay_page') or 'per_assay_page').strip().lower()
ASSAY_EXTRACTION_MAX_PAGE_CANDIDATE_IDS = max(
    16,
    int(getattr(constants, 'ASSAY_EXTRACTION_MAX_PAGE_CANDIDATE_IDS', 96) or 96),
)
ASSAY_EXTRACTION_MAX_MODEL_CONTENT_CHARS = max(
    2000,
    int(getattr(constants, 'ASSAY_EXTRACTION_MAX_MODEL_CONTENT_CHARS', 12000) or 12000),
)
ASSAY_EXTRACTION_CHUNK_HEADER_LINES = max(
    0,
    int(getattr(constants, 'ASSAY_EXTRACTION_CHUNK_HEADER_LINES', 8) or 8),
)
ASSAY_EXTRACTION_MAX_TABLE_ROWS_PER_CHUNK = max(
    0,
    int(getattr(constants, 'ASSAY_EXTRACTION_MAX_TABLE_ROWS_PER_CHUNK', 12) or 12),
)
ASSAY_EXTRACTION_DOCUMENT_CONTEXT_CHARS = max(
    0,
    int(getattr(constants, 'ASSAY_EXTRACTION_DOCUMENT_CONTEXT_CHARS', 3000) or 3000),
)


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


def _build_assay_model_content(page_content):
    text = str(page_content or '')
    tables = _extract_ocr_tables(text)
    if not tables:
        return text
    compact_text = re.sub(
        r'<table\b.*?</table>',
        '[OCR table parsed below]',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    compact_text = re.sub(r'\n{3,}', '\n\n', compact_text).strip()
    table_lines = []
    for table_index, table in enumerate(tables, 1):
        table_lines.append(f"Parsed OCR table {table_index}:")
        for row_index, row in enumerate(table, 1):
            cells = [re.sub(r'\s+', ' ', str(cell or '')).strip() for cell in row]
            table_lines.append(f"row {row_index}: " + " | ".join(cells))
    return (
        f"{compact_text}\n\n"
        "Parsed OCR table grids for the same page/chunk. Row and column order are preserved; "
        "use these grids as the authoritative table structure when the Markdown/HTML table is noisy:\n"
        "<PARSED_OCR_TABLE_GRIDS>\n"
        f"{chr(10).join(table_lines)}\n"
        "</PARSED_OCR_TABLE_GRIDS>"
    )


def _build_page_candidate_compound_ids(page_content, compound_id_list, max_candidates=96):
    official_ids = [str(item).strip() for item in (compound_id_list or []) if str(item).strip()]
    if not official_ids:
        return []

    page_text = str(page_content or '')
    if not page_text.strip():
        return official_ids[:max_candidates]

    matched = []
    id_prefix = r'(?:Example|Ex\.?|No\.?|Compound|Formula|实施例|化合物|编号)'

    for official_id in official_ids:
        if len(matched) >= max_candidates:
            break
        parts = parse_compound_id_parts(official_id)
        if not parts:
            continue
        core = str(parts.get('core') or '').strip()
        if not core:
            continue
        escaped_core = re.escape(core)
        pattern = re.compile(
            rf'(?<![0-9A-Za-z])(?:{id_prefix}\s*)?\(?\s*{escaped_core}\s*\)?(?![0-9A-Za-z])',
            flags=re.IGNORECASE,
        )
        if pattern.search(page_text) and official_id not in matched:
            matched.append(official_id)

    if len(matched) >= 2:
        return matched[:max_candidates]
    return official_ids[:max_candidates]


def _has_any_assay_records(assay_payload):
    return any(bool(values) for values in (assay_payload or {}).values())


def _split_long_assay_line(line, max_chars):
    line = str(line or '')
    if len(line) <= max_chars:
        return [line]

    if '|' in line:
        return [line]

    return _split_long_assay_text(line, max_chars)


def _split_long_assay_text(text, max_chars):
    text = str(text or '')
    if len(text) <= max_chars:
        return [text] if text else []

    sentence_units = [
        unit
        for unit in re.split(r'(?<=[。！？.!?])\s+', text)
        if unit
    ]
    if len(sentence_units) <= 1:
        sentence_units = [
            unit
            for unit in re.split(r'(?<=[;；])\s*', text)
            if unit
        ]
    if len(sentence_units) <= 1:
        sentence_units = [part for part in re.split(r'(\s+)', text) if part]

    chunks = []
    current = ''
    for unit in sentence_units:
        if len(unit) > max_chars:
            if current:
                chunks.append(current.rstrip())
                current = ''
            chunks.extend(_split_oversized_text_unit(unit, max_chars))
            continue
        separator = '' if not current or unit.isspace() or current.endswith((' ', '\n')) else ' '
        candidate = f"{current}{separator}{unit}" if current else unit
        if current and len(candidate) > max_chars:
            chunks.append(current.rstrip())
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current.rstrip())
    return [chunk for chunk in chunks if chunk] or [text]


def _split_oversized_text_unit(text, max_chars):
    text = str(text or '')
    if len(text) <= max_chars:
        return [text] if text else []
    parts = [part for part in re.split(r'(\s+)', text) if part]

    if len(parts) <= 1:
        return [text[index:index + max_chars] for index in range(0, len(text), max_chars)]

    chunks = []
    current = ''
    for part in parts:
        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ''
            chunks.extend(part[index:index + max_chars] for index in range(0, len(part), max_chars))
            continue
        if current and len(current) + len(part) > max_chars:
            chunks.append(current)
            current = part
        else:
            current += part
    if current:
        chunks.append(current)
    return chunks or [text]


def _is_assay_table_line(line):
    stripped = str(line or '').strip()
    if not stripped:
        return False
    return stripped.count('|') >= 2 and not stripped.startswith(('#', '* '))


def _pipe_table_cells(line):
    stripped = str(line or '').strip()
    if stripped.startswith('|') and stripped.endswith('|'):
        stripped = stripped[1:-1]
    return [cell.strip() for cell in stripped.split('|')]


def _is_pipe_table_delimiter(line):
    cells = _pipe_table_cells(line)
    return bool(cells) and all(re.fullmatch(r':?-{2,}:?', cell or '') for cell in cells)


def _is_html_table_start(line):
    return bool(re.search(r'<table\b', str(line or ''), flags=re.IGNORECASE))


def _is_html_table_end(line):
    return bool(re.search(r'</table\s*>', str(line or ''), flags=re.IGNORECASE))


def _is_probable_table_header(line):
    text = str(line or '').lower()
    return any(token in text for token in (
        'example',
        'compound',
        'formula',
        'id',
        'no.',
        'tr-fret',
        'ec50',
        'ic50',
        'assay',
        'activity',
        'inhibition',
        '实施例',
        '化合物',
        '编号',
    ))


def _is_probable_table_data_row(line):
    cells = [cell for cell in _pipe_table_cells(line) if cell]
    if len(cells) < 2:
        return False
    first_cell = cells[0]
    if re.fullmatch(r'(?:[A-Za-z]{0,4}\s*)?\d+[A-Za-z]?(?:[-–]\d+[A-Za-z]?)?', first_cell):
        return True
    return any(
        re.search(r'(?:[<>~=≤≥]?\s*)?\d+(?:\.\d+)?\s*(?:nM|uM|µM|mM|%|ng/mL|ug/mL|µg/mL)\b', cell, flags=re.IGNORECASE)
        for cell in cells[1:]
    )


def _is_valid_pipe_table_block(block_lines):
    table_lines = [line for line in block_lines if _is_assay_table_line(line)]
    if len(table_lines) < 2:
        return False
    cell_counts = [len(_pipe_table_cells(line)) for line in table_lines if not _is_pipe_table_delimiter(line)]
    if not cell_counts or max(cell_counts) < 2:
        return False
    common_count = max(set(cell_counts), key=cell_counts.count)
    consistent = sum(1 for count in cell_counts if count == common_count) >= max(2, len(cell_counts) // 2)
    has_delimiter = any(_is_pipe_table_delimiter(line) for line in table_lines[:3])
    if has_delimiter:
        return consistent
    if not consistent or len(table_lines) < 3:
        return False
    data_row_count = sum(1 for line in table_lines if _is_probable_table_data_row(line))
    return _is_probable_table_header(table_lines[0]) or data_row_count >= 2


def _append_text_blocks(blocks, text_lines):
    current = []
    for line in text_lines:
        if str(line or '').strip():
            current.append(line)
            continue
        if current:
            blocks.append({'type': 'text', 'lines': current})
            current = []
    if current:
        blocks.append({'type': 'text', 'lines': current})


def _build_assay_content_blocks(lines):
    blocks = []
    text_buffer = []
    index = 0
    while index < len(lines):
        line = lines[index]

        if _is_html_table_start(line):
            _append_text_blocks(blocks, text_buffer)
            text_buffer = []
            table_lines = [line]
            index += 1
            while index < len(lines):
                table_lines.append(lines[index])
                if _is_html_table_end(lines[index]):
                    index += 1
                    break
                index += 1
            blocks.append({'type': 'html_table', 'lines': table_lines})
            continue

        if _is_assay_table_line(line):
            candidate = []
            while index < len(lines) and _is_assay_table_line(lines[index]):
                candidate.append(lines[index])
                index += 1
            if _is_valid_pipe_table_block(candidate):
                _append_text_blocks(blocks, text_buffer)
                text_buffer = []
                blocks.append({'type': 'pipe_table', 'lines': candidate})
            else:
                text_buffer.extend(candidate)
            continue

        text_buffer.append(line)
        index += 1

    _append_text_blocks(blocks, text_buffer)
    return blocks


def _split_text_block_for_model(lines, max_chars):
    text = '\n'.join(lines).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = []
    current_len = 0
    for part in _split_long_assay_text(text, max_chars):
        for line in str(part).splitlines() or [str(part)]:
            line_len = len(line) + 1
            if current and current_len + line_len > max_chars:
                chunks.append('\n'.join(current).strip())
                current = []
                current_len = 0
            current.append(line)
            current_len += line_len
    if current:
        chunks.append('\n'.join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _pipe_table_header_lines(table_lines):
    delimiter_index = next(
        (idx for idx, line in enumerate(table_lines[:3]) if _is_pipe_table_delimiter(line)),
        None,
    )
    if delimiter_index is not None:
        return table_lines[: delimiter_index + 1]
    if table_lines and _is_probable_table_header(table_lines[0]):
        return [table_lines[0]]
    return []


def _split_pipe_table_block_for_model(lines, max_chars):
    table_lines = [str(line) for line in lines if str(line or '').strip()]
    if not table_lines:
        return []
    table_text = '\n'.join(table_lines).strip()
    row_limit = ASSAY_EXTRACTION_MAX_TABLE_ROWS_PER_CHUNK
    if len(table_text) <= max_chars and (row_limit <= 0 or len(table_lines) <= row_limit):
        return [table_text]

    header_lines = _pipe_table_header_lines(table_lines)
    rows = table_lines[len(header_lines):] if header_lines else table_lines
    chunks = []
    current = list(header_lines)
    current_len = sum(len(line) + 1 for line in current)
    min_payload_rows = 1 if header_lines else 0

    for row in rows:
        row_len = len(row) + 1
        if (
            len(current) > len(header_lines) + min_payload_rows - 1
            and (
                current_len + row_len > max_chars
                or (row_limit > 0 and len(current) - len(header_lines) >= row_limit)
            )
        ):
            chunks.append('\n'.join(current).strip())
            current = list(header_lines)
            current_len = sum(len(line) + 1 for line in current)
        current.append(row)
        current_len += row_len

    if len(current) > len(header_lines) or current:
        chunks.append('\n'.join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _split_html_table_block_for_model(lines, max_chars):
    table_text = '\n'.join(str(line) for line in lines if str(line or '').strip()).strip()
    if not table_text:
        return []

    row_matches = list(re.finditer(r'<tr\b.*?</tr\s*>', table_text, flags=re.IGNORECASE | re.DOTALL))
    if not row_matches:
        return [table_text]
    row_limit = ASSAY_EXTRACTION_MAX_TABLE_ROWS_PER_CHUNK
    if len(table_text) <= max_chars and (row_limit <= 0 or len(row_matches) <= row_limit):
        return [table_text]

    prefix = table_text[: row_matches[0].start()]
    suffix = table_text[row_matches[-1].end():]
    rows = [match.group(0) for match in row_matches]
    header_index = next(
        (
            idx
            for idx, row in enumerate(rows)
            if re.search(r'<th\b', row, flags=re.IGNORECASE) or _is_probable_table_header(re.sub(r'<[^>]+>', ' ', row))
        ),
        None,
    )
    header_rows = [rows[header_index]] if header_index is not None else []

    chunks = []
    current_rows = list(header_rows)
    current_len = len(prefix) + len(suffix) + sum(len(row) for row in current_rows)
    data_rows = [row for idx, row in enumerate(rows) if idx != header_index]

    for row in data_rows:
        row_len = len(row)
        if (
            len(current_rows) > len(header_rows)
            and (
                current_len + row_len > max_chars
                or (row_limit > 0 and len(current_rows) - len(header_rows) >= row_limit)
            )
        ):
            chunks.append(f"{prefix}{''.join(current_rows)}{suffix}".strip())
            current_rows = list(header_rows)
            current_len = len(prefix) + len(suffix) + sum(len(item) for item in current_rows)
        current_rows.append(row)
        current_len += row_len

    if len(current_rows) > len(header_rows) or current_rows:
        chunks.append(f"{prefix}{''.join(current_rows)}{suffix}".strip())
    return [chunk for chunk in chunks if chunk]


def _build_assay_chunk_header(lines, max_chars):
    if ASSAY_EXTRACTION_CHUNK_HEADER_LINES <= 0:
        return ''
    budget = min(600, max(120, max_chars // 3))
    selected = []
    current_len = 0
    for line in lines[:ASSAY_EXTRACTION_CHUNK_HEADER_LINES]:
        stripped = str(line or '').strip()
        if not stripped:
            continue
        if _is_html_table_start(stripped) or _is_assay_table_line(stripped):
            break
        remaining = budget - current_len
        if remaining <= 0:
            break
        if len(stripped) > remaining:
            selected.append(stripped[:remaining].rstrip())
            break
        selected.append(stripped)
        current_len += len(stripped) + 1
    return '\n'.join(selected).strip()


def _strip_assay_tables_for_context(page_content):
    text = str(page_content or '')
    if not text.strip():
        return ''
    without_html_tables = re.sub(
        r'<table\b.*?</table>',
        ' [OCR table omitted] ',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    kept_lines = []
    in_pipe_table = False
    for line in without_html_tables.splitlines():
        stripped = str(line or '').strip()
        if not stripped:
            if kept_lines and kept_lines[-1]:
                kept_lines.append('')
            in_pipe_table = False
            continue
        if _is_assay_table_line(stripped):
            if not in_pipe_table:
                kept_lines.append('[OCR table omitted]')
            in_pipe_table = True
            continue
        in_pipe_table = False
        kept_lines.append(stripped)
    context = re.sub(r'\n{3,}', '\n\n', '\n'.join(kept_lines)).strip()
    context = re.sub(r'(?:\[OCR table omitted\]\s*){2,}', '[OCR table omitted]\n', context)
    return context.strip()


def _build_assay_document_context(content_list, max_chars=ASSAY_EXTRACTION_DOCUMENT_CONTEXT_CHARS):
    if max_chars <= 0:
        return ''
    parts = []
    for page_offset, page_content in enumerate(content_list or [], 1):
        context = _strip_assay_tables_for_context(page_content)
        if context:
            parts.append(f"Selected assay page {page_offset} context:\n{context}")
    text = '\n\n'.join(parts).strip()
    if len(text) <= max_chars:
        return text
    head_budget = max_chars // 2
    tail_budget = max_chars - head_budget - 40
    return f"{text[:head_budget].rstrip()}\n\n[...context truncated...]\n\n{text[-tail_budget:].lstrip()}".strip()


def _attach_assay_document_context(chunk_content, document_context, max_chars=ASSAY_EXTRACTION_MAX_MODEL_CONTENT_CHARS):
    chunk_text = str(chunk_content or '')
    context = str(document_context or '').strip()
    if not context:
        return chunk_text
    block_prefix = (
        "ASSAY DOCUMENT CONTEXT from the selected assay pages. Use this for target/method/"
        "endpoint disambiguation, especially for continuation tables; extract records only "
        "from CURRENT PAGE/TABLE CHUNK below:\n"
    )
    budget = max(0, max_chars - len(chunk_text) - len(block_prefix) - 80)
    if budget <= 120:
        return chunk_text
    context_for_chunk = context if len(context) <= budget else context[:budget].rstrip()
    return (
        f"{block_prefix}<ASSAY_DOCUMENT_CONTEXT>\n"
        f"{context_for_chunk}\n"
        "</ASSAY_DOCUMENT_CONTEXT>\n\n"
        "<CURRENT_PAGE_TABLE_CHUNK>\n"
        f"{chunk_text}\n"
        "</CURRENT_PAGE_TABLE_CHUNK>"
    )


def _split_assay_content_for_model(page_content, max_chars=ASSAY_EXTRACTION_MAX_MODEL_CONTENT_CHARS):
    text = str(page_content or '')
    if not text.strip():
        return [text]

    raw_lines = text.splitlines()
    lines = [str(line) for line in raw_lines]
    header = _build_assay_chunk_header(lines, max_chars)
    chunks = []
    current = []
    current_len = 0

    block_chunks = []
    for block in _build_assay_content_blocks(lines):
        if block.get('type') == 'pipe_table':
            block_chunks.extend(_split_pipe_table_block_for_model(block.get('lines') or [], max_chars))
        elif block.get('type') == 'html_table':
            block_chunks.extend(_split_html_table_block_for_model(block.get('lines') or [], max_chars))
        else:
            block_chunks.extend(_split_text_block_for_model(block.get('lines') or [], max_chars))

    for chunk_text in block_chunks:
        is_table_chunk = bool(
            re.search(r'<table\b', chunk_text, flags=re.IGNORECASE)
            or _is_valid_pipe_table_block(str(chunk_text or '').splitlines())
        )
        if is_table_chunk:
            if current:
                chunks.append('\n'.join(current).strip())
                current = []
                current_len = 0
            chunks.append(chunk_text.strip())
            continue
        chunk_len = len(chunk_text) + 2
        if current and current_len + chunk_len > max_chars:
            chunks.append('\n'.join(current).strip())
            current = []
            current_len = 0
        current.append(chunk_text)
        current_len += chunk_len

    if current:
        chunks.append('\n'.join(current).strip())

    normalized_chunks = []
    for index, chunk in enumerate(chunks):
        if not chunk:
            continue
        if index > 0 and header and header not in chunk[: len(header) + 20]:
            chunk_with_header = f"{header}\n\n{chunk}"
            if len(chunk_with_header) <= max_chars:
                chunk = chunk_with_header
        normalized_chunks.append(chunk)
    return normalized_chunks or [text]


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


def _chunk_visual_review_payload(review_payload, max_items=ASSAY_VISUAL_VALUE_REVIEW_MAX_ITEMS_PER_CALL):
    chunks = []
    current = {}
    current_count = 0
    for assay_name, items in (review_payload or {}).items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if current_count >= max_items and current:
                chunks.append(current)
                current = {}
                current_count = 0
            current.setdefault(assay_name, []).append(item)
            current_count += 1
    if current:
        chunks.append(current)
    return chunks


def _merge_visual_review_reports(reports):
    merged = {'corrections': []}
    for report in reports:
        if not isinstance(report, dict):
            continue
        corrections = report.get('corrections')
        if isinstance(corrections, list):
            merged['corrections'].extend(item for item in corrections if isinstance(item, dict))
    return merged


def _review_assay_values_with_vision(pdf_file, page_numbers, review_payload, assay_dicts, audit_path=None):
    if not review_payload:
        return {}
    page_numbers = [int(page) for page in (page_numbers or [])]
    if len(page_numbers) > ASSAY_VISUAL_VALUE_REVIEW_MAX_PAGES:
        reports = []
        for start in range(0, len(page_numbers), ASSAY_VISUAL_VALUE_REVIEW_MAX_PAGES):
            page_batch = page_numbers[start:start + ASSAY_VISUAL_VALUE_REVIEW_MAX_PAGES]
            reports.append(_review_assay_values_with_vision(
                pdf_file,
                page_batch,
                review_payload,
                assay_dicts,
                audit_path=audit_path,
            ))
        return _merge_visual_review_reports(reports)

    with tempfile.TemporaryDirectory(prefix='biocheminsight_assay_visual_review_') as tmp_dir:
        image_path = os.path.join(tmp_dir, 'assay_pages.png')
        _render_assay_review_contact_sheet(pdf_file, page_numbers, image_path)
        review_chunks = _chunk_visual_review_payload(review_payload)
        reports = []
        for chunk_index, review_chunk in enumerate(review_chunks, 1):
            prompt = build_review_assay_values_prompt(assay_dicts, review_chunk)
            reports.append(run_vision_json_task(
                task_name='review_assay_values',
                image_file=image_path,
                prompt=prompt,
                parser=parse_review_assay_values_payload,
                audit_path=audit_path,
                metadata={
                    'pages': [int(page) for page in page_numbers],
                    'assay_count': len(assay_dicts or {}),
                    'review_item_count': sum(len(items or []) for items in (review_chunk or {}).values()),
                    'review_chunk_index': chunk_index,
                    'review_chunk_count': len(review_chunks),
                },
            ))
        return _merge_visual_review_reports(reports)


def _downgrade_low_confidence_visual_replacements(visual_report):
    if not isinstance(visual_report, dict):
        return {}
    corrections = visual_report.get('corrections')
    if not isinstance(corrections, list):
        return visual_report
    normalized_corrections = []
    for correction in corrections:
        if not isinstance(correction, dict):
            continue
        next_correction = dict(correction)
        action = str(next_correction.get('action') or '').strip().lower()
        confidence = str(next_correction.get('confidence') or 'medium').strip().lower()
        if action == 'replace' and confidence == 'low':
            next_correction['action'] = 'uncertain'
            evidence = str(next_correction.get('evidence') or '').strip()
            next_correction['evidence'] = (
                f"{evidence}; low-confidence replacement downgraded"
                if evidence
                else "low-confidence replacement downgraded"
            )
        normalized_corrections.append(next_correction)
    return {**visual_report, 'corrections': normalized_corrections}


def _assay_value_field(value, field='value'):
    if isinstance(value, dict):
        if field in value:
            return '' if value.get(field) is None else str(value.get(field)).strip()
        upper = field.upper()
        if upper in value:
            return '' if value.get(upper) is None else str(value.get(upper)).strip()
        if field == 'value':
            for key in ('VALUE', 'assay_value', 'ASSAY_VALUE'):
                if key in value:
                    return '' if value.get(key) is None else str(value.get(key)).strip()
        return ''
    if field == 'value':
        return '' if value is None else str(value).strip()
    return ''


def _normalize_visual_review_payload(review_payload, assay_dicts):
    if not isinstance(review_payload, dict):
        return {}
    normalized = {}
    for assay_name, items in review_payload.items():
        if not isinstance(items, list):
            continue
        assay_values = assay_dicts.get(assay_name, {}) if isinstance(assay_dicts, dict) else {}
        normalized_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            compound_id = str(item.get('compound_id') or '').strip()
            current = assay_values.get(compound_id) if isinstance(assay_values, dict) else None
            next_item = dict(item)
            if current is not None:
                next_item['ocr_value'] = _assay_value_field(current, 'value')
                next_item['unit'] = _assay_value_field(current, 'unit')
                next_item['method'] = _assay_value_field(current, 'method')
                next_item['description'] = _assay_value_field(current, 'description')
            normalized_items.append(next_item)
        if normalized_items:
            normalized[assay_name] = normalized_items
    return normalized


def _ensure_rich_assay_value(value, assay_name=''):
    if isinstance(value, dict):
        raw_match = value.get('assay_match') or value.get('ASSAY_MATCH') or {}
        if not isinstance(raw_match, dict):
            raw_match = {}
        return {
            'value': _assay_value_field(value, 'value'),
            'unit': _assay_value_field(value, 'unit'),
            'method': _assay_value_field(value, 'method') or str(assay_name or '').strip(),
            'description': _assay_value_field(value, 'description'),
            'confidence': _assay_value_field(value, 'confidence'),
            'reason': _assay_value_field(value, 'reason'),
            'assay_match': {
                'target': str(raw_match.get('target') or raw_match.get('TARGET') or '').strip(),
                'candidate': str(raw_match.get('candidate') or raw_match.get('CANDIDATE') or '').strip(),
                'compatible': raw_match.get('compatible') if isinstance(raw_match.get('compatible'), bool) else raw_match.get('COMPATIBLE'),
                'best_requested_assay': str(
                    raw_match.get('best_requested_assay') or raw_match.get('BEST_REQUESTED_ASSAY') or ''
                ).strip(),
                'reason': str(raw_match.get('reason') or raw_match.get('REASON') or '').strip(),
            },
        }
    return {
        'value': _assay_value_field(value, 'value'),
        'unit': '',
        'method': str(assay_name or '').strip(),
        'description': '',
        'confidence': '',
        'reason': '',
        'assay_match': {},
    }


def _apply_high_confidence_visual_corrections(assay_dicts, visual_report):
    if not isinstance(assay_dicts, dict) or not isinstance(visual_report, dict):
        return assay_dicts
    corrections = visual_report.get('corrections')
    if not isinstance(corrections, list):
        return assay_dicts
    merged = {
        assay_name: {
            compound_id: _ensure_rich_assay_value(value, assay_name)
            for compound_id, value in (assay_values or {}).items()
        }
        for assay_name, assay_values in assay_dicts.items()
        if isinstance(assay_values, dict)
    }
    for correction in corrections:
        if not isinstance(correction, dict):
            continue
        if str(correction.get('action') or '').strip().lower() != 'replace':
            continue
        if str(correction.get('confidence') or '').strip().lower() != 'high':
            continue
        assay_name = str(correction.get('assay_name') or '').strip()
        compound_id = str(correction.get('compound_id') or '').strip()
        visual_value = str(correction.get('visual_value') or '').strip()
        if not assay_name or not compound_id or not visual_value:
            continue
        if assay_name not in merged or compound_id not in merged[assay_name]:
            continue
        current = dict(merged[assay_name][compound_id])
        current['value'] = visual_value
        if correction.get('unit') is not None:
            current['unit'] = str(correction.get('unit') or '').strip()
        if correction.get('description') is not None:
            current['description'] = str(correction.get('description') or '').strip()
        current['confidence'] = 'high'
        current['reason'] = str(correction.get('evidence') or current.get('reason') or '').strip()
        merged[assay_name][compound_id] = current
    return merged


def _apply_visual_value_review(pdf_file, page_numbers, assay_dicts, ocr_context, audit_path=None):
    if not ASSAY_VISUAL_VALUE_REVIEW_ENABLED:
        return assay_dicts
    try:
        review_payload = identify_assay_visual_review_requests(
            ocr_context=ocr_context,
            assay_dicts=assay_dicts,
            parsed_tables=_extract_ocr_tables(ocr_context),
            audit_path=audit_path,
            metadata={'scope': 'assay_visual_review_planning', 'pages': [int(page) for page in page_numbers]},
        )
    except Exception as exc:
        print(f"Warning: assay visual review planning failed for pages {page_numbers}: {exc}")
        return assay_dicts
    if not review_payload:
        return assay_dicts
    review_payload = _normalize_visual_review_payload(review_payload, assay_dicts)
    if not review_payload:
        return assay_dicts

    try:
        visual_report = _review_assay_values_with_vision(pdf_file, page_numbers, review_payload, assay_dicts, audit_path=audit_path)
    except Exception as exc:
        print(f"Warning: assay visual value review failed for pages {page_numbers}: {exc}")
        return assay_dicts

    visual_report = _downgrade_low_confidence_visual_replacements(visual_report)
    fallback = _apply_high_confidence_visual_corrections(assay_dicts, visual_report)
    try:
        reconciled = reconcile_assay_values_with_visual_report(
            ocr_context=ocr_context,
            assay_dicts=fallback,
            visual_report=visual_report,
            audit_path=audit_path,
            metadata={'scope': 'assay_visual_reconciliation', 'pages': [int(page) for page in page_numbers]},
        )
    except Exception as exc:
        print(f"Warning: assay visual reconciliation failed for pages {page_numbers}: {exc}")
        return fallback
    return reconciled if isinstance(reconciled, dict) and reconciled else fallback


def build_alias_resolution_context(chunk, assay_name, raw_key, raw_value):
    lines = [line.strip() for line in str(chunk or '').splitlines() if line.strip()]
    key_text = str(raw_key or '').strip()
    value_text = _assay_value_field(raw_value, 'value')
    unit_text = _assay_value_field(raw_value, 'unit')
    method_text = _assay_value_field(raw_value, 'method')
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
        f"Observed unit: {unit_text}\n"
        f"Observed method: {method_text}\n"
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
        if not isinstance(assay_payload, dict):
            normalized[assay_name] = {}
            continue
        assay_values = {}
        for raw_key, raw_value in assay_payload.items():
            compound_id = str(raw_key or '').strip()
            if isinstance(raw_value, dict):
                value = raw_value.get('value') if 'value' in raw_value else raw_value.get('VALUE')
                assay_value = '' if value is None else str(value).strip()
                if compound_id and assay_value:
                    raw_match = raw_value.get('assay_match') or raw_value.get('ASSAY_MATCH') or {}
                    if not isinstance(raw_match, dict):
                        raw_match = {}
                    assay_values[compound_id] = {
                        'value': assay_value,
                        'unit': str(raw_value.get('unit') or raw_value.get('UNIT') or '').strip(),
                        'method': str(raw_value.get('method') or raw_value.get('METHOD') or assay_name).strip(),
                        'description': str(raw_value.get('description') or raw_value.get('DESCRIPTION') or '').strip(),
                        'confidence': str(raw_value.get('confidence') or raw_value.get('CONFIDENCE') or '').strip(),
                        'reason': str(raw_value.get('reason') or raw_value.get('REASON') or '').strip(),
                        'assay_match': {
                            'target': str(raw_match.get('target') or raw_match.get('TARGET') or '').strip(),
                            'candidate': str(raw_match.get('candidate') or raw_match.get('CANDIDATE') or '').strip(),
                            'compatible': raw_match.get('compatible') if isinstance(raw_match.get('compatible'), bool) else raw_match.get('COMPATIBLE'),
                            'best_requested_assay': str(
                                raw_match.get('best_requested_assay') or raw_match.get('BEST_REQUESTED_ASSAY') or ''
                            ).strip(),
                            'reason': str(raw_match.get('reason') or raw_match.get('REASON') or '').strip(),
                        },
                    }
                continue
            assay_value = '' if raw_value is None else str(raw_value).strip()
            if compound_id and assay_value:
                assay_values[compound_id] = assay_value
        normalized[assay_name] = assay_values
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

    class _PaddleOCRBatchError(RuntimeError):
        pass

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

    document_key = _build_ocr_document_key(pdf_file)
    failed_pages = []

    def fetch_page_markdowns(page_numbers):
        page_numbers = [int(page) for page in page_numbers]
        page_start = page_numbers[0]
        page_end = page_numbers[-1]
        report_progress(0, total_pages, f"📖 Calling PaddleOCR service for pages {page_start}-{page_end}")
        try:
            payload = request_pdf_to_markdown(
                pdf_file,
                page_start,
                page_end,
                lang,
                False,
                PADDLEOCR_SERVER_URL,
                document_key=document_key,
                page_number_offset=0,
                timeout_seconds=600,
            )
            content_list = _extract_payload_page_markdowns(payload)
            if not content_list:
                raise _PaddleOCRBatchError(
                    f"PaddleOCR returned empty markdown for pages {page_start}-{page_end}."
                )
            if len(content_list) != len(page_numbers):
                raise _PaddleOCRBatchError(
                    f"PaddleOCR page split mismatch for pages {page_start}-{page_end}: "
                    f"expected {len(page_numbers)}, got {len(content_list)}."
                )
            return content_list
        except (requests.RequestException, ValueError, _PaddleOCRBatchError) as exc:  # pragma: no cover - network dependant
            if len(page_numbers) > 1:
                midpoint = max(1, len(page_numbers) // 2)
                left_pages = page_numbers[:midpoint]
                right_pages = page_numbers[midpoint:]
                print(
                    f"Warning: PaddleOCR failed for pages {page_start}-{page_end}; "
                    f"retrying as {left_pages[0]}-{left_pages[-1]} and {right_pages[0]}-{right_pages[-1]}."
                )
                left_content = fetch_page_markdowns(left_pages)
                right_content = fetch_page_markdowns(right_pages)
                return left_content + right_content
            if len(page_numbers) == 1:
                failed_pages.append(page_start)
                print(
                    f"Warning: PaddleOCR failed for page {page_start}; continuing with blank markdown."
                )
                return [""]
            failed_pages.append(page_start)
            print(
                f"Warning: PaddleOCR failed for page {page_start}; continuing with blank markdown."
            )
            return [""]

    content_list = fetch_page_markdowns(list(range(assay_page_start, assay_page_end + 1)))

    if cache_key is not None and not failed_pages and len(content_list) <= ASSAY_PAGE_TEXT_CACHE_MAX_PAGES:
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
    if ASSAY_EXTRACTION_MODE in {'per_assay_page', 'page_assay', 'single_assay_page'}:
        report_progress(0, total_pages, f"📊 Processing {total_pages} pages with assay-scoped model calls")
        print(f"Processing {len(content_list)} pages with assay-scoped model calls for {len(assay_names)} assays.")
    else:
        report_progress(0, total_pages, f"📊 Dividing {total_pages} pages into {chunk_count} shared processing chunks")
        print(f"Total {chunk_count} shared chunks to process for {len(assay_names)} assays.")
    extraction_warnings = []
    model_audit_path = os.path.join(output_dir, 'model_calls.jsonl')

    if ASSAY_EXTRACTION_MODE in {'per_assay_page', 'page_assay', 'single_assay_page'}:
        page_content_chunks = [
            _split_assay_content_for_model(page_content)
            for page_content in content_list
        ]
        document_context = _build_assay_document_context(content_list)
        total_model_calls = max(1, sum(len(chunks) for chunks in page_content_chunks) * len(assay_names))
        completed_model_calls = 0
        for page_offset, page_content in enumerate(content_list):
            page_number = assay_page_start + page_offset
            chunks_for_page = page_content_chunks[page_offset]
            page_results = {assay_name: {} for assay_name in assay_names}
            for chunk_index, chunk_content in enumerate(chunks_for_page, 1):
                contextual_chunk_content = _attach_assay_document_context(chunk_content, document_context)
                model_chunk_content = _build_assay_model_content(contextual_chunk_content)
                candidate_ids = _build_page_candidate_compound_ids(
                    chunk_content,
                    compound_id_list,
                    max_candidates=ASSAY_EXTRACTION_MAX_PAGE_CANDIDATE_IDS,
                )
                chunk_label = (
                    f" chunk {chunk_index}/{len(chunks_for_page)}"
                    if len(chunks_for_page) > 1
                    else ""
                )
                assay_route = {assay_name: True for assay_name in assay_names}
                try:
                    assay_route = route_assays_for_content(
                        model_chunk_content,
                        assay_names,
                        retry=ASSAY_EXTRACTION_LLM_MAX_RETRIES,
                        timeout_seconds=ASSAY_EXTRACTION_LLM_TIMEOUT_SECONDS,
                        audit_path=model_audit_path,
                        metadata={
                            'scope': 'page_assay_route',
                            'page': page_number,
                            'chunk_index': chunk_index,
                            'chunk_count': len(chunks_for_page),
                            'candidate_id_count': len(candidate_ids or compound_id_list or []),
                            'chunk_chars': len(str(chunk_content or '')),
                            'model_chunk_chars': len(str(model_chunk_content or '')),
                        },
                    )
                except Exception as exc:
                    print(
                        f"Warning: assay routing failed for page {page_number}"
                        f"{chunk_label}; skipping this chunk: {exc}"
                    )
                    extraction_warnings.append({
                        'scope': 'page_assay_route',
                        'page': page_number,
                        'chunk_index': chunk_index,
                        'chunk_count': len(chunks_for_page),
                        'error_type': classify_exception(exc),
                        'error': str(exc),
                    })
                    completed_model_calls += len(assay_names)
                    continue
                for assay_name in assay_names:
                    if assay_route.get(assay_name) is False:
                        completed_model_calls += 1
                        report_progress(
                            min(total_pages, page_offset + 1),
                            total_pages,
                            f"↷ Skipping {assay_name} on page {page_number}{chunk_label} by assay routing ({completed_model_calls}/{total_model_calls})",
                        )
                        continue
                    completed_model_calls += 1
                    report_progress(
                        min(total_pages, page_offset + 1),
                        total_pages,
                        f"🔍 Extracting {assay_name} from page {page_number}{chunk_label} ({completed_model_calls}/{total_model_calls})",
                    )
                    try:
                        assay_result = content_to_dict(
                            model_chunk_content,
                            assay_name,
                            compound_id_list=candidate_ids or compound_id_list,
                            assay_context_names=assay_names,
                            retry=ASSAY_EXTRACTION_LLM_MAX_RETRIES,
                            timeout_seconds=ASSAY_EXTRACTION_LLM_TIMEOUT_SECONDS,
                            audit_path=model_audit_path,
                            metadata={
                                'scope': 'page_assay',
                                'page': page_number,
                                'chunk_index': chunk_index,
                                'chunk_count': len(chunks_for_page),
                                'candidate_id_count': len(candidate_ids or compound_id_list or []),
                                'chunk_chars': len(str(chunk_content or '')),
                                'model_chunk_chars': len(str(model_chunk_content or '')),
                            },
                        )
                        normalized_assay_result = _normalize_multi_assay_payload(
                            {assay_name: assay_result},
                            [assay_name],
                        ).get(assay_name, {})
                        if normalized_assay_result:
                            page_results[assay_name].update(normalized_assay_result)
                    except Exception as exc:
                        print(
                            f"Warning: assay extraction failed for page {page_number}"
                            f"{chunk_label}, assay {assay_name}: {exc}"
                        )
                        extraction_warnings.append({
                            'scope': 'page_assay',
                            'page': page_number,
                            'chunk_index': chunk_index,
                            'chunk_count': len(chunks_for_page),
                            'assay_name': assay_name,
                            'error_type': classify_exception(exc),
                            'error': str(exc),
                        })

            if not _has_any_assay_records(page_results):
                print(f"Page {page_number} produced no assay records.")
                continue

            reviewed_page_results = _apply_visual_value_review(
                pdf_file=pdf_file,
                page_numbers=[page_number],
                assay_dicts=page_results,
                ocr_context=page_content,
                audit_path=model_audit_path,
            )

            for assay_name in assay_names:
                page_assay_dict = reviewed_page_results.get(assay_name, {}) or {}
                if not page_assay_dict:
                    continue
                context_by_key = {
                    raw_key: build_alias_resolution_context(page_content, assay_name, raw_key, raw_value)
                    for raw_key, raw_value in page_assay_dict.items()
                }
                page_assay_dict = remap_assay_dict_to_official_ids(
                    page_assay_dict,
                    compound_id_list,
                    resolver_fn=resolve_compound_id_alias,
                    context_by_key=context_by_key,
                )
                assay_dicts[assay_name].update(page_assay_dict)

        for assay_name, assay_dict in assay_dicts.items():
            assay_name_clean = assay_name.replace(' ', '_').replace('/', '_')
            assay_dir = os.path.join(output_dir, assay_name_clean)
            os.makedirs(assay_dir, exist_ok=True)
            assay_json = os.path.join(assay_dir, 'assay_data.json')
            assay_json_named = os.path.join(assay_dir, f"{assay_name_clean}_assay_data.json")
            write_json_file(assay_json, assay_dict)
            write_json_file(assay_json_named, assay_dict)
            print(f"Saved per-assay page result for {assay_name} to {assay_json}")

        if extraction_warnings:
            warnings_path = os.path.join(output_dir, 'assay_extraction_warnings.json')
            write_json_file(warnings_path, extraction_warnings)
            print(f"Assay extraction warnings saved to {warnings_path}")

        report_progress(total_pages, total_pages, "✅ Finished multi-assay extraction")
        return assay_dicts

    for idx, start in enumerate(range(0, len(content_list), pages_per_chunk), 1):
        chunk = "\n\n".join(content_list[start:start + pages_per_chunk])
        processed_pages = min(total_pages, idx * pages_per_chunk)
        report_progress(processed_pages, total_pages, f"🔍 Analyzing shared chunk {idx} of {chunk_count}")
        print(f"Processing shared chunk {idx}/{chunk_count}...")
        print('Shared chunk content preview:', chunk[:1000])

        try:
            model_chunk = _build_assay_model_content(chunk)
            chunk_multi_assay_dict = content_to_multi_assay_dict(
                model_chunk,
                assay_names,
                compound_id_list=compound_id_list,
                retry=ASSAY_EXTRACTION_LLM_MAX_RETRIES,
                timeout_seconds=ASSAY_EXTRACTION_LLM_TIMEOUT_SECONDS,
                audit_path=model_audit_path,
                metadata={
                    'scope': 'chunk',
                    'chunk_index': idx,
                    'pages': list(range(assay_page_start + start, assay_page_start + min(len(content_list), start + pages_per_chunk))),
                    'chunk_chars': len(str(chunk or '')),
                    'model_chunk_chars': len(str(model_chunk or '')),
                },
            )
        except Exception as exc:
            if len(content_list[start:start + pages_per_chunk]) <= 1:
                print(f"Warning: shared assay chunk {idx}/{chunk_count} failed: {exc}")
                extraction_warnings.append({
                    'scope': 'chunk',
                    'chunk_index': idx,
                    'pages': list(range(assay_page_start + start, assay_page_start + min(len(content_list), start + pages_per_chunk))),
                    'error_type': classify_exception(exc),
                    'error': str(exc),
                })
                continue
            print(
                f"Warning: shared assay chunk {idx}/{chunk_count} failed: {exc}; "
                "retrying page-by-page."
            )
            extraction_warnings.append({
                'scope': 'chunk_retry',
                'chunk_index': idx,
                'pages': list(range(assay_page_start + start, assay_page_start + min(len(content_list), start + pages_per_chunk))),
                'error_type': classify_exception(exc),
                'error': str(exc),
                'action': 'retrying page-by-page',
            })
            chunk_multi_assay_dict = {assay_name: {} for assay_name in assay_names}
            for page_offset, page_content in enumerate(content_list[start:start + pages_per_chunk]):
                page_number = assay_page_start + start + page_offset
                try:
                    model_page_content = _build_assay_model_content(page_content)
                    page_result = content_to_multi_assay_dict(
                        model_page_content,
                        assay_names,
                        compound_id_list=compound_id_list,
                        retry=ASSAY_EXTRACTION_LLM_MAX_RETRIES,
                        timeout_seconds=ASSAY_EXTRACTION_LLM_TIMEOUT_SECONDS,
                        audit_path=model_audit_path,
                        metadata={
                            'scope': 'page_retry',
                            'chunk_index': idx,
                            'page': page_number,
                            'page_chars': len(str(page_content or '')),
                            'model_page_chars': len(str(model_page_content or '')),
                        },
                    )
                    page_result = _normalize_multi_assay_payload(page_result, assay_names)
                    for assay_name, assay_payload in page_result.items():
                        if assay_payload:
                            chunk_multi_assay_dict.setdefault(assay_name, {}).update(assay_payload)
                except Exception as page_exc:
                    print(f"Warning: shared assay page {page_number} failed: {page_exc}")
                    extraction_warnings.append({
                        'scope': 'page',
                        'chunk_index': idx,
                        'page': page_number,
                        'error_type': classify_exception(page_exc),
                        'error': str(page_exc),
                    })
                    continue

        normalized_chunk = _normalize_multi_assay_payload(chunk_multi_assay_dict, assay_names)
        if not any(normalized_chunk.get(assay_name) for assay_name in assay_names):
            print(f"Shared chunk {idx}/{chunk_count} produced no assay records.")
            continue

        chunk_page_numbers = list(range(assay_page_start + start, assay_page_start + min(len(content_list), start + pages_per_chunk)))
        normalized_chunk = _apply_visual_value_review(
            pdf_file=pdf_file,
            page_numbers=chunk_page_numbers,
            assay_dicts=normalized_chunk,
            ocr_context=chunk,
            audit_path=model_audit_path,
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

    if extraction_warnings:
        warnings_path = os.path.join(output_dir, 'assay_extraction_warnings.json')
        write_json_file(warnings_path, extraction_warnings)
        print(f"Assay extraction warnings saved to {warnings_path}")

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
