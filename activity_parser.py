import os
import sys
from typing import Optional

import requests

from utils.pdf_utils import dots_ocr
from utils.llm_utils import content_to_dict, configure_genai
from utils.file_utils import write_json_file, read_text_file
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

GEMINI_API_KEY = getattr(constants, 'GEMINI_API_KEY', None)
GEMINI_MODEL_NAME = getattr(constants, 'GEMINI_MODEL_NAME', 'gemini-2.0-flash')
DEFAULT_OCR_ENGINE = (getattr(constants, 'DEFAULT_OCR_ENGINE', 'paddleocr') or 'paddleocr').strip().lower()
PADDLEOCR_SERVER_URL: Optional[str] = getattr(constants, 'PADDLEOCR_SERVER_URL', None)
SUPPORTED_OCR_ENGINES = {'paddleocr', 'dots_ocr'}

if DEFAULT_OCR_ENGINE not in SUPPORTED_OCR_ENGINES:
    DEFAULT_OCR_ENGINE = 'paddleocr'

def extract_activity_data(
    pdf_file,
    assay_page_start,
    assay_page_end,
    assay_name,
    compound_id_list,
    output_dir,
    pages_per_chunk=3,
    lang='en',
    ocr_engine=DEFAULT_OCR_ENGINE,
    ocr_server='http://localhost:8001',
    progress_callback=None,
):
    """
    根据PDF指定页码范围解析数据：
    
    1. 将指定页码范围上传到配置好的 OCR 服务（PaddleOCR 或 DotsOCR），并获取 Markdown 结果。
    2. 根据参数 pages_per_chunk，将多个连续页面的 Markdown 内容组合为一个 chunk，
       每个 chunk 内部的内容通过页码信息分隔，保持原有页面结构。
    3. 针对每个 chunk 调用 content_to_dict 进行数据提取，并合并各chunk的结果。
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

    assay_dict = {}
    content_list = []
    
    total_pages = assay_page_end - assay_page_start + 1
    
    def report_progress(current: int, total: int, message: str) -> None:
        if progress_callback:
            try:
                progress_callback(current, total, message)
            except TypeError:
                # Fallback for legacy callbacks that only accept the message argument
                progress_callback(message)  # type: ignore[call-arg]

    report_progress(0, total_pages, f"🧪 Starting assay extraction for pages {assay_page_start}-{assay_page_end} ({total_pages} pages)")

    ocr_engine_normalized = (ocr_engine or DEFAULT_OCR_ENGINE).strip().lower()
    if ocr_engine_normalized not in SUPPORTED_OCR_ENGINES:
        raise ValueError(f"Unsupported OCR engine '{ocr_engine}'. Supported engines: {', '.join(sorted(SUPPORTED_OCR_ENGINES))}.")

    if ocr_engine_normalized == 'paddleocr':
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
        except (requests.RequestException, OSError) as exc:  # pragma: no cover - network dependant
            raise RuntimeError(f"PaddleOCR service request failed: {exc}") from exc

        payload = response.json()
        markdown_text = payload.get('markdown', '') if isinstance(payload, dict) else ''
        if not markdown_text:
            report_progress(0, total_pages, "⚠️ PaddleOCR returned empty markdown content")
        else:
            separator = '\n\n-#-#-#-#-\n\n'
            content_pages = [segment.strip() for segment in markdown_text.split(separator) if segment.strip()]
            content_list.extend(content_pages)

    elif ocr_engine_normalized == 'dots_ocr':
        # 使用 dots_ocr 解析 PDF
        for aps in range(assay_page_start, assay_page_end + 1):
            current_page_idx = aps - assay_page_start + 1
            report_progress(current_page_idx, total_pages, f"📄 Processing page {aps} ({current_page_idx} of {total_pages}) with Dots OCR")
            # 将指定页码的内容转为 Markdown，假设返回一个列表 [markdown文件路径]
            assay_md_files = dots_ocr(pdf_file, output_dir, page_start=aps, page_end=aps)
            assay_md_file = assay_md_files[0]
            content = read_text_file(assay_md_file)
            content_list.append(content)
    
    chunks = []
    for i in range(0, len(content_list), pages_per_chunk):
        group_pages = content_list[i:i + pages_per_chunk]
        chunk_text = "\n\n".join(group_pages)
        chunks.append(chunk_text)

    report_progress(0, total_pages, f"📊 Dividing {total_pages} pages into {len(chunks)} processing chunks")
    print(f"Total {len(chunks)} chunks to process.")
        
    # 针对每个 chunk 调用 content_to_dict 进行提取
    for idx, chunk in enumerate(chunks, 1):
        processed_pages = min(total_pages, idx * pages_per_chunk)
        report_progress(processed_pages, total_pages, f"🔍 Analyzing chunk {idx} of {len(chunks)}")
        print(f"Processing chunk {idx}/{len(chunks)}...")
        print('Chunk content preview:', chunk[:1000])  # Preview first 1000 characters
        chunk_assay_dict = content_to_dict(chunk, assay_name, compound_id_list=compound_id_list)
        if chunk_assay_dict:
            assay_dict.update(chunk_assay_dict)
        else:
            print(f"Warning: Chunk {idx} returned empty results.")

    print(f"Extracted total assay data entries: {len(assay_dict)}")

    # 保存提取结果至 JSON 文件
    output_json = f'{output_dir}/assay_data.json'
    print(f"Saving assay data to {output_json}")
    write_json_file(output_json, assay_dict)

    report_progress(total_pages, total_pages, "✅ Finished assay extraction")

    return assay_dict
