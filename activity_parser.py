import os
import sys
from utils.pdf_utils import pdf_to_markdown, dots_ocr
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

def extract_activity_data(pdf_file, assay_page_start, assay_page_end, assay_name,
                          compound_id_list, output_dir, pages_per_chunk=3, lang='en', ocr_engine='paddleocr', ocr_server='http://localhost:8001'):
    """
    根据PDF指定页码范围解析数据：
    
    1. 使用 pdf_to_markdown 将 PDF 中 assay_page_start 到 assay_page_end 页转换为 Markdown 格式，
       返回一个字典，键为页码，值为对应页面内容。
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
    """

    assay_dict = {}
    content_list = []

    if ocr_engine == 'paddleocr':
        # 使用 paddleocr 解析 PDF
        print(f"Processing pages with PaddleOCR...")
        # 将指定页码的内容转为 Markdown，假设返回一个字典 {页码: markdown文本}
        assay_md_file = pdf_to_markdown(pdf_file, output_dir, page_start=assay_page_start,
                                                page_end=assay_page_end, lang=lang)
        
        content = read_text_file(assay_md_file)
        content_list.append([line.strip() for line in content.split('\n\n-#-#-#-#-\n\n') if line.strip()])

    elif ocr_engine == 'dots_ocr':
        # 使用 dots_ocr 解析 PDF
        for aps in range(assay_page_start, assay_page_end + 1):
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

    print(f"Total {len(chunks)} chunks to process.")
        
    # 针对每个 chunk 调用 content_to_dict 进行提取
    for idx, chunk in enumerate(chunks, 1):
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

    return assay_dict