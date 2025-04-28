# from utils.pdf_utils import pdf_to_markdown
# from utils.llm_utils import content_to_dict, configure_genai
# from utils.file_utils import read_text_file, write_json_file
# from constants import GEMINI_API_KEY, GEMINI_MODEL_NAME

# def extract_activity_data(pdf_file, assay_page_start, assay_page_end, assay_name, compound_id_list, output_dir, lang='en'):
#     # Parse the PDF file to Markdown

#     # 如果assay_page_start是list，则取第一个元素
#     if isinstance(assay_page_start, list):
#         assay_page_start = assay_page_start[0]
#     if isinstance(assay_page_end, list):
#         assay_page_end = assay_page_end[0]

#     assay_md_file = pdf_to_markdown(pdf_file, output_dir, page_start=assay_page_start, page_end=assay_page_end, lang=lang)
#     print(f"Markdown file created: {assay_md_file}")

#     # Read the content of the Markdown file
#     content = read_text_file(assay_md_file)
#     print(f"Extracted content from {assay_md_file}:\n{content}")

#     # Configure the AI client
#     configure_genai(GEMINI_API_KEY)

#     assay_dict = content_to_dict(content, assay_name, compound_id_list=compound_id_list, model_name=GEMINI_MODEL_NAME)

#     # Save assay_dict to JSON file
#     output_json = f'{output_dir}/assay_data.json'
#     write_json_file(output_json, assay_dict)

#     return assay_dict


from utils.pdf_utils import pdf_to_markdown
from utils.llm_utils import content_to_dict, configure_genai
from utils.file_utils import write_json_file, read_text_file
from constants import GEMINI_API_KEY, GEMINI_MODEL_NAME

def extract_activity_data(pdf_file, assay_page_start, assay_page_end, assay_name,
                          compound_id_list, output_dir, pages_per_chunk=3, lang='en'):
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

    # # 如果assay_page_start是list，则取第一个元素
    # if isinstance(assay_page_start, list):
    #     assay_page_start = assay_page_start[0]
    # if isinstance(assay_page_end, list):
    #     assay_page_end = assay_page_end[0]

    assay_dict = {}
    for aps, ape in zip(assay_page_start, assay_page_end):

        # 将指定页码的内容转为 Markdown，假设返回一个字典 {页码: markdown文本}
        assay_md_file = pdf_to_markdown(pdf_file, output_dir, page_start=aps,
                                                page_end=ape, lang=lang)
        
        content = read_text_file(assay_md_file)
        content_list = [line.strip() for line in content.split('\n\n-#-#-#-#-\n\n') if line.strip()]

        chunks = []
        for i in range(0, len(content_list), pages_per_chunk):
            group_pages = content_list[i:i + pages_per_chunk]
            chunk_text = "\n\n".join(group_pages)
            chunks.append(chunk_text)

        print(f"Total {len(chunks)} chunks to process.")
        
        # 配置 AI 客户端
        configure_genai(GEMINI_API_KEY)
        
        # 针对每个 chunk 调用 content_to_dict 进行提取
        for idx, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {idx}/{len(chunks)}...")
            print('Chunk content preview:', chunk[:1000])  # Preview first 1000 characters
            chunk_assay_dict = content_to_dict(chunk, assay_name, compound_id_list=compound_id_list, 
                                                model_name=GEMINI_MODEL_NAME)
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
