# activity_parser.py

from utils.pdf_utils import pdf_to_markdown
from utils.llm_utils import content_to_dict, configure_genai
from utils.file_utils import read_text_file, write_json_file
from constants import GEMINI_API_KEY

def extract_activity_data(pdf_file, assay_page_start, assay_page_end, assay_name, compound_id_list, output_dir):
    # Parse the PDF file to Markdown
    assay_md_file = pdf_to_markdown(pdf_file, output_dir, page_start=assay_page_start, page_end=assay_page_end)

    # Read the content of the Markdown file
    content = read_text_file(assay_md_file)

    # Configure the AI client
    configure_genai(GEMINI_API_KEY)

    assay_dict = content_to_dict(content, assay_name, compound_id_list=compound_id_list)

    # Save assay_dict to JSON file
    output_json = f'{output_dir}/assay_data.json'
    write_json_file(output_json, assay_dict)

    return assay_dict