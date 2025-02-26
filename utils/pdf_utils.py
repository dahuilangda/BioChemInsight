import os
import shutil
import PyPDF2
import fitz
import uuid
from pathlib import Path
try:
    from .pdf2md.parser import parse_file
    from .pdf2md.writer import Writer
    from .file_utils import create_directory, get_file_basename
except ImportError:
    from pdf2md.parser import parse_file
    from pdf2md.writer import Writer
    from file_utils import create_directory, get_file_basename

def generate_uuid_directory(base_dir="output"):
    """
    Creates a unique directory for each run based on UUID and returns the path.
    """
    unique_id = str(uuid.uuid4())
    output_dir = os.path.join(base_dir, unique_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_pdf_page_range(input_path, output_dir, page_start, page_end):
    """
    Saves a specific page range from the input PDF to a new PDF file.
    """
    page_start = page_start - 1

    with open(input_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        pdf_writer = PyPDF2.PdfWriter()

        for page_num in range(page_start, page_end):
            pdf_writer.add_page(pdf_reader.pages[page_num])

        output_file = os.path.join(output_dir, f'pdf_page_{page_start + 1}_{page_end}.pdf')
        with open(output_file, 'wb') as file_out:
            pdf_writer.write(file_out)
    return output_file

def split_pdf_to_images(input_path, images_dir, page_start=1, page_end=None):
    """
    Splits a PDF into individual page images.
    """

    create_directory(images_dir)

    with open(input_path, 'rb') as file:
        doc = fitz.open(file)
        if page_end is None:
            page_end = len(doc)
    
        for page_num in range(page_start-1, page_end):
            output_file = os.path.join(images_dir, f'page_{page_num+1}.png')
            page = doc.load_page(page_num)
            matrix = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=matrix, clip=page.rect)
            pix.save(output_file, 'png')

def parse_pdf(filename: str, output_dir: str, lang: str='en') -> str:
    """
    Parses a PDF file into Markdown format and returns the path to the Markdown file.
    """

    base_name = get_file_basename(filename)
    blocks = parse_file(filename, lang)
    writer = Writer(filename, blocks)
    md_file = os.path.join(output_dir, f'{base_name}.md')
    writer.write_markdown()
    print(f'Markdown saved to {md_file}')
    return md_file

def pdf_to_markdown(input_file, output_dir, page_start, page_end, lang='en'):
    """
    Converts a PDF file to Markdown format.
    """

    image_dir = os.path.join(output_dir, 'activity_images')
    create_directory(image_dir)

    for current_page in range(page_start, page_end + 1):
        pdf_output_path = save_pdf_page_range(input_file, image_dir, current_page, current_page)
        parse_pdf(pdf_output_path, output_dir, lang)

    # merge all markdown files into one
    merged_file = os.path.join(output_dir, 'activity.md')
    if os.path.exists(merged_file):
        os.remove(merged_file)

    markdown_files = Path(image_dir).rglob('*.md')
    with open(merged_file, 'w') as output_file:
        for file in markdown_files:
            with open(file, 'r') as f:
                content = f.readlines()
                content = content[3:]
                content = '\n'.join(content)
                output_file.write(content)
    return merged_file

if __name__ == '__main__':
    pdf_to_markdown('../data/sample.pdf', '../data/output', 270, 272, 'en')