import os
import shutil
import PyPDF2
import fitz
import uuid
from .file_utils import create_directory

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

def save_pdf_single_page(input_path, output_dir, page_start, page_end):
    """
    Saves a single page from the input PDF to a new PDF file list.
    """
    page_start = page_start - 1
    page_end = page_end - 1
    files = []
    for page_num in range(page_start, page_end + 1):
        with open(input_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pdf_writer = PyPDF2.PdfWriter()

            pdf_writer.add_page(pdf_reader.pages[page_num])
            output_file = os.path.join(output_dir, f'pdf_page_{page_num + 1}.pdf')
            with open(output_file, 'wb') as file_out:
                pdf_writer.write(file_out)
            files.append(output_file)
    return files

def split_pdf_to_images(input_path, images_dir, page_start=1, page_end=None):
    """
    Splits a PDF into individual page images.
    """

    create_directory(images_dir)

    with open(input_path, 'rb') as file:
        doc = fitz.open(file)
        try:
            if page_end is None:
                page_end = len(doc)

            for page_num in range(page_start - 1, page_end):
                output_file = os.path.join(images_dir, f'page_{page_num+1}.png')
                page = doc.load_page(page_num)
                matrix = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=matrix, clip=page.rect)
                pix.save(output_file, 'png')
                pix = None
                page = None
                if ((page_num - (page_start - 1) + 1) % 8) == 0:
                    try:
                        import gc
                        gc.collect()
                    except Exception:
                        pass
        finally:
            doc.close()
