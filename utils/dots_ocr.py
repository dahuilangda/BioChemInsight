#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A command-line tool to parse PDF files into Markdown format using the dots.ocr library.

This script extracts layout information, text, and tables from a PDF and compiles them
into a single Markdown file. It connects to a backend inference engine (like vLLM)
to perform the analysis. It also includes an optional post-processing step to replace
base64 image data with a text placeholder.

Prerequisites:
- Python 3.8+
- The 'dots_ocr' library and its dependencies must be installed.
  (e.g., pip install dots-ocr)

Example Usage:
    python dots_ocr.py /path/to/your/document.pdf

    # Enable image post-processing
    python dots_ocr.py /path/to/your/document.pdf --post-process-images

    # Specify an output file and connect to a different server
    python dots_ocr.py "assets/showcase_origin/sample.pdf" -o /path/to/output.md --server_ip 192.168.1.100
"""

import argparse
import logging
import os
import shutil
import tempfile
import json
import re 
from pathlib import Path
from typing import List, Dict, Any

from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from constants import DOTSOCR_SERVER_IP, DOTSOCR_SERVER_PORT, DOTSOCR_PROMPT_MODE

print(f"Using DotsOCR server at {DOTSOCR_SERVER_IP}:{DOTSOCR_SERVER_PORT} with prompt mode '{DOTSOCR_PROMPT_MODE}'")


def post_process_images(markdown_text: str) -> str:
    """
    Finds and replaces base64 image data in markdown text with a placeholder tag.
    This makes the output cleaner and more usable for language models.
    Args:
        markdown_text (str): The input markdown string.
    Returns:
        str: The markdown text with base64 images replaced.
    """
    for match in re.finditer(r'!\[\]\(data:image;base64,[^)]+\)', markdown_text):
        placeholder = "<base64_image_placeholder>"
        markdown_text = markdown_text.replace(match.group(0), placeholder)
        logging.debug(f"Replaced base64 image data with placeholder at position {match.start()}")
    num_replacements = markdown_text.count("<base64_image_placeholder>")
    logging.info(f"Post-processed {num_replacements} base64 image(s) in the markdown text.")
    processed_text = markdown_text.replace("<base64_image_placeholder>", "<img src='[base64 image data removed]' alt='Base64 Image'>")
    logging.debug("Final markdown text after post-processing:\n" + processed_text[:500] + "...")
    logging.info("Post-processing complete. Base64 images replaced with placeholder tags.")
    
    return processed_text


class PdfParser:
    """
    A wrapper class for the DotsOCRParser to handle PDF to Markdown conversion.
    """

    def __init__(self, server_ip: str, server_port: int, min_pixels: int, max_pixels: int):
        """
        Initializes the PDF parser with connection and processing settings.

        Args:
            server_ip (str): The IP address of the inference server.
            server_port (int): The port of the inference server.
            min_pixels (int): The minimum number of pixels for an input image dimension.
            max_pixels (int): The maximum number of pixels for an input image dimension.
        """
        logging.info(f"Initializing DotsOCRParser for server at {server_ip}:{server_port}")
        self.parser = DotsOCRParser(
            ip=server_ip,
            port=server_port,
            dpi=200,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        self.temp_dir = None

    def process_pdf(self, pdf_path: Path, prompt_mode: str) -> str:
        """
        Parses a PDF file and returns the combined Markdown content.

        This method orchestrates the creation of a temporary directory, calls the
        parser's high-level API, processes the results, and handles cleanup.

        Args:
            pdf_path (Path): The path to the input PDF file.
            prompt_mode (str): The prompt mode to use for parsing.

        Returns:
            str: A string containing the combined Markdown content of all pages.

        Raises:
            FileNotFoundError: If the input PDF file does not exist.
            ValueError: If the parser returns no results.
            Exception: For other processing errors.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"Input file not found: {pdf_path}")

        self.temp_dir = Path(tempfile.mkdtemp(prefix="dots_ocr_cli_"))
        logging.info(f"Created temporary directory: {self.temp_dir}")
        
        try:
            filename_stem = pdf_path.stem
            
            # Use the high-level API from DotsOCRParser
            results: List[Dict[str, Any]] = self.parser.parse_pdf(
                input_path=str(pdf_path),
                filename=filename_stem,
                prompt_mode=prompt_mode,
                save_dir=str(self.temp_dir)
            )

            if not results:
                raise ValueError("Parsing failed: The parser returned no results.")
            
            logging.info(f"Successfully parsed {len(results)} page(s). Aggregating Markdown content.")

            all_md_content = []
            
            # 先按页码排序，再进行迭代，以修复原代码中的变量作用域错误
            sorted_results = sorted(results, key=lambda r: r.get('page_no', float('inf')))

            for i, result in enumerate(sorted_results):
                md_path = result.get('md_content_path')
                if md_path and os.path.exists(md_path):
                    with open(md_path, 'r', encoding='utf-8') as f:
                        all_md_content.append(f.read())
                else:
                    logging.warning(f"No Markdown content file found for page {result.get('page_no', i + 1)}")
            
            # 使用分隔符连接每一页的内容
            return "\n\n---\n\n".join(all_md_content)

        except Exception as e:
            logging.error(f"An error occurred during PDF processing: {e}")
            raise  # Re-raise the exception after logging
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Removes the temporary directory if it exists.
        """
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logging.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except OSError as e:
                logging.error(f"Failed to remove temporary directory {self.temp_dir}: {e}")
        self.temp_dir = None


def main():
    """
    Main function to parse command-line arguments and run the PDF parsing process.
    """
    parser = argparse.ArgumentParser(
        description="Parse a PDF file into Markdown using the dots.ocr library.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example Commands:
  # Basic usage with a local file
  python %(prog)s "my_document.pdf"

  # Enable image post-processing to replace base64 data
  python %(prog)s "my_document.pdf" --post-process-images

  # Specify an output file path
  python %(prog)s "my_document.pdf" -o "output/my_document.md"

  # Connect to a different server and use a different prompt
  python %(prog)s "my_document.pdf" --server_ip 10.0.0.5 --prompt_mode prompt_ocr
"""
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the input PDF file."
    )
    parser.add_argument(
        "-o", "--output_path",
        type=Path,
        default=None,
        help="Path to save the output Markdown file. \n(default: <input_filename>.md in the same directory)"
    )
    parser.add_argument(
        "--server_ip",
        type=str,
        default=DOTSOCR_SERVER_IP,
        help=f"IP address of the inference server. (default: {DOTSOCR_SERVER_IP})"
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=DOTSOCR_SERVER_PORT,
        help=f"Port of the inference server. (default: {DOTSOCR_SERVER_PORT})"
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default=DOTSOCR_PROMPT_MODE,
        choices=["prompt_layout_all_en", "prompt_layout_only_en", "prompt_ocr"],
        help=f"The prompt mode to use for parsing. (default: {DOTSOCR_PROMPT_MODE})"
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=MIN_PIXELS,
        help=f"Minimum image dimension in pixels. (default: {MIN_PIXELS})"
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=MAX_PIXELS,
        help=f"Maximum image dimension in pixels. (default: {MAX_PIXELS})"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level. (default: INFO)"
    )
    
    parser.add_argument(
        '--post-process-images',
        action='store_true',
        help="Enable post-processing to replace base64 images with a placeholder tag."
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Determine output path if not specified
    output_path = args.output_path
    if output_path is None:
        output_path = args.input_path.with_suffix(".md")
    
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize and run the parser
        pdf_parser = PdfParser(
            server_ip=args.server_ip,
            server_port=args.server_port,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels
        )
        markdown_content = pdf_parser.process_pdf(args.input_path, args.prompt_mode)
        # If the flag is set, apply the image post-processing function
        if args.post_process_images:
            logging.info("Image post-processing enabled.")
            markdown_content = post_process_images(markdown_content)
        
        # Save the final markdown content
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        logging.info(f"✅ Successfully completed parsing.")
        print(f"\nOutput saved to: {output_path.resolve()}")

    except FileNotFoundError as e:
        logging.error(e)
        exit(1)
    except Exception as e:
        logging.critical(f"A critical error occurred: {e}")
        print("\n❌ Parsing failed. Check the log for details.")
        exit(1)


if __name__ == "__main__":
    main()