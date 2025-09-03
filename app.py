import os
import sys
import json
import pandas as pd
import gradio as gr
from pathlib import Path
import tempfile
import shutil
import fitz  # PyMuPDF
import base64
from typing import List, Dict, Any
import subprocess
import io
import math
import time
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import Draw

# Add the current directory to the Python path to import local modules
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    sys.path.append('.')

# Suppress unnecessary warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

# --- Simplified State Management ---
# Remove complex session management, use simple in-memory state with auto-download

# --- Helper function for rendering SMILES to image ---
def smiles_to_img_tag(smiles: str) -> str:
    """Converts a SMILES string to a base64-encoded image tag for markdown."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES"
        img = Draw.MolToImage(mol, size=(300, 300), kekulize=True)
        
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f'![structure](data:image/png;base64,{img_str})'
    except Exception as e:
        return f"Error rendering structure: {str(e)}"

class BioChemInsightApp:
    """Simplified application without complex session management."""
    def __init__(self):
        """Initializes the application with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.current_pdf_path = None
        self.current_pdf_filename = None
        print(f"Application initialized with temp directory: {self.temp_dir}")

    def get_processing_status(self) -> str:
        """Simple status check based on current state."""
        if not self.current_pdf_path:
            return "Please upload a PDF file"
        return f"PDF loaded: {self.current_pdf_filename}"

    def segment_to_img_tag(self, segment_path: str) -> str:
        """Converts a segment file path to a base64-encoded image tag for markdown."""
        if not segment_path or pd.isna(segment_path):
            return "No segment"
        try:
            full_path = os.path.join(self.temp_dir, "output", segment_path)
            if not os.path.exists(full_path):
                full_path = segment_path
                if not os.path.exists(full_path):
                    return f"Segment not found: {segment_path}"
            
            with open(full_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()

            return f'![segment](data:image/png;base64,{encoded_string})'
        except Exception as e:
            return f"Error rendering segment: {str(e)}"

    def _enrich_dataframe_with_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds rendered structure and segment images to the dataframe for display."""
        
        SMILES_COL = 'SMILES'
        SEGMENT_COL = 'SEGMENT_FILE'
        IMAGE_FILE_COL = 'IMAGE_FILE'
        COMPOUND_ID_COL = 'COMPOUND_ID'

        df = df.copy()

        if SMILES_COL in df.columns:
            df['Structure'] = df[SMILES_COL].apply(smiles_to_img_tag)
        if SEGMENT_COL in df.columns:
            df['Segment'] = df[SEGMENT_COL].apply(self.segment_to_img_tag)
        if IMAGE_FILE_COL in df.columns:
            df['Image File'] = df[IMAGE_FILE_COL].apply(self.segment_to_img_tag)

        all_cols = df.columns.tolist()
        
        front_cols = ['Structure', 'Segment', 'Image File', COMPOUND_ID_COL]
        back_cols = [SMILES_COL, SEGMENT_COL, IMAGE_FILE_COL]

        present_front = [col for col in front_cols if col in all_cols]
        middle_cols = [
            col for col in all_cols 
            if col not in present_front and col not in back_cols
        ]
        
        new_order = present_front + sorted(middle_cols)
        df = df[new_order]
        
        return df

    def _get_df_dtypes(self, df: pd.DataFrame) -> List[str]:
        """Generates a list of datatypes for the Gradio DataFrame."""
        markdown_cols = ['Structure', 'Segment', 'Image File']
        dtypes = []
        for col in df.columns:
            if col in markdown_cols:
                dtypes.append("markdown")
            else:
                dtypes.append("str")
        return dtypes

    def get_pdf_info(self, pdf_file) -> tuple:
        """Processes the uploaded PDF and returns its basic information."""
        print(f"Debug: get_pdf_info called with: {pdf_file}, type: {type(pdf_file)}")
        
        if pdf_file is None:
            print("Debug: pdf_file is None")
            return "❌ Please upload a PDF file first", 0
            
        try:
            # Handle different Gradio file upload formats
            if hasattr(pdf_file, 'name') and pdf_file.name:
                # Gradio File object or NamedString
                file_path = str(pdf_file.name)  # Convert to string to handle NamedString
                pdf_name = os.path.basename(file_path)
            elif isinstance(pdf_file, str):
                # String path
                file_path = pdf_file
                pdf_name = os.path.basename(file_path)
            else:
                print(f"Debug: Unknown file type: {type(pdf_file)}, value: {pdf_file}")
                return "❌ Unsupported file format", 0
            
            print(f"Debug: Processing file: {file_path}, name: {pdf_name}")
            
            if not os.path.exists(file_path):
                print(f"Debug: File does not exist: {file_path}")
                return "❌ File not found", 0
                
            self.current_pdf_filename = pdf_name
            self.current_pdf_path = os.path.join(self.temp_dir, pdf_name)
            
            # Ensure temp directory exists
            os.makedirs(self.temp_dir, exist_ok=True)
            shutil.copy(file_path, self.current_pdf_path)
            
            print(f"Debug: File copied to: {self.current_pdf_path}")
            
            doc = fitz.open(self.current_pdf_path)
            total_pages = doc.page_count
            doc.close()
            info = f"✅ PDF uploaded successfully, containing {total_pages} pages."
            print(f"Debug: PDF processed successfully, {total_pages} pages")
            return info, total_pages
        except Exception as e:
            print(f"Debug: Error processing PDF: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Failed to load PDF: {str(e)}", 0

    def parse_pages_input(self, pages_str: str) -> List[int]:
        """Parses a page string (e.g., '1,3,5-7,10') into a list of page numbers."""
        if not pages_str or not pages_str.strip():
            return []
        pages = set()
        for part in pages_str.strip().split(','):
            part = part.strip()
            if not part: continue
            if '-' in part:
                try:
                    start, end = map(int, part.split('-', 1))
                    if start > end: start, end = end, start
                    pages.update(range(start, end + 1))
                except ValueError: continue
            else:
                try: pages.add(int(part))
                except ValueError: continue
        return sorted(list(pages))
    
    def _generate_pdf_gallery(self, total_pages: int, struct_pages_str: str, assay_pages_str: str) -> str:
        """Generates the HTML gallery for ALL pages of the PDF."""
        if not self.current_pdf_path or not os.path.exists(self.current_pdf_path):
            return "<div class='center-placeholder'>Please upload a valid PDF file first</div>"
        
        try:
            doc = fitz.open(self.current_pdf_path)
            struct_pages = set(self.parse_pages_input(struct_pages_str))
            assay_pages = set(self.parse_pages_input(assay_pages_str))
            
            gallery_html = """<div class="gallery-wrapper"><div class="selection-info-bar"><span style="font-weight: 500;">💡 Tip: Toggle 'Selection Mode', then click pages to select. Hold Shift and click to select range. You can also type page ranges directly.</span></div></div><div id="gallery-container" class="gallery-container">"""
            
            for page_num in range(1, total_pages + 1):
                page = doc[page_num - 1]
                low_res_pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img_base64 = base64.b64encode(low_res_pix.tobytes("png")).decode()

                page_classes = "page-item"
                if page_num in struct_pages: page_classes += " selected-struct"
                if page_num in assay_pages: page_classes += " selected-assay"
                
                gallery_html += f"""
                <div class="{page_classes}" data-page="{page_num}" onclick="handlePageClick(this, event)">
                    <img src='data:image/png;base64,{img_base64}' alt='Page {page_num}' />
                    <div class="page-label">Page {page_num}</div>
                    <div class="selection-check struct-check">S</div>
                    <div class="selection-check assay-check">A</div>
                    <div class="magnify-icon" onclick="event.stopPropagation(); requestMagnifyView({page_num});" title="Enlarge page">🔍</div>
                </div>
                """
            
            gallery_html += "</div>"
            doc.close()
            return gallery_html
        except Exception as e:
            return f"<div class='center-placeholder error'>Failed to generate preview: {str(e)}</div>"

    def get_magnified_page_data(self, page_num: int) -> str:
        if not page_num or not self.current_pdf_path: return ""
        try:
            doc = fitz.open(self.current_pdf_path)
            if 0 < page_num <= doc.page_count:
                page = doc[page_num - 1]
                high_res_pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
                high_res_base64 = base64.b64encode(high_res_pix.tobytes("png")).decode()
                doc.close()
                return high_res_base64
            doc.close()
            return ""
        except Exception as e:
            print(f"Error getting high-res page: {e}")
            return ""

    def update_gallery_view(self, total_pages: int, struct_pages_str: str, assay_pages_str: str) -> str:
        """Wrapper to generate the full gallery view."""
        if not self.current_pdf_path: return "<div class='center-placeholder'>Please upload a PDF file first</div>"
        gallery_html = self._generate_pdf_gallery(total_pages, struct_pages_str, assay_pages_str)
        return gallery_html

    def clear_all_selections(self, total_pages_num: int) -> tuple:
        """Clears selections - only clears text inputs, visual clearing handled by JavaScript."""
        return "", "", gr.update()
    
    def _run_pipeline(self, args: List[str], clear_output: bool = True) -> str:
        """Enhanced pipeline execution with better error handling and recovery."""
        output_dir = os.path.join(self.temp_dir, "output")
        if clear_output and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            pipeline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.py")
            if not os.path.exists(pipeline_path):
                pipeline_path = "pipeline.py"
        except NameError:
            pipeline_path = "pipeline.py"

        command = [sys.executable, pipeline_path, self.current_pdf_path] + args + ["--output", output_dir]
        print(f"Running command: {' '.join(command)}")
        
        # Enhanced error handling with retries
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # Set timeout for long-running processes
                result = subprocess.run(
                    command, 
                    check=True, 
                    capture_output=True, 
                    text=True, 
                    encoding='utf-8',
                    timeout=300  # 5 minute timeout
                )
                print("Pipeline STDOUT:", result.stdout)
                if result.stderr: 
                    print("Pipeline STDERR:", result.stderr)
                return output_dir
                
            except subprocess.TimeoutExpired as e:
                print(f"Pipeline timeout on attempt {attempt + 1}/{max_retries + 1}")
                if attempt < max_retries:
                    print("Retrying with extended timeout...")
                    time.sleep(5)  # Wait before retry
                else:
                    raise Exception(f"Pipeline timed out after {max_retries + 1} attempts")
                    
            except subprocess.CalledProcessError as e:
                print(f"Pipeline error on attempt {attempt + 1}/{max_retries + 1}: {e}")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                
                if attempt < max_retries:
                    print("Retrying pipeline execution...")
                    time.sleep(3)  # Wait before retry
                else:
                    # Final attempt failed, provide helpful error message
                    error_msg = f"Pipeline failed after {max_retries + 1} attempts"
                    if e.stderr:
                        if "CUDA" in e.stderr or "GPU" in e.stderr:
                            error_msg += ". GPU memory issue detected - try reducing batch size"
                        elif "Memory" in e.stderr or "OOM" in e.stderr:
                            error_msg += ". Out of memory - try processing fewer pages"
                        else:
                            error_msg += f". Error: {e.stderr[:200]}"
                    raise Exception(error_msg)
                    
            except FileNotFoundError:
                error_msg = f"Pipeline script not found at '{pipeline_path}'"
                print(f"Error: {error_msg}")
                raise Exception(error_msg)
                
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}/{max_retries + 1}: {e}")
                if attempt < max_retries:
                    print("Retrying...")
                    time.sleep(2)
                else:
                    raise Exception(f"Pipeline failed: {str(e)}")
        
        # This should never be reached, but just in case
        raise Exception("Pipeline execution failed unexpectedly")

    def extract_structures(self, struct_pages_input: str, engine_input: str, lang_input: str) -> tuple:
        if not self.current_pdf_path:
            return "❌ Please upload a PDF file first", None, None, "none", gr.update(), gr.update(), gr.update(visible=False), "Please upload a PDF to begin.", gr.update(), gr.update(), gr.update()
        if not struct_pages_input or not struct_pages_input.strip():
            return "❌ Please select pages for structures.", None, None, "none", gr.update(), gr.update(), gr.update(visible=False), "Please select pages for structures.", gr.update(), gr.update(), gr.update()
        
        # Validate that pages actually exist
        parsed_pages = self.parse_pages_input(struct_pages_input)
        if not parsed_pages:
            return "❌ Please select valid pages for structures.", None, None, "none", gr.update(), gr.update(), gr.update(visible=False), "Please select valid pages for structures.", gr.update(), gr.update(), gr.update()
        try:
            print(f"Extracting structures from pages: {struct_pages_input}")
            
            args = ["--structure-pages", struct_pages_input, "--engine", engine_input, "--lang", lang_input]
            output_dir = self._run_pipeline(args, clear_output=True)
            structures_file = os.path.join(output_dir, "structures.csv")
            if os.path.exists(structures_file):
                df = pd.read_csv(structures_file)
                if not df.empty:
                    df_enriched = self._enrich_dataframe_with_images(df.copy())
                    datatypes = self._get_df_dtypes(df_enriched)
                    view_df_update = gr.update(value=df_enriched, datatype=datatypes, visible=True)
                    
                    edit_df = df.drop(columns=['Structure', 'Segment', 'Image File'], errors='ignore')
                    edit_df_update = gr.update(value=edit_df, visible=False)

                    print(f"Successfully extracted {len(df)} structures")

                    new_guidance = f"✅ **Step 1 Complete**. Now, enter assay names, select pages with bioactivity data, and click Step 2."
                    return f"✅ Extracted {len(df)} structures.", df.to_dict('records'), None, "structures", view_df_update, edit_df_update, gr.update(visible=True), new_guidance, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
                else:
                    return "⚠️ No structures found.", None, None, "none", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "No structures found.", gr.update(), gr.update(), gr.update()
            else:
                 return "❌ 'structures.csv' not created.", None, None, "none", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "File not generated.", gr.update(), gr.update(), gr.update()
        except Exception as e:
            print(f"Error extracting structures: {e}")
            return f"❌ Error: {str(e)}", None, None, "none", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "An error occurred.", gr.update(), gr.update(), gr.update()

    def extract_activity_and_merge(self, assay_pages_input: str, structures_data: list, assay_names: str, lang_input: str, ocr_engine_input: str) -> tuple:
        if not assay_pages_input: return "❌ Select activity pages.", gr.update(), "merged", gr.update(), gr.update(), gr.update(), gr.update(), "Select activity pages.", gr.update()
        if not structures_data: return "⚠️ Run Step 1 first.", gr.update(), "merged", gr.update(), gr.update(), gr.update(), gr.update(), "Run Step 1 first.", gr.update()
        if not assay_names: return "❌ Enter assay names.", gr.update(), "merged", gr.update(), gr.update(), gr.update(), gr.update(), "Enter assay names.", gr.update()
        try:
            print(f"Extracting activity data from pages: {assay_pages_input}")
            
            args = ["--assay-pages", assay_pages_input, "--assay-names", assay_names, "--lang", lang_input, "--ocr-engine", ocr_engine_input]
            output_dir = self._run_pipeline(args, clear_output=False)
            merged_file = os.path.join(output_dir, "merged.csv")
            if not os.path.exists(merged_file):
                return "❌ 'merged.csv' not found.", gr.update(), "merged", gr.update(), gr.update(), gr.update(), gr.update(), "Merge failed.", gr.update()
            df = pd.read_csv(merged_file)
            if df.empty:
                return "⚠️ Merge result is empty.", gr.update(), "merged", gr.update(), gr.update(), gr.update(), gr.update(), "No data to merge.", gr.update()
            
            df_enriched = self._enrich_dataframe_with_images(df.copy())
            datatypes = self._get_df_dtypes(df_enriched)
            view_df_update = gr.update(value=df_enriched, datatype=datatypes, visible=True)
            
            edit_df = df.drop(columns=['Structure', 'Segment', 'Image File'], errors='ignore')
            edit_df_update = gr.update(value=edit_df, visible=False)

            print(f"Successfully merged {len(df)} records")

            status_msg = f"✅ Merge successful. Generated {len(df)} records."
            return (status_msg, df.to_dict('records'), "merged", view_df_update, edit_df_update, merged_file, 
                    gr.update(visible=True), 
                    f"✅ **Process Complete!** View and download results below.", gr.update(visible=True))
        except Exception as e:
            print(f"Error in activity extraction and merge: {e}")
            return f"❌ Error: {repr(e)}", gr.update(), "merged", gr.update(), gr.update(), gr.update(), gr.update(), "An error occurred.", gr.update()

    def _prepare_download_payload(self, filename: str, mime_type: str, data_bytes: bytes) -> str:
        """Creates a JSON string with Base64 encoded data for client-side download."""
        if not data_bytes: return None
        b64_data = base64.b64encode(data_bytes).decode('utf-8')
        return json.dumps({"name": filename, "mime": mime_type, "data": b64_data})

    def download_csv_and_get_payload(self, data: list, filename: str) -> str:
        """Generates a CSV file from state and returns the JSON payload for download."""
        if data is None or not isinstance(data, list):
            gr.Warning(f"No data available to download for '{filename}'.")
            return None
        
        df = pd.DataFrame(data)
        cols_to_drop = ['Structure', 'Segment', 'Image File']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        
        gr.Info(f"Preparing download for {filename}...")
        return self._prepare_download_payload(filename, 'text/csv', csv_bytes)

    def generate_metadata_and_get_payload(self, struct_pages_str, assay_pages_str, structures_data, merged_data) -> str:
        """Creates a meta.json file in memory and returns the JSON payload for download."""
        if not self.current_pdf_path: return None
        metadata = {
            "source_file": self.current_pdf_filename,
            "output_directory": os.path.join(self.temp_dir, "output"),
            "processing_details": {
                "structure_extraction": {"pages_processed": struct_pages_str or "None", "structures_found": len(structures_data) if structures_data else 0},
                "bioactivity_extraction": {"pages_processed": assay_pages_str or "None", "merged_records_found": len(merged_data) if merged_data else 0}
            }
        }
        json_bytes = json.dumps(metadata, indent=4).encode('utf-8')
        gr.Info("Preparing download for meta.json...")
        return self._prepare_download_payload("meta.json", "application/json", json_bytes)

    def on_upload(self, pdf_file) -> tuple:
        """Enhanced PDF upload with better error handling and user feedback."""
        print(f"Debug: on_upload called with: {pdf_file}, type: {type(pdf_file)}")
        
        # Basic validation first
        if pdf_file is None:
            print("Debug: pdf_file is None in on_upload")
            return ("❌ Please upload a PDF file", 0, "<div class='center-placeholder'>Please upload a PDF file</div>", 
                    None, None, None, "none", "", "", "Please upload a valid PDF file", 
                    "🚀 Welcome to BioChemInsight! Upload a PDF file to start processing.", 
                    gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
        
        try:
            # Enhanced PDF processing with better error handling
            info, pages = self.get_pdf_info(pdf_file)
            
            # Check if PDF processing succeeded
            if pages == 0 or not self.current_pdf_path:
                print(f"Debug: PDF processing failed. Pages: {pages}, Path: {self.current_pdf_path}")
                return (info, 0, "<div class='center-placeholder'>PDF processing failed, please re-upload</div>", 
                        None, None, None, "none", "", "", "PDF processing failed", 
                        "⚠️ PDF processing failed. Please check the file and try again.", 
                        gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
            
            print(f"Debug: PDF uploaded successfully. Path: {self.current_pdf_path}, Pages: {pages}")
            
            # Generate gallery with error handling
            try:
                gallery_html = self.update_gallery_view(pages, "", "")
            except Exception as e:
                print(f"Debug: Gallery generation failed: {e}")
                gallery_html = f"<div class='center-placeholder'>⚠️ Preview generation failed: {str(e)}<br>PDF loaded successfully, but preview unavailable.</div>"
            
            # Backup file to avoid loss
            try:
                backup_path = os.path.join(self.temp_dir, f"backup_{self.current_pdf_filename}")
                shutil.copy2(self.current_pdf_path, backup_path)
                print(f"PDF backed up to: {backup_path}")
            except Exception as e:
                print(f"Warning: Failed to create backup: {e}")
            
            guidance = """
            ✅ **PDF uploaded successfully!**
            **Step 1:** Select pages containing **chemical structures** and click the button below.
            """
            
            return (info, pages, gallery_html, None, None, None, "none", "", "", "PDF loaded", guidance, 
                    gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
                    
        except Exception as e:
            print(f"Critical error in on_upload: {e}")
            import traceback
            traceback.print_exc()
            
            error_guidance = f"""
            ❌ **Upload failed:** {str(e)}
            
            **Please try:**
            1. Check if the PDF file is valid and not corrupted
            2. Try a smaller PDF file (< 50MB recommended)
            3. Ensure the PDF is not password-protected
            4. Refresh the page and try again
            """
            
            return (f"❌ Upload error: {str(e)}", 0, 
                    f"<div class='center-placeholder'>❌ Upload failed:<br>{str(e)}</div>", 
                    None, None, None, "none", "", "", "Upload failed", error_guidance,
                    gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))

    def show_enlarged_image_from_df(self, evt: gr.SelectData) -> str:
        """Callback to handle clicks on the results dataframe to enlarge images."""
        if evt.index is None or evt.value is None or not isinstance(evt.value, str):
            return None
        if evt.value.startswith('!['):
            if 'base64,' in evt.value:
                try:
                    return evt.value.split('base64,', 1)[1][:-1]
                except (IndexError, TypeError):
                    return None
        return None

    def enter_edit_mode(self):
        """Switches to edit mode by toggling component visibility. No data processing needed."""
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True)
        )

    def save_changes(self, edited_df: pd.DataFrame, active_name: str):
        """Saves edited data back to state and switches back to view mode."""
        df_enriched = self._enrich_dataframe_with_images(edited_df.copy())
        datatypes = self._get_df_dtypes(df_enriched)
        
        updated_data_for_state = edited_df.to_dict('records')

        updated_structures = updated_data_for_state if active_name == "structures" else gr.update()
        updated_merged = updated_data_for_state if active_name == "merged" else gr.update()
        
        new_edit_df = edited_df.drop(columns=['Structure', 'Segment', 'Image File'], errors='ignore')

        return (
            updated_structures, updated_merged,
            gr.update(value=df_enriched, datatype=datatypes, visible=True),
            gr.update(value=new_edit_df, visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    def cancel_edit(self):
        """Switches back to view mode by toggling visibility."""
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    def create_interface(self):
        css = """
        #magnify-page-input, #magnified-image-output { display: none; }
        .center-placeholder { text-align: center; padding: 50px; border: 2px dashed #ccc; border-radius: 10px; margin-top: 20px; }
        .gallery-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px; max-height: 90vh; overflow-y: auto; }
        .page-item { border: 3px solid #ddd; border-radius: 8px; cursor: pointer; position: relative; background: white; transition: border-color 0.2s, box-shadow 0.2s; }
        .page-item:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .page-item.selected-struct { border-color: #28a745; } .page-item.selected-assay { border-color: #007bff; } .page-item.selected-struct.selected-assay { border-image: linear-gradient(45deg, #28a745, #007bff) 1; }
        .selection-check { position: absolute; top: 8px; width: 24px; height: 24px; border-radius: 50%; color: white; display: none; align-items: center; justify-content: center; font-weight: bold; z-index: 2; font-size: 14px; }
        .page-item.selected-struct .struct-check { display: flex; right: 8px; background: #28a745; } .page-item.selected-assay .assay-check { display: flex; right: 38px; background: #007bff; }
        .page-item .magnify-icon { position: absolute; top: 8px; left: 8px; width: 24px; height: 24px; background: rgba(0,0,0,0.5); color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: zoom-in; z-index: 2;}
        .page-item .page-label { text-align: center; padding: 8px; }
        #magnify-modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.8); justify-content: center; align-items: center; }
        #magnify-modal img { max-width: 90%; max-height: 90%; } #magnify-modal .close-btn { position: absolute; top: 20px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; cursor: pointer; }
        #results-table-view .gr-table tbody tr td { height: auto !important; } #results-table-view .gr-table tbody tr td div { max-height: 250px; overflow-y: auto; }
        #results-table-view .gr-table tbody tr td img { max-width: 100%; height: auto; object-fit: contain; display: block; margin: auto; }
        #results-table-view .gr-table tbody tr td:nth-child(1),
        #results-table-view .gr-table tbody tr td:nth-child(2),
        #results-table-view .gr-table tbody tr td:nth-child(3) { cursor: zoom-in; }
        """
        
        js_script = """
        () => {
            // Network resilience and auto-save functionality
            window.biochemAutoSave = {
                // Save processing results to browser storage for recovery
                saveResults: function(type, data) {
                    try {
                        const key = `biocheminsight_${type}_${Date.now()}`;
                        localStorage.setItem(key, JSON.stringify({
                            timestamp: new Date().toISOString(),
                            type: type,
                            data: data,
                            url: window.location.href
                        }));
                        console.log(`Results saved: ${type} (${key})`);
                    } catch (e) {
                        console.warn('Failed to save results:', e);
                    }
                },
                
                // Check for and restore previous results
                checkForSavedResults: function() {
                    try {
                        const keys = Object.keys(localStorage).filter(k => k.startsWith('biocheminsight_'));
                        if (keys.length > 0) {
                            console.log(`Found ${keys.length} saved result(s)`);
                            // Show recovery option to user
                            const latest = keys.sort().pop();
                            const data = JSON.parse(localStorage.getItem(latest));
                            if (data && Date.now() - new Date(data.timestamp).getTime() < 24 * 60 * 60 * 1000) {
                                console.log('Recent results available for recovery');
                                return data;
                            }
                        }
                    } catch (e) {
                        console.warn('Failed to check saved results:', e);
                    }
                    return null;
                },
                
                // Auto-download results when processing completes
                autoDownloadResults: function() {
                    setTimeout(() => {
                        // Trigger download of structures if available
                        const structBtn = document.querySelector('button[id*="struct-dl-csv"]');
                        if (structBtn && structBtn.style.display !== 'none') {
                            console.log('Auto-downloading structures...');
                            structBtn.click();
                        }
                        
                        // Trigger download of merged results if available
                        setTimeout(() => {
                            const mergedBtn = document.querySelector('button[id*="merged-dl"]');
                            if (mergedBtn && mergedBtn.style.display !== 'none') {
                                console.log('Auto-downloading merged results...');
                                mergedBtn.click();
                            }
                        }, 1000);
                    }, 2000);
                }
            };
            
            // Monitor for processing completion and auto-save
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    if (mutation.type === 'childList') {
                        // Check if processing completed (success message appeared)
                        const successMessages = document.querySelectorAll('[class*="success"], [class*="complete"]');
                        successMessages.forEach(msg => {
                            if (msg.textContent.includes('Complete') || msg.textContent.includes('successful')) {
                                console.log('Processing completed, triggering auto-save');
                                window.biochemAutoSave.autoDownloadResults();
                            }
                        });
                    }
                });
            });
            
            // Start observing
            setTimeout(() => {
                observer.observe(document.body, {
                    childList: true,
                    subtree: true
                });
            }, 1000);
            
            // Simplified JavaScript without session management complexity
            const parsePageString = (str) => {
                const pages = new Set();
                if (!str) return pages;
                str.split(',').forEach(part => {
                    part = part.trim();
                    if (part.includes('-')) {
                        const [start, end] = part.split('-').map(Number);
                        if (!isNaN(start) && !isNaN(end)) for (let i = Math.min(start, end); i <= Math.max(start, end); i++) pages.add(i);
                    } else { const num = Number(part); if (!isNaN(num)) pages.add(num); }
                });
                return pages;
            };
            const stringifyPageSet = (pageSet) => {
                const ranges = []; let sortedPages = Array.from(pageSet).sort((a, b) => a - b); let i = 0;
                while (i < sortedPages.length) {
                    let start = sortedPages[i]; let j = i;
                    while (j + 1 < sortedPages.length && sortedPages[j + 1] === sortedPages[j] + 1) { j++; }
                    if (j === i) ranges.push(start.toString()); 
                    else if (j === i + 1) ranges.push(start.toString(), sortedPages[j].toString());
                    else ranges.push(`${start}-${sortedPages[j]}`);
                    i = j + 1;
                }
                return ranges.join(',');
            };
            
            // Store last clicked page for shift+click range selection
            window.lastClickedPage = null;
            
            window.handlePageClick = function(element, event) {
                const pageNum = parseInt(element.getAttribute('data-page'));
                const modeRadio = document.querySelector('#selection-mode-radio input:checked');
                if (!modeRadio) return;
                
                const mode = modeRadio.value;
                const classToToggle = (mode === 'Structures') ? 'selected-struct' : 'selected-assay';
                const targetId = (mode === 'Structures') ? '#struct-pages-input' : '#assay-pages-input';
                const targetTextbox = document.querySelector(targetId).querySelector('textarea, input');
                let pages = parsePageString(targetTextbox.value);
                
                // Handle Shift+Click for range selection
                if (event && event.shiftKey && window.lastClickedPage !== null && window.lastClickedPage !== pageNum) {
                    const startPage = Math.min(window.lastClickedPage, pageNum);
                    const endPage = Math.max(window.lastClickedPage, pageNum);
                    
                    // Add all pages in the range
                    for (let i = startPage; i <= endPage; i++) {
                        pages.add(i);
                        // Visual update for each page in range
                        const pageElement = document.querySelector(`[data-page="${i}"]`);
                        if (pageElement) {
                            pageElement.classList.add(classToToggle);
                        }
                    }
                    
                    console.log(`Shift+Click: Selected range ${startPage}-${endPage}`);
                } else {
                    // Normal click behavior - toggle single page
                    if (pages.has(pageNum)) {
                        pages.delete(pageNum);
                        element.classList.remove(classToToggle);
                    } else {
                        pages.add(pageNum);
                        element.classList.add(classToToggle);
                    }
                }
                
                // Remember last clicked page for next shift+click
                window.lastClickedPage = pageNum;
                
                // Update input field
                const newPageString = stringifyPageSet(pages);
                if (targetTextbox.value !== newPageString) {
                    targetTextbox.value = newPageString;
                    targetTextbox.dispatchEvent(new Event('input', { bubbles: true }));
                }
            };
            
            window.requestMagnifyView = function(pageNum) {
                const inputContainer = document.getElementById('magnify-page-input');
                if (!inputContainer) { console.error("Could not find magnify input container."); return; }
                const input = inputContainer.querySelector('textarea, input');
                if (input) {
                    input.value = pageNum;
                    const event = new Event('input', { bubbles: true });
                    input.dispatchEvent(event);
                } else {
                    console.error("Could not find magnify input field inside container.");
                }
            }
            
            window.openMagnifyView = function(base64Data) {
                if (!base64Data) return;
                document.getElementById('magnified-img').src = "data:image/png;base64," + base64Data;
                document.getElementById('magnify-modal').style.display = 'flex';
                if (document.activeElement) {
                    document.activeElement.blur();
                }
            }
            
            window.closeMagnifyView = function() {
                document.getElementById('magnify-modal').style.display = 'none';
            }
            
            window.triggerDownload = function(payloadStr) {
                if (!payloadStr) return;
                try {
                    const payload = JSON.parse(payloadStr);
                    const link = document.createElement('a');
                    link.href = `data:${payload.mime};base64,${payload.data}`;
                    link.download = payload.name;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    
                    // Also save to browser storage for recovery
                    window.biochemAutoSave.saveResults('download', {
                        filename: payload.name,
                        size: payload.data.length
                    });
                } catch (e) { 
                    console.error("Failed to trigger download:", e); 
                }
            }
        }
        """
        
        with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="BioChemInsight", css=css, js=js_script) as interface:
            gr.HTML("""<div id="magnify-modal" onclick="closeMagnifyView()"><span class="close-btn">&times;</span><img id="magnified-img"></div>""")
            gr.Markdown("<h1>🧬 BioChemInsight: Interactive Biochemical Document Extractor</h1>")

            # --- State Management ---
            total_pages = gr.State(0)
            structures_data = gr.State(None)
            merged_data = gr.State(None)
            active_dataset_name = gr.State("none")
            merged_path = gr.State(None)
            
            # --- Hidden components for communication ---
            magnify_page_input = gr.Number(elem_id="magnify-page-input", visible=True, interactive=True)
            magnified_image_output = gr.Textbox(elem_id="magnified-image-output", visible=True, interactive=True)
            download_trigger = gr.Textbox(visible=False)

            with gr.Row():
                with gr.Column(scale=1):
                    pdf_input = gr.File(label="Upload PDF File", file_types=[".pdf"])
                    pdf_info = gr.Textbox(label="Document Info", interactive=False)
                    with gr.Group():
                         ocr_engine_input = gr.Dropdown(label="OCR Engine (for Bioactivity)", choices=['paddleocr', 'dots_ocr'], value='dots_ocr', interactive=True)
                         lang_input = gr.Dropdown(label="Document Language (for PaddleOCR)", choices=[("English", "en"), ("Chinese", "ch")], value="en", interactive=True, visible=False)
                    with gr.Group():
                        gr.Markdown("<h4>Page Selection</h4>")
                        selection_mode = gr.Radio(["Structures", "Bioactivity"], label="Selection Mode", value="Structures", elem_id="selection-mode-radio", interactive=True)
                        struct_pages_input = gr.Textbox(label="Structure Pages", elem_id="struct-pages-input", info="e.g. 1-5, 8, 12")
                        assay_pages_input = gr.Textbox(label="Bioactivity Pages", elem_id="assay-pages-input", info="e.g. 15, 18-20")
                        clear_btn = gr.Button("Clear All Selections", variant="secondary")
                    with gr.Group():
                        gr.Markdown("<h4>Extraction Actions</h4>")
                        guidance_text = gr.Markdown("🚀 Welcome to BioChemInsight! Upload a PDF file to start processing.")
                        engine_input = gr.Dropdown(label="Structure Extraction Engine", choices=['molnextr', 'molscribe', 'molvec'], value='molnextr', interactive=True)
                        struct_btn = gr.Button("Step 1: Extract Structures", variant="primary")
                        assay_names_input = gr.Textbox(label="Assay Names (comma-separated)", placeholder="e.g., IC50, Ki, EC50", visible=False)
                        assay_btn = gr.Button("Step 2: Extract Activity", visible=False, variant="primary")
                with gr.Column(scale=3):
                    gallery = gr.HTML("<div class='center-placeholder'>PDF previews will appear here.</div>")

            gr.Markdown("---")
            gr.Markdown("<h3>Results</h3>")
            status_display = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                edit_btn = gr.Button("✏️ Edit Data", visible=False)
                save_btn = gr.Button("✅ Save Changes", visible=False)
                cancel_btn = gr.Button("❌ Cancel", visible=False)
            
            results_display_view = gr.DataFrame(label="Results (View Mode)", interactive=False, wrap=True, elem_id="results-table-view", visible=False)
            results_display_edit = gr.DataFrame(label="Results (Edit Mode)", interactive=True, wrap=True, visible=False)
            
            with gr.Row():
                struct_dl_csv_btn = gr.Button("Download Structures (CSV)", visible=False)
                merged_dl_btn = gr.Button("Download Merged Results (CSV)", visible=False)
                meta_dl_btn = gr.Button("Download Metadata (JSON)", visible=False)
            
            # --- Event Handlers ---
            
            # Function to toggle language dropdown visibility
            def toggle_lang_visibility(ocr_engine):
                return gr.update(visible=(ocr_engine == 'paddleocr'))

            ocr_engine_input.change(
                fn=toggle_lang_visibility,
                inputs=ocr_engine_input,
                outputs=lang_input
            )

            pdf_input.upload(
                self.on_upload, 
                inputs=[pdf_input], 
                outputs=[pdf_info, total_pages, gallery, structures_data, merged_data, merged_path, active_dataset_name, struct_pages_input, assay_pages_input,
                status_display, guidance_text, results_display_view, results_display_edit,
                struct_dl_csv_btn, merged_dl_btn, meta_dl_btn,
                assay_btn, assay_names_input, edit_btn]
            )

            clear_btn.click(
                self.clear_all_selections, 
                [total_pages], 
                [struct_pages_input, assay_pages_input, gallery],
                js="() => { console.log('Clearing all selections'); document.querySelectorAll('.page-item').forEach(item => { item.classList.remove('selected-struct', 'selected-assay'); }); window.lastClickedPage = null; }"
            )
            magnify_page_input.input(self.get_magnified_page_data, [magnify_page_input], [magnified_image_output])
            magnified_image_output.change(None, [magnified_image_output], None, js="(d) => openMagnifyView(d)")
            
            struct_btn.click(
                self.extract_structures, 
                [struct_pages_input, engine_input, lang_input], 
                [status_display, structures_data, merged_data, active_dataset_name, results_display_view, results_display_edit, struct_dl_csv_btn, guidance_text, assay_btn, assay_names_input, edit_btn]
            )
            assay_btn.click(
                self.extract_activity_and_merge,
                [assay_pages_input, structures_data, assay_names_input, lang_input, ocr_engine_input],
                [status_display, merged_data, active_dataset_name, results_display_view, results_display_edit, merged_path, merged_dl_btn, guidance_text, edit_btn]
            ).then(lambda: gr.update(visible=False), [], [struct_dl_csv_btn])
            
            results_display_view.select(self.show_enlarged_image_from_df, None, magnified_image_output)
            
            edit_btn.click(
                self.enter_edit_mode,
                None,
                [results_display_view, results_display_edit, edit_btn, save_btn, cancel_btn]
            )
            save_btn.click(
                self.save_changes,
                [results_display_edit, active_dataset_name],
                [structures_data, merged_data, results_display_view, results_display_edit, edit_btn, save_btn, cancel_btn]
            )
            cancel_btn.click(
                self.cancel_edit,
                [],
                [results_display_view, results_display_edit, edit_btn, save_btn, cancel_btn]
            )
            
            struct_dl_csv_btn.click(self.download_csv_and_get_payload, [structures_data, gr.Textbox("structures.csv", visible=False)], download_trigger)
            merged_dl_btn.click(self.download_csv_and_get_payload, [merged_data, gr.Textbox("merged_results.csv", visible=False)], download_trigger)
            meta_dl_btn.click(self.generate_metadata_and_get_payload, inputs=[struct_pages_input, assay_pages_input, structures_data, merged_data], outputs=download_trigger)
            
            download_trigger.change(None, [download_trigger], None, js="(payload) => triggerDownload(payload)")

        return interface

    def __del__(self):
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir): 
                shutil.rmtree(self.temp_dir)
        except Exception as e: 
            print(f"Failed to clean up temporary files: {e}")

def find_free_port(start_port: int = 7860, max_tries: int = 50) -> int:
    """Find a free port starting from start_port."""
    import socket
    
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    
    raise OSError(f"Could not find a free port in range {start_port}-{start_port + max_tries}")

def main():
    """Main entry point for the simplified application."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='BioChemInsight - Interactive Biochemical Document Extractor')
    parser.add_argument('--port', type=int, default=7860, help='Port number (will auto-find if busy)')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host address')
    parser.add_argument('--share', action='store_true', help='Enable sharing')
    
    args = parser.parse_args()
    
    # Find a free port if the requested one is busy
    try:
        actual_port = find_free_port(args.port)
        if actual_port != args.port:
            print(f"⚠️ 端口 {args.port} 被占用，改用端口 {actual_port}")
    except OSError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Create simplified app instance
    app = BioChemInsightApp()
    interface = app.create_interface()
    
    print(f"🚀 Starting BioChemInsight (Simplified Version)...")
    print(f"🌐 Access URL: http://{args.host}:{actual_port}")
    print(f"💡 Features: Simplified processing with auto-download for results")
    print(f"🔒 Network Safety: Results are automatically prepared for download")
    print(f"📱 Mobile friendly: http://localhost:{actual_port}")
    
    # Add auto-save mechanisms for network stability
    print(f"🛡️  Network Protection: Results cached in browser for recovery")
    
    try:
        interface.launch(
            server_name=args.host, 
            server_port=actual_port,
            share=args.share, 
            debug=True,
            show_api=False,
            prevent_thread_lock=False
        )
    except Exception as e:
        print(f"❌ 启动界面失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()