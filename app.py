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

# --- Helper function for rendering SMILES to image ---
def smiles_to_img_tag(smiles: str) -> str:
    """Converts a SMILES string to a base64-encoded image tag for markdown."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES"
        # --- Increased image size for better visibility ---
        img = Draw.MolToImage(mol, size=(300, 300), kekulize=True)
        
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f'![structure](data:image/png;base64,{img_str})'
    except Exception as e:
        return f"Error rendering structure: {str(e)}"

class BioChemInsightApp:
    """Encapsulates the core logic and UI of the BioChemInsight application."""
    def __init__(self):
        """Initializes the application and creates a temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.current_pdf_path = None
        self.current_pdf_filename = None
        print(f"Created temporary directory: {self.temp_dir}")


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
        if 'SMILES' in df.columns:
            df['Structure'] = df['SMILES'].apply(smiles_to_img_tag)
        if 'segment_file' in df.columns:
            df['Segment'] = df['segment_file'].apply(self.segment_to_img_tag)
        
        all_cols = df.columns.tolist()
        
        front_cols = ['COMPOUND_ID', 'Structure', 'Segment']
        back_cols = ['SMILES', 'segment_file', 'page']

        present_front = [col for col in front_cols if col in all_cols]
        
        middle_cols = [
            col for col in all_cols 
            if col not in present_front and col not in back_cols
        ]
        
        new_order = present_front + sorted(middle_cols)
        df = df[new_order]
        
        return df

    def get_pdf_info(self, pdf_file: gr.File) -> tuple:
        """Processes the uploaded PDF and returns its basic information."""
        if pdf_file is None:
            return "❌ Please upload a PDF file first", 0
        try:
            pdf_name = os.path.basename(pdf_file.name)
            self.current_pdf_filename = pdf_name
            self.current_pdf_path = os.path.join(self.temp_dir, pdf_name)
            shutil.copy(pdf_file.name, self.current_pdf_path)
            doc = fitz.open(self.current_pdf_path)
            total_pages = doc.page_count
            doc.close()
            info = f"✅ PDF uploaded successfully, containing {total_pages} pages."
            return info, total_pages
        except Exception as e:
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
            
            gallery_html = """<div class="gallery-wrapper"><div class="selection-info-bar"><span style="font-weight: 500;">Hint: Use the 'Selection Mode' toggle, then click pages to select. You can also type page ranges directly.</span></div></div><div id="gallery-container" class="gallery-container">"""
            
            # Loop through all pages instead of a slice
            for page_num in range(1, total_pages + 1):
                page = doc[page_num - 1]
                low_res_pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img_base64 = base64.b64encode(low_res_pix.tobytes("png")).decode()

                page_classes = "page-item"
                if page_num in struct_pages: page_classes += " selected-struct"
                if page_num in assay_pages: page_classes += " selected-assay"
                
                gallery_html += f"""
                <div class="{page_classes}" data-page="{page_num}" onclick="handlePageClick(this)">
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
        """Clears selections and re-renders the full gallery."""
        gallery_html = self.update_gallery_view(total_pages_num, "", "")
        return "", "", gallery_html
    
    def _run_pipeline(self, args: List[str], clear_output: bool = True) -> str:
        output_dir = os.path.join(self.temp_dir, "output")
        if clear_output and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        pipeline_path = "pipeline.py"
        command = [sys.executable, pipeline_path, self.current_pdf_path] + args + ["--output", output_dir]
        print(f"Running command: {' '.join(command)}")
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            print("Pipeline STDOUT:", result.stdout)
            if result.stderr: print("Pipeline STDERR:", result.stderr)
            return output_dir
        except subprocess.CalledProcessError as e:
            print(f"Error running pipeline: {e}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
            raise e
        except FileNotFoundError:
            print(f"Error: 'pipeline.py' not found.")
            raise

    def extract_structures(self, struct_pages_input: str, engine_input: str, lang_input: str) -> tuple:
        """Runs Step 1 (Structure Extraction) and updates the UI."""
        if not self.current_pdf_path:
            return "❌ Please upload a PDF file first", None, None, gr.update(visible=False), "Please upload a PDF to begin.", gr.update(visible=False), gr.update(visible=False)
        if not struct_pages_input:
            return "❌ Please select pages to process for structures.", None, None, gr.update(visible=False), "Please select pages containing chemical structures.", gr.update(visible=False), gr.update(visible=False)
            
        try:
            args = ["--structure-pages", struct_pages_input, "--engine", engine_input, "--lang", lang_input]
            output_dir = self._run_pipeline(args, clear_output=True)
            
            structures_file = os.path.join(output_dir, "structures.csv")
            if os.path.exists(structures_file):
                df = pd.read_csv(structures_file)
                if not df.empty:
                    df_enriched = self._enrich_dataframe_with_images(df)
                    new_guidance = "✅ **Step 1 Complete.** Now, enter assay names, select pages with bioactivity data, and click Step 2."
                    return f"✅ Successfully extracted {len(df)} structures.", df.to_dict('records'), df_enriched, gr.update(visible=True), new_guidance, gr.update(visible=True), gr.update(visible=True)
                else:
                    return "⚠️ Pipeline ran, but no structures were found.", None, None, gr.update(visible=False), "No structures found. Please check your page selection and try again.", gr.update(visible=False), gr.update(visible=False)
            else:
                 return "❌ Pipeline finished, but 'structures.csv' was not created.", None, None, gr.update(visible=False), "An error occurred. The structures file was not generated.", gr.update(visible=False), gr.update(visible=False)
        except Exception as e:
            return f"❌ Error during structure extraction: {str(e)}", None, None, gr.update(visible=False), "An unexpected error occurred. See console for details.", gr.update(visible=False), gr.update(visible=False)

    def extract_activity_and_merge(self, assay_pages_input: str, structures_data: list, assay_names: str, lang_input: str) -> tuple:
        """Runs Step 2 (Activity Extraction and Merge) and updates the UI."""
        if not assay_pages_input: return "❌ Select activity pages.", None, None, None, gr.update(visible=False), gr.update(visible=False), "Please select pages with activity data."
        if not structures_data: return "⚠️ Run Step 1 first.", None, None, None, gr.update(visible=False), gr.update(visible=False), "Structure data missing. Please run Step 1 first."
        if not assay_names: return "❌ Enter assay names.", None, None, None, gr.update(visible=False), gr.update(visible=False), "Please provide the names of the assays to extract (e.g., IC50)."

        try:
            args = ["--assay-pages", assay_pages_input, "--assay-names", assay_names, "--lang", lang_input]
            output_dir = self._run_pipeline(args, clear_output=False)
            
            merged_file = os.path.join(output_dir, "merged.csv")
            
            if not os.path.exists(merged_file):
                return "❌ Merge failed. 'merged.csv' not found.", None, None, None, gr.update(visible=False), gr.update(visible=False), "Merge step failed."

            df = pd.read_csv(merged_file)
            if df.empty:
                return "⚠️ Merge complete, but the resulting file is empty.", None, None, None, gr.update(visible=False), gr.update(visible=False), "The pipeline found no data to merge."
            
            df_enriched = self._enrich_dataframe_with_images(df)
            status_msg = f"✅ Merge successful. Generated {len(df)} merged records."
            
            return (
                status_msg, df.to_dict('records'), df_enriched,
                merged_file, 
                gr.update(visible=True), # merged_dl_btn
                gr.update(visible=True), # meta_dl_btn
                "✅ **Process Complete!** View and download results below."
            )

        except Exception as e:
            return f"❌ Error: {repr(e)}", None, None, None, gr.update(visible=False), gr.update(visible=False), "An unexpected error occurred."
    
    def _prepare_download_payload(self, filename: str, mime_type: str, data_bytes: bytes) -> str:
        """Creates a JSON string with Base64 encoded data for client-side download."""
        if not data_bytes:
            return None
        b64_data = base64.b64encode(data_bytes).decode('utf-8')
        return json.dumps({"name": filename, "mime": mime_type, "data": b64_data})

    def download_csv_and_get_payload(self, data: list, filename: str) -> str:
        """Generates a CSV file, reads it, and returns the JSON payload for download."""
        if data is None or not isinstance(data, list):
            print(f"Download triggered for '{filename}', but data was invalid. Aborting.")
            gr.Warning(f"No data available to download for '{filename}'.")
            return None
        
        df = pd.DataFrame(data)
        cols_to_drop = ['Structure', 'Segment']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        
        print(f"Preparing in-memory CSV for {filename}...")
        gr.Info(f"Preparing download for {filename}...")
        return self._prepare_download_payload(filename, 'text/csv', csv_bytes)

    def generate_metadata_and_get_payload(self, struct_pages_str, assay_pages_str, structures_data, merged_data) -> str:
        """Creates a meta.json file in memory and returns the JSON payload for download."""
        if not self.current_pdf_path:
            print("Cannot generate metadata, PDF path not set.")
            return None
        
        metadata = {
            "source_file": self.current_pdf_filename,
            "output_directory": os.path.join(self.temp_dir, "output"),
            "processing_details": {
                "structure_extraction": {
                    "pages_processed": struct_pages_str or "None",
                    "structures_found": len(structures_data) if structures_data else 0
                },
                "bioactivity_extraction": {
                    "pages_processed": assay_pages_str or "None",
                    "merged_records_found": len(merged_data) if merged_data else 0
                }
            }
        }
        
        json_bytes = json.dumps(metadata, indent=4).encode('utf-8')
        
        print(f"Preparing in-memory JSON for meta.json...")
        gr.Info(f"Preparing download for meta.json...")
        return self._prepare_download_payload("meta.json", "application/json", json_bytes)

    def on_upload(self, pdf_file: gr.File) -> tuple:
        """Handles PDF upload and generates the initial full gallery."""
        info, pages = self.get_pdf_info(pdf_file)
        gallery_html = self.update_gallery_view(pages, "", "")
        guidance = "✅ PDF loaded. **Step 1:** Please select pages containing **chemical structures** and click the button below."
        
        # Updated return tuple to remove components related to page navigation
        return (
            info, pages, gallery_html,
            None, None, None, "", "",
            "Status...", guidance, gr.update(value=None),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
            gr.update(visible=False), gr.update(visible=False)
        )

    def create_interface(self):
        # --- CSS can be simplified by removing page navigation styles ---
        css = """
        #magnify-page-input, #magnified-image-output { display: none; }
        .center-placeholder { text-align: center; padding: 50px; border: 2px dashed #ccc; border-radius: 10px; margin-top: 20px; }
        .gallery-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px; max-height: 80vh; overflow-y: auto; } /* Increased height */
        .page-item { border: 3px solid #ddd; border-radius: 8px; cursor: pointer; position: relative; background: white; transition: border-color 0.2s; }
        .page-item.selected-struct { border-color: #28a745; }
        .page-item.selected-assay { border-color: #007bff; }
        .page-item.selected-struct.selected-assay { border-image: linear-gradient(45deg, #28a745, #007bff) 1; }
        .selection-check { position: absolute; top: 8px; width: 24px; height: 24px; border-radius: 50%; color: white; display: none; align-items: center; justify-content: center; font-weight: bold; z-index: 2; font-size: 14px; }
        .page-item.selected-struct .struct-check { display: flex; right: 8px; background: #28a745; }
        .page-item.selected-assay .assay-check { display: flex; right: 38px; background: #007bff; }
        .page-item .magnify-icon { position: absolute; top: 8px; left: 8px; width: 24px; height: 24px; background: rgba(0,0,0,0.5); color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: zoom-in; z-index: 2;}
        .page-item .page-label { text-align: center; padding: 8px; }
        #magnify-modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.8); justify-content: center; align-items: center; }
        #magnify-modal img { max-width: 90%; max-height: 90%; } #magnify-modal .close-btn { position: absolute; top: 20px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; cursor: pointer; }
        """
        
        js_script = """
        () => {
            const parsePageString = (str) => {
                const pages = new Set();
                if (!str) return pages;
                str.split(',').forEach(part => {
                    part = part.trim();
                    if (part.includes('-')) {
                        const [start, end] = part.split('-').map(Number);
                        if (!isNaN(start) && !isNaN(end)) {
                            for (let i = Math.min(start, end); i <= Math.max(start, end); i++) pages.add(i);
                        }
                    } else {
                        const num = Number(part);
                        if (!isNaN(num)) pages.add(num);
                    }
                });
                return pages;
            };

            const stringifyPageSet = (pageSet) => {
                const ranges = [];
                let sortedPages = Array.from(pageSet).sort((a, b) => a - b);
                let i = 0;
                while (i < sortedPages.length) {
                    let start = sortedPages[i];
                    let j = i;
                    while (j + 1 < sortedPages.length && sortedPages[j + 1] === sortedPages[j] + 1) {
                        j++;
                    }
                    if (j === i) {
                        ranges.push(start.toString());
                    } else if (j === i + 1) {
                         ranges.push(start.toString(), sortedPages[j].toString());
                    }
                    else {
                        ranges.push(`${start}-${sortedPages[j]}`);
                    }
                    i = j + 1;
                }
                return ranges.join(',');
            };

            window.handlePageClick = function(element) {
                const pageNum = parseInt(element.getAttribute('data-page'));
                const modeRadio = document.querySelector('#selection-mode-radio input:checked');
                if (!modeRadio) return;
                const mode = modeRadio.value;

                const classToToggle = (mode === 'Structures') ? 'selected-struct' : 'selected-assay';
                element.classList.toggle(classToToggle);

                const targetId = (mode === 'Structures') ? '#struct-pages-input' : '#assay-pages-input';
                const targetTextbox = document.querySelector(targetId).querySelector('textarea, input');
                
                let pages = parsePageString(targetTextbox.value);
                
                if (pages.has(pageNum)) {
                    pages.delete(pageNum);
                } else {
                    pages.add(pageNum);
                }
                
                const newPageString = stringifyPageSet(pages);
                
                if (targetTextbox.value !== newPageString) {
                    targetTextbox.value = newPageString;
                    targetTextbox.dispatchEvent(new Event('input', { bubbles: true }));
                }
            };

            window.requestMagnifyView = function(pageNum) {
                const inputContainer = document.getElementById('magnify-page-input');
                const input = inputContainer.querySelector('textarea, input');
                if (input) {
                    input.value = pageNum;
                    const event = new Event('input', { bubbles: true });
                    input.dispatchEvent(event);
                } else {
                    console.error("Could not find magnify input field.");
                }
            }

            window.openMagnifyView = function(base64Data) {
                if (!base64Data) return;
                document.getElementById('magnified-img').src = "data:image/png;base64," + base64Data;
                document.getElementById('magnify-modal').style.display = 'flex';
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
                } catch (e) {
                    console.error("Failed to trigger download:", e);
                }
            }
        }
        """
        with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="BioChemInsight", css=css, js=js_script) as interface:
            gr.HTML("""<div id="magnify-modal" onclick="closeMagnifyView()"><span class="close-btn">&times;</span><img id="magnified-img"></div>""")
            gr.Markdown("<h1>🧬 BioChemInsight: Interactive Biochemical Document Extractor</h1>")

            total_pages = gr.State(0)
            structures_data, merged_data, merged_path = gr.State(None), gr.State(None), gr.State(None)
            
            magnify_page_input = gr.Number(elem_id="magnify-page-input", visible=True, interactive=True)
            magnified_image_output = gr.Textbox(elem_id="magnified-image-output", visible=True, interactive=True)
            
            download_trigger = gr.Textbox(visible=False)

            with gr.Row():
                with gr.Column(scale=1):
                    pdf_input = gr.File(label="Upload PDF File", file_types=[".pdf"])
                    pdf_info = gr.Textbox(label="Document Info", interactive=False)
                    
                    with gr.Group():
                         lang_input = gr.Dropdown(
                            label="Document Language",
                            choices=[("English", "en"), ("Chinese", "ch")],
                            value="en",
                            interactive=True
                        )

                    with gr.Group():
                        gr.Markdown("<h4>Page Selection</h4>")
                        selection_mode = gr.Radio(["Structures", "Bioactivity"], label="Selection Mode", value="Structures", elem_id="selection-mode-radio", interactive=True)
                        struct_pages_input = gr.Textbox(label="Structure Pages", elem_id="struct-pages-input", info="e.g. 1-5, 8, 12")
                        assay_pages_input = gr.Textbox(label="Bioactivity Pages", elem_id="assay-pages-input", info="e.g. 15, 18-20")
                        clear_btn = gr.Button("Clear All Selections", variant="secondary")
                        
                    with gr.Group():
                        gr.Markdown("<h4>Extraction Actions</h4>")
                        guidance_text = gr.Markdown("Please upload a PDF to begin.")
                        # engine_input = gr.Radio(
                        #     label="Structure Extraction Engine",
                        #     choices=['molnextr', 'molscribe', 'molvec'],
                        #     value='molnextr',
                        #     interactive=True
                        # )
                        engine_input = gr.Dropdown(
                            label="Structure Extraction Engine",
                            choices=['molnextr', 'molscribe', 'molvec'],
                            value='molnextr',
                            interactive=True
                        )
                        struct_btn = gr.Button("Step 1: Extract Structures", variant="primary")
                        assay_names_input = gr.Textbox(label="Assay Names (comma-separated)", placeholder="e.g., IC50, Ki, EC50", visible=False)
                        assay_btn = gr.Button("Step 2: Extract Activity", visible=False, variant="primary")

                with gr.Column(scale=3):
                    gallery = gr.HTML("<div class='center-placeholder'>PDF previews will appear here.</div>")

            gr.Markdown("---")
            gr.Markdown("<h3>Results</h3>")
            
            status_display = gr.Textbox(label="Status", interactive=False)
            results_display = gr.DataFrame(label="Results", interactive=True, wrap=True, datatype=["markdown", "markdown"])
            
            with gr.Row():
                struct_dl_csv_btn = gr.Button("Download Structures (CSV)", visible=False)
                merged_dl_btn = gr.Button("Download Merged Results (CSV)", visible=False)
                meta_dl_btn = gr.Button("Download Metadata (JSON)", visible=False)

            # Updated outputs list for the upload event
            pdf_input.upload(self.on_upload, [pdf_input], [
                pdf_info, total_pages, gallery,
                structures_data, merged_data, merged_path, struct_pages_input, assay_pages_input,
                status_display, guidance_text, results_display,
                struct_dl_csv_btn, merged_dl_btn, meta_dl_btn,
                assay_btn, assay_names_input
            ])

            # Updated inputs list for the clear button event
            clear_btn.click(self.clear_all_selections, [total_pages], [struct_pages_input, assay_pages_input, gallery])
            
            magnify_page_input.input(self.get_magnified_page_data, [magnify_page_input], [magnified_image_output])
            magnified_image_output.change(None, [magnified_image_output], None, js="(d) => openMagnifyView(d)")
            
            struct_btn.click(
                self.extract_structures, 
                [struct_pages_input, engine_input, lang_input], 
                [status_display, structures_data, results_display, struct_dl_csv_btn, guidance_text, assay_btn, assay_names_input]
            )
            
            assay_btn.click(
                self.extract_activity_and_merge,
                [assay_pages_input, structures_data, assay_names_input, lang_input],
                [status_display, merged_data, results_display, merged_path, merged_dl_btn, meta_dl_btn, guidance_text]
            ).then(
                lambda: gr.update(visible=False), 
                [],
                [struct_dl_csv_btn]
            )
            
            struct_dl_csv_btn.click(
                self.download_csv_and_get_payload, 
                [structures_data, gr.Textbox("structures.csv", visible=False)], 
                download_trigger
            )
            merged_dl_btn.click(
                self.download_csv_and_get_payload, 
                [merged_data, gr.Textbox("merged_results.csv", visible=False)], 
                download_trigger
            )
            meta_dl_btn.click(
                self.generate_metadata_and_get_payload,
                inputs=[struct_pages_input, assay_pages_input, structures_data, merged_data],
                outputs=download_trigger
            )
            
            download_trigger.change(None, [download_trigger], None, js="(payload) => triggerDownload(payload)")

        return interface

    def __del__(self):
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir): 
                shutil.rmtree(self.temp_dir)
        except Exception as e: 
            print(f"Failed to clean up temporary files: {e}")

def main():
    app = BioChemInsightApp()
    interface = app.create_interface()
    interface.launch(server_name="0.0.0.0", share=False, debug=True)

if __name__ == "__main__":
    main()