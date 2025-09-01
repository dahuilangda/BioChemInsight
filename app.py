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
import uuid
import threading
import time
from datetime import datetime, timedelta

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

# --- Session Management ---
class SessionManager:
    """Manages user sessions with UUID-based state persistence."""
    
    def __init__(self, session_dir: str = "./sessions"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)
        self.active_sessions = {}
        self.cleanup_interval = 3600  # 1 hour cleanup interval
        self.session_timeout = 86400  # 24 hours session timeout
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_sessions, daemon=True)
        self.cleanup_thread.start()
    
    def create_session(self) -> str:
        """Creates a new session and returns its UUID."""
        session_id = str(uuid.uuid4())
        session_path = self.session_dir / session_id
        session_path.mkdir(exist_ok=True)
        
        session_data = {
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'status': 'created',
            'current_pdf_path': None,
            'structures_data': None,
            'merged_data': None,
            'progress': {}
        }
        
        self._save_session_data(session_id, session_data)
        self.active_sessions[session_id] = session_data
        return session_id
    
    def get_session(self, session_id: str) -> dict:
        """Retrieves session data by UUID."""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try to load from disk
        session_path = self.session_dir / session_id
        if session_path.exists():
            try:
                with open(session_path / "session.json", 'r') as f:
                    session_data = json.load(f)
                self.active_sessions[session_id] = session_data
                return session_data
            except Exception as e:
                print(f"Failed to load session {session_id}: {e}")
        
        return None
    
    def update_session(self, session_id: str, updates: dict):
        """Updates session data."""
        session_data = self.get_session(session_id)
        if session_data:
            session_data.update(updates)
            session_data['last_activity'] = datetime.now().isoformat()
            self._save_session_data(session_id, session_data)
            self.active_sessions[session_id] = session_data
    
    def _save_session_data(self, session_id: str, session_data: dict):
        """Saves session data to disk."""
        session_path = self.session_dir / session_id
        session_path.mkdir(exist_ok=True)
        
        with open(session_path / "session.json", 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def _cleanup_expired_sessions(self):
        """Background thread to cleanup expired sessions."""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(seconds=self.session_timeout)
                
                for session_id in list(self.active_sessions.keys()):
                    session_data = self.active_sessions[session_id]
                    last_activity = datetime.fromisoformat(session_data['last_activity'])
                    
                    if last_activity < cutoff_time:
                        # Remove from memory
                        del self.active_sessions[session_id]
                        
                        # Remove from disk
                        session_path = self.session_dir / session_id
                        if session_path.exists():
                            shutil.rmtree(session_path)
                
                # Cleanup disk sessions not in memory
                for session_path in self.session_dir.iterdir():
                    if session_path.is_dir():
                        session_file = session_path / "session.json"
                        if session_file.exists():
                            try:
                                with open(session_file, 'r') as f:
                                    session_data = json.load(f)
                                last_activity = datetime.fromisoformat(session_data['last_activity'])
                                
                                if last_activity < cutoff_time:
                                    shutil.rmtree(session_path)
                            except Exception:
                                # If we can't read the session file, remove it
                                shutil.rmtree(session_path)
                
            except Exception as e:
                print(f"Session cleanup error: {e}")
            
            time.sleep(self.cleanup_interval)

# Global session manager
session_manager = SessionManager()

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
    """Encapsulates the core logic and UI of the BioChemInsight application."""
    def __init__(self, session_id: str = None):
        """Initializes the application and creates a temporary directory."""
        self.session_id = session_id  # Don't create session automatically
        self.temp_dir = tempfile.mkdtemp()
        self.current_pdf_path = None
        self.current_pdf_filename = None
        
        # Only load session data if session_id is provided
        if self.session_id:
            session_data = session_manager.get_session(self.session_id)
            if session_data:
                print(f"Debug: Restoring session {self.session_id[:8]}...")
                # Try to restore PDF path from session
                pdf_path = session_data.get('current_pdf_path')
                if pdf_path and os.path.exists(pdf_path):
                    self.current_pdf_path = pdf_path
                    self.current_pdf_filename = session_data.get('current_pdf_filename', os.path.basename(pdf_path))
                    print(f"Debug: Restored PDF: {self.current_pdf_filename}")
                else:
                    print(f"Debug: Session {self.session_id[:8]} has no valid PDF file")
            else:
                print(f"Debug: Session {self.session_id[:8]} not found, will create new session on first action")

    def _update_session_state(self, **kwargs):
        """Updates the session state with new data. Creates session if not exists."""
        # Create session on first update (usually when PDF is uploaded)
        if not self.session_id:
            self.session_id = session_manager.create_session()
            print(f"Debug: Created new session: {self.session_id}")
        
        session_manager.update_session(self.session_id, kwargs)
    
    def get_session_status(self) -> dict:
        """Gets current session status and progress."""
        if not self.session_id:
            return {'session_id': None, 'status': 'no_session'}
            
        session_data = session_manager.get_session(self.session_id)
        if session_data:
            return {
                'session_id': self.session_id,
                'status': session_data.get('status', 'created'),
                'progress': session_data.get('progress', {}),
                'has_pdf': bool(session_data.get('current_pdf_path')),
                'has_structures': bool(session_data.get('structures_data')),
                'has_merged': bool(session_data.get('merged_data'))
            }
        return {'session_id': self.session_id, 'status': 'new'}
    
    def restore_session_data(self) -> tuple:
        """Restores session data from saved state."""
        session_data = session_manager.get_session(self.session_id)
        if not session_data:
            return "❌ No session data found", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "No session data available."
        
        try:
            status_msg = f"🔄 Restoring session data (Status: {session_data.get('status', 'unknown')})"
            
            # Restore structures data if available
            structures_data = session_data.get('structures_data')
            merged_data = session_data.get('merged_data')
            
            structures_update = gr.update(value=structures_data) if structures_data else gr.update()
            merged_update = gr.update(value=merged_data) if merged_data else gr.update()
            
            # Show appropriate UI elements based on session state
            status = session_data.get('status', 'created')
            if status == 'structures_extracted':
                guidance = f"✅ Session restored. Structures data available. Continue with Step 2 (Session: {self.session_id[:8]}...)."
                return status_msg, structures_update, merged_update, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), guidance
            elif status == 'completed':
                guidance = f"✅ Session restored. Process completed. All data available (Session: {self.session_id[:8]}...)."
                return status_msg, structures_update, merged_update, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), guidance
            else:
                guidance = f"✅ Session restored (Status: {status}) (Session: {self.session_id[:8]}...)."
                return status_msg, structures_update, merged_update, gr.update(), gr.update(), gr.update(), guidance
                
        except Exception as e:
            return f"❌ Error restoring session: {str(e)}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "Failed to restore session data."

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
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            print("Pipeline STDOUT:", result.stdout)
            if result.stderr: print("Pipeline STDERR:", result.stderr)
            return output_dir
        except subprocess.CalledProcessError as e:
            print(f"Error running pipeline: {e}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
            raise e
        except FileNotFoundError:
            print(f"Error: '{pipeline_path}' not found.")
            raise

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
            # Update session with current progress
            self._update_session_state(
                status='extracting_structures',
                progress={'step': 'extracting_structures', 'struct_pages': struct_pages_input}
            )
            
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

                    # Update session with results
                    self._update_session_state(
                        status='structures_extracted',
                        structures_data=df.to_dict('records'),
                        progress={'step': 'structures_extracted', 'struct_count': len(df)}
                    )

                    new_guidance = f"✅ **Step 1 Complete** (Session: {self.session_id[:8]}...). Now, enter assay names, select pages with bioactivity data, and click Step 2."
                    return f"✅ Extracted {len(df)} structures.", df.to_dict('records'), None, "structures", view_df_update, edit_df_update, gr.update(visible=True), new_guidance, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
                else:
                    return "⚠️ No structures found.", None, None, "none", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "No structures found.", gr.update(), gr.update(), gr.update()
            else:
                 return "❌ 'structures.csv' not created.", None, None, "none", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "File not generated.", gr.update(), gr.update(), gr.update()
        except Exception as e:
            self._update_session_state(
                status='error',
                progress={'step': 'error', 'error': str(e)}
            )
            return f"❌ Error: {str(e)}", None, None, "none", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "An error occurred.", gr.update(), gr.update(), gr.update()

    def extract_activity_and_merge(self, assay_pages_input: str, structures_data: list, assay_names: str, lang_input: str, ocr_engine_input: str) -> tuple:
        if not assay_pages_input: return "❌ Select activity pages.", gr.update(), "merged", gr.update(), gr.update(), gr.update(), gr.update(), "Select activity pages.", gr.update()
        if not structures_data: return "⚠️ Run Step 1 first.", gr.update(), "merged", gr.update(), gr.update(), gr.update(), gr.update(), "Run Step 1 first.", gr.update()
        if not assay_names: return "❌ Enter assay names.", gr.update(), "merged", gr.update(), gr.update(), gr.update(), gr.update(), "Enter assay names.", gr.update()
        try:
            # Update session with current progress
            self._update_session_state(
                status='extracting_activity',
                progress={'step': 'extracting_activity', 'assay_pages': assay_pages_input, 'assay_names': assay_names}
            )
            
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

            # Update session with final results
            self._update_session_state(
                status='completed',
                merged_data=df.to_dict('records'),
                progress={'step': 'completed', 'records_count': len(df)}
            )

            status_msg = f"✅ Merge successful. Generated {len(df)} records (Session: {self.session_id[:8]}...)."
            return (status_msg, df.to_dict('records'), "merged", view_df_update, edit_df_update, merged_file, 
                    gr.update(visible=True), 
                    f"✅ **Process Complete!** (Session: {self.session_id[:8]}...) View and download results below.", gr.update(visible=True))
        except Exception as e:
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

    def set_session_id(self, session_id: str) -> tuple:
        """Updates the app's session ID and tries to restore session data."""
        if session_id and session_id != self.session_id:
            self.session_id = session_id
            print(f"Debug: Setting session ID to {session_id[:8]}...")
            
            # Try to restore session data
            session_data = session_manager.get_session(session_id)
            if session_data:
                print(f"Debug: Found existing session data for {session_id[:8]}")
                
                # Restore PDF if available
                pdf_path = session_data.get('current_pdf_path')
                pdf_filename = session_data.get('current_pdf_filename')
                
                if pdf_path and os.path.exists(pdf_path):
                    self.current_pdf_path = pdf_path
                    self.current_pdf_filename = pdf_filename
                    
                    # Generate page count for UI
                    try:
                        import fitz
                        doc = fitz.open(pdf_path)
                        total_pages = doc.page_count
                        doc.close()
                        
                        print(f"Debug: Successfully restored PDF with {total_pages} pages")
                        
                        # Get status and restore appropriate UI state
                        status = session_data.get('status', 'created')
                        structures_data = session_data.get('structures_data')
                        merged_data = session_data.get('merged_data')
                        
                        # Generate gallery view
                        gallery_html = self.update_gallery_view(total_pages, "", "")
                        
                        # Prepare UI updates based on session status
                        if status == 'structures_extracted' and structures_data:
                            df = pd.DataFrame(structures_data)
                            df_enriched = self._enrich_dataframe_with_images(df.copy())
                            datatypes = self._get_df_dtypes(df_enriched)
                            guidance = f"✅ Session restored with {len(structures_data)} structures. Continue with Step 2 (Session: {session_id[:8]}...)."
                            
                            return (
                                f"✅ Session {session_id[:8]} restored with {total_pages} page PDF", 
                                total_pages, gallery_html, structures_data, None, None, "structures",
                                "", "", f"Structures found: {len(structures_data)}", guidance,
                                gr.update(value=df_enriched, datatype=datatypes, visible=True), 
                                gr.update(value=df.drop(columns=['Structure', 'Segment', 'Image File'], errors='ignore'), visible=False),
                                gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                                gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), 
                                session_id
                            )
                        elif status == 'completed' and merged_data:
                            df = pd.DataFrame(merged_data)
                            df_enriched = self._enrich_dataframe_with_images(df.copy())
                            datatypes = self._get_df_dtypes(df_enriched)
                            guidance = f"✅ Session completely restored with {len(merged_data)} merged records (Session: {session_id[:8]}...)."
                            
                            return (
                                f"✅ Session {session_id[:8]} completely restored with {total_pages} page PDF", 
                                total_pages, gallery_html, structures_data, merged_data, None, "merged",
                                "", "", f"Process completed: {len(merged_data)} records", guidance,
                                gr.update(value=df_enriched, datatype=datatypes, visible=True), 
                                gr.update(value=df.drop(columns=['Structure', 'Segment', 'Image File'], errors='ignore'), visible=False),
                                gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                                gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), 
                                session_id
                            )
                        else:
                            # PDF uploaded but no processing done yet
                            guidance = f"✅ Session {session_id[:8]} restored. PDF loaded, ready to begin Step 1."
                            return (
                                f"✅ Session {session_id[:8]} restored with PDF: {pdf_filename}", 
                                total_pages, gallery_html, None, None, None, "none",
                                "", "", "PDF loaded", guidance,
                                gr.update(value=None, visible=False), 
                                gr.update(value=None, visible=False),
                                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                                session_id
                            )
                            
                    except Exception as e:
                        print(f"Debug: Error reading restored PDF: {e}")
                        return ("⚠️ Session found but PDF is not accessible", 0, "<div class='center-placeholder'>PDF file is not accessible</div>", 
                                None, None, None, "none", "", "", "PDF access error", f"Session {session_id[:8]} found but PDF file is missing",
                                gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "")
                else:
                    return (f"✅ Session {session_id[:8]} restored (no PDF)", 0, "<div class='center-placeholder'>No PDF in session</div>", 
                            None, None, None, "none", "", "", "No PDF data", f"Session {session_id[:8]} restored but contains no PDF data",
                            gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), session_id)
            else:
                return (f"⚠️ Session {session_id[:8]} not found", 0, "<div class='center-placeholder'>Session not found</div>", 
                        None, None, None, "none", "", "", "Session not found", f"Session {session_id[:8]} not found - will create new session on next upload",
                        gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "")
        
        return ("No session change", 0, "<div class='center-placeholder'>No session change</div>", 
                None, None, None, "none", "", "", "No change", "No session change needed",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "")

    def check_session_on_load(self) -> tuple:
        """Checks for existing session and restores UI state on initial page load."""
        print(f"Debug: check_session_on_load called, current session_id: {self.session_id}")
        
        # If no session ID set, return default state
        if not self.session_id:
            print("Debug: No session ID, returning default state")
            return (
                "🚀 Welcome to BioChemInsight! Please upload a PDF file to start processing. A new session will be created automatically after upload.",
                0, "<div class='center-placeholder'>PDF previews will appear here.</div>", 
                None, None, None, "none", "", "", "Status...", 
                "🚀 Welcome to BioChemInsight! Please upload a PDF file to start processing. A new session will be created automatically after upload.",
                gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            )
        
        # If session ID exists, try to restore session
        print(f"Debug: Attempting to restore session {self.session_id[:8]}...")
        session_data = session_manager.get_session(self.session_id)
        
        if not session_data:
            print(f"Debug: Session {self.session_id[:8]} not found")
            return (
                f"⚠️ Session {self.session_id[:8]} not found - will create new session on upload",
                0, "<div class='center-placeholder'>Session not found. Please upload a PDF file.</div>", 
                None, None, None, "none", "", "", "Session not found", 
                f"Session {self.session_id[:8]} not found - will create new session on next upload",
                gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            )
        
        print(f"Debug: Found session data for {self.session_id[:8]}")
        
        # Check if session has PDF data
        pdf_path = session_data.get('current_pdf_path')
        pdf_filename = session_data.get('current_pdf_filename')
        
        if not pdf_path or not os.path.exists(pdf_path):
            print(f"Debug: Session {self.session_id[:8]} has no valid PDF file")
            return (
                f"✅ Session {self.session_id[:8]} found but no PDF data",
                0, "<div class='center-placeholder'>Please upload a PDF file to continue with this session.</div>", 
                None, None, None, "none", "", "", "No PDF in session", 
                f"Session {self.session_id[:8]} restored - upload a PDF to continue",
                gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            )
        
        # Restore PDF data
        self.current_pdf_path = pdf_path
        self.current_pdf_filename = pdf_filename
        
        try:
            import fitz
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            doc.close()
            
            print(f"Debug: Successfully restored PDF {pdf_filename} with {total_pages} pages")
            
            # Generate gallery
            gallery_html = self.update_gallery_view(total_pages, "", "")
            
            # Get session status and data
            status = session_data.get('status', 'pdf_uploaded')
            structures_data = session_data.get('structures_data')
            merged_data = session_data.get('merged_data')
            
            print(f"Debug: Session status: {status}, structures: {bool(structures_data)}, merged: {bool(merged_data)}")
            
            # Restore UI based on session status
            if status == 'completed' and merged_data:
                df = pd.DataFrame(merged_data)
                df_enriched = self._enrich_dataframe_with_images(df.copy())
                datatypes = self._get_df_dtypes(df_enriched)
                guidance = f"✅ Session {self.session_id[:8]} fully restored - Process completed with {len(merged_data)} records!"
                
                return (
                    f"✅ Session {self.session_id[:8]} restored: {pdf_filename} ({total_pages} pages)",
                    total_pages, gallery_html, structures_data, merged_data, None, "merged",
                    "", "", f"Process completed: {len(merged_data)} records", guidance,
                    gr.update(value=df_enriched, datatype=datatypes, visible=True), 
                    gr.update(value=df.drop(columns=['Structure', 'Segment', 'Image File'], errors='ignore'), visible=False),
                    gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                    gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
                )
            elif status == 'structures_extracted' and structures_data:
                df = pd.DataFrame(structures_data)
                df_enriched = self._enrich_dataframe_with_images(df.copy())
                datatypes = self._get_df_dtypes(df_enriched)
                guidance = f"✅ Session {self.session_id[:8]} restored with {len(structures_data)} structures - Continue with Step 2!"
                
                return (
                    f"✅ Session {self.session_id[:8]} restored: {pdf_filename} ({total_pages} pages)",
                    total_pages, gallery_html, structures_data, None, None, "structures",
                    "", "", f"Structures found: {len(structures_data)}", guidance,
                    gr.update(value=df_enriched, datatype=datatypes, visible=True), 
                    gr.update(value=df.drop(columns=['Structure', 'Segment', 'Image File'], errors='ignore'), visible=False),
                    gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                    gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
                )
            else:
                # PDF uploaded but no processing done yet
                guidance = f"✅ Session {self.session_id[:8]} restored with PDF loaded - Ready to begin Step 1!"
                return (
                    f"✅ Session {self.session_id[:8]} restored: {pdf_filename} ({total_pages} pages)",
                    total_pages, gallery_html, None, None, None, "none",
                    "", "", "PDF loaded", guidance,
                    gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                )
                
        except Exception as e:
            print(f"Debug: Error restoring PDF: {e}")
            return (
                f"⚠️ Session {self.session_id[:8]} found but PDF error: {str(e)}",
                0, "<div class='center-placeholder'>PDF file is not accessible</div>", 
                None, None, None, "none", "", "", "PDF access error", 
                f"Session {self.session_id[:8]} found but PDF file has issues",
                gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            )

    def get_current_session_id(self) -> str:
        """Returns the current session ID for UI updates."""
        return self.session_id or ""

    def on_upload(self, pdf_file) -> tuple:
        """Handles PDF upload and generates the initial full gallery."""
        print(f"Debug: on_upload called with: {pdf_file}, type: {type(pdf_file)}")
        
        # Basic validation first
        if pdf_file is None:
            print("Debug: pdf_file is None in on_upload")
            return ("❌ Please upload a PDF file", 0, "<div class='center-placeholder'>Please upload a PDF file</div>", 
                    None, None, None, "none", "", "", "Please upload a valid PDF file", "Please upload a PDF file to start processing.", 
                    gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "")
        
        info, pages = self.get_pdf_info(pdf_file)
        
        # Check if PDF processing succeeded
        if pages == 0 or not self.current_pdf_path:
            print(f"Debug: PDF processing failed. Pages: {pages}, Path: {self.current_pdf_path}")
            return (info, 0, "<div class='center-placeholder'>PDF processing failed, please re-upload</div>", 
                    None, None, None, "none", "", "", "PDF processing failed", "Please re-upload the PDF file.", 
                    gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "")
        
        print(f"Debug: PDF uploaded successfully. Path: {self.current_pdf_path}, Pages: {pages}")
        
        # Update session state
        try:
            self._update_session_state(
                current_pdf_path=self.current_pdf_path,
                current_pdf_filename=self.current_pdf_filename,
                status='pdf_uploaded',
                progress={'step': 'pdf_uploaded', 'total_pages': pages}
            )
        except Exception as e:
            print(f"Debug: Session update failed: {e}")
        
        # Generate gallery
        try:
            gallery_html = self.update_gallery_view(pages, "", "")
        except Exception as e:
            print(f"Debug: Gallery generation failed: {e}")
            gallery_html = "<div class='center-placeholder'>页面预览生成失败</div>"
        
        guidance = "✅ PDF uploaded successfully. **Step 1:** Select pages containing **chemical structures** and click the button below."
        return (info, pages, gallery_html, None, None, None, "none", "", "", "Status...", guidance, 
                gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                self.session_id or "")

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
        # Check for session ID from URL on interface creation
        def get_url_session_id():
            """This will be replaced by JavaScript to get session ID from URL"""
            return ""
        
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
            // Initialize session restoration on page load
            window.initializeSession = function() {
                const urlSessionId = window.getSessionIdFromURL();
                console.log('initializeSession called, URL session ID:', urlSessionId);
                
                if (urlSessionId) {
                    console.log('Found session ID in URL:', urlSessionId);
                    
                    // Update the hidden session input with URL session ID
                    const hiddenInput = document.getElementById('session-id-hidden');
                    if (hiddenInput) {
                        const textarea = hiddenInput.querySelector('textarea');
                        if (textarea) {
                            console.log('Current textarea value:', textarea.value);
                            console.log('URL session ID:', urlSessionId);
                            
                            if (textarea.value !== urlSessionId) {
                                textarea.value = urlSessionId;
                                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                                console.log('Updated hidden input with session ID from URL');
                            }
                        } else {
                            console.log('Textarea not found in hidden input');
                        }
                    } else {
                        console.log('Hidden input element not found');
                    }
                    
                    // Trigger initial load to restore session state
                    setTimeout(() => {
                        const loadButton = document.querySelector('#initial-load-trigger button');
                        if (loadButton) {
                            console.log('Clicking initial load button');
                            loadButton.click();
                        } else {
                            console.log('Initial load button not found');
                        }
                    }, 1000);
                } else {
                    console.log('No session ID found in URL - triggering initial load for fresh start');
                    setTimeout(() => {
                        const loadButton = document.querySelector('#initial-load-trigger button');
                        if (loadButton) {
                            console.log('Clicking initial load button for fresh start');
                            loadButton.click();
                        }
                    }, 500);
                }
            };
            
            // Run session initialization when page loads - multiple strategies to ensure it runs
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', window.initializeSession);
            } else {
                // DOM already loaded
                setTimeout(window.initializeSession, 100);
            }
            
            // Also run when Gradio finishes loading (backup)
            setTimeout(window.initializeSession, 1000);
            setTimeout(window.initializeSession, 2000);
            
            // Session management functions
            window.getSessionIdFromURL = function() {
                const urlParams = new URLSearchParams(window.location.search);
                return urlParams.get('session_id');
            }
            
            window.updateURLWithSessionId = function(sessionId) {
                try {
                    const url = new URL(window.location);
                    url.searchParams.set('session_id', sessionId);
                    
                    // Safari-compatible URL update
                    if (window.history && window.history.replaceState) {
                        window.history.replaceState({sessionId: sessionId}, '', url.toString());
                        console.log('URL updated with replaceState');
                    } else {
                        // Fallback for older browsers
                        console.log('Using fallback URL update method');
                        window.location.hash = `session_id=${sessionId}`;
                    }
                    
                    // Update page title with session info
                    document.title = `BioChemInsight - 会话 ${sessionId.substring(0, 8)}`;
                    
                    console.log('URL updated with session ID:', sessionId.substring(0, 8));
                    console.log('Current URL:', window.location.href);
                } catch (error) {
                    console.error('Error updating URL:', error);
                    // Ultimate fallback - just log the session ID
                    console.log('Session ID for manual URL update:', sessionId);
                }
            }
            
            window.getSessionIdFromPage = function() {
                const hiddenInput = document.getElementById('session-id-hidden');
                if (hiddenInput) {
                    const textarea = hiddenInput.querySelector('textarea');
                    if (textarea) {
                        return textarea.value;
                    }
                }
                return null;
            }
            
            // Function to be called immediately after PDF upload
            window.updateURLAfterUpload = function() {
                console.log('updateURLAfterUpload called');
                const sessionId = window.getSessionIdFromPage();
                console.log('Session ID from page:', sessionId);
                
                if (sessionId) {
                    const currentSessionId = window.getSessionIdFromURL();
                    console.log('Current session ID from URL:', currentSessionId);
                    
                    if (!currentSessionId || currentSessionId !== sessionId) {
                        console.log('Updating URL with session ID:', sessionId);
                        window.updateURLWithSessionId(sessionId);
                        
                        console.log('New session created:', sessionId.substring(0, 8));
                        console.log('Session URL:', sessionUrl);
                    } else {
                        console.log('Session ID unchanged, no URL update needed');
                    }
                } else {
                    console.log('No session ID found on page');
                }
            }
            
            // Function to be called when user starts processing (kept for compatibility)
            window.updateURLOnProcessStart = function() {
                // No longer needed as URL is updated on upload, but kept for compatibility
                return true;
            }
            
            // Auto-save functionality for long-running tasks
            window.autoSave = function() {
                const sessionId = window.getSessionIdFromURL() || window.getSessionIdFromPage();
                if (sessionId) {
                    const saveData = {
                        sessionId: sessionId,
                        timestamp: new Date().toISOString(),
                        url: window.location.href
                    };
                    localStorage.setItem('biocheminsight_session', JSON.stringify(saveData));
                }
            }
            
            // Auto-save on page unload
            window.addEventListener('beforeunload', window.autoSave);
            
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
                } catch (e) { console.error("Failed to trigger download:", e); }
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
            # Hidden session ID for JavaScript access
            session_id_hidden = gr.Textbox(value=self.session_id or "", visible=False, elem_id="session-id-hidden")
            # Hidden trigger for session restoration
            session_restore_trigger = gr.Textbox(visible=False, elem_id="session-restore-trigger")
            # Initial load trigger to handle session restoration on page load
            initial_load_trigger = gr.Button("Load", visible=False, elem_id="initial-load-trigger")

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
                        guidance_text = gr.Markdown("🚀 Welcome to BioChemInsight! Please upload a PDF file to start processing. A new session will be created automatically after upload.")
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
            
            # Initial load check for session restoration
            initial_load_trigger.click(
                self.check_session_on_load,
                inputs=[],
                outputs=[pdf_info, total_pages, gallery, structures_data, merged_data, merged_path, active_dataset_name, 
                        struct_pages_input, assay_pages_input, status_display, guidance_text, 
                        results_display_view, results_display_edit,
                        struct_dl_csv_btn, merged_dl_btn, meta_dl_btn,
                        assay_btn, assay_names_input, edit_btn]
            )
            
            # Session restoration from URL
            session_restore_trigger.change(
                self.set_session_id,
                inputs=[session_restore_trigger],
                outputs=[pdf_info, total_pages, gallery, structures_data, merged_data, merged_path, active_dataset_name, 
                        struct_pages_input, assay_pages_input, status_display, guidance_text, 
                        results_display_view, results_display_edit,
                        struct_dl_csv_btn, merged_dl_btn, meta_dl_btn,
                        assay_btn, assay_names_input, edit_btn, session_id_hidden]
            )

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
                assay_btn, assay_names_input, edit_btn, session_id_hidden]
            )
            
            # Add URL update when session_id_hidden changes
            session_id_hidden.change(
                fn=None,
                inputs=[session_id_hidden],
                js="""
                (sessionId) => {
                    console.log('Session ID changed to:', sessionId);
                    
                    if (sessionId && sessionId.length > 0) {
                        try {
                            const url = new URL(window.location);
                            url.searchParams.set('session_id', sessionId);
                            window.history.replaceState({sessionId: sessionId}, '', url.toString());
                            document.title = `BioChemInsight - Session ${sessionId.substring(0, 8)}`;
                            
                            console.log('URL updated successfully:', url.toString());
                            
                        } catch (error) {
                            console.error('Error updating URL:', error);
                        }
                    } else {
                        console.log('No session ID provided for URL update');
                    }
                    
                    return sessionId;
                }
                """
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
    """Main entry point for the application with enhanced session routing."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='BioChemInsight - Interactive Biochemical Document Extractor')
    parser.add_argument('--session-id', type=str, help='Session ID to resume')
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
    
    # Check if session ID was provided via command line
    session_id = args.session_id
    
    # If no session ID provided, check for environment variable (useful for URL routing)
    if not session_id:
        session_id = os.environ.get('SESSION_ID')
    
    # Create default app instance (without session_id, will create on PDF upload)
    default_app = BioChemInsightApp(session_id=session_id if session_id else None)
    interface = default_app.create_interface()
    
    print(f"🚀 Starting BioChemInsight...")
    print(f"🌐 Access URL: http://{args.host}:{actual_port}")
    print(f"💡 Tip: Session will be created automatically after PDF upload with UUID in address bar")
    print(f"📱 Mobile friendly: http://localhost:{actual_port}")
    print(f"🔄 Session management: Each PDF upload creates a new session")
    
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