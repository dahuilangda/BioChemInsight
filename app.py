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
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw

# Add the current directory to the Python path to import local modules
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    sys.path.append('.')

# Import pipeline functions directly for better progress tracking
from pipeline import extract_structures, extract_assay, get_total_pages
from structure_parser import extract_structures_from_pdf
from activity_parser import extract_activity_data

# Suppress unnecessary warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

# --- Simplified State Management ---
# Remove complex session management, use simple in-memory state with auto-download

# --- Helper function for rendering SMILES to image ---
def smiles_to_img_tag(smiles: str) -> str:
    """Converts a SMILES string to a base64-encoded image tag for markdown with optimized size."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES"
        # 增大尺寸提高清晰度，从150x150增加到200x200
        img = Draw.MolToImage(mol, size=(300, 300), kekulize=True)
        
        buffered = io.BytesIO()
        img.save(buffered, format="PNG", optimize=True, quality=90)
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
        self._last_structure_cache = {}  # 缓存结构提取结果，避免重复点击时UI震动
        print(f"Application initialized with temp directory: {self.temp_dir}")

    def get_processing_status(self) -> str:
        """Simple status check based on current state."""
        if not self.current_pdf_path:
            return "Please upload a PDF file"
        return f"PDF loaded: {self.current_pdf_filename}"

    def segment_to_img_tag(self, segment_path: str) -> str:
        """Converts a segment file path to a base64-encoded image tag for markdown with optimized compression."""
        if not segment_path or pd.isna(segment_path):
            return "No segment"
        try:
            full_path = os.path.join(self.temp_dir, "output", segment_path)
            if not os.path.exists(full_path):
                full_path = segment_path
                if not os.path.exists(full_path):
                    return f"Segment not found: {segment_path}"
            
            # 使用PIL压缩图片，但保持更大尺寸因为这是完整的页面图片
            from PIL import Image
            with Image.open(full_path) as img:
                # 增大压缩图片尺寸，最大宽度提升到400px，因为是完整的页面图片需要看清细节
                img.thumbnail((1500, 1000), Image.Resampling.LANCZOS)
                
                buffered = io.BytesIO()
                img.save(buffered, format="PNG", optimize=True, quality=90)
                encoded_string = base64.b64encode(buffered.getvalue()).decode()

            return f'![segment](data:image/png;base64,{encoded_string})'
        except Exception as e:
            return f"Error rendering segment: {str(e)}"

    def _enrich_dataframe_with_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds rendered structure and segment images to the dataframe for display with compressed images."""
        
        SMILES_COL = 'SMILES'
        SEGMENT_COL = 'SEGMENT_FILE'
        IMAGE_FILE_COL = 'IMAGE_FILE'
        COMPOUND_ID_COL = 'COMPOUND_ID'

        df = df.copy()

        # 恢复图片显示，但使用压缩的图片尺寸
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
    
    
    def _extract_structures_direct(self, pages_str: str, engine: str, progress_callback=None) -> str:
        """Direct structure extraction with precise progress tracking."""
        output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Parse page numbers
        pages = self.parse_pages_input(pages_str)
        if not pages:
            raise ValueError("No valid pages specified")
        
        # Group consecutive pages for processing
        def group_consecutive_pages(pages):
            pages = sorted(pages)
            groups = []
            current_group = [pages[0]]
            
            for i in range(1, len(pages)):
                if pages[i] == pages[i-1] + 1:  # 连续页面
                    current_group.append(pages[i])
                else:  # 不连续，开始新组
                    groups.append(current_group)
                    current_group = [pages[i]]
            groups.append(current_group)
            return groups
        
        page_groups = group_consecutive_pages(pages)
        total_pages = len(pages)
        processed_pages = 0
        
        try:
            if progress_callback:
                progress_callback(f"🔧 初始化结构提取引擎 ({engine})...", 0.02)
            
            # Import the extraction function and patch it for progress tracking
            from structure_parser import extract_structures_from_pdf
            
            all_structures = []
            
            # Process each group of consecutive pages
            for group_idx, group in enumerate(page_groups):
                start_page = min(group)
                end_page = max(group)
                group_pages = len(group)
                
                # 计算组的进度范围：5% - 75%，总共70%分配给结构提取
                group_start_progress = 0.05 + (processed_pages / total_pages) * 0.70
                group_end_progress = 0.05 + ((processed_pages + group_pages) / total_pages) * 0.70
                
                if progress_callback:
                    progress_callback(f"📖 处理页面组 {group_idx + 1}/{len(page_groups)} (页面 {start_page}-{end_page})", 
                                    group_start_progress)
                
                # Create group output directory
                group_output_dir = os.path.join(output_dir, f"structures_group_{group_idx}")
                os.makedirs(group_output_dir, exist_ok=True)
                
                # Extract structures for this group with detailed progress
                try:
                    structures = self._extract_single_group(
                        pdf_file=self.current_pdf_path,
                        start_page=start_page,
                        end_page=end_page,
                        output_dir=group_output_dir,
                        engine=engine,
                        progress_callback=lambda page, total, msg: progress_callback(
                            f"🧪 正在处理第 {page} 页，共 {total_pages} 页 - {msg}", 
                            # 更精确的进度计算：在组的进度范围内分配
                            group_start_progress + ((page - start_page) / group_pages) * (group_end_progress - group_start_progress)
                        ) if progress_callback else None
                    )
                    
                    if structures:
                        # Add page info to each structure
                        for structure in structures:
                            if isinstance(structure, dict):
                                structure['source_pages'] = list(group)
                                structure['group_id'] = group_idx
                        all_structures.extend(structures if isinstance(structures, list) else [structures])
                    
                    processed_pages += group_pages
                    
                except Exception as e:
                    print(f"Error processing group {group_idx + 1}: {e}")
                    processed_pages += group_pages
                    continue
            
            if progress_callback:
                progress_callback("💼 处理提取结果...", 0.78)
            
            # Remove duplicates and save results
            if all_structures:
                # Convert to DataFrame format for pipeline compatibility
                # 去重：基于COMPOUND_ID和SMILES的组合进行去重，保留不同的化合物ID
                seen_combinations = set()
                unique_structures = []
                
                for structure in all_structures:
                    if isinstance(structure, dict):
                        compound_id = structure.get('COMPOUND_ID', '')
                        smiles = structure.get('SMILES', '')
                        # 创建唯一标识：COMPOUND_ID + SMILES的组合
                        combination_key = f"{compound_id}_{smiles}"
                        if combination_key not in seen_combinations:
                            seen_combinations.add(combination_key)
                            unique_structures.append(structure)
                    else:
                        # 兼容旧格式
                        structure_str = str(structure)
                        if structure_str not in seen_combinations:
                            seen_combinations.add(structure_str)
                            unique_structures.append(structure)
                
                if unique_structures:
                    if progress_callback:
                        progress_callback("💾 保存结构数据...", 0.90)
                    
                    structures_df = pd.DataFrame(unique_structures)
                    structure_csv = os.path.join(output_dir, 'structures.csv')
                    structures_df.to_csv(structure_csv, index=False, encoding='utf-8-sig')
                    
                    if progress_callback:
                        progress_callback(f"✅ 提取完成! 找到 {len(structures_df)} 个结构", 1.0)
                    
                    print(f"Chemical structures saved to {structure_csv} ({len(structures_df)} unique structures)")
                    return output_dir
            
            if progress_callback:
                progress_callback("⚠️ 未找到化学结构", 1.0)
            print("No structures were extracted")
            return output_dir
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ 提取失败: {str(e)}", 1.0)
            raise e

    def _extract_single_group(self, pdf_file, start_page, end_page, output_dir, engine, progress_callback=None):
        """Extract structures from a single group of consecutive pages with progress tracking."""
        from utils.pdf_utils import split_pdf_to_images
        from utils.file_utils import create_directory
        from decimer_segmentation import get_expanded_masks, apply_masks
        from utils.image_utils import save_box_image
        import cv2
        from PIL import Image
        import torch
        
        images_dir = os.path.join(output_dir, 'structure_images')
        segmented_dir = os.path.join(output_dir, 'segment')
        
        # Prepare directories
        if os.path.exists(segmented_dir):
            shutil.rmtree(segmented_dir)
        create_directory(segmented_dir)
        
        # Split PDF to images
        if progress_callback:
            progress_callback(start_page, end_page, "准备页面图像")
        
        extraction_start_page = max(1, start_page - 1)
        split_pdf_to_images(pdf_file, images_dir, page_start=extraction_start_page, page_end=end_page)
        
        # Load model based on engine
        if progress_callback:
            progress_callback(start_page, end_page, f"加载 {engine} 模型")
        
        model = None
        if engine == 'molscribe':
            from molscribe import MolScribe
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth', local_dir="./models")
            model = MolScribe(ckpt_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        elif engine == 'molnextr':
            from utils.MolNexTR import molnextr
            BASE_ = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                '/app/models/molnextr_best.pth',
                f'{BASE_}/models/molnextr_best.pth',
            ]
            
            ckpt_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    ckpt_path = path
                    break
            
            if ckpt_path is None:
                from huggingface_hub import hf_hub_download
                ckpt_path = hf_hub_download('CYF200127/MolNexTR', 'molnextr_best.pth', 
                                          repo_type='dataset', local_dir="./models")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = molnextr(ckpt_path, device)
        
        # Process each page
        data_list = []
        total_pages_in_group = end_page - start_page + 1
        
        for page_idx, i in enumerate(range(start_page, end_page + 1)):
            try:
                scanned_page_file_path = os.path.join(images_dir, f'page_{i}.png')
                page = cv2.imread(scanned_page_file_path)
                
                if page is None:
                    print(f"Warning: Could not read image for page {i}")
                    continue
                
                # Get structure segments
                masks = get_expanded_masks(page)
                segments, bboxes = apply_masks(page, masks)
                
                if len(segments) == 0:
                    if progress_callback:
                        progress_callback(i, total_pages_in_group, f"页面 {i} 无结构")
                    continue
                
                # Sort segments
                from structure_parser import sort_segments_bboxes
                segments, bboxes, masks = sort_segments_bboxes(segments, bboxes, masks)
                
                # Process each segment
                for idx, segment in enumerate(segments):
                    # 更准确的进度计算：页面进度 + 页面内结构进度
                    page_progress = page_idx / total_pages_in_group
                    segment_progress_within_page = (idx + 1) / len(segments) / total_pages_in_group
                    current_progress = page_progress + segment_progress_within_page
                    
                    if progress_callback:
                        # 阶段1：保存图片（占每个结构处理时间的30%）
                        progress_callback(i, total_pages_in_group, f"页面 {i} 结构 {idx + 1}/{len(segments)} - 保存图片")
                    
                    output_name = os.path.join(segmented_dir, f'highlight_{i}_{idx}.png')
                    segment_name = os.path.join(segmented_dir, f'segment_{i}_{idx}.png')
                    
                    save_box_image(bboxes, masks, idx, page, output_name)
                    
                    # 创建左右合并图片：前一页完整图（左）+ 当前高亮结构区域（右）
                    prev_page_path = os.path.join(images_dir, f'page_{i-1}.png')
                    if os.path.exists(prev_page_path):
                        # 加载刚保存的高亮图片和前一页完整图片
                        current_highlight_img = cv2.imread(output_name)
                        prev_page_img = cv2.imread(prev_page_path)

                        # 确保两个图片都成功加载
                        if current_highlight_img is not None and prev_page_img is not None:
                            # 调整前一页图片大小以匹配高亮图片的高度，保持宽高比
                            ch_height = current_highlight_img.shape[0]
                            pp_height, pp_width, _ = prev_page_img.shape
                            scale_ratio = ch_height / pp_height
                            new_pp_width = int(pp_width * scale_ratio)
                            resized_prev_page = cv2.resize(prev_page_img, (new_pp_width, ch_height))
                            
                            # 水平拼接：前一页完整图（左）+ 当前高亮结构区域（右）
                            combined_img = cv2.hconcat([resized_prev_page, current_highlight_img])
                            # 用合并后的图片覆盖原来的高亮图片
                            cv2.imwrite(output_name, combined_img)
                    
                    # Ensure segment is a proper numpy array for PIL
                    try:
                        if not isinstance(segment, np.ndarray):
                            continue
                        
                        # Ensure the array has the right shape and dtype
                        if len(segment.shape) != 3:
                            continue
                            
                        # Handle both RGB (3 channels) and RGBA (4 channels) images
                        if segment.shape[2] == 4:
                            # Convert RGBA to RGB by removing alpha channel
                            segment = segment[:, :, :3]
                        elif segment.shape[2] != 3:
                            continue
                        
                        # Convert to uint8 if needed
                        if segment.dtype != np.uint8:
                            if segment.max() <= 1.0:
                                segment = (segment * 255).astype(np.uint8)
                            else:
                                segment = segment.astype(np.uint8)
                        
                        segment_image = Image.fromarray(segment)
                        segment_image.save(segment_name)
                        
                    except Exception as e:
                        continue
                    
                    # Extract SMILES using the model
                    if progress_callback:
                        progress_callback(i, total_pages_in_group, f"页面 {i} 结构 {idx + 1}/{len(segments)} - 提取SMILES")
                    
                    try:
                        if engine == 'molscribe':
                            smiles = model.predict_image_file(segment_name, return_atoms_bonds=True, return_confidence=True).get('smiles')
                        elif engine == 'molnextr':
                            smiles = model.predict_final_results(segment_name, return_atoms_bonds=True, return_confidence=True).get('predicted_smiles')
                        elif engine == 'molvec':
                            # Use molvec for extraction
                            smiles = self._extract_with_molvec(segment_name)
                        else:
                            smiles = None
                        
                        if smiles and smiles.strip():
                            if progress_callback:
                                progress_callback(i, total_pages_in_group, f"页面 {i} 结构 {idx + 1}/{len(segments)} - 识别化合物ID")
                            
                            # Get compound ID using the highlight image path (not SMILES)
                            from utils.llm_utils import structure_to_id, get_compound_id_from_description
                            cpd_id_ = structure_to_id(output_name)
                            cpd_id = get_compound_id_from_description(cpd_id_)
                            if '```json' in cpd_id:
                                cpd_id = cpd_id.split('```json\n')[1].split('\n```')[0]
                                cpd_id = cpd_id.replace('{"COMPOUND_ID": "', '').replace('"}', '')
                            
                            if progress_callback:
                                progress_callback(i, total_pages_in_group, f"页面 {i} 结构 {idx + 1}/{len(segments)} - 完成")
                            
                            data_list.append({
                                'COMPOUND_ID': cpd_id,
                                'SMILES': smiles,
                                'PAGE': i,
                                'SEGMENT_INDEX': idx,
                                'SEGMENT_FILE': segment_name,  # 使用完整路径
                                'IMAGE_FILE': output_name     # 使用完整路径
                            })
                    except Exception as e:
                        print(f"Error extracting SMILES from segment {idx} on page {i}: {e}")
                        continue
            
            except Exception as e:
                print(f"Error processing page {i}: {e}")
                continue
        
        return data_list

    def _extract_with_molvec(self, image_path):
        """Extract SMILES using molvec."""
        try:
            import subprocess
            import os
            
            # Use molvec jar file
            molvec_jar = os.path.join(os.path.dirname(__file__), 'bin', 'molvec-0.9.9-SNAPSHOT-jar-with-dependencies.jar')
            if not os.path.exists(molvec_jar):
                return None
            
            # Run molvec (无timeout限制，允许长时间处理)
            result = subprocess.run([
                'java', '-jar', molvec_jar, 
                '-f', image_path,
                '-o', 'smiles'
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            
        except Exception as e:
            print(f"Molvec extraction error: {e}")
        
        return None

    def _extract_activity_direct(self, pages_str: str, assay_names: str, lang: str, ocr_engine: str, structures_data: list, progress_callback=None) -> str:
        """Direct activity extraction with precise progress tracking."""
        output_dir = os.path.join(self.temp_dir, "output")
        
        # Parse page numbers and assay names
        pages = self.parse_pages_input(pages_str)
        assay_list = [name.strip() for name in assay_names.split(',') if name.strip()]
        
        if not pages:
            raise ValueError("No valid pages specified")
        if not assay_list:
            raise ValueError("No assay names specified")
        
        # Get compound IDs from structures data
        compound_ids = []
        if structures_data:
            for item in structures_data:
                if isinstance(item, dict) and 'COMPOUND_ID' in item:
                    compound_ids.append(item['COMPOUND_ID'])
        
        total_steps = len(assay_list) + 2  # +2 for setup and merging
        current_step = 0
        
        try:
            if progress_callback:
                progress_callback(f"🔧 初始化活性数据提取 ({ocr_engine})...", current_step / total_steps)
            
            current_step += 1
            
            # Extract activity data for each assay
            all_assay_data = {}
            
            for assay_idx, assay_name in enumerate(assay_list):
                if progress_callback:
                    progress_callback(f"🧪 处理测定 {assay_idx + 1}/{len(assay_list)}: {assay_name}", 
                                    (current_step + assay_idx) / total_steps)
                
                # Determine page range for this assay
                page_start = min(pages)
                page_end = max(pages)
                
                # Extract activity data
                def page_progress_callback(message):
                    if progress_callback:
                        progress_callback(f"🧪 {assay_name}: {message}")
                
                assay_data = extract_activity_data(
                    pdf_file=self.current_pdf_path,
                    assay_page_start=page_start,
                    assay_page_end=page_end,
                    assay_name=assay_name,
                    compound_id_list=compound_ids,
                    output_dir=output_dir,
                    lang=lang,
                    ocr_engine=ocr_engine,
                    progress_callback=page_progress_callback
                )
                
                if assay_data:
                    all_assay_data[assay_name] = assay_data
            
            current_step = total_steps - 1
            
            if progress_callback:
                progress_callback("🔗 合并结构和活性数据...", current_step / total_steps)
            
            # Merge structures and activity data
            self._merge_data(structures_data, all_assay_data, output_dir)
            
            if progress_callback:
                progress_callback("✅ 活性数据提取完成!", 1.0)
            
            return output_dir
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ 提取失败: {str(e)}", 1.0)
            raise e

    def _merge_data(self, structures_data: list, assay_data: dict, output_dir: str):
        """Merge structures and activity data into a single CSV."""
        if not structures_data or not assay_data:
            return
        
        # Convert structures to dataframe
        structures_df = pd.DataFrame(structures_data)
        
        # Create merged dataframe
        merged_records = []
        
        for _, structure in structures_df.iterrows():
            compound_id = structure.get('COMPOUND_ID', '')
            
            # Base record with structure info
            record = structure.to_dict()
            
            # Add activity data for each assay
            for assay_name, assay_dict in assay_data.items():
                if compound_id in assay_dict:
                    activity_info = assay_dict[compound_id]
                    if isinstance(activity_info, dict):
                        # Add activity data with assay prefix
                        for key, value in activity_info.items():
                            record[f"{assay_name}_{key}"] = value
                    else:
                        record[assay_name] = activity_info
                else:
                    record[assay_name] = None
            
            merged_records.append(record)
        
        # Save merged data
        if merged_records:
            merged_df = pd.DataFrame(merged_records)
            merged_file = os.path.join(output_dir, 'merged.csv')
            merged_df.to_csv(merged_file, index=False, encoding='utf-8-sig')
            print(f"Merged data saved to {merged_file}")

    def _run_pipeline(self, args: List[str], clear_output: bool = True, progress_callback=None) -> str:
        """Enhanced pipeline execution with direct function calls and precise progress tracking."""
        output_dir = os.path.join(self.temp_dir, "output")
        if clear_output and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Parse arguments to determine operation type
            if "--structure-pages" in args:
                # Structure extraction
                struct_idx = args.index("--structure-pages")
                pages_str = args[struct_idx + 1]
                
                engine_idx = args.index("--engine") if "--engine" in args else None
                engine = args[engine_idx + 1] if engine_idx else 'molnextr'
                
                # Create enhanced progress callback
                def enhanced_progress(message, progress_val=None):
                    if progress_callback:
                        if progress_val is not None:
                            progress_callback(message)
                        else:
                            progress_callback(message)
                
                return self._extract_structures_direct(pages_str, engine, enhanced_progress)
                
            elif "--assay-pages" in args:
                # Activity extraction (requires existing structures)
                assay_idx = args.index("--assay-pages")
                pages_str = args[assay_idx + 1]
                
                names_idx = args.index("--assay-names")
                assay_names = args[names_idx + 1]
                
                lang_idx = args.index("--lang") if "--lang" in args else None
                lang = args[lang_idx + 1] if lang_idx else 'en'
                
                ocr_idx = args.index("--ocr-engine") if "--ocr-engine" in args else None
                ocr_engine = args[ocr_idx + 1] if ocr_idx else 'paddleocr'
                
                # Load existing structures
                structures_file = os.path.join(output_dir, "structures.csv")
                if os.path.exists(structures_file):
                    structures_df = pd.read_csv(structures_file)
                    structures_data = structures_df.to_dict('records')
                else:
                    structures_data = []
                
                # Create enhanced progress callback
                def enhanced_progress(message, progress_val=None):
                    if progress_callback:
                        progress_callback(message)
                
                return self._extract_activity_direct(pages_str, assay_names, lang, ocr_engine, structures_data, enhanced_progress)
                
            else:
                raise ValueError("Unknown pipeline operation")
                
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ 处理失败: {str(e)}")
            raise e

    def extract_structures(self, struct_pages_input: str, engine_input: str, lang_input: str, progress=gr.Progress()) -> tuple:
        if not self.current_pdf_path:
            return "❌ Please upload a PDF file first", None, None, "none", gr.update(), gr.update(), gr.update(visible=False), "Please upload a PDF to begin.", gr.update(), gr.update(), gr.update()
        if not struct_pages_input or not struct_pages_input.strip():
            return "❌ Please select pages for structures.", None, None, "none", gr.update(), gr.update(), gr.update(visible=False), "Please select pages for structures.", gr.update(), gr.update(), gr.update()
        
        # Validate that pages actually exist
        parsed_pages = self.parse_pages_input(struct_pages_input)
        if not parsed_pages:
            return "❌ Please select valid pages for structures.", None, None, "none", gr.update(), gr.update(), gr.update(visible=False), "Please select valid pages for structures.", gr.update(), gr.update(), gr.update()
        
        # 检查是否已经有相同参数的结果存在，避免重复处理造成UI震动
        cache_key = f"{struct_pages_input}_{engine_input}_{lang_input}"
        if hasattr(self, '_last_structure_cache') and self._last_structure_cache.get('key') == cache_key:
            cached_result = self._last_structure_cache.get('result')
            if cached_result:
                progress(1.0, desc="✅ 使用缓存结果")
                return cached_result
        
        try:
            print(f"Extracting structures from pages: {struct_pages_input}")
            
            # Create progress callback that updates Gradio progress with proper values
            progress_steps = len(parsed_pages) + 3  # setup, processing pages, finalization
            current_step = 0
            
            def progress_callback(message, progress_val=None):
                nonlocal current_step
                if progress_val is not None:
                    # 确保进度值在0-1之间
                    clamped_progress = max(0.0, min(1.0, progress_val))
                    progress(clamped_progress, desc=message)
                else:
                    current_step += 1
                    step_progress = min(1.0, current_step / progress_steps)
                    progress(step_progress, desc=message)
            
            progress(0.1, desc="🚀 准备提取化学结构...")
            
            args = ["--structure-pages", struct_pages_input, "--engine", engine_input, "--lang", lang_input]
            output_dir = self._run_pipeline(args, clear_output=True, progress_callback=progress_callback)
            
            progress(0.8, desc="📊 处理结果数据...")
            
            structures_file = os.path.join(output_dir, "structures.csv")
            if os.path.exists(structures_file):
                df = pd.read_csv(structures_file)
                if not df.empty:
                    progress(0.9, desc="🎨 生成结构图像...")
                    
                    df_enriched = self._enrich_dataframe_with_images(df.copy())
                    datatypes = self._get_df_dtypes(df_enriched)
                    view_df_update = gr.update(value=df_enriched, datatype=datatypes, visible=True)
                    
                    edit_df = df.drop(columns=['Structure', 'Segment', 'Image File'], errors='ignore')
                    edit_df_update = gr.update(value=edit_df, visible=False)

                    print(f"Successfully extracted {len(df)} structures")

                    progress(1.0, desc="✅ 结构提取完成!")
                    
                    new_guidance = f"✅ **Step 1 Complete**. Now, enter assay names, select pages with bioactivity data, and click Step 2."
                    result = f"✅ Extracted {len(df)} structures.", df.to_dict('records'), None, "structures", view_df_update, edit_df_update, gr.update(visible=True), new_guidance, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
                    
                    # 缓存结果以避免重复点击时UI震动
                    if not hasattr(self, '_last_structure_cache'):
                        self._last_structure_cache = {}
                    self._last_structure_cache['key'] = cache_key
                    self._last_structure_cache['result'] = result
                    
                    return result
                else:
                    progress(1.0, desc="⚠️ 未找到结构")
                    return "⚠️ No structures found.", None, None, "none", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "No structures found.", gr.update(), gr.update(), gr.update()
            else:
                progress(1.0, desc="❌ 结果文件未生成")
                return "❌ 'structures.csv' not created.", None, None, "none", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "File not generated.", gr.update(), gr.update(), gr.update()
        except Exception as e:
            print(f"Error extracting structures: {e}")
            progress(1.0, desc=f"❌ 错误: {str(e)}")
            return f"❌ Error: {str(e)}", None, None, "none", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "An error occurred.", gr.update(), gr.update(), gr.update()

    def extract_activity_and_merge(self, assay_pages_input: str, structures_data: list, assay_names: str, lang_input: str, ocr_engine_input: str, progress=gr.Progress()) -> tuple:
        if not assay_pages_input: return "❌ Select activity pages.", gr.update(), "merged", gr.update(), gr.update(), gr.update(), gr.update(), "Select activity pages.", gr.update()
        if not structures_data: return "⚠️ Run Step 1 first.", gr.update(), "merged", gr.update(), gr.update(), gr.update(), gr.update(), "Run Step 1 first.", gr.update()
        if not assay_names: return "❌ Enter assay names.", gr.update(), "merged", gr.update(), gr.update(), gr.update(), gr.update(), "Enter assay names.", gr.update()
        
        try:
            print(f"Extracting activity data from pages: {assay_pages_input}")
            
            # Parse assay names for better progress tracking
            assay_list = [name.strip() for name in assay_names.split(',') if name.strip()]
            progress_steps = len(assay_list) + 3  # setup, processing assays, merging
            current_step = 0
            
            # Create progress callback that updates Gradio progress with proper values  
            def progress_callback(message, progress_val=None):
                nonlocal current_step
                if progress_val is not None:
                    # 确保进度值在0-1之间
                    clamped_progress = max(0.0, min(1.0, progress_val))
                    progress(clamped_progress, desc=message)
                else:
                    current_step += 1
                    step_progress = min(1.0, current_step / progress_steps)
                    progress(step_progress, desc=message)
            
            progress(0.1, desc="🧪 准备提取生物活性数据...")
            
            args = ["--assay-pages", assay_pages_input, "--assay-names", assay_names, "--lang", lang_input, "--ocr-engine", ocr_engine_input]
            output_dir = self._run_pipeline(args, clear_output=False, progress_callback=progress_callback)
            
            progress(0.8, desc="🔗 合并结构和活性数据...")
            
            merged_file = os.path.join(output_dir, "merged.csv")
            if not os.path.exists(merged_file):
                progress(1.0, desc="❌ 合并失败")
                return "❌ 'merged.csv' not found.", gr.update(), "merged", gr.update(), gr.update(), gr.update(), gr.update(), "Merge failed.", gr.update()
            
            df = pd.read_csv(merged_file)
            if df.empty:
                progress(1.0, desc="⚠️ 合并结果为空")
                return "⚠️ Merge result is empty.", gr.update(), "merged", gr.update(), gr.update(), gr.update(), gr.update(), "No data to merge.", gr.update()
            
            progress(0.9, desc="🎨 生成最终结果...")
            
            df_enriched = self._enrich_dataframe_with_images(df.copy())
            datatypes = self._get_df_dtypes(df_enriched)
            view_df_update = gr.update(value=df_enriched, datatype=datatypes, visible=True)
            
            edit_df = df.drop(columns=['Structure', 'Segment', 'Image File'], errors='ignore')
            edit_df_update = gr.update(value=edit_df, visible=False)

            print(f"Successfully merged {len(df)} records")

            progress(1.0, desc="✅ 数据合并完成!")

            status_msg = f"✅ Merge successful. Generated {len(df)} records."
            return (status_msg, df.to_dict('records'), "merged", view_df_update, edit_df_update, merged_file, 
                    gr.update(visible=True), 
                    f"✅ **Process Complete!** View and download results below.", gr.update(visible=True))
        except Exception as e:
            print(f"Error in activity extraction and merge: {e}")
            progress(1.0, desc=f"❌ 错误: {str(e)}")
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
        
        /* Network resilience styles */
        #connection-indicator {
            position: fixed !important;
            top: 10px !important;
            right: 10px !important;
            padding: 8px 12px !important;
            border-radius: 6px !important;
            font-size: 12px !important;
            font-weight: bold !important;
            z-index: 9999 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        }
        
        /* Enhanced progress bar styling */
        .progress-container {
            width: 100%;
            background-color: #f0f0f0;
            border-radius: 10px;
            margin: 10px 0;
            overflow: hidden;
            height: 25px;
            position: relative;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            border-radius: 10px;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 13px;
            font-weight: bold;
            position: relative;
            max-width: 100%;
            min-width: 0%;
        }
        
        .progress-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #333;
            font-size: 12px;
            font-weight: 600;
            z-index: 2;
            text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        /* 确保Gradio原生进度条不超出边界 */
        .gr-block .gr-progress {
            max-width: 100% !important;
            overflow: hidden !important;
        }
        
        .gr-block .gr-progress > div {
            max-width: 100% !important;
        }
        
        /* Improved button layout */
        .gr-button {
            width: 100%;
            margin: 5px 0;
        }
        
        .gr-dropdown, .gr-textbox {
            width: 100%;
            margin: 5px 0;
        }
        
        /* Enhanced processing indicator */
        .processing-indicator {
            background: linear-gradient(90deg, #f0f0f0, #e0e0e0, #f0f0f0);
            background-size: 200% 100%;
            animation: shimmer 2s infinite;
            border-radius: 6px;
            padding: 12px 16px;
            margin: 10px 0;
            border: 1px solid #ddd;
            font-size: 14px;
            font-weight: 500;
        }
        
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        
        /* Improved button states */
        .gr-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        /* Better spacing for form elements */
        .gr-form > * {
            margin-bottom: 8px;
        }
        
        /* Recovery notification */
        .recovery-notification {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 12px;
            border-radius: 6px;
            margin: 10px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .recovery-notification .close-btn {
            margin-left: auto;
            cursor: pointer;
            font-weight: bold;
        }
        
        /* Timeout warning styles with better alignment */
        .timeout-warning {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 12px;
            border-radius: 6px;
            margin: 10px 0;
        }
        
        .timeout-warning strong {
            display: block;
            margin-bottom: 5px;
        }
        
        /* Improved column layout */
        .gr-row {
            display: flex;
            align-items: stretch;
        }
        
        .gr-column {
            display: flex;
            flex-direction: column;
        }
        
        /* Progress text formatting */
        .progress-detail {
            font-size: 13px;
            color: #666;
            margin: 5px 0;
            padding: 8px 12px;
            background: #f8f9fa;
            border-radius: 4px;
            border-left: 3px solid #007bff;
        }
        
        .progress-detail .current-page {
            font-weight: bold;
            color: #007bff;
        }
        
        .progress-detail .total-pages {
            color: #6c757d;
        }
        """
        
        js_script = """
        () => {
            // Enhanced network resilience and recovery functionality
            window.biochemAutoSave = {
                // Save processing results and state to browser storage for recovery
                saveResults: function(type, data) {
                    try {
                        const key = `biocheminsight_${type}_${Date.now()}`;
                        const saveData = {
                            timestamp: new Date().toISOString(),
                            type: type,
                            data: data,
                            url: window.location.href,
                            userAgent: navigator.userAgent
                        };
                        localStorage.setItem(key, JSON.stringify(saveData));
                        console.log(`Results saved: ${type} (${key})`);
                        
                        // Also save current state for recovery
                        this.saveCurrentState();
                    } catch (e) {
                        console.warn('Failed to save results:', e);
                    }
                },
                
                // Save current page state for recovery
                saveCurrentState: function() {
                    try {
                        const structPages = document.querySelector('#struct-pages-input textarea, #struct-pages-input input')?.value || '';
                        const assayPages = document.querySelector('#assay-pages-input textarea, #assay-pages-input input')?.value || '';
                        const assayNames = document.querySelector('[placeholder*="IC50"]')?.value || '';
                        
                        const state = {
                            structPages,
                            assayPages,
                            assayNames,
                            timestamp: new Date().toISOString()
                        };
                        
                        localStorage.setItem('biocheminsight_current_state', JSON.stringify(state));
                    } catch (e) {
                        console.warn('Failed to save state:', e);
                    }
                },
                
                // Restore previous state
                restoreState: function() {
                    try {
                        const saved = localStorage.getItem('biocheminsight_current_state');
                        if (saved) {
                            const state = JSON.parse(saved);
                            const age = Date.now() - new Date(state.timestamp).getTime();
                            
                            // Only restore if less than 1 hour old
                            if (age < 60 * 60 * 1000) {
                                const structInput = document.querySelector('#struct-pages-input textarea, #struct-pages-input input');
                                const assayInput = document.querySelector('#assay-pages-input textarea, #assay-pages-input input');
                                const assayNamesInput = document.querySelector('[placeholder*="IC50"]');
                                
                                if (structInput && state.structPages) {
                                    structInput.value = state.structPages;
                                    structInput.dispatchEvent(new Event('input', { bubbles: true }));
                                }
                                if (assayInput && state.assayPages) {
                                    assayInput.value = state.assayPages;
                                    assayInput.dispatchEvent(new Event('input', { bubbles: true }));
                                }
                                if (assayNamesInput && state.assayNames) {
                                    assayNamesInput.value = state.assayNames;
                                    assayNamesInput.dispatchEvent(new Event('input', { bubbles: true }));
                                }
                                
                                console.log('State restored from previous session');
                                return true;
                            }
                        }
                    } catch (e) {
                        console.warn('Failed to restore state:', e);
                    }
                    return false;
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
                
                // Clean old saved data
                cleanOldData: function() {
                    try {
                        const keys = Object.keys(localStorage).filter(k => k.startsWith('biocheminsight_'));
                        const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000; // 7 days
                        
                        keys.forEach(key => {
                            try {
                                const data = JSON.parse(localStorage.getItem(key));
                                if (new Date(data.timestamp).getTime() < cutoff) {
                                    localStorage.removeItem(key);
                                    console.log(`Cleaned old data: ${key}`);
                                }
                            } catch (e) {
                                localStorage.removeItem(key); // Remove corrupted data
                            }
                        });
                    } catch (e) {
                        console.warn('Failed to clean old data:', e);
                    }
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
                },
                
                // Show connection status
                updateConnectionStatus: function() {
                    let indicator = document.getElementById('connection-indicator');
                    if (!indicator) {
                        indicator = document.createElement('div');
                        indicator.id = 'connection-indicator';
                        indicator.style.cssText = `
                            position: fixed;
                            top: 10px;
                            right: 10px;
                            padding: 8px 12px;
                            border-radius: 6px;
                            font-size: 12px;
                            font-weight: bold;
                            z-index: 9999;
                            transition: all 0.3s ease;
                        `;
                        document.body.appendChild(indicator);
                    }
                    
                    if (navigator.onLine) {
                        indicator.textContent = '🟢 在线';
                        indicator.style.backgroundColor = '#d4edda';
                        indicator.style.color = '#155724';
                        indicator.style.border = '1px solid #c3e6cb';
                    } else {
                        indicator.textContent = '🔴 离线';
                        indicator.style.backgroundColor = '#f8d7da';
                        indicator.style.color = '#721c24';
                        indicator.style.border = '1px solid #f5c6cb';
                    }
                }
            };
            
            // Network status monitoring
            window.addEventListener('online', () => {
                console.log('网络恢复连接');
                window.biochemAutoSave.updateConnectionStatus();
            });
            
            window.addEventListener('offline', () => {
                console.log('网络连接断开');
                window.biochemAutoSave.updateConnectionStatus();
                window.biochemAutoSave.saveCurrentState();
            });
            
            // Initialize on load
            setTimeout(() => {
                window.biochemAutoSave.updateConnectionStatus();
                window.biochemAutoSave.cleanOldData();
                
                // Try to restore previous state
                if (window.biochemAutoSave.restoreState()) {
                    console.log('Previous session state restored');
                }
            }, 1000);
            
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
            
            // Auto-save current state periodically
            setInterval(() => {
                if (navigator.onLine) {
                    window.biochemAutoSave.saveCurrentState();
                }
            }, 60000); // Every minute
            
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
                
                // Save state after selection change
                window.biochemAutoSave.saveCurrentState();
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
                        
                        # Engine selection with full width
                        engine_input = gr.Dropdown(
                            label="Structure Extraction Engine", 
                            choices=['molnextr', 'molscribe', 'molvec'], 
                            value='molnextr', 
                            interactive=True
                        )
                        
                        # Action buttons with consistent spacing
                        struct_btn = gr.Button("Step 1: Extract Structures", variant="primary")
                        
                        assay_names_input = gr.Textbox(
                            label="Assay Names (comma-separated)", 
                            placeholder="e.g., IC50, Ki, EC50", 
                            visible=False
                        )
                        
                        assay_btn = gr.Button("Step 2: Extract Activity", visible=False, variant="primary")
                with gr.Column(scale=3):
                    gallery = gr.HTML("<div class='center-placeholder'>PDF previews will appear here.</div>")

            gr.Markdown("---")
            gr.Markdown("<h3>Results</h3>")
            # gr.Markdown("💡 **提示**: 点击表格中的 📊 Structure、🖼️ Segment 或 📷 Image File 单元格查看对应图片")
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
    """Main entry point for the enhanced application with network resilience."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='BioChemInsight - Enhanced Biochemical Document Extractor')
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
    
    # Create enhanced app instance
    app = BioChemInsightApp()
    interface = app.create_interface()
    
    print(f"🚀 启动 BioChemInsight (网络增强版)...")
    print(f"🌐 访问地址: http://{args.host}:{actual_port}")
    
    try:
        interface.launch(
            server_name=args.host, 
            server_port=actual_port,
            share=args.share, 
            debug=True,
            show_api=False,
            prevent_thread_lock=False,
            # Additional stability settings
            inbrowser=False,
            quiet=False
        )
    except Exception as e:
        print(f"❌ 启动界面失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()