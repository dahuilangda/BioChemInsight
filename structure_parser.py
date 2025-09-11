import os

import shutil
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image

import torch
from decimer_segmentation import get_expanded_masks, apply_masks

from constants import MOLVEC
from utils.image_utils import save_box_image
from utils.pdf_utils import split_pdf_to_images
from utils.file_utils import create_directory
from utils.llm_utils import structure_to_id, get_compound_id_from_description


def sort_segments_bboxes(segments, bboxes, masks, same_row_pixel_threshold=50):
    """
    Sorts segments and bounding boxes in "reading order"
    """
    sorted_bboxes = sorted(bboxes, key=lambda bbox: bbox[0])

    rows = []
    current_row = [sorted_bboxes[0]]
    for bbox in sorted_bboxes[1:]:
        if abs(bbox[0] - current_row[-1][0]) < same_row_pixel_threshold:
            current_row.append(bbox)
        else:
            rows.append(sorted(current_row, key=lambda x: x[1]))
            current_row = [bbox]
    rows.append(sorted(current_row, key=lambda x: x[1]))

    sorted_bboxes = [bbox for row in rows for bbox in row]
    sorted_segments = [segments[bboxes.index(bbox)] for bbox in sorted_bboxes]
    sorted_masks = [masks[:, :, bboxes.index(bbox)] for bbox in sorted_bboxes]
    sorted_masks = np.stack(sorted_masks, axis=-1)
    return sorted_segments, sorted_bboxes, sorted_masks

def extract_structures_from_pdf(pdf_file, page_start, page_end, output, engine='molnextr', progress_callback=None):
    images_dir = os.path.join(output, 'structure_images')
    segmented_dir = os.path.join(output, 'segment')

    # Prepare directories
    shutil.rmtree(segmented_dir, ignore_errors=True)
    create_directory(segmented_dir)

    # If page_start is > 1, start extracting one page earlier to get context.
    extraction_start_page = max(1, page_start - 1)
    split_pdf_to_images(pdf_file, images_dir, page_start=extraction_start_page, page_end=page_end)

    if engine == 'molscribe':
        from molscribe import MolScribe
        from huggingface_hub import hf_hub_download
        # Load MolScribe model
        print('Loading MolScribe model...')
        # ckpt_path = hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth', local_dir="./models", local_files_only=True)
        ckpt_path = hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth', local_dir="./models")
        model = MolScribe(ckpt_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    elif engine == 'molvec':
        from rdkit import Chem
    elif engine == 'molnextr':
        from utils.MolNexTR import molnextr
        BASE_ = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            '/app/models/molnextr_best.pth',  # Docker环境
            f'{BASE_}/models/molnextr_best.pth',     # 本地相对路径
        ]
        
        ckpt_path = None
        for path in possible_paths:
            if os.path.exists(path):
                ckpt_path = path
                break
        
        # 如果本地没有找到，尝试下载
        if ckpt_path is None:
            try:
                from huggingface_hub import hf_hub_download
                print('正在下载 MolNexTR 模型，这可能需要几分钟...')
                ckpt_path = hf_hub_download('CYF200127/MolNexTR', 'molnextr_best.pth', 
                                          repo_type='dataset', local_dir="./models")
                print(f'模型下载完成: {ckpt_path}')
            except Exception as e:
                print(f'模型下载失败: {e}')
                print('请手动下载模型文件或使用其他引擎 (molscribe/molvec)')
                raise FileNotFoundError(f'MolNexTR model not found. Please download it first or use another engine. Error: {e}')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Loading MolNexTR model from: {ckpt_path}')
        model = molnextr(ckpt_path, device)
    else:
        raise ValueError(f'Invalid engine: {engine}, must be "molscribe" or "molvec"')
    
    data_list = []
    total_pages = page_end - page_start + 1
    
    for page_idx, i in enumerate(range(page_start, page_end + 1)):
        if progress_callback:
            progress_callback(page_idx + 1, total_pages, f"Processing page {i}")
        
        try:
            scanned_page_file_path = os.path.join(images_dir, f'page_{i}.png')

            page = cv2.imread(scanned_page_file_path)
            if page is None:
                print(f"Warning: Could not read image for page {i} at {scanned_page_file_path}. Skipping.")
                continue

            masks = get_expanded_masks(page)
            segments, bboxes = apply_masks(page, masks)
            
            if len(segments) > 0:
                segments, bboxes, masks = sort_segments_bboxes(segments, bboxes, masks)

            for idx, segment in enumerate(segments):
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
                    
                    # Only proceed with SMILES extraction if image was saved successfully
                    if not os.path.exists(segment_name):
                        continue
                        
                except Exception as e:
                    print(f"Error extracting SMILES from segment {idx} on page {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                # Extract SMILES from the saved segment image
                smiles = ''
                try:
                    if engine == 'molscribe':
                        smiles = model.predict_image_file(segment_name, return_atoms_bonds=True, return_confidence=True).get('smiles')
                    elif engine == 'molnextr':
                        smiles = model.predict_final_results(segment_name, return_atoms_bonds=True, return_confidence=True).get('predicted_smiles')
                    elif engine == 'molvec':
                        if Chem is None:
                            print(f"Error: RDKit not available for molvec engine on segment {idx} page {i}")
                            smiles = ''
                        else:
                            cmd = f'java -jar {MOLVEC} -f {segment_name} -o {segment_name}.sdf'
                            os.popen(cmd).read()
                            try:
                                sdf = Chem.SDMolSupplier(f'{segment_name}.sdf')
                                if len(sdf) != 0:
                                    smiles = Chem.MolToSmiles(sdf[0])
                            except Exception as e:
                                print(f"Error reading SDF for segment {idx} on page {i}: {e}")
                                smiles = ''
                except Exception as e:
                    smiles = ''

                cpd_id_ = structure_to_id(output_name)
                cpd_id = get_compound_id_from_description(cpd_id_)
                if '```json' in cpd_id:
                    cpd_id = cpd_id.split('```json\n')[1].split('\n```')[0]
                    cpd_id = cpd_id.replace('{"COMPOUND_ID": "', '').replace('"}', '')

                row_data = {
                    'PAGE_NUM': i,
                    'COMPOUND_ID': cpd_id,
                    'SMILES': smiles,
                    'IMAGE_FILE': output_name,
                    'SEGMENT_FILE': segment_name
                }
                data_list.append(row_data)
        except Exception as e:
            print(f"Error processing page {i}: {e}")
            continue

    return data_list