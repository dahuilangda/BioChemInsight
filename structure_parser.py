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

def extract_structures_from_pdf(pdf_file, page_start, page_end, output, engine='molnextr'):
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
        ckpt_path = hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth', local_dir="./models", local_files_only=True)
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
    for i in tqdm(range(page_start, page_end + 1), desc='Extracting structures'):
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
                
                prev_page_path = os.path.join(images_dir, f'page_{i-1}.png')
                if os.path.exists(prev_page_path):
                    # Load the highlight image we just saved and the full previous page
                    current_highlight_img = cv2.imread(output_name)
                    prev_page_img = cv2.imread(prev_page_path)

                    # Proceed only if both images were loaded successfully
                    if current_highlight_img is not None and prev_page_img is not None:
                        # Resize previous page to match the HEIGHT of the highlight image, maintaining aspect ratio
                        ch_height = current_highlight_img.shape[0]
                        pp_height, pp_width, _ = prev_page_img.shape
                        scale_ratio = ch_height / pp_height
                        new_pp_width = int(pp_width * scale_ratio)
                        resized_prev_page = cv2.resize(prev_page_img, (new_pp_width, ch_height))
                        # Horizontally stack the previous page to the left of the current structure highlight
                        combined_img = cv2.hconcat([resized_prev_page, current_highlight_img])
                        # Overwrite the original highlight image with our new combined image
                        cv2.imwrite(output_name, combined_img)

                segment_image = Image.fromarray(segment)
                segment_image.save(segment_name)

                if engine == 'molscribe':
                    smiles = model.predict_image_file(segment_name, return_atoms_bonds=True, return_confidence=True).get('smiles')
                elif engine == 'molnextr':
                    smiles = model.predict_final_results(segment_name, return_atoms_bonds=True, return_confidence=True).get('predicted_smiles')
                elif engine == 'molvec':
                    cmd = f'java -jar {MOLVEC} -f {segment_name} -o {segment_name}.sdf'
                    os.popen(cmd).read()
                    try:
                        sdf = Chem.SDMolSupplier(f'{segment_name}.sdf')
                        if len(sdf) != 0:
                            smiles = Chem.MolToSmiles(sdf[0])
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