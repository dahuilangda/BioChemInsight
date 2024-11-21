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

def extract_structures_from_pdf(pdf_file, page_start, page_end, output, engine='molvec'):
    images_dir = os.path.join(output, 'structure_images')
    segmented_dir = os.path.join(output, 'segment')

    # Prepare directories
    shutil.rmtree(segmented_dir, ignore_errors=True)
    create_directory(segmented_dir)

    # Split PDF into images
    split_pdf_to_images(pdf_file, images_dir, page_start=page_start, page_end=page_end)

    if engine == 'molscribe':
        from molscribe import MolScribe
        from huggingface_hub import hf_hub_download
        # Load MolScribe model
        print('Loading MolScribe model...')
        ckpt_path = hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth')
        model = MolScribe(ckpt_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    elif engine == 'molvec':
        from rdkit import Chem
    else:
        raise ValueError(f'Invalid engine: {engine}, must be "molscribe" or "molvec"')
    
    data_list = []
    for i in tqdm(range(page_start, page_end + 1), desc='Extracting structures'):
        try:
            scanned_page_file_path = os.path.join(images_dir, f'page_{i}.png')

            page = cv2.imread(scanned_page_file_path)
            masks = get_expanded_masks(page)
            segments, bboxes = apply_masks(page, masks)

            if len(segments) > 0:
                segments, bboxes, masks = sort_segments_bboxes(segments, bboxes, masks)

            for idx, segment in enumerate(segments):
                output_name = os.path.join(segmented_dir, f'highlight_{i}_{idx}.png')
                segment_name = os.path.join(segmented_dir, f'segment_{i}_{idx}.png')

                save_box_image(bboxes, masks, idx, page, output_name)

                segment_image = Image.fromarray(segment)
                segment_image.save(segment_name)

                if engine == 'molscribe':
                    smiles = model.predict_image_file(segment_name, return_atoms_bonds=True, return_confidence=True).get('smiles')
                elif engine == 'molvec':
                    cmd = f'java -jar {MOLVEC} -f {segment_name} -o {segment_name}.sdf'
                    os.popen(cmd).read()
                    try:
                        sdf = Chem.SDMolSupplier(f'{segment_name}.sdf')
                        if len(sdf) != 0:
                            smiles = Chem.MolToSmiles(sdf[0])
                    except Exception as e:
                        smiles = ''

                cpd_id_discription = structure_to_id(output_name)
                cpd_id = get_compound_id_from_description(cpd_id_discription)
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