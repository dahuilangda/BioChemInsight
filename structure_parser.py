import os
import shutil
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import signal
import subprocess
import threading
import time

import torch
from decimer_segmentation import get_expanded_masks, apply_masks

from constants import MOLVEC
from utils.image_utils import save_box_image
from utils.pdf_utils import split_pdf_to_images
from utils.file_utils import create_directory
from utils.llm_utils import structure_to_id, get_compound_id_from_description

# 并行处理相关导入
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import concurrent.futures

# 全局预测锁，避免模型并发冲突
predict_lock = threading.Lock()

# 超时设置（秒）
MODEL_TIMEOUT = 300  # 5分钟
MOLECULE_PROCESSING_TIMEOUT = 60  # 1分钟
PAGE_PROCESSING_TIMEOUT = 600  # 10分钟



def sort_segments_bboxes(segments, bboxes, masks, same_row_pixel_threshold=50):
    """
    Sorts segments and bounding boxes in "reading order"
    """
    bbox_with_indices = [(bbox, idx) for idx, bbox in enumerate(bboxes)]
    sorted_bbox_with_indices = sorted(bbox_with_indices, key=lambda x: x[0][0])  # Sort by x

    rows = []
    current_row = [sorted_bbox_with_indices[0]]
    for bbox_with_idx in sorted_bbox_with_indices[1:]:
        if abs(bbox_with_idx[0][0] - current_row[-1][0][0]) < same_row_pixel_threshold:
            current_row.append(bbox_with_idx)
        else:
            rows.append(sorted(current_row, key=lambda x: x[0][1]))  # sort by y
            current_row = [bbox_with_idx]
    rows.append(sorted(current_row, key=lambda x: x[0][1]))

    sorted_bboxes = [bbox_with_idx[0] for row in rows for bbox_with_idx in row]
    sorted_indices = [bbox_with_idx[1] for row in rows for bbox_with_idx in row]

    sorted_segments = [segments[idx] for idx in sorted_indices]
    sorted_masks = [masks[:, :, idx] for idx in sorted_indices]
    sorted_masks = np.stack(sorted_masks, axis=-1)

    return sorted_segments, sorted_bboxes, sorted_masks


def batch_structure_to_id(image_files, batch_size=4):
    """
    批量调用structure_to_id函数处理多个图像文件
    """
    results = [None] * len(image_files)
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_index = {executor.submit(structure_to_id, image_file): i
                           for i, image_file in enumerate(image_files)}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                # 添加超时控制
                cpd_id = future.result(timeout=MODEL_TIMEOUT)
                if '```json' in cpd_id:
                    cpd_id = cpd_id.split('```json\n')[1].split('\n```')[0]
                    cpd_id = cpd_id.replace('{"COMPOUND_ID": "', '').replace('"}', '')
                results[index] = cpd_id
            except TimeoutError:
                print(f"Timeout processing image {image_files[index]}")
                results[index] = None
            except Exception as e:
                print(f"Error processing image {image_files[index]}: {e}")
                results[index] = None
    return results


def batch_process_structure_ids(data_list, all_image_files, all_segment_info, batch_size=4):
    """
    批量处理结构图像以获取化合物ID
    """
    if not all_image_files:
        return data_list

    print(f"Processing {len(all_image_files)} images for compound IDs...")
    cpd_ids = batch_structure_to_id(all_image_files, batch_size)
    print(f"Received {len(cpd_ids)} compound IDs from batch processing")

    for i, (data_idx, page_num, segment_idx) in enumerate(all_segment_info):
        if i < len(cpd_ids) and cpd_ids[i] is not None:
            cpd_id = cpd_ids[i]
            if isinstance(cpd_id, str) and '```json' in cpd_id:
                try:
                    cpd_id = cpd_id.split('```json\n')[1].split('\n```')[0]
                    cpd_id = cpd_id.replace('{"COMPOUND_ID": "', '').replace('"}', '')
                except:
                    pass
            if data_idx < len(data_list):
                image_file = all_image_files[i] if i < len(all_image_files) else "unknown"
                print(f"Assigning compound ID '{cpd_id}' to data item {data_idx} (page {page_num}, segment {segment_idx}) from image {image_file}")
                data_list[data_idx]['COMPOUND_ID'] = cpd_id
            else:
                print(f"Warning: data_idx {data_idx} is out of range")
        else:
            print(f"Warning: No compound ID for item {i} (data_idx {data_idx})")
    return data_list


def process_segment(engine, model, MOLVEC, segment, idx, i, segmented_dir, output_name, prev_page_path):
    """处理单个分割区域"""
    try:
        segment_name = os.path.join(segmented_dir, f'segment_{i}_{idx}.png')

        # 拼接前一页和当前高亮
        if os.path.exists(prev_page_path):
            current_highlight_img = cv2.imread(output_name)
            prev_page_img = cv2.imread(prev_page_path)
            if current_highlight_img is not None and prev_page_img is not None:
                ch_height = current_highlight_img.shape[0]
                pp_height, pp_width, _ = prev_page_img.shape
                scale_ratio = ch_height / pp_height
                new_pp_width = int(pp_width * scale_ratio)
                resized_prev_page = cv2.resize(prev_page_img, (new_pp_width, ch_height))
                combined_img = cv2.hconcat([resized_prev_page, current_highlight_img])
                cv2.imwrite(output_name, combined_img)

        if not isinstance(segment, np.ndarray) or len(segment.shape) != 3:
            return None
        if segment.shape[2] == 4:
            segment = segment[:, :, :3]
        elif segment.shape[2] != 3:
            return None
        if segment.dtype != np.uint8:
            if segment.max() <= 1.0:
                segment = (segment * 255).astype(np.uint8)
            else:
                segment = segment.astype(np.uint8)

        segment_image = Image.fromarray(segment)
        segment_image.save(segment_name)
        if not os.path.exists(segment_name):
            return None

        smiles = ''
        # 模型调用必须串行
        with predict_lock:
            try:
                if engine == 'molscribe':
                    # 使用超时控制的线程来运行模型预测
                    def predict_molscribe():
                        return model.predict_image_file(segment_name, return_atoms_bonds=True, return_confidence=True).get('smiles')
                    
                    # 创建一个线程来运行预测
                    import threading
                    result_container = [None]
                    exception_container = [None]
                    
                    def run_predict():
                        try:
                            result_container[0] = predict_molscribe()
                        except Exception as e:
                            exception_container[0] = e
                    
                    predict_thread = threading.Thread(target=run_predict)
                    predict_thread.start()
                    predict_thread.join(timeout=MODEL_TIMEOUT)
                    
                    if predict_thread.is_alive():
                        print(f"Timeout processing segment {idx} on page {i} with molscribe")
                        smiles = ''
                    elif exception_container[0]:
                        raise exception_container[0]
                    else:
                        smiles = result_container[0]
                        
                elif engine == 'molnextr':
                    # 使用超时控制的线程来运行模型预测
                    def predict_molnextr():
                        return model.predict_final_results(segment_name, return_atoms_bonds=True, return_confidence=True).get('predicted_smiles')
                    
                    # 创建一个线程来运行预测
                    import threading
                    result_container = [None]
                    exception_container = [None]
                    
                    def run_predict():
                        try:
                            result_container[0] = predict_molnextr()
                        except Exception as e:
                            exception_container[0] = e
                    
                    predict_thread = threading.Thread(target=run_predict)
                    predict_thread.start()
                    predict_thread.join(timeout=MODEL_TIMEOUT)
                    
                    if predict_thread.is_alive():
                        print(f"Timeout processing segment {idx} on page {i} with molnextr")
                        smiles = ''
                    elif exception_container[0]:
                        raise exception_container[0]
                    else:
                        smiles = result_container[0]
                elif engine == 'molvec':
                    from rdkit import Chem
                    cmd = f'java -jar {MOLVEC} -f {segment_name} -o {segment_name}.sdf'
                    try:
                        subprocess.run(cmd, shell=True, timeout=MOLECULE_PROCESSING_TIMEOUT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        sdf = Chem.SDMolSupplier(f'{segment_name}.sdf')
                        if len(sdf) != 0:
                            smiles = Chem.MolToSmiles(sdf[0])
                    except subprocess.TimeoutExpired:
                        print(f"Timeout processing segment {idx} on page {i} with molvec")
                        smiles = ''
                    except Exception as e:
                        print(f"Error reading SDF for segment {idx} on page {i}: {e}")
                        smiles = ''
            except Exception as e:
                print(f"Error processing segment {idx} on page {i}: {e}")
                smiles = ''

        row_data = {
            'PAGE_NUM': i,
            'SMILES': smiles,
            'IMAGE_FILE': output_name,
            'SEGMENT_FILE': segment_name
        }
        return row_data
    except Exception as e:
        print(f"Error processing segment {idx} on page {i}: {e}")
        return None


def process_page(engine, model, MOLVEC, i, scanned_page_file_path, segmented_dir, images_dir, progress_callback=None, total_pages=None, page_idx=None):
    """处理单个页面"""
    # 创建一个线程来运行页面处理
    result_container = [None]
    exception_container = [None]
    
    def run_process_page():
        try:
            if progress_callback and page_idx is not None and total_pages is not None:
                progress_callback(page_idx + 1, total_pages, f"Processing page {i}")

            page = cv2.imread(scanned_page_file_path)
            if page is None:
                print(f"Warning: Could not read image for page {i}")
                result_container[0] = ([], [], [])
                return

            masks = get_expanded_masks(page)
            segments, bboxes = apply_masks(page, masks)
            if len(segments) > 0:
                segments, bboxes, masks = sort_segments_bboxes(segments, bboxes, masks)

            page_data_list = []
            image_files = []
            segment_info = []

            for idx, segment in enumerate(segments):
                output_name = os.path.join(segmented_dir, f'highlight_{i}_{idx}.png')
                box_coords_path = os.path.join(segmented_dir, f'highlight_{i}_{idx}.json')
                try:
                    save_box_image(bboxes, masks, idx, page, output_name)
                    
                    # Save the bounding box coordinates to a JSON file as an object
                    with open(box_coords_path, 'w') as f_json:
                        import json
                        json.dump({"box": bboxes[idx]}, f_json)

                except Exception as e:
                    print(f"Warning: Failed to save boxed image or coords for segment {idx} on page {i}: {e}")

                prev_page_path = os.path.join(images_dir, f'page_{i-1}.png')
                row_data = process_segment(engine, model, MOLVEC, segment, idx, i, segmented_dir, output_name, prev_page_path)
                if row_data:
                    row_data['BOX_COORDS_FILE'] = box_coords_path
                    row_data['PAGE_IMAGE_FILE'] = scanned_page_file_path
                    page_data_list.append(row_data)
                    image_files.append(output_name)
                    segment_info.append((len(page_data_list) - 1, i, idx))

            result_container[0] = (page_data_list, image_files, segment_info)
        except Exception as e:
            print(f"Error processing page {i}: {e}")
            exception_container[0] = e
            result_container[0] = ([], [], [])
    
    # 在单独的线程中运行页面处理
    process_thread = threading.Thread(target=run_process_page)
    process_thread.start()
    process_thread.join(timeout=PAGE_PROCESSING_TIMEOUT)
    
    # 检查线程是否超时
    if process_thread.is_alive():
        print(f"Timeout processing page {i}, terminating...")
        # 注意：这里无法真正终止线程，但至少可以返回默认值
        return [], [], []
    elif exception_container[0]:
        raise exception_container[0]
    else:
        return result_container[0]


def extract_structures_from_pdf(pdf_file, page_start, page_end, output, engine='molnextr', progress_callback=None, batch_size=4):
    images_dir = os.path.join(output, 'structure_images')
    segmented_dir = os.path.join(output, 'segment')

    shutil.rmtree(segmented_dir, ignore_errors=True)
    create_directory(segmented_dir)

    extraction_start_page = max(1, page_start - 1)
    split_pdf_to_images(pdf_file, images_dir, page_start=extraction_start_page, page_end=page_end)

    if engine == 'molscribe':
        from molscribe import MolScribe
        from huggingface_hub import hf_hub_download
        print('Loading MolScribe model...')
        ckpt_path = hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth', local_dir="./models")
        model = MolScribe(ckpt_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    elif engine == 'molvec':
        from rdkit import Chem
        model = None
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
            try:
                from huggingface_hub import hf_hub_download
                print('正在下载 MolNexTR 模型...')
                ckpt_path = hf_hub_download('CYF200127/MolNexTR', 'molnextr_best.pth',
                                            repo_type='dataset', local_dir="./models")
            except Exception as e:
                raise FileNotFoundError(f'MolNexTR model not found. Error: {e}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Loading MolNexTR model from: {ckpt_path}')
        model = molnextr(ckpt_path, device)
    else:
        raise ValueError(f'Invalid engine: {engine}')

    data_list = []
    all_image_files = []
    all_segment_info = []
    total_pages = page_end - page_start + 1

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_page = {}
        page_order = []
        for page_idx, i in enumerate(range(page_start, page_end + 1)):
            scanned_page_file_path = os.path.join(images_dir, f'page_{i}.png')
            if os.path.exists(scanned_page_file_path):
                future = executor.submit(
                    process_page,
                    engine, model, MOLVEC, i, scanned_page_file_path,
                    segmented_dir, images_dir, progress_callback, total_pages, page_idx
                )
                future_to_page[future] = (i, page_idx)
                page_order.append((i, page_idx))

        page_results = {}
        for future in as_completed(future_to_page):
            page_num, page_idx = future_to_page[future]
            try:
                # 添加超时控制
                page_data, image_files, segment_info = future.result(timeout=MODEL_TIMEOUT)
                page_results[page_idx] = (page_data, image_files, segment_info)
            except TimeoutError:
                print(f"Timeout collecting results for page {page_num}")
                page_results[page_idx] = ([], [], [])
            except Exception as e:
                print(f"Error collecting results for page {page_num}: {e}")
                page_results[page_idx] = ([], [], [])

        data_list_offset = 0
        for page_num, page_idx in page_order:
            if page_idx in page_results:
                page_data, image_files, segment_info = page_results[page_idx]
                adjusted_segment_info = []
                for local_data_idx, p_num, segment_idx in segment_info:
                    global_data_idx = data_list_offset + local_data_idx
                    adjusted_segment_info.append((global_data_idx, p_num, segment_idx))
                data_list.extend(page_data)
                all_image_files.extend(image_files)
                all_segment_info.extend(adjusted_segment_info)
                data_list_offset += len(page_data)

    data_list = batch_process_structure_ids(data_list, all_image_files, all_segment_info, batch_size)
    return data_list
