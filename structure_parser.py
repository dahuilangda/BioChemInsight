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

# 并行处理相关导入
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures


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


def batch_structure_to_id(image_files, batch_size=4):
    """
    批量调用structure_to_id函数处理多个图像文件
    
    Args:
        image_files: 图像文件路径列表
        batch_size: 批处理大小，默认为4
    
    Returns:
        化合物ID列表
    """
    results = []
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(structure_to_id, image_file): image_file 
                         for image_file in image_files}
        
        # 收集结果
        for future in as_completed(future_to_file):
            image_file = future_to_file[future]
            try:
                cpd_id = future.result()
                if '```json' in cpd_id:
                    cpd_id = cpd_id.split('```json\n')[1].split('\n```')[0]
                    cpd_id = cpd_id.replace('{"COMPOUND_ID": "', '').replace('"}', '')
                results.append(cpd_id)
            except Exception as e:
                print(f"Error processing image {image_file}: {e}")
                results.append(None)
    
    return results

def batch_process_structure_ids(data_list, all_image_files, all_segment_info, batch_size=4):
    """
    批量处理结构图像以获取化合物ID
    
    Args:
        data_list: 包含页面数据的列表
        all_image_files: 所有需要处理的图像文件路径列表
        all_segment_info: 分段信息列表
        batch_size: 批处理大小，默认为4
        
    Returns:
        更新后的数据列表，包含化合物ID信息
    """
    if not all_image_files:
        return data_list
        
    # 使用已有的batch_structure_to_id函数批量获取化合物ID
    cpd_ids = batch_structure_to_id(all_image_files, batch_size)
    
    # 将获取到的化合物ID关联到对应的数据项
    for i, (data_idx, page_num, segment_idx) in enumerate(all_segment_info):
        if i < len(cpd_ids) and cpd_ids[i]:
            # 处理返回的JSON格式
            cpd_id = cpd_ids[i]
            if '```json' in cpd_id:
                try:
                    cpd_id = cpd_id.split('```json\n')[1].split('\n```')[0]
                    cpd_id = cpd_id.replace('{"COMPOUND_ID": "', '').replace('"}', '')
                except:
                    pass  # 如果解析失败，保持原始值
            
            # 更新数据列表中的条目
            if data_idx < len(data_list):
                data_list[data_idx]['COMPOUND_ID'] = cpd_id
                
    return data_list


def process_segment(engine, model, MOLVEC, segment, idx, i, segmented_dir, output_name, prev_page_path):
    """处理单个分割区域的函数，用于并行执行"""
    try:
        segment_name = os.path.join(segmented_dir, f'segment_{i}_{idx}.png')

        # 创建左右合并图片：前一页完整图（左）+ 当前高亮结构区域（右）
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
                return None
                
            # Ensure the array has the right shape and dtype
            if len(segment.shape) != 3:
                return None
                
            # Handle both RGB (3 channels) and RGBA (4 channels) images
            if segment.shape[2] == 4:
                # Convert RGBA to RGB by removing alpha channel
                segment = segment[:, :, :3]
            elif segment.shape[2] != 3:
                return None
            
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
                return None
                
        except Exception as e:
            print(f"Error extracting SMILES from segment {idx} on page {i}: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Extract SMILES from the saved segment image
        smiles = ''
        try:
            if engine == 'molscribe':
                smiles = model.predict_image_file(segment_name, return_atoms_bonds=True, return_confidence=True).get('smiles')
            elif engine == 'molnextr':
                smiles = model.predict_final_results(segment_name, return_atoms_bonds=True, return_confidence=True).get('predicted_smiles')
            elif engine == 'molvec':
                from rdkit import Chem
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

        # cpd_id_ = structure_to_id(output_name)
        # cpd_id = get_compound_id_from_description(cpd_id_)
        # cpd_id = structure_to_id(output_name)

        # if '```json' in cpd_id:
        #     cpd_id = cpd_id.split('```json\\n')[1].split('\\n```')[0]
        #     cpd_id = cpd_id.replace('{\"COMPOUND_ID\": \"', '').replace('\"}', '')

        # 返回处理结果，但不调用structure_to_id，将在后续批量处理
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
    """处理单个页面的函数，用于并行执行"""
    try:
        if progress_callback and page_idx is not None and total_pages is not None:
            progress_callback(page_idx + 1, total_pages, f"Processing page {i}")
        
        page = cv2.imread(scanned_page_file_path)
        if page is None:
            print(f"Warning: Could not read image for page {i} at {scanned_page_file_path}. Skipping.")
            return []

        masks = get_expanded_masks(page)
        segments, bboxes = apply_masks(page, masks)
        
        if len(segments) > 0:
            segments, bboxes, masks = sort_segments_bboxes(segments, bboxes, masks)

        page_data_list = []
        image_files = []  # 收集所有需要处理的图像文件
        segment_info = []  # 保存分割区域信息用于后续关联
        
        for idx, segment in enumerate(segments):
            output_name = os.path.join(segmented_dir, f'highlight_{i}_{idx}.png')
            try:
                save_box_image(bboxes, masks, idx, page, output_name)
            except Exception as e:
                print(f"Warning: Failed to save boxed image for segment {idx} on page {i}: {e}")
                # 创建一个简单的替代图像
                try:
                    # 创建一个带边框的简单图像作为替代
                    segment_copy = segment.copy()
                    if len(segment_copy.shape) == 3 and segment_copy.shape[2] >= 3:
                        # 在图像上绘制一个绿色边框
                        cv2.rectangle(segment_copy, (10, 10), (segment_copy.shape[1]-10, segment_copy.shape[0]-10), (0, 255, 0), 2)
                    cv2.imwrite(output_name, segment_copy)
                    print(f"Created fallback boxed image for segment {idx} on page {i}")
                except Exception as fallback_e:
                    print(f"Error: Failed to create fallback boxed image for segment {idx} on page {i}: {fallback_e}")
                    # 如果连替代方案都失败，就跳过这个分割区域
                    continue
            
            # 处理分割区域
            prev_page_path = os.path.join(images_dir, f'page_{i-1}.png')
            row_data = process_segment(engine, model, MOLVEC, segment, idx, i, segmented_dir, output_name, prev_page_path)
            if row_data:
                page_data_list.append(row_data)
                image_files.append(output_name)  # 收集图像文件
                segment_info.append((len(page_data_list) - 1, i, idx))  # 记录索引信息
                
        return page_data_list, image_files, segment_info
    except Exception as e:
        print(f"Error processing page {i}: {e}")
        return [], [], []

def extract_structures_from_pdf(pdf_file, page_start, page_end, output, engine='molnextr', progress_callback=None, batch_size=4):
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
    all_image_files = []
    all_segment_info = []
    total_pages = page_end - page_start + 1
    
    # 使用线程池并行处理页面
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        # 提交所有页面处理任务
        future_to_page = {}
        for page_idx, i in enumerate(range(page_start, page_end + 1)):
            scanned_page_file_path = os.path.join(images_dir, f'page_{i}.png')
            if os.path.exists(scanned_page_file_path):
                future = executor.submit(
                    process_page, 
                    engine, model, MOLVEC, i, scanned_page_file_path, 
                    segmented_dir, images_dir, progress_callback, total_pages, page_idx
                )
                future_to_page[future] = i
        
        # 收集结果
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                page_data, image_files, segment_info = future.result()
                data_list.extend(page_data)
                all_image_files.extend(image_files)
                all_segment_info.extend(segment_info)
            except Exception as e:
                print(f"Error collecting results for page {page_num}: {e}")
                continue

    # 批量处理化合物ID
    data_list = batch_process_structure_ids(data_list, all_image_files, all_segment_info, batch_size)
    
    return data_list