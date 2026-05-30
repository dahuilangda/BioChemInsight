import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64,garbage_collection_threshold:0.8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
import shutil
import json
import re
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import signal
import threading
import ctypes
import time

import torch
from utils.structure_recognition import StructureRecognizer, normalize_segment_array

import constants as project_constants
from utils.image_utils import save_box_image
from utils.pdf_utils import split_pdf_to_images
from utils.file_utils import create_directory
from utils.llm_utils import (
    build_label_only_structure_role_review_prompt,
    build_review_structure_id_prompt,
    build_structure_to_id_prompt,
    classify_structure_candidate,
    parse_visual_structure_id_payload,
    run_vision_json_task,
    structure_to_id,
)

# 并行处理相关导入
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import concurrent.futures

# 全局 GPU 锁，避免 MolNexTR 并发冲突和显存碎片化
predict_lock = threading.Lock()

# 超时设置（秒）。These are intentionally configurable because a slow or
# unresponsive visual model server can otherwise make every page wait for the
# old 10-minute hard limit.
MODEL_TIMEOUT = int(getattr(project_constants, 'STRUCTURE_MODEL_TIMEOUT_SECONDS', 180))
PAGE_PROCESSING_TIMEOUT = int(getattr(project_constants, 'STRUCTURE_PAGE_PROCESSING_TIMEOUT_SECONDS', 240))
STRUCTURE_FILTER_ENABLED = bool(getattr(project_constants, 'STRUCTURE_FILTER_ENABLED', True))
SAVE_FILTERED_STRUCTURES = bool(getattr(project_constants, 'SAVE_FILTERED_STRUCTURES', True))
DEFAULT_STRUCTURE_FILTER_STRICTNESS = str(getattr(project_constants, 'STRUCTURE_FILTER_STRICTNESS', 'strict')).strip().lower()
DEFAULT_STRUCTURE_PAGE_WORKERS = int(getattr(project_constants, 'STRUCTURE_PAGE_WORKERS', 0) or 0)
DEFAULT_STRUCTURE_ID_BATCH_SIZE = int(getattr(project_constants, 'STRUCTURE_ID_BATCH_SIZE', 0) or 0)
DEFAULT_STRUCTURE_ID_MAX_INFLIGHT = int(getattr(project_constants, 'STRUCTURE_ID_MAX_INFLIGHT', 0) or 0)
DEFAULT_STRUCTURE_PAGE_MAX_INFLIGHT = int(getattr(project_constants, 'STRUCTURE_PAGE_MAX_INFLIGHT', 0) or 0)
STRUCTURE_ID_VERIFIER_ENABLED = bool(getattr(project_constants, 'STRUCTURE_ID_VERIFIER_ENABLED', True))


def bbox_yxyx_to_xyxy(bbox):
    if bbox is None or len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox}")
    y1, x1, y2, x2 = [int(v) for v in bbox]
    return x1, y1, x2, y2


def get_available_memory_gb():
    try:
        if os.path.exists('/proc/meminfo'):
            with open('/proc/meminfo', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        parts = line.split()
                        if len(parts) >= 2:
                            kb_value = int(parts[1])
                            return kb_value / (1024 * 1024)
    except Exception:
        pass
    return None


def resolve_structure_runtime_settings(batch_size=4, page_workers=None, id_batch_size=None):
    requested_batch_size = max(1, int(batch_size or 1))
    requested_page_workers = int(page_workers or DEFAULT_STRUCTURE_PAGE_WORKERS or 0)
    requested_id_batch_size = int(id_batch_size or DEFAULT_STRUCTURE_ID_BATCH_SIZE or 0)

    available_memory_gb = get_available_memory_gb()
    if requested_page_workers > 0:
        resolved_page_workers = requested_page_workers
    else:
        if torch.cuda.is_available():
            memory_cap = 1
        elif available_memory_gb is None:
            memory_cap = 2
        elif available_memory_gb < 8:
            memory_cap = 1
        elif available_memory_gb < 16:
            memory_cap = 2
        elif available_memory_gb < 32:
            memory_cap = 3
        else:
            memory_cap = 4
        resolved_page_workers = min(requested_batch_size, memory_cap)

    if requested_id_batch_size > 0:
        resolved_id_batch_size = requested_id_batch_size
    else:
        resolved_id_batch_size = min(requested_batch_size, 4)

    resolved_page_workers = max(1, resolved_page_workers)
    resolved_id_batch_size = max(1, resolved_id_batch_size)

    print(
        "Structure runtime settings: "
        f"requested_batch_size={requested_batch_size}, "
        f"page_workers={resolved_page_workers}, "
        f"id_batch_size={resolved_id_batch_size}, "
        f"available_memory_gb={available_memory_gb:.2f}" if available_memory_gb is not None
        else "Structure runtime settings: "
             f"requested_batch_size={requested_batch_size}, "
             f"page_workers={resolved_page_workers}, "
             f"id_batch_size={resolved_id_batch_size}, "
             "available_memory_gb=unknown"
    )
    return resolved_page_workers, resolved_id_batch_size


def parse_compound_id_response(raw_value):
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        value = str(raw_value).strip()
        return value or None

    value = raw_value.strip()
    if not value:
        return None

    if '```' in value:
        fence_parts = value.split('```')
        for part in fence_parts:
            stripped_part = part.strip()
            if stripped_part.startswith('{') and stripped_part.endswith('}'):
                value = stripped_part
                break
            if '\n' in stripped_part:
                maybe_json = stripped_part.split('\n', 1)[1].strip()
                if maybe_json.startswith('{') and maybe_json.endswith('}'):
                    value = maybe_json
                    break

    try:
        payload = json.loads(value)
        if isinstance(payload, dict):
            for key in ('COMPOUND_ID', 'compound_id', 'VALUE', 'value', 'ID', 'id', 'answer', 'Answer'):
                if key in payload:
                    candidate = payload[key]
                    if candidate is None:
                        continue
                    return parse_compound_id_response(str(candidate).strip())
    except Exception:
        pass

    compact_value = re.sub(r'\s+', ' ', value).strip()
    if not compact_value:
        return None
    if compact_value.lower() in {'none', 'null'} or compact_value.lower().startswith('error:'):
        return None
    return compact_value


def parse_structure_id_response(raw_value):
    """Parse legacy string or structured visual-ID model output."""
    metadata = {
        'compound_id': None,
        'visual_role': '',
        'id_source': '',
        'evidence': '',
        'confidence': '',
        'raw_response': raw_value if isinstance(raw_value, str) else '',
    }
    if raw_value is None:
        return metadata

    value = str(raw_value).strip() if not isinstance(raw_value, dict) else raw_value
    payload = value if isinstance(value, dict) else None
    if isinstance(value, str) and value:
        json_text = value
        if '```' in json_text:
            for part in json_text.split('```'):
                stripped_part = part.strip()
                if stripped_part.startswith('{') and stripped_part.endswith('}'):
                    json_text = stripped_part
                    break
                if '\n' in stripped_part:
                    maybe_json = stripped_part.split('\n', 1)[1].strip()
                    if maybe_json.startswith('{') and maybe_json.endswith('}'):
                        json_text = maybe_json
                        break
        try:
            maybe_payload = json.loads(json_text)
            if isinstance(maybe_payload, dict):
                payload = maybe_payload
        except Exception:
            payload = None

    if isinstance(payload, dict):
        for key in ('COMPOUND_ID', 'compound_id', 'VALUE', 'value', 'ID', 'id', 'answer', 'Answer'):
            if key in payload:
                metadata['compound_id'] = parse_compound_id_response(payload.get(key))
                break
        metadata['visual_role'] = str(payload.get('VISUAL_ROLE') or payload.get('visual_role') or '').strip()
        metadata['id_source'] = str(payload.get('ID_SOURCE') or payload.get('id_source') or '').strip()
        metadata['evidence'] = str(payload.get('EVIDENCE') or payload.get('evidence') or '').strip()
        metadata['confidence'] = str(payload.get('CONFIDENCE') or payload.get('confidence') or 'medium').strip()
        contract_error = str(payload.get('OUTPUT_CONTRACT_ERROR') or payload.get('output_contract_error') or '').strip()
        if not metadata['raw_response']:
            metadata['raw_response'] = str(payload.get('RAW_RESPONSE') or payload.get('raw_response') or '').strip()
        if not metadata['raw_response']:
            metadata['raw_response'] = json.dumps(payload, ensure_ascii=False)
        if contract_error:
            metadata['output_contract_error'] = contract_error
        return metadata

    return metadata


def is_nonfinal_compound_id(value):
    """Return True when an extracted ID is not a target/final compound for downstream SAR."""
    compact_value = parse_compound_id_response(value)
    if not compact_value:
        return False
    return bool(re.match(r'^\s*(intermediate|int\.?|preparation|embodiment)\b', compact_value, flags=re.IGNORECASE))


def is_usable_final_compound_id(value):
    """Return True when an extracted ID is usable as a final target compound ID."""
    compact_value = parse_compound_id_response(value)
    if not compact_value:
        return False
    if compact_value.lower() in {'none', 'null', 'unknown', 'n/a', 'na'}:
        return False
    if is_nonfinal_compound_id(compact_value):
        return False
    return True


def is_nonfinal_visual_role(value):
    role = str(value or '').strip().lower()
    return role in {'reactant', 'intermediate', 'reagent_or_condition'}


def is_label_only_product_role(id_result):
    """Return True for product decisions grounded only in a nearby label."""
    if not isinstance(id_result, dict):
        return False
    role = str(id_result.get('visual_role') or '').strip().lower()
    source = str(id_result.get('id_source') or '').strip().lower()
    evidence = str(id_result.get('evidence') or '').strip().lower()
    if role not in {'product', 'final_product'} or source != 'local_label':
        return False
    product_context_terms = (
        'title', 'obtained', 'afforded', 'final', 'last arrow', 'peak',
        'enantiomer', 'result', 'product text', 'heading governs',
    )
    return not any(term in evidence for term in product_context_terms)


def review_label_only_structure_role(image_file, initial_result):
    """Second-pass role check for label-only product/final-product decisions."""
    compound_id = str(initial_result.get('compound_id') or '').strip()
    if not compound_id:
        return initial_result

    base_prompt = build_structure_to_id_prompt()
    review_prompt = build_label_only_structure_role_review_prompt(base_prompt, compound_id)
    try:
        def _parser(response_text):
            payload = parse_visual_structure_id_payload(response_text)
            payload['RAW_RESPONSE'] = response_text or ''
            return payload

        reviewed_payload = run_vision_json_task(
            task_name='structure_to_id',
            image_file=image_file,
            prompt=review_prompt,
            parser=_parser,
            metadata={'scope': 'label_only_role_review', 'compound_id': compound_id},
        )
        reviewed = parse_structure_id_response(reviewed_payload)
    except Exception as e:
        print(f"Warning: second-pass structure role review failed for {image_file}: {e}")
        return initial_result
    if reviewed.get('compound_id'):
        return reviewed
    return initial_result


def verify_structure_id_with_visual_review(image_file, initial_result):
    """Second-pass visual verifier for Compound ID attachment and completeness."""
    if not STRUCTURE_ID_VERIFIER_ENABLED:
        return initial_result
    if not isinstance(initial_result, dict):
        return initial_result

    compound_id = str(initial_result.get('compound_id') or '').strip()
    if not compound_id or compound_id.lower() == 'none':
        return initial_result

    base_prompt = build_structure_to_id_prompt()
    review_payload = {
        'COMPOUND_ID': compound_id,
        'VISUAL_ROLE': initial_result.get('visual_role', ''),
        'ID_SOURCE': initial_result.get('id_source', ''),
        'EVIDENCE': initial_result.get('evidence', ''),
        'CONFIDENCE': initial_result.get('confidence', ''),
    }
    review_prompt = build_review_structure_id_prompt(base_prompt, review_payload)
    def _parser(response_text):
        payload = parse_visual_structure_id_payload(response_text)
        payload['RAW_RESPONSE'] = response_text or ''
        return payload

    reviewed_payload = run_vision_json_task(
        task_name='structure_to_id',
        image_file=image_file,
        prompt=review_prompt,
        parser=_parser,
        metadata={'scope': 'structure_id_verifier', 'compound_id': compound_id},
    )
    reviewed = parse_structure_id_response(reviewed_payload)
    reviewed_id_text = str(reviewed_payload.get('COMPOUND_ID') or '').strip() if isinstance(reviewed_payload, dict) else ''

    if reviewed.get('compound_id') is not None or reviewed_id_text.lower() == 'none':
        previous_raw = initial_result.get('raw_response', '')
        reviewed['initial_compound_id'] = compound_id
        reviewed['initial_id_evidence'] = initial_result.get('evidence', '')
        reviewed['initial_id_confidence'] = initial_result.get('confidence', '')
        if previous_raw and reviewed.get('raw_response') != previous_raw:
            reviewed['initial_raw_response'] = previous_raw
        return reviewed
    return initial_result


def resolve_structure_id_candidate(
    image_file,
    id_extract_fn=structure_to_id,
):
    raw_result = id_extract_fn(image_file)
    if isinstance(raw_result, str):
        try:
            strict_payload = parse_visual_structure_id_payload(raw_result)
            strict_payload['RAW_RESPONSE'] = raw_result or ''
            raw_result = strict_payload
        except Exception as e:
            raw_result = {
                'COMPOUND_ID': 'None',
                'VISUAL_ROLE': 'unknown',
                'ID_SOURCE': 'none',
                'EVIDENCE': f'invalid structure_to_id contract: {e}',
                'CONFIDENCE': 'low',
                'RAW_RESPONSE': raw_result or '',
                'OUTPUT_CONTRACT_ERROR': str(e),
            }
    id_result = parse_structure_id_response(raw_result)
    if is_label_only_product_role(id_result):
        id_result = review_label_only_structure_role(image_file, id_result)
    id_result = verify_structure_id_with_visual_review(image_file, id_result)
    return id_result


def batch_structure_to_id(image_jobs, batch_size=4):
    """
    批量解析结构 ID。
    """
    results = [None] * len(image_jobs)
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        max_inflight = DEFAULT_STRUCTURE_ID_MAX_INFLIGHT if DEFAULT_STRUCTURE_ID_MAX_INFLIGHT > 0 else max(batch_size, batch_size * 2)
        future_to_index = {}
        submit_idx = 0

        while submit_idx < len(image_jobs) and len(future_to_index) < max_inflight:
            future = executor.submit(resolve_structure_id_candidate, **image_jobs[submit_idx])
            future_to_index[future] = submit_idx
            submit_idx += 1

        while future_to_index:
            done_futures, _ = concurrent.futures.wait(
                future_to_index.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done_futures:
                index = future_to_index.pop(future)
                try:
                    cpd_id = future.result(timeout=MODEL_TIMEOUT)
                    results[index] = cpd_id
                except TimeoutError:
                    print(f"Timeout processing image job {image_jobs[index]}")
                    results[index] = None
                except Exception as e:
                    print(f"Error processing image job {image_jobs[index]}: {e}")
                    results[index] = None

                if submit_idx < len(image_jobs):
                    next_future = executor.submit(resolve_structure_id_candidate, **image_jobs[submit_idx])
                    future_to_index[next_future] = submit_idx
                    submit_idx += 1
    return results


def batch_process_structure_ids(data_list, all_image_files, all_segment_info, batch_size=4):
    """
    批量处理结构图像以获取化合物ID
    """
    if not all_image_files:
        return data_list

    pending_image_files = []
    pending_segment_info = []

    for i, (data_idx, page_num, segment_idx) in enumerate(all_segment_info):
        if data_idx >= len(data_list):
            print(f"Warning: data_idx {data_idx} is out of range")
            continue

        existing_id = parse_compound_id_response(data_list[data_idx].get('COMPOUND_ID'))
        if existing_id and existing_id.lower() != 'none':
            data_list[data_idx]['COMPOUND_ID'] = existing_id
            continue

        if i < len(all_image_files):
            pending_image_files.append(all_image_files[i])
            pending_segment_info.append((data_idx, page_num, segment_idx))

    if not pending_image_files:
        print("All compound IDs were already resolved.")
        return data_list

    print(f"Processing {len(pending_image_files)} images for compound IDs...")
    chunk_size = max(batch_size, batch_size * 4)
    processed_count = 0
    for start in range(0, len(pending_image_files), chunk_size):
        end = min(len(pending_image_files), start + chunk_size)
        chunk_image_files = pending_image_files[start:end]
        chunk_segment_info = pending_segment_info[start:end]
        id_results = batch_structure_to_id(
            [{'image_file': image_file} for image_file in chunk_image_files],
            batch_size,
        )
        print(f"Received {len(id_results)} compound ID results from batch processing chunk {start}:{end}")

        for i, (data_idx, page_num, segment_idx) in enumerate(chunk_segment_info):
            if i < len(id_results) and id_results[i] is not None:
                id_result = id_results[i] if isinstance(id_results[i], dict) else parse_structure_id_response(id_results[i])
                cpd_id = id_result.get('compound_id')
                image_file = chunk_image_files[i] if i < len(chunk_image_files) else "unknown"
                data_list[data_idx]['VISUAL_ROLE'] = id_result.get('visual_role', '')
                data_list[data_idx]['ID_SOURCE'] = id_result.get('id_source', '')
                data_list[data_idx]['ID_EVIDENCE'] = id_result.get('evidence', '')
                data_list[data_idx]['ID_CONFIDENCE'] = id_result.get('confidence', '')
                data_list[data_idx]['ID_RAW_RESPONSE'] = id_result.get('raw_response', '')
                data_list[data_idx]['INITIAL_COMPOUND_ID'] = id_result.get('initial_compound_id', '')
                data_list[data_idx]['INITIAL_ID_EVIDENCE'] = id_result.get('initial_id_evidence', '')
                data_list[data_idx]['INITIAL_ID_CONFIDENCE'] = id_result.get('initial_id_confidence', '')
                data_list[data_idx]['INITIAL_ID_RAW_RESPONSE'] = id_result.get('initial_raw_response', '')
                if cpd_id is not None:
                    print(f"Assigning compound ID '{cpd_id}' to data item {data_idx} (page {page_num}, segment {segment_idx}) from image {image_file}")
                    data_list[data_idx]['COMPOUND_ID'] = cpd_id
                else:
                    print(f"Warning: No compound ID for item {processed_count + i} (data_idx {data_idx})")
            else:
                print(f"Warning: No compound ID for item {processed_count + i} (data_idx {data_idx})")
        processed_count += len(chunk_segment_info)
        gc.collect()
    return data_list


def resolve_structure_id_jobs(id_jobs, batch_size=4):
    """
    使用可变 row 引用流式回填化合物 ID，避免依赖全局 data_idx 累积。
    """
    if not id_jobs:
        return 0

    pending_jobs = []
    for job in id_jobs:
        row = job.get('row') or {}
        existing_id = parse_compound_id_response(row.get('COMPOUND_ID'))
        if existing_id and existing_id.lower() != 'none':
            row['COMPOUND_ID'] = existing_id
            continue
        image_file = job.get('image_file')
        if image_file:
            pending_jobs.append(job)

    if not pending_jobs:
        print("All compound IDs in streamed jobs were already resolved.")
        return 0

    print(f"Processing {len(pending_jobs)} streamed images for compound IDs...")
    chunk_size = max(batch_size, batch_size * 4)
    resolved_count = 0
    for start in range(0, len(pending_jobs), chunk_size):
        end = min(len(pending_jobs), start + chunk_size)
        chunk_jobs = pending_jobs[start:end]
        chunk_image_jobs = [
            {
                'image_file': job['image_file'],
            }
            for job in chunk_jobs
        ]
        id_results = batch_structure_to_id(chunk_image_jobs, batch_size)
        print(f"Received {len(id_results)} compound ID results from streamed batch chunk {start}:{end}")

        for i, job in enumerate(chunk_jobs):
            row = job.get('row') or {}
            page_num = job.get('page_num')
            segment_idx = job.get('segment_idx')
            image_file = job.get('image_file', 'unknown')
            if i < len(id_results) and id_results[i] is not None:
                id_result = id_results[i] if isinstance(id_results[i], dict) else parse_structure_id_response(id_results[i])
                cpd_id = id_result.get('compound_id')
                row['VISUAL_ROLE'] = id_result.get('visual_role', '')
                row['ID_SOURCE'] = id_result.get('id_source', '')
                row['ID_EVIDENCE'] = id_result.get('evidence', '')
                row['ID_CONFIDENCE'] = id_result.get('confidence', '')
                row['ID_RAW_RESPONSE'] = id_result.get('raw_response', '')
                row['INITIAL_COMPOUND_ID'] = id_result.get('initial_compound_id', '')
                row['INITIAL_ID_EVIDENCE'] = id_result.get('initial_id_evidence', '')
                row['INITIAL_ID_CONFIDENCE'] = id_result.get('initial_id_confidence', '')
                row['INITIAL_ID_RAW_RESPONSE'] = id_result.get('initial_raw_response', '')
                if cpd_id is not None:
                    row['COMPOUND_ID'] = cpd_id
                    print(f"Assigning compound ID '{cpd_id}' to streamed row (page {page_num}, segment {segment_idx}) from image {image_file}")
                    resolved_count += 1
                else:
                    print(f"Warning: No streamed compound ID for page {page_num}, segment {segment_idx}")
            else:
                print(f"Warning: No streamed compound ID for page {page_num}, segment {segment_idx}")
        gc.collect()
    return resolved_count


def classify_segment_image(
    segment_name,
    idx,
    page_num,
    structure_filter_strictness=DEFAULT_STRUCTURE_FILTER_STRICTNESS,
):
    if not STRUCTURE_FILTER_ENABLED:
        return {
            'structure_type': 'complete_compound',
            'is_complete_compound': True,
            'reason': 'structure_filter_disabled',
            'raw_response': '',
            'border_contact': {'suspicious': False, 'sides': [], 'ratios': {}},
            'strictness': structure_filter_strictness,
        }

    try:
        result = classify_structure_candidate(segment_name, strictness=structure_filter_strictness)
        structure_type = result.get('structure_type', 'uncertain')
        is_complete_compound = bool(result.get('is_complete_compound'))

        print(
            f"Segment {idx} on page {page_num} classified as {structure_type} "
            f"(complete={is_complete_compound})"
        )
        return result
    except Exception as e:
        print(f"Error classifying segment {idx} on page {page_num}: {e}")
        return {
            'structure_type': 'uncertain',
            'is_complete_compound': False,
            'reason': f'classification_error: {e}',
            'raw_response': '',
            'border_contact': {'suspicious': False, 'sides': [], 'ratios': {}},
            'strictness': structure_filter_strictness,
        }


def process_segment(
    recognizer,
    segment,
    idx,
    i,
    segmented_dir,
    output_name,
    prev_page_path,
    segment_name=None,
    id_image_name=None,
    structure_filter_strictness=DEFAULT_STRUCTURE_FILTER_STRICTNESS,
):
    """处理单个分割区域"""
    try:
        segment_name = segment_name or os.path.join(segmented_dir, f'segment_{i}_{idx}.png')

        # Keep a current-page-only highlight for visual ID extraction.  The
        # two-page stitched image is useful for cross-page display/context, but
        # it halves effective resolution for small labels on the current page.
        if id_image_name and os.path.exists(output_name):
            try:
                shutil.copyfile(output_name, id_image_name)
            except Exception as e:
                print(f"Warning: Failed to save ID highlight image {id_image_name}: {e}")

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

        if segment is not None:
            segment = normalize_segment_array(segment)
            if segment is None:
                return None
            cv2.imwrite(segment_name, segment)

        if not os.path.exists(segment_name):
            return None

        classification = classify_segment_image(
            segment_name,
            idx,
            i,
            structure_filter_strictness=structure_filter_strictness,
        )
        structure_type = classification.get('structure_type', 'uncertain')
        is_complete_compound = bool(classification.get('is_complete_compound'))
        structure_filter_reason = classification.get('reason', '')
        structure_filter_raw_response = classification.get('raw_response', '')
        border_contact = classification.get('border_contact') or {}
        structure_filter_strictness = classification.get('strictness', structure_filter_strictness)
        border_contact_sides = ','.join(border_contact.get('sides', [])) if isinstance(border_contact, dict) else ''

        if not is_complete_compound:
            print(f"Skipping segment {idx} on page {i} because it is classified as {structure_type}")
            return {
                'PAGE_NUM': i,
                'SMILES': '',
                'IMAGE_FILE': output_name,
                'SEGMENT_FILE': segment_name,
                'STRUCTURE_TYPE': structure_type,
                'IS_COMPLETE_COMPOUND': False,
                'STRUCTURE_FILTER_REASON': structure_filter_reason,
                'STRUCTURE_FILTER_RAW_RESPONSE': structure_filter_raw_response,
                'STRUCTURE_FILTER_BORDER_SIDES': border_contact_sides,
                'STRUCTURE_FILTER_STRICTNESS': structure_filter_strictness,
                'FILTERED_OUT': True,
            }

        smiles = ''
        molblock = ''
        # 模型调用必须串行
        with predict_lock:
            try:
                prediction = recognizer.predict_segment_file(segment_name)
                smiles = prediction.smiles
                molblock = prediction.molblock
                print(
                    f"Structure recognition finished segment {idx} on page {i} in "
                    f"{prediction.elapsed_seconds:.2f}s"
                )
            except Exception as e:
                print(f"Error processing segment {idx} on page {i}: {e}")
                smiles = ''
                molblock = ''

        row_data = {
            'PAGE_NUM': i,
            'SMILES': smiles,
            'IMAGE_FILE': output_name,
            'SEGMENT_FILE': segment_name,
            'STRUCTURE_TYPE': structure_type,
            'IS_COMPLETE_COMPOUND': is_complete_compound,
            'STRUCTURE_FILTER_REASON': structure_filter_reason,
            'STRUCTURE_FILTER_RAW_RESPONSE': structure_filter_raw_response,
            'STRUCTURE_FILTER_BORDER_SIDES': border_contact_sides,
            'STRUCTURE_FILTER_STRICTNESS': structure_filter_strictness,
            'FILTERED_OUT': False,
        }
        if molblock:
            row_data['MOLBLOCK'] = molblock
        return row_data
    except Exception as e:
        print(f"Error processing segment {idx} on page {i}: {e}")
        return None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def process_page(
    recognizer,
    i,
    scanned_page_file_path,
    segmented_dir,
    images_dir,
    progress_callback=None,
    total_pages=None,
    page_idx=None,
    structure_filter_strictness=DEFAULT_STRUCTURE_FILTER_STRICTNESS,
):
    """处理单个页面"""
    # 创建一个线程来运行页面处理
    result_container = [None]
    exception_container = [None]
    
    def run_process_page():
        page = None
        masks = None
        segments = None
        bboxes = None
        detected_structures = None
        try:
            if progress_callback and page_idx is not None and total_pages is not None:
                progress_callback(page_idx + 1, total_pages, f"Processing page {i}")

            page = cv2.imread(scanned_page_file_path)
            if page is None:
                print(f"Warning: Could not read image for page {i}")
                result_container[0] = ([], [], [], [])
                return

            detected_structures = recognizer.detect_segments(page)
            segments = [item.image for item in detected_structures.structures]
            bboxes = [item.bbox for item in detected_structures.structures]
            masks = detected_structures.masks

            page_data_list = []
            filtered_page_data_list = []
            image_files = []
            segment_info = []
            segment_jobs = []
            prev_page_path = os.path.join(images_dir, f'page_{i-1}.png')

            for idx, segment in enumerate(segments):
                segment = normalize_segment_array(segment)
                if segment is None:
                    continue
                output_name = os.path.join(segmented_dir, f'highlight_{i}_{idx}.png')
                id_image_name = os.path.join(segmented_dir, f'id_highlight_{i}_{idx}.png')
                segment_name = os.path.join(segmented_dir, f'segment_{i}_{idx}.png')
                box_coords_path = os.path.join(segmented_dir, f'highlight_{i}_{idx}.json')
                try:
                    cv2.imwrite(segment_name, segment)
                    save_box_image(bboxes, masks, idx, page, output_name)

                    # Save the bounding box coordinates to a JSON file using built-in ints
                    with open(box_coords_path, 'w') as f_json:
                        import json
                        bbox_coords = np.asarray(bboxes[idx]).astype(int).tolist()
                        json.dump(
                            {
                                "box": bbox_coords,
                                "page_width": int(page.shape[1]),
                                "page_height": int(page.shape[0]),
                            },
                            f_json,
                        )

                except Exception as e:
                    print(f"Warning: Failed to save boxed image or coords for segment {idx} on page {i}: {e}")

                segment_jobs.append({
                    'idx': idx,
                    'segment_name': segment_name,
                    'output_name': output_name,
                    'id_image_name': id_image_name,
                    'box_coords_path': box_coords_path,
                    'prev_page_path': prev_page_path,
                })
                segments[idx] = None

            try:
                del page, masks, segments, bboxes, detected_structures
                page = None
                masks = None
                segments = None
                bboxes = None
                detected_structures = None
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for job in segment_jobs:
                row_data = process_segment(
                    recognizer,
                    None,
                    job['idx'],
                    i,
                    segmented_dir,
                    job['output_name'],
                    job['prev_page_path'],
                    segment_name=job['segment_name'],
                    id_image_name=job.get('id_image_name'),
                    structure_filter_strictness=structure_filter_strictness,
                )
                if row_data:
                    row_data['BOX_COORDS_FILE'] = job['box_coords_path']
                    row_data['PAGE_IMAGE_FILE'] = scanned_page_file_path
                    if row_data.get('FILTERED_OUT'):
                        filtered_page_data_list.append(row_data)
                    else:
                        page_data_list.append(row_data)
                        id_image_file = job.get('id_image_name') or ''
                        image_files.append(id_image_file if os.path.exists(id_image_file) else job['output_name'])
                        segment_info.append((len(page_data_list) - 1, i, job['idx']))

            result_container[0] = (page_data_list, filtered_page_data_list, image_files, segment_info)
        except Exception as e:
            print(f"Error processing page {i}: {e}")
            exception_container[0] = e
            result_container[0] = ([], [], [], [])
        finally:
            try:
                del page, masks, segments, bboxes, detected_structures
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 在单独的线程中运行页面处理
    process_thread = threading.Thread(target=run_process_page)
    process_thread.daemon = True
    process_thread.start()
    process_thread.join(timeout=PAGE_PROCESSING_TIMEOUT)

    # 检查线程是否超时
    if process_thread.is_alive():
        print(f"Timeout processing page {i}, terminating...")
        # Force-raise exception in the stuck thread so it unwinds instead of
        # leaking forever.  ctypes is the standard CPython trick for this.
        try:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(process_thread.ident),
                ctypes.py_object(TimeoutError),
            )
        except Exception:
            pass
        return [], [], [], []
    elif exception_container[0]:
        raise exception_container[0]
    else:
        return result_container[0]


def extract_structures_from_pdf(
    pdf_file,
    page_start,
    page_end,
    output,
    progress_callback=None,
    batch_size=4,
    page_workers=None,
    id_batch_size=None,
    structure_filter_strictness=DEFAULT_STRUCTURE_FILTER_STRICTNESS,
):
    images_dir = os.path.join(output, 'structure_images')
    segmented_dir = os.path.join(output, 'segment')

    shutil.rmtree(segmented_dir, ignore_errors=True)
    create_directory(segmented_dir)

    extraction_start_page = max(1, page_start - 1)
    split_pdf_to_images(pdf_file, images_dir, page_start=extraction_start_page, page_end=page_end)

    recognizer = StructureRecognizer()

    data_list = []
    filtered_data_list = []
    pending_id_jobs = []
    total_pages = page_end - page_start + 1
    resolved_page_workers, resolved_id_batch_size = resolve_structure_runtime_settings(
        batch_size=batch_size,
        page_workers=page_workers,
        id_batch_size=id_batch_size,
    )

    with ThreadPoolExecutor(max_workers=resolved_page_workers) as executor:
        flush_job_threshold = max(8, resolved_id_batch_size * 4)

        def flush_pending_id_jobs(flush_all=False):
            nonlocal pending_id_jobs
            if not pending_id_jobs:
                return 0
            if not flush_all and len(pending_id_jobs) < flush_job_threshold:
                return 0
            if flush_all:
                jobs_to_process = pending_id_jobs
                pending_id_jobs = []
            else:
                jobs_to_process = pending_id_jobs[:flush_job_threshold]
                pending_id_jobs = pending_id_jobs[flush_job_threshold:]
            return resolve_structure_id_jobs(jobs_to_process, resolved_id_batch_size)

        def merge_page_result(page_result, current_offset):
            nonlocal pending_id_jobs
            page_data, filtered_page_data, image_files, segment_info = page_result
            adjusted_segment_info = []
            for local_data_idx, p_num, segment_idx in segment_info:
                global_data_idx = current_offset + local_data_idx
                adjusted_segment_info.append((global_data_idx, p_num, segment_idx))
            data_list.extend(page_data)
            filtered_data_list.extend(filtered_page_data)
            for local_idx, page_num, segment_idx in segment_info:
                if local_idx >= len(page_data):
                    continue
                image_file = image_files[local_idx] if local_idx < len(image_files) else ''
                pending_id_jobs.append({
                    'row': page_data[local_idx],
                    'image_file': image_file,
                    'page_num': page_num,
                    'segment_idx': segment_idx,
                })
            flush_pending_id_jobs(flush_all=False)
            return current_offset + len(page_data)

        pages_to_process = []
        for page_idx, i in enumerate(range(page_start, page_end + 1)):
            scanned_page_file_path = os.path.join(images_dir, f'page_{i}.png')
            if os.path.exists(scanned_page_file_path):
                pages_to_process.append((i, page_idx, scanned_page_file_path))

        max_inflight = (
            DEFAULT_STRUCTURE_PAGE_MAX_INFLIGHT
            if DEFAULT_STRUCTURE_PAGE_MAX_INFLIGHT > 0
            else max(1, resolved_page_workers + 1)
        )
        pending_futures = {}
        page_results = {}
        next_flush_page_idx = 0
        data_list_offset = 0
        submit_cursor = 0

        while submit_cursor < len(pages_to_process) and len(pending_futures) < max_inflight:
            page_num, page_idx, scanned_page_file_path = pages_to_process[submit_cursor]
            future = executor.submit(
                process_page,
                recognizer, page_num, scanned_page_file_path,
                segmented_dir, images_dir, progress_callback, total_pages, page_idx, structure_filter_strictness
            )
            pending_futures[future] = (page_num, page_idx)
            submit_cursor += 1

        while pending_futures:
            done_futures, _ = concurrent.futures.wait(
                pending_futures.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done_futures:
                page_num, page_idx = pending_futures.pop(future)
                try:
                    page_results[page_idx] = future.result(timeout=MODEL_TIMEOUT)
                except TimeoutError:
                    print(f"Timeout collecting results for page {page_num}")
                    page_results[page_idx] = ([], [], [], [])
                except Exception as e:
                    print(f"Error collecting results for page {page_num}: {e}")
                    page_results[page_idx] = ([], [], [], [])

                while next_flush_page_idx in page_results:
                    data_list_offset = merge_page_result(page_results.pop(next_flush_page_idx), data_list_offset)
                    next_flush_page_idx += 1

                if submit_cursor < len(pages_to_process):
                    next_page_num, next_page_idx, scanned_page_file_path = pages_to_process[submit_cursor]
                    next_future = executor.submit(
                        process_page,
                        recognizer, next_page_num, scanned_page_file_path,
                        segmented_dir, images_dir, progress_callback, total_pages, next_page_idx, structure_filter_strictness
                    )
                    pending_futures[next_future] = (next_page_num, next_page_idx)
                    submit_cursor += 1

        while next_flush_page_idx in page_results:
            data_list_offset = merge_page_result(page_results.pop(next_flush_page_idx), data_list_offset)
            next_flush_page_idx += 1

    flush_pending_id_jobs(flush_all=True)

    final_data_list = []
    for row in data_list:
        nonfinal_role = isinstance(row, dict) and is_nonfinal_visual_role(row.get('VISUAL_ROLE'))
        unusable_id = isinstance(row, dict) and not is_usable_final_compound_id(row.get('COMPOUND_ID'))
        if isinstance(row, dict) and (nonfinal_role or unusable_id):
            filtered_row = dict(row)
            filtered_row['FILTERED_OUT'] = True
            filtered_row['IS_COMPLETE_COMPOUND'] = False
            filtered_row['STRUCTURE_TYPE'] = 'non_final_or_unidentified_compound'
            reason = str(filtered_row.get('STRUCTURE_FILTER_REASON') or '').strip()
            if nonfinal_role:
                intermediate_reason = f"Visual model classified this structure role as {filtered_row.get('VISUAL_ROLE')}; only target/final-product structures are kept"
            else:
                intermediate_reason = 'Only structures with usable target-compound IDs are kept; unidentified structures and IDs explicitly labeled as intermediates, preparations, or embodiments are excluded'
            filtered_row['STRUCTURE_FILTER_REASON'] = (
                f"{reason}; {intermediate_reason}" if reason else intermediate_reason
            )
            filtered_data_list.append(filtered_row)
            continue
        final_data_list.append(row)
    data_list = final_data_list

    if SAVE_FILTERED_STRUCTURES and filtered_data_list:
        filtered_csv = os.path.join(output, 'filtered_structures.csv')
        pd.DataFrame(filtered_data_list).to_csv(filtered_csv, index=False, encoding='utf-8-sig')
        print(f"Filtered structures saved to {filtered_csv} ({len(filtered_data_list)} items)")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return data_list, filtered_data_list
