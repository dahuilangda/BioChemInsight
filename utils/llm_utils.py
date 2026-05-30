import os
import warnings
warnings.filterwarnings("ignore")

import time
import requests
from functools import wraps
import json
import base64
import logging
import re
import sys
import signal
import threading

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import PIL.Image
    import PIL.ImageDraw
    import PIL.ImageFont
except ImportError:
    logger.error("Pillow (PIL) package not found. Please install it: pip install Pillow")
    logger.info("Testing of structure_to_id with dummy image creation will be skipped.")
    PIL = None

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(SCRIPT_DIR) != '' and os.path.exists(os.path.join(SCRIPT_DIR, '..', 'constants.py')):
         sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
         import constants
         sys.path.pop(0)
    else:
        import constants
except ImportError:
    logger.error("constants.py not found. Please ensure it's in the same directory, parent directory, or your PYTHONPATH.")
    sys.exit(1)

from utils.skill_prompt_loader import load_merged_skill_json, render_skill_prompt_with_examples
from utils.compound_id_utils import canonicalize_alias_token, parse_compound_id_parts
from utils.model_harness import (
    ModelContractError,
    classify_exception,
    extract_json_content,
    parse_validated_json_object,
    require_confidence_value,
    require_json_object,
    require_required_keys,
    run_json_task,
    run_with_retries,
)


LLM_TEXT_MODEL_NAME = getattr(constants, 'LLM_OPENAI_COMPATIBLE_MODEL_NAME', None)
LLM_TEXT_MODEL_URL = getattr(constants, 'LLM_OPENAI_COMPATIBLE_MODEL_URL', None)
LLM_TEXT_MODEL_KEY = getattr(constants, 'LLM_OPENAI_COMPATIBLE_MODEL_KEY', None)

# Visual Model Configuration
VISUAL_MODEL_NAME = getattr(constants, 'VISUAL_MODEL_NAME', None)
VISUAL_MODEL_URL = getattr(constants, 'VISUAL_MODEL_URL', None)
VISUAL_MODEL_KEY = getattr(constants, 'VISUAL_MODEL_KEY', None)
STRUCTURE_FILTER_STRICTNESS = getattr(constants, 'STRUCTURE_FILTER_STRICTNESS', 'strict')
VISION_MODEL_TIMEOUT_SECONDS = int(getattr(constants, 'VISION_MODEL_TIMEOUT_SECONDS', 120))
VISION_MODEL_MAX_RETRIES = max(1, int(getattr(constants, 'VISION_MODEL_MAX_RETRIES', 1)))
VISION_MODEL_OUTER_TIMEOUT_PADDING_SECONDS = max(1, int(getattr(constants, 'VISION_MODEL_OUTER_TIMEOUT_PADDING_SECONDS', 10)))
VISION_MODEL_CONCURRENCY = max(1, int(getattr(constants, 'VISION_MODEL_CONCURRENCY', 2)))
LLM_MODEL_TIMEOUT_SECONDS = int(getattr(constants, 'LLM_MODEL_TIMEOUT_SECONDS', 180))
LLM_MODEL_OUTER_TIMEOUT_PADDING_SECONDS = max(1, int(getattr(constants, 'LLM_MODEL_OUTER_TIMEOUT_PADDING_SECONDS', 10)))
ASSAY_VALUE_VERIFIER_MAX_ITEMS = max(1, int(getattr(constants, 'ASSAY_VALUE_VERIFIER_MAX_ITEMS', 4) or 4))
ASSAY_MATCH_VERIFIER_MAX_ITEMS = max(1, int(getattr(constants, 'ASSAY_MATCH_VERIFIER_MAX_ITEMS', 8) or 8))
ASSAY_COMPOUND_ID_VERIFIER_MAX_ITEMS = max(1, int(getattr(constants, 'ASSAY_COMPOUND_ID_VERIFIER_MAX_ITEMS', 8) or 8))
ASSAY_EXTRACTION_PROMPT_MAX_COMPOUND_IDS = max(
    0,
    int(getattr(constants, 'ASSAY_EXTRACTION_PROMPT_MAX_COMPOUND_IDS', 128) or 128),
)
LOG_PREVIEW_CHARS = 8192

HTTP_PROXY = getattr(constants, 'HTTP_PROXY', '')
HTTPS_PROXY = getattr(constants, 'HTTPS_PROXY', '')
_visual_model_semaphore = threading.BoundedSemaphore(VISION_MODEL_CONCURRENCY)

if LLM_TEXT_MODEL_NAME and LLM_TEXT_MODEL_URL and LLM_TEXT_MODEL_KEY:
    pass
else:
    raise ValueError("No LLM model configured for get_compound_id_from_description. Please set LLM_OPENAI_COMPATIBLE_MODEL_NAME, LLM_OPENAI_COMPATIBLE_MODEL_URL, and LLM_OPENAI_COMPATIBLE_MODEL_KEY in constants.py.")

if VISUAL_MODEL_KEY and VISUAL_MODEL_URL and VISUAL_MODEL_NAME:
    logger.info("Using OpenAI-compatible visual model: %s", VISUAL_MODEL_NAME)
else:
    raise ValueError("No visual model configured for structure_to_id. Please set VISUAL_MODEL_NAME, VISUAL_MODEL_URL, and VISUAL_MODEL_KEY in constants.py.")

def proxy_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        original_http_proxy = os.environ.get('http_proxy')
        original_https_proxy = os.environ.get('https_proxy')
        if HTTP_PROXY:
            os.environ['http_proxy'] = HTTP_PROXY
        if HTTPS_PROXY:
            os.environ['https_proxy'] = HTTPS_PROXY

        result = func(*args, **kwargs)

        if original_http_proxy is None:
            os.environ.pop('http_proxy', None)
        elif HTTP_PROXY:
            os.environ['http_proxy'] = original_http_proxy

        if original_https_proxy is None:
            os.environ.pop('https_proxy', None)
        elif HTTPS_PROXY:
            os.environ['https_proxy'] = original_https_proxy
        return result
    return wrapper

def cost_time(func):
    """
    Decorator to ensure a function takes at least 1.5 seconds.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        cost = end - start
        if cost < 1.5:
            time.sleep(1.5 - cost)
        return result
    return wrapper

def require_openai():
    if OpenAI is None:
        raise ImportError("openai package not found. Please install it: pip install openai")
    return OpenAI


def sanitize_model_response_text(response_text):
    if not isinstance(response_text, str):
        return ''
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
    response_text = re.sub(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', r'\1', response_text, flags=re.DOTALL)
    return response_text.strip()


def preview_text(value, limit=LOG_PREVIEW_CHARS):
    text = '' if value is None else str(value)
    return text[:limit] + ('...' if len(text) > limit else '')


def build_content_to_dict_prompt(content, assay_name, compound_id_list=None, assay_context_names=None):
    if compound_id_list is None:
        compound_id_list_block = (
            '本次主抽取 prompt 未提供化合物ID allowlist。仍必须从 OCR 表格/正文中可见的最终 '
            'Example / Compound / No. / ID / 实施例 / 化合物 单元格抽取 compound_id，'
            '并用页面内可见的完整 ID 文本作为 JSON key；不要因为没有 allowlist 就返回空对象。'
            '后续步骤会再做规范 ID 校验。\n\n开始提取数据...\n\n'
        )
    else:
        compound_id_list = list(dict.fromkeys(compound_id_list))
        compounds = ', '.join([f'"{cid}"' for cid in compound_id_list])
        compound_id_list_block = '化合物ID列表如下，解析时请不要超出此列表范围：\n'
        compound_id_list_block += f"{compounds}\n\n"
        compound_id_list_block += '\n开始提取数据: \n'
    assay_context_names = [str(item).strip() for item in (assay_context_names or []) if str(item).strip()]
    if str(assay_name or '').strip() and str(assay_name).strip() not in assay_context_names:
        assay_context_names = [str(assay_name).strip(), *assay_context_names]
    assay_context_names = list(dict.fromkeys(assay_context_names))
    requested_assays_context_block = (
        '本次任务请求的所有 assay 名称如下。当前只抽取目标 assay，但必须用这个列表判断'
        '候选列最应该归属哪个 assay：\n'
        f"{json.dumps(assay_context_names, ensure_ascii=False)}\n\n"
    )

    return render_skill_prompt_with_examples(
        'biocheminsight-text-models',
        'references/content_to_dict_prompt.md',
        'references/examples/content_to_dict_examples.md',
        {
            'ASSAY_NAME': assay_name,
            'MARKDOWN_TEXT': content,
            'COMPOUND_ID_LIST_BLOCK': compound_id_list_block,
            'REQUESTED_ASSAYS_CONTEXT_BLOCK': requested_assays_context_block,
        },
    )


def build_content_to_multi_assay_dict_prompt(content, assay_names, compound_id_list=None):
    assay_names = [str(name).strip() for name in (assay_names or []) if str(name).strip()]
    if compound_id_list is None:
        compound_id_list_block = (
            '本次主抽取 prompt 未提供化合物ID allowlist。仍必须从 OCR 表格/正文中可见的最终 '
            'Example / Compound / No. / ID / 实施例 / 化合物 单元格抽取 compound_id，'
            '并用页面内可见的完整 ID 文本作为 JSON key；不要因为没有 allowlist 就返回空对象。'
            '后续步骤会再做规范 ID 校验。\n\n开始提取数据...\n\n'
        )
    else:
        compound_id_list = list(dict.fromkeys(compound_id_list))
        compounds = ', '.join([f'"{cid}"' for cid in compound_id_list])
        compound_id_list_block = '化合物ID列表如下，解析时请不要超出此列表范围：\n'
        compound_id_list_block += f"{compounds}\n\n"
        compound_id_list_block += '\n开始提取数据: \n'

    return render_skill_prompt_with_examples(
        'biocheminsight-text-models',
        'references/content_to_multi_assay_dict_prompt.md',
        'references/examples/content_to_multi_assay_dict_examples.md',
        {
            'ASSAY_NAMES_JSON': json.dumps(assay_names, ensure_ascii=False),
            'MARKDOWN_TEXT': content,
            'COMPOUND_ID_LIST_BLOCK': compound_id_list_block,
        },
    )


def build_route_assays_for_content_prompt(content, assay_names):
    assay_names = [str(name).strip() for name in (assay_names or []) if str(name).strip()]
    return render_skill_prompt_with_examples(
        'biocheminsight-text-models',
        'references/route_assays_for_content_prompt.md',
        None,
        {
            'ASSAY_NAMES_JSON': json.dumps(assay_names, ensure_ascii=False, indent=2),
            'OCR_CONTEXT': str(content or ''),
        },
    )


def build_reconcile_detected_assay_names_prompt(assay_names, page_decisions=None, ocr_context=''):
    assay_names = [str(name).strip() for name in (assay_names or []) if str(name).strip()]
    return render_skill_prompt_with_examples(
        'biocheminsight-text-models',
        'references/reconcile_detected_assay_names_prompt.md',
        None,
        {
            'ASSAY_NAMES_JSON': json.dumps(assay_names, ensure_ascii=False, indent=2),
            'PAGE_DECISIONS_JSON': json.dumps(page_decisions or {}, ensure_ascii=False, indent=2),
            'OCR_CONTEXT': str(ocr_context or ''),
        },
    )


def build_verify_assay_match_assignments_prompt(content, assay_name, assay_context_names, assay_payload):
    assay_context_names = [str(item).strip() for item in (assay_context_names or []) if str(item).strip()]
    if str(assay_name or '').strip() and str(assay_name).strip() not in assay_context_names:
        assay_context_names = [str(assay_name).strip(), *assay_context_names]
    assay_context_names = list(dict.fromkeys(assay_context_names))
    return render_skill_prompt_with_examples(
        'biocheminsight-text-models',
        'references/verify_assay_match_assignments_prompt.md',
        None,
        {
            'ASSAY_NAME': assay_name,
            'ASSAY_NAMES_JSON': json.dumps(assay_context_names, ensure_ascii=False, indent=2),
            'ASSAY_PAYLOAD_JSON': json.dumps(assay_payload or {}, ensure_ascii=False, indent=2),
            'OCR_CONTEXT': str(content or ''),
        },
    )


def build_verify_compound_id_assignments_prompt(content, compound_id_list, assay_payload):
    allowlist = [str(item).strip() for item in (compound_id_list or []) if str(item).strip()]
    return render_skill_prompt_with_examples(
        'biocheminsight-text-models',
        'references/verify_compound_id_assignments_prompt.md',
        None,
        {
            'COMPOUND_ID_LIST_JSON': json.dumps(allowlist, ensure_ascii=False, indent=2),
            'ASSAY_PAYLOAD_JSON': json.dumps(assay_payload or {}, ensure_ascii=False, indent=2),
            'OCR_CONTEXT': str(content or ''),
        },
    )


def build_verify_assay_value_assignments_prompt(content, assay_name, assay_context_names, assay_payload):
    assay_context_names = [str(item).strip() for item in (assay_context_names or []) if str(item).strip()]
    if str(assay_name or '').strip() and str(assay_name).strip() not in assay_context_names:
        assay_context_names = [str(assay_name).strip(), *assay_context_names]
    assay_context_names = list(dict.fromkeys(assay_context_names))
    return render_skill_prompt_with_examples(
        'biocheminsight-text-models',
        'references/verify_assay_value_assignments_prompt.md',
        None,
        {
            'ASSAY_NAME': assay_name,
            'ASSAY_NAMES_JSON': json.dumps(assay_context_names, ensure_ascii=False, indent=2),
            'ASSAY_PAYLOAD_JSON': json.dumps(assay_payload or {}, ensure_ascii=False, indent=2),
            'OCR_CONTEXT': str(content or ''),
        },
    )


def build_identify_assay_visual_review_requests_prompt(ocr_context, assay_dicts, parsed_tables=None):
    return render_skill_prompt_with_examples(
        'biocheminsight-text-models',
        'references/identify_assay_visual_review_requests_prompt.md',
        None,
        {
            'OCR_CONTEXT': str(ocr_context or ''),
            'ASSAY_DICTS_JSON': json.dumps(assay_dicts or {}, ensure_ascii=False, indent=2),
            'PARSED_TABLES_JSON': json.dumps(parsed_tables or [], ensure_ascii=False, indent=2),
        },
    )


def build_reconcile_assay_values_with_visual_report_prompt(ocr_context, assay_dicts, visual_report):
    return render_skill_prompt_with_examples(
        'biocheminsight-text-models',
        'references/reconcile_assay_values_with_visual_report_prompt.md',
        None,
        {
            'OCR_CONTEXT': str(ocr_context or ''),
            'ASSAY_DICTS_JSON': json.dumps(assay_dicts or {}, ensure_ascii=False, indent=2),
            'VISUAL_REPORT_JSON': json.dumps(visual_report or {}, ensure_ascii=False, indent=2),
        },
    )


def build_description_to_id_prompt(description):
    return render_skill_prompt_with_examples(
        'biocheminsight-text-models',
        'references/get_compound_id_from_description_prompt.md',
        'references/examples/get_compound_id_from_description_examples.md',
        {
            'DESCRIPTION': description,
        },
    )


def build_resolve_compound_id_alias_prompt(raw_id, compound_id_list, context=''):
    allowlist = [str(item).strip() for item in (compound_id_list or []) if str(item).strip()]
    allowlist_json = json.dumps(allowlist, ensure_ascii=False)
    return render_skill_prompt_with_examples(
        'biocheminsight-text-models',
        'references/resolve_compound_id_alias_prompt.md',
        'references/examples/resolve_compound_id_alias_examples.md',
        {
            'RAW_ID': str(raw_id or ''),
            'OPTIONAL_CONTEXT': str(context or ''),
            'ALLOWLIST': allowlist_json,
        },
    )


def normalize_filter_strictness(value):
    normalized = str(value or 'strict').strip().lower()
    if normalized in {'strict', 'balanced', 'permissive'}:
        return normalized
    return 'strict'


def build_strictness_instruction(strictness):
    strictness = normalize_filter_strictness(strictness)
    if strictness == 'permissive':
        return (
            "Strictness mode: permissive.\n"
            "- Allow complete_compound when the full molecule appears plausibly complete.\n"
            "- Use uncertain only when incompleteness or ambiguity is substantial."
        )
    if strictness == 'balanced':
        return (
            "Strictness mode: balanced.\n"
            "- Use standard conservative chemistry-document judgment.\n"
            "- Prefer fragment or uncertain when clear incompleteness cues exist, but do not over-block on weak hints alone."
        )
    return (
        "Strictness mode: strict.\n"
        "- If completeness cannot be confirmed confidently, prefer uncertain instead of complete_compound.\n"
        "- Border-touching or near-cropping cues are high-risk review signals, not automatic rejection; reject only when chemistry is actually clipped or completeness cannot be visually confirmed."
    )


def build_classify_structure_prompt(strictness='strict'):
    base_prompt = render_skill_prompt_with_examples(
        'biocheminsight-vision-models',
        'references/classify_structure_candidate_prompt.md',
        'references/examples/classify_structure_candidate_examples.md',
    )
    return f"{base_prompt}\n\n{build_strictness_instruction(strictness)}".strip()


def build_crop_check_prompt(border_sides_text, strictness='strict'):
    base_prompt = render_skill_prompt_with_examples(
        'biocheminsight-vision-models',
        'references/classify_structure_crop_check_prompt.md',
        'references/examples/classify_structure_crop_check_examples.md',
        {
            'BORDER_SIDES': border_sides_text,
        },
    )
    return f"{base_prompt}\n\n{build_strictness_instruction(strictness)}".strip()


def build_border_review_prompt(border_sides_text, strictness='strict'):
    base_prompt = render_skill_prompt_with_examples(
        'biocheminsight-vision-models',
        'references/classify_structure_border_review_prompt.md',
        'references/examples/classify_structure_border_review_examples.md',
        {
            'BORDER_SIDES': border_sides_text,
        },
    )
    return f"{base_prompt}\n\n{build_strictness_instruction(strictness)}".strip()


def build_structure_to_id_prompt():
    return render_skill_prompt_with_examples(
        'biocheminsight-vision-models',
        'references/structure_to_id_prompt.md',
        'references/examples/structure_to_id_examples.md',
    )


def build_review_assay_values_prompt(assay_dicts, review_payload):
    return render_skill_prompt_with_examples(
        'biocheminsight-vision-models',
        'references/review_assay_values_prompt.md',
        None,
        {
            'ASSAY_DICTS_JSON': json.dumps(assay_dicts or {}, ensure_ascii=False, indent=2),
            'REVIEW_PAYLOAD_JSON': json.dumps(review_payload or {}, ensure_ascii=False, indent=2),
        },
    )


def build_label_only_structure_role_review_prompt(base_prompt, compound_id):
    return render_skill_prompt_with_examples(
        'biocheminsight-vision-models',
        'references/review_label_only_structure_role_prompt.md',
        None,
        {
            'BASE_STRUCTURE_TO_ID_PROMPT': str(base_prompt or ''),
            'COMPOUND_ID': str(compound_id or ''),
        },
    )


def build_review_structure_id_prompt(base_prompt, initial_result):
    return render_skill_prompt_with_examples(
        'biocheminsight-vision-models',
        'references/review_structure_id_prompt.md',
        None,
        {
            'BASE_STRUCTURE_TO_ID_PROMPT': str(base_prompt or ''),
            'INITIAL_RESULT_JSON': json.dumps(initial_result or {}, ensure_ascii=False, indent=2),
        },
    )


TEXT_MODEL_RUNTIME = load_merged_skill_json(
    'biocheminsight-model-common',
    'references/runtime.json',
    'biocheminsight-text-models',
    'references/runtime.json',
)
TEXT_MODEL_OUTPUT_SCHEMAS = load_merged_skill_json(
    'biocheminsight-model-common',
    'references/output_schemas.json',
    'biocheminsight-text-models',
    'references/output_schemas.json',
)
VISION_MODEL_RUNTIME = load_merged_skill_json(
    'biocheminsight-model-common',
    'references/runtime.json',
    'biocheminsight-vision-models',
    'references/runtime.json',
)
VISION_MODEL_OUTPUT_SCHEMAS = load_merged_skill_json(
    'biocheminsight-model-common',
    'references/output_schemas.json',
    'biocheminsight-vision-models',
    'references/output_schemas.json',
)


def get_task_runtime(skill_runtime, task_name):
    tasks = skill_runtime.get('tasks', {}) if isinstance(skill_runtime, dict) else {}
    task_runtime = tasks.get(task_name, {}) if isinstance(tasks, dict) else {}
    return task_runtime if isinstance(task_runtime, dict) else {}


def get_retry_delays(skill_runtime, task_name, channel=None):
    task_runtime = get_task_runtime(skill_runtime, task_name)
    delays = task_runtime.get('retry_delays_seconds', [])
    if not delays and channel:
        channel_defaults = skill_runtime.get(channel, {}).get('defaults', {}) if isinstance(skill_runtime, dict) else {}
        delays = channel_defaults.get('retry_delays_seconds', [])
    if isinstance(delays, list):
        return [float(value) for value in delays]
    return []


def get_task_temperature(skill_runtime, task_name, channel=None, default=0.0):
    task_runtime = get_task_runtime(skill_runtime, task_name)
    value = task_runtime.get('temperature')
    if value is None and channel:
        channel_defaults = skill_runtime.get(channel, {}).get('defaults', {}) if isinstance(skill_runtime, dict) else {}
        value = channel_defaults.get('temperature', default)
    if value is None:
        value = default
    try:
        return float(value)
    except Exception:
        return float(default)


def get_system_prompt(skill_runtime, channel, key, default):
    channel_config = skill_runtime.get(channel, {}) if isinstance(skill_runtime, dict) else {}
    prompts = channel_config.get('system_prompts', {}) if isinstance(channel_config, dict) else {}
    value = prompts.get(key, default) if isinstance(prompts, dict) else default
    return value if isinstance(value, str) and value.strip() else default


def sleep_before_retry(attempt, retry, delays):
    if attempt >= retry - 1:
        return
    delay = delays[attempt] if attempt < len(delays) else (1 + attempt)
    time.sleep(max(0.0, float(delay)))


def validate_required_keys(payload, schema):
    if not isinstance(payload, dict):
        return False
    required_keys = schema.get('required_keys', []) if isinstance(schema, dict) else []
    return all(key in payload for key in required_keys)


def parse_structure_classification_payload(response_text, schema_name, task_name):
    schema = VISION_MODEL_OUTPUT_SCHEMAS.get(schema_name, {})
    payload = parse_validated_json_object(response_text, schema, task_name)
    structure_type = normalize_structure_type(payload.get('structure_type'))
    allowed = set(schema.get('allowed_structure_types') or [])
    if structure_type not in allowed:
        raise ValueError(f"{task_name} payload has invalid structure_type: {payload.get('structure_type')!r}")
    is_complete_compound = payload.get('is_complete_compound')
    if not isinstance(is_complete_compound, bool):
        raise ValueError(f"{task_name} payload has non-boolean is_complete_compound")
    confidence = require_confidence_value(payload.get('confidence'), task_name)
    reason = str(payload.get('reason') or '').strip()
    if not reason:
        raise ValueError(f"{task_name} payload has empty reason")
    if is_complete_compound and structure_type != 'complete_compound':
        raise ValueError(f"{task_name} payload is_complete_compound=true requires complete_compound")
    if structure_type == 'complete_compound' and not is_complete_compound:
        raise ValueError(f"{task_name} payload complete_compound requires is_complete_compound=true")
    return {
        'structure_type': structure_type,
        'is_complete_compound': is_complete_compound,
        'confidence': confidence,
        'reason': reason,
    }


def parse_crop_check_payload(response_text):
    task_name = 'crop_check'
    schema = VISION_MODEL_OUTPUT_SCHEMAS.get('crop_check', {})
    payload = parse_validated_json_object(response_text, schema, task_name)
    crop_status = str(payload.get('crop_status') or '').strip().lower()
    allowed = set(schema.get('allowed_crop_statuses') or [])
    if crop_status not in allowed:
        raise ValueError(f"{task_name} payload has invalid crop_status: {payload.get('crop_status')!r}")
    is_cropped = payload.get('is_cropped')
    if not isinstance(is_cropped, bool):
        raise ValueError(f"{task_name} payload has non-boolean is_cropped")
    confidence = require_confidence_value(payload.get('confidence'), task_name)
    reason = str(payload.get('reason') or '').strip()
    if not reason:
        raise ValueError(f"{task_name} payload has empty reason")
    if crop_status == 'fragment' and not is_cropped:
        raise ValueError(f"{task_name} payload fragment requires is_cropped=true")
    if crop_status == 'not_cropped' and is_cropped:
        raise ValueError(f"{task_name} payload not_cropped requires is_cropped=false")
    return {
        'crop_status': crop_status,
        'is_cropped': is_cropped,
        'confidence': confidence,
        'reason': reason,
    }


def validate_rich_assay_value(value, expected_assay_name=None, requested_assay_names=None):
    schema = TEXT_MODEL_OUTPUT_SCHEMAS.get('rich_assay_value', {})
    if not validate_required_keys(value, schema):
        return False
    confidence = str((value or {}).get('confidence') or '').strip().lower()
    if confidence not in {'high', 'medium', 'low'}:
        return False
    match_schema = TEXT_MODEL_OUTPUT_SCHEMAS.get('rich_assay_value_match', {})
    assay_match = (value or {}).get('assay_match')
    if not validate_required_keys(assay_match, match_schema):
        return False
    if assay_match.get('compatible') is not True:
        return False
    if not str(assay_match.get('reason') or '').strip():
        return False
    best_requested_assay = str(assay_match.get('best_requested_assay') or '').strip()
    if not best_requested_assay:
        return False
    requested = [str(item).strip() for item in (requested_assay_names or []) if str(item).strip()]
    if requested and best_requested_assay not in requested:
        return False
    if expected_assay_name is not None and best_requested_assay != str(expected_assay_name or '').strip():
        return False
    return True


def _normalize_assay_match_object(value):
    raw_match = {}
    if isinstance(value, dict):
        raw_match = value.get('assay_match') or value.get('ASSAY_MATCH') or {}
    if not isinstance(raw_match, dict):
        raw_match = {}
    return {
        'target': str(raw_match.get('target') or raw_match.get('TARGET') or '').strip(),
        'candidate': str(raw_match.get('candidate') or raw_match.get('CANDIDATE') or '').strip(),
        'compatible': raw_match.get('compatible') if isinstance(raw_match.get('compatible'), bool) else raw_match.get('COMPATIBLE'),
        'best_requested_assay': str(
            raw_match.get('best_requested_assay') or raw_match.get('BEST_REQUESTED_ASSAY') or ''
        ).strip(),
        'reason': str(raw_match.get('reason') or raw_match.get('REASON') or '').strip(),
    }


def _extract_assay_value_text(value):
    """Accept legacy string values and rich value/confidence/reason objects."""
    if isinstance(value, dict):
        for key in ('value', 'VALUE', 'assay_value', 'ASSAY_VALUE'):
            if key in value:
                candidate = value.get(key)
                return '' if candidate is None else str(candidate).strip()
        return ''
    if value is None:
        return ''
    return str(value).strip()


def _normalize_assay_value_object(value, expected_assay_name=None, requested_assay_names=None):
    if isinstance(value, dict):
        raw_value = None
        for key in ('value', 'VALUE', 'assay_value', 'ASSAY_VALUE'):
            if key in value:
                raw_value = value.get(key)
                break
        assay_value = '' if raw_value is None else str(raw_value).strip()
        if not assay_value:
            return None
        normalized = {
            'value': assay_value,
            'unit': str(value.get('unit') or value.get('UNIT') or '').strip(),
            'method': str(value.get('method') or value.get('METHOD') or '').strip(),
            'description': str(value.get('description') or value.get('DESCRIPTION') or '').strip(),
            'confidence': str(value.get('confidence') or value.get('CONFIDENCE') or '').strip(),
            'reason': str(value.get('reason') or value.get('REASON') or '').strip(),
            'assay_match': _normalize_assay_match_object(value),
        }
        if not validate_rich_assay_value(normalized, expected_assay_name, requested_assay_names):
            return None
        return normalized
    assay_value = _extract_assay_value_text(value)
    if not assay_value:
        return None
    normalized = {
        'value': assay_value,
        'unit': '',
        'method': '',
        'description': '',
        'confidence': '',
        'reason': '',
        'assay_match': {},
    }
    if not validate_rich_assay_value(normalized, expected_assay_name, requested_assay_names):
        return None
    return normalized


def normalize_single_assay_dict_payload(payload, expected_assay_name=None, requested_assay_names=None):
    if not isinstance(payload, dict):
        return {}
    normalized = {}
    for raw_key, raw_value in payload.items():
        compound_id = str(raw_key or '').strip()
        assay_value = _normalize_assay_value_object(raw_value, expected_assay_name, requested_assay_names)
        if compound_id and assay_value:
            normalized[compound_id] = assay_value
    return normalized


def normalize_multi_assay_dict_payload(payload, assay_names):
    assay_names = [str(name).strip() for name in (assay_names or []) if str(name).strip()]
    normalized = {assay_name: {} for assay_name in assay_names}
    if not isinstance(payload, dict):
        return normalized
    for assay_name in assay_names:
        assay_payload = payload.get(assay_name, {})
        normalized[assay_name] = normalize_single_assay_dict_payload(
            assay_payload,
            expected_assay_name=assay_name,
            requested_assay_names=assay_names,
        )
    return normalized


def require_multi_assay_top_level_keys(payload, assay_names, task_name='content_to_multi_assay_dict'):
    assay_names = [str(name).strip() for name in (assay_names or []) if str(name).strip()]
    if not isinstance(payload, dict):
        raise ModelContractError(f"{task_name} returned {type(payload).__name__}, expected object")
    missing = [assay_name for assay_name in assay_names if assay_name not in payload]
    if missing:
        raise ModelContractError(f"{task_name} payload missing requested assay keys: {', '.join(missing)}")


def verify_assay_match_assignments(
    content,
    assay_name,
    assay_context_names,
    assay_payload,
    retry=2,
    audit_path=None,
    metadata=None,
    timeout_seconds=None,
):
    requested_assays = [str(item).strip() for item in (assay_context_names or []) if str(item).strip()]
    if str(assay_name or '').strip() and str(assay_name).strip() not in requested_assays:
        requested_assays = [str(assay_name).strip(), *requested_assays]
    requested_assays = list(dict.fromkeys(requested_assays))
    if not isinstance(assay_payload, dict) or not assay_payload:
        return {}
    if len(requested_assays) <= 1:
        return assay_payload

    compound_ids = [str(key) for key in assay_payload.keys()]
    if len(compound_ids) > ASSAY_MATCH_VERIFIER_MAX_ITEMS:
        verified = {}
        batch_count = (len(compound_ids) + ASSAY_MATCH_VERIFIER_MAX_ITEMS - 1) // ASSAY_MATCH_VERIFIER_MAX_ITEMS
        for batch_index, start in enumerate(range(0, len(compound_ids), ASSAY_MATCH_VERIFIER_MAX_ITEMS), 1):
            batch_ids = compound_ids[start:start + ASSAY_MATCH_VERIFIER_MAX_ITEMS]
            batch_payload = {compound_id: assay_payload[compound_id] for compound_id in batch_ids}
            verified.update(verify_assay_match_assignments(
                content,
                assay_name,
                requested_assays,
                batch_payload,
                retry=retry,
                audit_path=audit_path,
                timeout_seconds=timeout_seconds,
                metadata={
                    **(metadata or {}),
                    'match_verifier_batch_index': batch_index,
                    'match_verifier_batch_count': batch_count,
                    'match_verifier_batch_size': len(batch_ids),
                },
            ))
        return verified

    requested_set = set(requested_assays)

    def _parser(response_text):
        payload = parse_validated_json_object(
            response_text,
            TEXT_MODEL_OUTPUT_SCHEMAS.get('verify_assay_match_assignments', {}),
            'verify_assay_match_assignments',
        )
        missing = [compound_id for compound_id in compound_ids if compound_id not in payload]
        if missing:
            raise ModelContractError(
                "verify_assay_match_assignments payload missing compound IDs: "
                + ", ".join(missing[:10])
            )
        verified = {}
        for compound_id in compound_ids:
            decision = payload.get(compound_id)
            if not isinstance(decision, dict):
                raise ModelContractError(f"verify_assay_match_assignments item {compound_id!r} is not an object")
            if not isinstance(decision.get('compatible_current'), bool):
                raise ModelContractError(f"verify_assay_match_assignments item {compound_id!r} has non-boolean compatible_current")
            best_assay = str(decision.get('best_requested_assay') or '').strip()
            if best_assay not in requested_set:
                raise ModelContractError(
                    f"verify_assay_match_assignments item {compound_id!r} has invalid best_requested_assay: {best_assay!r}"
                )
            confidence = require_confidence_value(
                decision.get('confidence'),
                'verify_assay_match_assignments',
                key=f'{compound_id}.confidence',
            )
            reason = str(decision.get('reason') or '').strip()
            if not reason:
                raise ModelContractError(f"verify_assay_match_assignments item {compound_id!r} has empty reason")
            if decision.get('compatible_current') is True and best_assay == str(assay_name or '').strip():
                next_value = dict(assay_payload[compound_id])
                next_match = dict(next_value.get('assay_match') or {})
                next_match['best_requested_assay'] = best_assay
                next_match['verified_by'] = 'verify_assay_match_assignments'
                next_match['verification_confidence'] = confidence
                next_match['verification_reason'] = reason
                next_value['assay_match'] = next_match
                verified[compound_id] = next_value
        return verified

    return run_text_json_task(
        task_name='verify_assay_match_assignments',
        prompt=build_verify_assay_match_assignments_prompt(content, assay_name, requested_assays, assay_payload),
        parser=_parser,
        retry=retry,
        audit_path=audit_path,
        timeout_seconds=timeout_seconds,
        metadata={
            'assay_name': assay_name,
            'assay_context_count': len(requested_assays),
            'compound_id_count': len(compound_ids),
            'content_chars': len(str(content or '')),
            **(metadata or {}),
        },
    )


def verify_compound_id_assignments(
    content,
    compound_id_list,
    assay_payload,
    retry=2,
    audit_path=None,
    metadata=None,
    timeout_seconds=None,
):
    allowlist = [str(item).strip() for item in (compound_id_list or []) if str(item).strip()]
    allowed = set(allowlist)
    if not isinstance(assay_payload, dict) or not assay_payload:
        return {}
    if not allowed:
        return assay_payload

    current_ids = [str(key) for key in assay_payload.keys()]
    if len(current_ids) > ASSAY_COMPOUND_ID_VERIFIER_MAX_ITEMS:
        verified = {}
        batch_count = (len(current_ids) + ASSAY_COMPOUND_ID_VERIFIER_MAX_ITEMS - 1) // ASSAY_COMPOUND_ID_VERIFIER_MAX_ITEMS
        for batch_index, start in enumerate(range(0, len(current_ids), ASSAY_COMPOUND_ID_VERIFIER_MAX_ITEMS), 1):
            batch_ids = current_ids[start:start + ASSAY_COMPOUND_ID_VERIFIER_MAX_ITEMS]
            batch_payload = {compound_id: assay_payload[compound_id] for compound_id in batch_ids}
            verified.update(verify_compound_id_assignments(
                content,
                allowlist,
                batch_payload,
                retry=retry,
                audit_path=audit_path,
                timeout_seconds=timeout_seconds,
                metadata={
                    **(metadata or {}),
                    'compound_id_verifier_batch_index': batch_index,
                    'compound_id_verifier_batch_count': batch_count,
                    'compound_id_verifier_batch_size': len(batch_ids),
                },
            ))
        return verified

    def _is_alias_compatible(current_id, canonical_id):
        current_token = canonicalize_alias_token(current_id)
        canonical_token = canonicalize_alias_token(canonical_id)
        if current_token and current_token == canonical_token:
            return True
        current_parts = parse_compound_id_parts(current_id)
        canonical_parts = parse_compound_id_parts(canonical_id)
        if current_parts and canonical_parts:
            current_core = canonicalize_alias_token(current_parts.get('core'))
            canonical_core = canonicalize_alias_token(canonical_parts.get('core'))
            return bool(current_core and current_core == canonical_core)
        return False

    def _parser(response_text):
        payload = parse_validated_json_object(
            response_text,
            TEXT_MODEL_OUTPUT_SCHEMAS.get('verify_compound_id_assignments', {}),
            'verify_compound_id_assignments',
        )
        missing = [compound_id for compound_id in current_ids if compound_id not in payload]
        if missing:
            raise ModelContractError(
                "verify_compound_id_assignments payload missing compound IDs: "
                + ", ".join(missing[:10])
            )

        verified = {}
        for current_id in current_ids:
            decision = payload.get(current_id)
            if not isinstance(decision, dict):
                raise ModelContractError(f"verify_compound_id_assignments item {current_id!r} is not an object")
            if not isinstance(decision.get('valid_current_id'), bool):
                raise ModelContractError(f"verify_compound_id_assignments item {current_id!r} has non-boolean valid_current_id")
            canonical_id = str(decision.get('canonical_compound_id') or '').strip()
            if canonical_id.lower() == 'none':
                canonical_id = 'None'
            if canonical_id != 'None' and canonical_id not in allowed:
                raise ModelContractError(
                    f"verify_compound_id_assignments item {current_id!r} has canonical_compound_id outside allowlist: {canonical_id!r}"
                )
            confidence = require_confidence_value(
                decision.get('confidence'),
                'verify_compound_id_assignments',
                key=f'{current_id}.confidence',
            )
            reason = str(decision.get('reason') or '').strip()
            if not reason:
                raise ModelContractError(f"verify_compound_id_assignments item {current_id!r} has empty reason")
            if decision.get('valid_current_id') is not True or canonical_id == 'None':
                continue
            if not _is_alias_compatible(current_id, canonical_id):
                continue
            next_value = dict(assay_payload[current_id])
            next_match = dict(next_value.get('assay_match') or {})
            next_match['compound_id_verified_by'] = 'verify_compound_id_assignments'
            next_match['compound_id_verification_confidence'] = confidence
            next_match['compound_id_verification_reason'] = reason
            next_value['assay_match'] = next_match
            if canonical_id not in verified:
                verified[canonical_id] = next_value
        return verified

    return run_text_json_task(
        task_name='verify_compound_id_assignments',
        prompt=build_verify_compound_id_assignments_prompt(content, allowlist, assay_payload),
        parser=_parser,
        retry=retry,
        audit_path=audit_path,
        timeout_seconds=timeout_seconds,
        metadata={
            'compound_id_count': len(current_ids),
            'allowlist_count': len(allowlist),
            'content_chars': len(str(content or '')),
            **(metadata or {}),
        },
    )


def route_assays_for_content(
    content,
    assay_names,
    retry=2,
    audit_path=None,
    metadata=None,
    timeout_seconds=None,
):
    assay_names = [str(name).strip() for name in (assay_names or []) if str(name).strip()]
    assay_names = list(dict.fromkeys(assay_names))
    if not assay_names:
        return {}
    if len(assay_names) == 1:
        return {assay_names[0]: True}
    assay_set = set(assay_names)

    def _parser(response_text):
        payload = parse_validated_json_object(
            response_text,
            TEXT_MODEL_OUTPUT_SCHEMAS.get('route_assays_for_content', {}),
            'route_assays_for_content',
        )
        decisions = payload.get('assays')
        if not isinstance(decisions, dict):
            raise ModelContractError("route_assays_for_content payload has invalid assays object")
        missing = [assay_name for assay_name in assay_names if assay_name not in decisions]
        if missing:
            raise ModelContractError(
                "route_assays_for_content payload missing requested assays: "
                + ", ".join(missing[:10])
            )
        routed = {}
        for assay_name in assay_names:
            decision = decisions.get(assay_name)
            if not isinstance(decision, dict):
                raise ModelContractError(f"route_assays_for_content item {assay_name!r} is not an object")
            if not isinstance(decision.get('extract'), bool):
                raise ModelContractError(f"route_assays_for_content item {assay_name!r} has non-boolean extract")
            require_confidence_value(
                decision.get('confidence'),
                'route_assays_for_content',
                key=f'{assay_name}.confidence',
            )
            reason = str(decision.get('reason') or '').strip()
            if not reason:
                raise ModelContractError(f"route_assays_for_content item {assay_name!r} has empty reason")
            if assay_name not in assay_set:
                raise ModelContractError(f"route_assays_for_content returned unexpected assay: {assay_name!r}")
            routed[assay_name] = bool(decision.get('extract'))
        return routed

    return run_text_json_task(
        task_name='route_assays_for_content',
        prompt=build_route_assays_for_content_prompt(content, assay_names),
        parser=_parser,
        retry=retry,
        audit_path=audit_path,
        timeout_seconds=timeout_seconds,
        metadata={
            'assay_context_count': len(assay_names),
            'content_chars': len(str(content or '')),
            **(metadata or {}),
        },
    )


def verify_assay_value_assignments(
    content,
    assay_name,
    assay_context_names,
    assay_payload,
    retry=2,
    audit_path=None,
    metadata=None,
    timeout_seconds=None,
):
    requested_assays = [str(item).strip() for item in (assay_context_names or []) if str(item).strip()]
    if str(assay_name or '').strip() and str(assay_name).strip() not in requested_assays:
        requested_assays = [str(assay_name).strip(), *requested_assays]
    requested_assays = list(dict.fromkeys(requested_assays))
    if not isinstance(assay_payload, dict) or not assay_payload:
        return {}

    compound_ids = [str(key) for key in assay_payload.keys()]
    if len(compound_ids) > ASSAY_VALUE_VERIFIER_MAX_ITEMS:
        verified = {}
        for batch_index, start in enumerate(range(0, len(compound_ids), ASSAY_VALUE_VERIFIER_MAX_ITEMS), 1):
            batch_ids = compound_ids[start:start + ASSAY_VALUE_VERIFIER_MAX_ITEMS]
            batch_payload = {compound_id: assay_payload[compound_id] for compound_id in batch_ids}
            verified.update(verify_assay_value_assignments(
                content,
                assay_name,
                requested_assays,
                batch_payload,
                retry=retry,
                audit_path=audit_path,
                timeout_seconds=timeout_seconds,
                metadata={
                    **(metadata or {}),
                    'value_verifier_batch_index': batch_index,
                    'value_verifier_batch_count': (len(compound_ids) + ASSAY_VALUE_VERIFIER_MAX_ITEMS - 1) // ASSAY_VALUE_VERIFIER_MAX_ITEMS,
                    'value_verifier_batch_size': len(batch_ids),
                },
            ))
        return verified

    def _parser(response_text):
        payload = parse_validated_json_object(
            response_text,
            TEXT_MODEL_OUTPUT_SCHEMAS.get('verify_assay_value_assignments', {}),
            'verify_assay_value_assignments',
        )
        missing = [compound_id for compound_id in compound_ids if compound_id not in payload]
        if missing:
            raise ModelContractError(
                "verify_assay_value_assignments payload missing compound IDs: "
                + ", ".join(missing[:10])
            )

        verified = {}
        for compound_id in compound_ids:
            decision = payload.get(compound_id)
            if not isinstance(decision, dict):
                raise ModelContractError(f"verify_assay_value_assignments item {compound_id!r} is not an object")
            if not isinstance(decision.get('valid_assay_value'), bool):
                raise ModelContractError(f"verify_assay_value_assignments item {compound_id!r} has non-boolean valid_assay_value")
            corrected_value = str(decision.get('corrected_value') or '').strip()
            if corrected_value.lower() == 'none':
                corrected_value = 'None'
            corrected_unit = str(decision.get('corrected_unit') or '').strip()
            confidence = require_confidence_value(
                decision.get('confidence'),
                'verify_assay_value_assignments',
                key=f'{compound_id}.confidence',
            )
            reason = str(decision.get('reason') or '').strip()
            if not reason:
                raise ModelContractError(f"verify_assay_value_assignments item {compound_id!r} has empty reason")
            if decision.get('valid_assay_value') is not True or corrected_value == 'None':
                continue
            next_value = dict(assay_payload[compound_id])
            original_value = str(next_value.get('value') or '').strip()
            original_unit = str(next_value.get('unit') or '').strip()
            if corrected_value:
                next_value['value'] = corrected_value
            if corrected_unit:
                next_value['unit'] = corrected_unit
            next_match = dict(next_value.get('assay_match') or {})
            next_match['value_verified_by'] = 'verify_assay_value_assignments'
            next_match['value_verification_confidence'] = confidence
            next_match['value_verification_reason'] = reason
            if original_value and original_value != str(next_value.get('value') or '').strip():
                next_match['original_value'] = original_value
            if original_unit and original_unit != str(next_value.get('unit') or '').strip():
                next_match['original_unit'] = original_unit
            next_value['assay_match'] = next_match
            verified[compound_id] = next_value
        return verified

    return run_text_json_task(
        task_name='verify_assay_value_assignments',
        prompt=build_verify_assay_value_assignments_prompt(content, assay_name, requested_assays, assay_payload),
        parser=_parser,
        retry=retry,
        audit_path=audit_path,
        timeout_seconds=timeout_seconds,
        metadata={
            'assay_name': assay_name,
            'assay_context_count': len(requested_assays),
            'compound_id_count': len(compound_ids),
            'content_chars': len(str(content or '')),
            **(metadata or {}),
        },
    )


def _normalize_confidence_text(value, default='medium'):
    normalized = str(value or '').strip().lower()
    if normalized in {'high', 'medium', 'low'}:
        return normalized
    return default


def extract_confident_compound_id(payload, fallback=None, none_value=None):
    if not isinstance(payload, dict):
        return fallback
    confidence = _normalize_confidence_text(payload.get("CONFIDENCE"), default='medium')
    if confidence == 'low':
        return none_value
    compound_id = payload.get("COMPOUND_ID", fallback)
    if compound_id is None:
        return none_value
    compound_id_text = str(compound_id).strip()
    if not compound_id_text or compound_id_text.lower() == 'none':
        return none_value
    return compound_id_text


def parse_compound_id_payload(response_text, schema_name, task_name, allowed_ids=None, none_value=None):
    schema = TEXT_MODEL_OUTPUT_SCHEMAS.get(schema_name, {})
    payload = parse_validated_json_object(response_text, schema, task_name)
    confidence = require_confidence_value(payload.get('CONFIDENCE'), task_name, key='CONFIDENCE')
    reason = str(payload.get('REASON') or '').strip()
    if not reason:
        raise ValueError(f"{task_name} payload has empty REASON")
    compound_id = payload.get('COMPOUND_ID')
    if compound_id is None:
        raise ValueError(f"{task_name} payload has null COMPOUND_ID")
    compound_id_text = str(compound_id).strip()
    if not compound_id_text:
        raise ValueError(f"{task_name} payload has empty COMPOUND_ID")
    if compound_id_text.lower() in {'none', 'null', 'unknown', 'n/a', 'na'}:
        return none_value
    if confidence == 'low':
        return none_value
    if allowed_ids is not None:
        allowed = {str(item).strip() for item in allowed_ids if str(item).strip()}
        if compound_id_text not in allowed:
            raise ValueError(f"{task_name} returned COMPOUND_ID outside allowlist: {compound_id_text!r}")
    return compound_id_text


def parse_visual_structure_id_payload(response_text):
    task_name = 'structure_to_id'
    schema = VISION_MODEL_OUTPUT_SCHEMAS.get('structure_to_id', {})
    payload = parse_validated_json_object(response_text, schema, task_name)

    confidence = require_confidence_value(payload.get('CONFIDENCE'), task_name, key='CONFIDENCE')
    visual_role = str(payload.get('VISUAL_ROLE') or '').strip().lower()
    id_source = str(payload.get('ID_SOURCE') or '').strip().lower()
    allowed_roles = set(schema.get('allowed_visual_roles') or [])
    allowed_sources = set(schema.get('allowed_id_sources') or [])
    if visual_role not in allowed_roles:
        raise ValueError(f"{task_name} payload has invalid VISUAL_ROLE: {payload.get('VISUAL_ROLE')!r}")
    if id_source not in allowed_sources:
        raise ValueError(f"{task_name} payload has invalid ID_SOURCE: {payload.get('ID_SOURCE')!r}")

    compound_id = payload.get('COMPOUND_ID')
    if compound_id is None:
        raise ValueError(f"{task_name} payload has null COMPOUND_ID")
    compound_id_text = str(compound_id).strip()
    if not compound_id_text:
        raise ValueError(f"{task_name} payload has empty COMPOUND_ID")
    if compound_id_text.lower() in {'null', 'unknown', 'n/a', 'na'}:
        raise ValueError(f"{task_name} payload must use string \"None\" for no identifier")

    evidence = str(payload.get('EVIDENCE') or '').strip()
    if not evidence:
        raise ValueError(f"{task_name} payload has empty EVIDENCE")
    if compound_id_text.lower() == 'none' and id_source != 'none':
        raise ValueError(f"{task_name} payload COMPOUND_ID=None requires ID_SOURCE=none")

    return {
        'COMPOUND_ID': 'None' if compound_id_text.lower() == 'none' else compound_id_text,
        'VISUAL_ROLE': visual_role,
        'ID_SOURCE': id_source,
        'EVIDENCE': evidence,
        'CONFIDENCE': confidence,
    }


def parse_review_assay_values_payload(response_text):
    task_name = 'review_assay_values'
    schema = VISION_MODEL_OUTPUT_SCHEMAS.get(task_name, {})
    payload = parse_validated_json_object(response_text, schema, task_name)
    corrections = payload.get('corrections')
    if not isinstance(corrections, list):
        raise ValueError(f"{task_name} payload corrections must be a list")
    item_schema = {'required_keys': schema.get('correction_required_keys', [])}
    normalized = []
    for index, item in enumerate(corrections):
        if not isinstance(item, dict):
            raise ValueError(f"{task_name} correction {index} must be an object")
        require_required_keys(item, item_schema, task_name)
        action = str(item.get('action') or '').strip().lower()
        if action not in {'keep', 'replace', 'uncertain'}:
            raise ValueError(f"{task_name} correction {index} has invalid action: {item.get('action')!r}")
        confidence = require_confidence_value(item.get('confidence'), task_name)
        normalized.append({
            'assay_name': str(item.get('assay_name') or '').strip(),
            'compound_id': str(item.get('compound_id') or '').strip(),
            'current_value': str(item.get('current_value') or '').strip(),
            'visual_value': str(item.get('visual_value') or '').strip(),
            'unit': str(item.get('unit') or '').strip(),
            'description': str(item.get('description') or '').strip(),
            'action': action,
            'confidence': confidence,
            'evidence': str(item.get('evidence') or '').strip(),
        })
    return {'corrections': normalized}


@proxy_decorator
# @cost_time
def content_to_dict(
    content,
    assay_name,
    compound_id_list=None,
    assay_context_names=None,
    retry=3,
    audit_path=None,
    metadata=None,
    timeout_seconds=None,
):
    """
    Converts Markdown text to a dictionary using the configured OpenAI-compatible model.
    """
    if not LLM_TEXT_MODEL_KEY or not LLM_TEXT_MODEL_URL or not LLM_TEXT_MODEL_NAME:
        raise ValueError("OpenAI-compatible text model is not configured.")

    logger.info("Converting content to dict using OpenAI-compatible model.")
    prompt_compound_id_list = compound_id_list
    if (
        ASSAY_EXTRACTION_PROMPT_MAX_COMPOUND_IDS > 0
        and compound_id_list
        and len(compound_id_list) > ASSAY_EXTRACTION_PROMPT_MAX_COMPOUND_IDS
    ):
        prompt_compound_id_list = None

    def _parser(response_text):
        payload = parse_validated_json_object(
            response_text,
            TEXT_MODEL_OUTPUT_SCHEMAS.get('content_to_dict', {}),
            'content_to_dict',
        )
        requested_assays = assay_context_names or [assay_name]
        normalized = normalize_single_assay_dict_payload(
            payload,
            expected_assay_name=assay_name,
            requested_assay_names=requested_assays,
        )
        if compound_id_list:
            normalized = verify_compound_id_assignments(
                content,
                compound_id_list,
                normalized,
                retry=2,
                audit_path=audit_path,
                timeout_seconds=timeout_seconds,
                metadata={
                    'source_task': 'content_to_dict',
                    **(metadata or {}),
                },
            )
            allowed = {str(item).strip() for item in compound_id_list if str(item).strip()}
            outside = [key for key in normalized if key not in allowed]
            for key in outside:
                normalized.pop(key, None)
        normalized = verify_assay_match_assignments(
            content,
            assay_name,
            requested_assays,
            normalized,
            retry=2,
            audit_path=audit_path,
            timeout_seconds=timeout_seconds,
            metadata={
                'source_task': 'content_to_dict',
                **(metadata or {}),
            },
        )
        normalized = verify_assay_value_assignments(
            content,
            assay_name,
            requested_assays,
            normalized,
            retry=1,
            audit_path=audit_path,
            timeout_seconds=timeout_seconds,
            metadata={
                'source_task': 'content_to_dict',
                **(metadata or {}),
            },
        )
        return normalized

    return run_text_json_task(
        task_name='content_to_dict',
        prompt=build_content_to_dict_prompt(content, assay_name, prompt_compound_id_list, assay_context_names),
        parser=_parser,
        retry=retry,
        audit_path=audit_path,
        timeout_seconds=timeout_seconds,
        metadata={
            'assay_name': assay_name,
            'compound_id_count': len(compound_id_list or []),
            'prompt_compound_id_count': len(prompt_compound_id_list or []),
            'assay_context_count': len(assay_context_names or []),
            'content_chars': len(str(content or '')),
            **(metadata or {}),
        },
    )


@proxy_decorator
def identify_assay_visual_review_requests(ocr_context, assay_dicts, parsed_tables=None, retry=2, audit_path=None, metadata=None):
    """
    Ask the text model to decide which extracted assay cells need visual reread.
    The decision is intentionally model-owned and context-based: the model sees
    the whole OCR chunk, parsed table grids, assay names, and extracted values.
    """
    if not isinstance(assay_dicts, dict) or not assay_dicts:
        return {}

    prompt = build_identify_assay_visual_review_requests_prompt(ocr_context, assay_dicts, parsed_tables)

    def _parser(response_text):
        payload = parse_validated_json_object(
            response_text,
            TEXT_MODEL_OUTPUT_SCHEMAS.get('identify_assay_visual_review_requests', {}),
            'identify_assay_visual_review_requests',
        )
        return payload

    return run_text_json_task(
        task_name='identify_assay_visual_review_requests',
        prompt=prompt,
        parser=_parser,
        retry=retry,
        audit_path=audit_path,
        metadata={
            'assay_count': len(assay_dicts or {}),
            'ocr_context_chars': len(str(ocr_context or '')),
            **(metadata or {}),
        },
    )


@proxy_decorator
def reconcile_assay_values_with_visual_report(ocr_context, assay_dicts, visual_report, retry=2, audit_path=None, metadata=None):
    """
    Let the text model reconcile the original OCR/text extraction with the
    visual audit report. The vision model reports observations; the text model
    owns the final structured assay dictionary.
    """
    if not isinstance(assay_dicts, dict) or not assay_dicts:
        return assay_dicts
    if not isinstance(visual_report, dict) or not visual_report:
        return assay_dicts

    prompt = build_reconcile_assay_values_with_visual_report_prompt(ocr_context, assay_dicts, visual_report)

    def _parser(response_text):
        payload = parse_validated_json_object(
            response_text,
            TEXT_MODEL_OUTPUT_SCHEMAS.get('reconcile_assay_values_with_visual_report', {}),
            'reconcile_assay_values_with_visual_report',
        )
        normalized = {}
        for assay_name, assay_payload in payload.items():
            normalized_assay_name = str(assay_name)
            normalized[normalized_assay_name] = normalize_single_assay_dict_payload(
                assay_payload,
                expected_assay_name=normalized_assay_name,
                requested_assay_names=list((assay_dicts or {}).keys()),
            )
        return normalized

    return run_text_json_task(
        task_name='reconcile_assay_values_with_visual_report',
        prompt=prompt,
        parser=_parser,
        retry=retry,
        audit_path=audit_path,
        metadata={
            'assay_count': len(assay_dicts or {}),
            'visual_correction_count': len((visual_report or {}).get('corrections') or []),
            'ocr_context_chars': len(str(ocr_context or '')),
            **(metadata or {}),
        },
    )


@proxy_decorator
def content_to_multi_assay_dict(
    content,
    assay_names,
    compound_id_list=None,
    retry=3,
    audit_path=None,
    metadata=None,
    timeout_seconds=None,
):
    """
    Converts Markdown/OCR content into a nested dictionary for multiple requested assay names.
    """
    if not LLM_TEXT_MODEL_KEY or not LLM_TEXT_MODEL_URL or not LLM_TEXT_MODEL_NAME:
        raise ValueError("OpenAI-compatible text model is not configured.")

    logger.info("Converting content to multi-assay dict using OpenAI-compatible model.")
    prompt_compound_id_list = compound_id_list
    if (
        ASSAY_EXTRACTION_PROMPT_MAX_COMPOUND_IDS > 0
        and compound_id_list
        and len(compound_id_list) > ASSAY_EXTRACTION_PROMPT_MAX_COMPOUND_IDS
    ):
        prompt_compound_id_list = None

    def _parser(response_text):
        try:
            assay_dict = parse_validated_json_object(
                response_text,
                TEXT_MODEL_OUTPUT_SCHEMAS.get('content_to_multi_assay_dict', {}),
                'content_to_multi_assay_dict',
            )
        except Exception as exc:
            raise ModelContractError(f"content_to_multi_assay_dict invalid output contract: {exc}") from exc
        require_multi_assay_top_level_keys(
            assay_dict,
            assay_names,
            'content_to_multi_assay_dict',
        )
        normalized = normalize_multi_assay_dict_payload(assay_dict, assay_names)
        for assay_name, assay_payload in list(normalized.items()):
            if compound_id_list:
                assay_payload = verify_compound_id_assignments(
                    content,
                    compound_id_list,
                    assay_payload,
                    retry=2,
                    audit_path=audit_path,
                    timeout_seconds=timeout_seconds,
                    metadata={
                        'source_task': 'content_to_multi_assay_dict',
                        **(metadata or {}),
                    },
                )
                allowed = {str(item).strip() for item in compound_id_list if str(item).strip()}
                for key in [key for key in assay_payload if key not in allowed]:
                    assay_payload.pop(key, None)
            assay_payload = verify_assay_match_assignments(
                content,
                assay_name,
                assay_names,
                assay_payload,
                retry=2,
                audit_path=audit_path,
                timeout_seconds=timeout_seconds,
                metadata={
                    'source_task': 'content_to_multi_assay_dict',
                    **(metadata or {}),
                },
            )
            normalized[assay_name] = verify_assay_value_assignments(
                content,
                assay_name,
                assay_names,
                assay_payload,
                retry=1,
                audit_path=audit_path,
                timeout_seconds=timeout_seconds,
                metadata={
                    'source_task': 'content_to_multi_assay_dict',
                    **(metadata or {}),
                },
            )
        return normalized

    return run_text_json_task(
        task_name='content_to_multi_assay_dict',
        prompt=build_content_to_multi_assay_dict_prompt(content, assay_names, prompt_compound_id_list),
        parser=_parser,
        retry=retry,
        audit_path=audit_path,
        timeout_seconds=timeout_seconds,
        metadata={
            'assay_names': assay_names,
            'compound_id_count': len(compound_id_list or []),
            'prompt_compound_id_count': len(prompt_compound_id_list or []),
            'content_chars': len(str(content or '')),
            **(metadata or {}),
        },
    )


def encode_image_to_base64_data_uri(image_path):
    """Encodes an image file to a base64 data URI."""
    try:
        with open(image_path, 'rb') as image_file_obj:
            encoded_image_bytes = base64.b64encode(image_file_obj.read())
        encoded_image_text = encoded_image_bytes.decode("utf-8")

        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".jpg" or ext == ".jpeg": mime_type = "image/jpeg"
        elif ext == ".png": mime_type = "image/png"
        elif ext == ".webp": mime_type = "image/webp"
        elif ext == ".gif": mime_type = "image/gif"
        else:
            mime_type = "application/octet-stream"
            logger.warning("Unknown MIME type for %s, using default %s.", image_path, mime_type)
        return f"data:{mime_type};base64,{encoded_image_text}"
    except FileNotFoundError: logger.error("Image file not found at %s", image_path); raise
    except Exception as e: logger.error("Error encoding image %s: %s", image_path, e); raise


def call_visual_model(image_file, prompt, retries=None):
    """Call the configured visual model with a hard outer timeout guard and retry.

    The SDK-level ``timeout`` parameter is not always reliable (e.g. slow
    servers that accept the TCP connection but never send a response).  We
    wrap the actual call inside a daemon thread and join it with a slightly
    larger timeout so that callers never block indefinitely.

    On timeout or API error the call is retried up to *retries* times with a
    short back-off.  If every attempt fails the last exception is re-raised.
    """
    retries = VISION_MODEL_MAX_RETRIES if retries is None else max(1, int(retries))
    last_exc = None
    for attempt in range(1, retries + 1):
        result = [None]
        exc = [None]
        outer_timeout = VISION_MODEL_TIMEOUT_SECONDS + VISION_MODEL_OUTER_TIMEOUT_PADDING_SECONDS
        acquired = _visual_model_semaphore.acquire(timeout=outer_timeout)
        if not acquired:
            last_exc = TimeoutError(
                f"Waiting for visual model slot exceeded {outer_timeout}s "
                f"(concurrency={VISION_MODEL_CONCURRENCY}, attempt {attempt}/{retries})"
            )
            logger.warning("%s", last_exc)
            continue

        def _run():
            try:
                result[0] = _call_visual_model_inner(image_file, prompt)
            except Exception as e:
                exc[0] = e
            finally:
                try:
                    _visual_model_semaphore.release()
                except ValueError:
                    pass

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=outer_timeout)

        if t.is_alive():
            last_exc = TimeoutError(
                f"Visual model call exceeded {outer_timeout}s (attempt {attempt}/{retries})"
            )
            logger.warning("%s", last_exc)
        elif exc[0] is not None:
            last_exc = exc[0]
            logger.warning("Visual model error on attempt %s/%s: %s", attempt, retries, last_exc)
        else:
            return result[0]

        # Back-off before next retry
        if attempt < retries:
            wait = min(5 * attempt, 15)
            logger.info("Retrying visual model call in %ss ...", wait)
            time.sleep(wait)

    raise last_exc


def _call_visual_model_inner(image_file, prompt):
    response_text = None
    actual_model_name = VISUAL_MODEL_NAME
    json_system_prompt = get_system_prompt(
        VISION_MODEL_RUNTIME,
        'vision',
        'chemistry_vision',
        "You are a careful vision assistant.",
    )

    if not VISUAL_MODEL_KEY:
        raise ValueError("VISUAL_MODEL_KEY is not configured in constants.py.")
    if not actual_model_name:
        raise ValueError("VISUAL_MODEL_NAME is required.")

    logger.info("Using OpenAI-compatible visual model: %s", actual_model_name)
    try:
        client = require_openai()(api_key=VISUAL_MODEL_KEY, base_url=VISUAL_MODEL_URL, timeout=VISION_MODEL_TIMEOUT_SECONDS)
        image_base64_uri = encode_image_to_base64_data_uri(image_file)
        task_name = 'visual_id_extraction'
        lowered_prompt = prompt.lower() if isinstance(prompt, str) else ''
        if 'structure_type' in lowered_prompt or 'crop_status' in lowered_prompt:
            task_name = 'visual_json_classification'
        temperature = get_task_temperature(VISION_MODEL_RUNTIME, task_name, channel='vision', default=0.0)
        messages = [
            {"role": "system", "content": json_system_prompt},
            {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_base64_uri}},
        ]}]
        logger.info("Sending prompt and image to OpenAI-compatible model '%s'.", actual_model_name)
        completion = client.chat.completions.create(model=actual_model_name, messages=messages, temperature=temperature)
        response_text = completion.choices[0].message.content
    except Exception as e:
        logger.error("Error with OpenAI-compatible visual model '%s': %s", actual_model_name, e)
        raise

    return sanitize_model_response_text(response_text)


def run_text_json_task(
    *,
    task_name,
    prompt,
    parser,
    retry=3,
    audit_path=None,
    metadata=None,
    timeout_seconds=None,
):
    if not LLM_TEXT_MODEL_KEY or not LLM_TEXT_MODEL_URL or not LLM_TEXT_MODEL_NAME:
        raise ValueError("OpenAI-compatible text model is not configured.")

    retry_delays = get_retry_delays(TEXT_MODEL_RUNTIME, task_name, channel='text')
    json_system_prompt = get_system_prompt(
        TEXT_MODEL_RUNTIME,
        'text',
        'json_extraction',
        "You are a precise information extraction assistant. Return JSON only.",
    )
    temperature = get_task_temperature(TEXT_MODEL_RUNTIME, task_name, channel='text', default=0.0)
    request_timeout = timeout_seconds or LLM_MODEL_TIMEOUT_SECONDS

    def _operation():
        logger.info("Calling LLM '%s' at '%s' for %s.", LLM_TEXT_MODEL_NAME, LLM_TEXT_MODEL_URL, task_name)
        result = [None]
        exc = [None]

        def _run():
            try:
                client = require_openai()(
                    api_key=LLM_TEXT_MODEL_KEY,
                    base_url=LLM_TEXT_MODEL_URL,
                    timeout=request_timeout,
                )
                response = client.chat.completions.create(
                    model=LLM_TEXT_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": json_system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )
                result[0] = sanitize_model_response_text(response.choices[0].message.content or '')
            except Exception as error:
                exc[0] = error

        outer_timeout = request_timeout + LLM_MODEL_OUTER_TIMEOUT_PADDING_SECONDS
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=outer_timeout)
        if thread.is_alive():
            raise TimeoutError(f"Text model call exceeded {outer_timeout}s")
        if exc[0] is not None:
            raise exc[0]
        return result[0] or ''

    return run_json_task(
        task_name=task_name,
        channel='text',
        operation=_operation,
        parser=parser,
        retry=retry,
        retry_delays=retry_delays,
        audit_path=audit_path,
        metadata={
            'model': LLM_TEXT_MODEL_NAME,
            'url': LLM_TEXT_MODEL_URL,
            'prompt_chars': len(str(prompt or '')),
            'timeout_seconds': request_timeout,
            'outer_timeout_seconds': request_timeout + LLM_MODEL_OUTER_TIMEOUT_PADDING_SECONDS,
            **(metadata or {}),
        },
    )


def run_vision_json_task(
    *,
    task_name,
    image_file,
    prompt,
    parser,
    retry=None,
    audit_path=None,
    metadata=None,
):
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image file for {task_name} not found: {image_file}")

    def _operation():
        return call_visual_model(image_file, prompt, retries=1)

    return run_json_task(
        task_name=task_name,
        channel='vision',
        operation=_operation,
        parser=parser,
        retry=VISION_MODEL_MAX_RETRIES if retry is None else retry,
        retry_delays=get_retry_delays(VISION_MODEL_RUNTIME, task_name, channel='vision'),
        audit_path=audit_path,
        metadata={
            'model': VISUAL_MODEL_NAME,
            'url': VISUAL_MODEL_URL,
            'image_file': str(image_file),
            'prompt_chars': len(str(prompt or '')),
            **(metadata or {}),
        },
    )


def normalize_structure_type(value):
    if not isinstance(value, str):
        return 'uncertain'

    normalized = value.strip().lower().replace('-', '_').replace(' ', '_')
    alias_map = {
        'complete': 'complete_compound',
        'complete_molecule': 'complete_compound',
        'complete_compound': 'complete_compound',
        'full_compound': 'complete_compound',
        'full_molecule': 'complete_compound',
        'molecule': 'complete_compound',
        'specific_molecule': 'complete_compound',
        'markush': 'markush',
        'markush_structure': 'markush',
        'fragment': 'fragment',
        'partial': 'fragment',
        'partial_structure': 'fragment',
        'noise': 'noise',
        'artifact': 'noise',
        'non_molecule': 'noise',
        'non_molecular': 'noise',
        'uncertain': 'uncertain',
        'unknown': 'uncertain',
        'ambiguous': 'uncertain',
    }
    if normalized in alias_map:
        return alias_map[normalized]
    if 'markush' in normalized or 'rgroup' in normalized or 'r_group' in normalized:
        return 'markush'
    if 'fragment' in normalized or 'partial' in normalized or 'substituent' in normalized:
        return 'fragment'
    if 'noise' in normalized or 'artifact' in normalized or 'reaction' in normalized or 'text' in normalized:
        return 'noise'
    if 'complete' in normalized or 'full' in normalized or 'specific' in normalized:
        return 'complete_compound'
    return 'uncertain'


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'true', 'yes', '1', 'y'}:
            return True
        if normalized in {'false', 'no', '0', 'n'}:
            return False
    return None


def analyze_border_contact(image_file, dark_threshold=245, band_width=4, ratio_threshold=0.035):
    """
    Detects whether non-white drawing content strongly touches the image border,
    which is a useful cue for cropped/partial structures.
    """
    if PIL is None:
        return {
            'suspicious': False,
            'sides': [],
            'ratios': {},
        }

    try:
        img = PIL.Image.open(image_file).convert('L')
        width, height = img.size
        if width <= 0 or height <= 0:
            return {'suspicious': False, 'sides': [], 'ratios': {}}

        band_width = max(1, min(band_width, width, height))
        pixels = img.load()

        def ratio_for_side(side):
            dark = 0
            total = 0
            if side == 'left':
                for x in range(band_width):
                    for y in range(height):
                        total += 1
                        if pixels[x, y] < dark_threshold:
                            dark += 1
            elif side == 'right':
                for x in range(max(0, width - band_width), width):
                    for y in range(height):
                        total += 1
                        if pixels[x, y] < dark_threshold:
                            dark += 1
            elif side == 'top':
                for y in range(band_width):
                    for x in range(width):
                        total += 1
                        if pixels[x, y] < dark_threshold:
                            dark += 1
            elif side == 'bottom':
                for y in range(max(0, height - band_width), height):
                    for x in range(width):
                        total += 1
                        if pixels[x, y] < dark_threshold:
                            dark += 1
            return dark / total if total else 0.0

        ratios = {
            'left': ratio_for_side('left'),
            'right': ratio_for_side('right'),
            'top': ratio_for_side('top'),
            'bottom': ratio_for_side('bottom'),
        }
        suspicious_sides = [side for side, ratio in ratios.items() if ratio >= ratio_threshold]
        return {
            'suspicious': bool(suspicious_sides),
            'sides': suspicious_sides,
            'ratios': ratios,
        }
    except Exception as e:
        logger.warning("Failed to analyze border contact for %s: %s", image_file, e)
        return {'suspicious': False, 'sides': [], 'ratios': {}}


@proxy_decorator
def classify_structure_candidate(image_file, prompt=None, strictness=None):
    """
    Classifies a candidate structure image so that only complete compounds proceed downstream.
    """
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image file for classify_structure_candidate not found: {image_file}")

    strictness = normalize_filter_strictness(strictness or STRUCTURE_FILTER_STRICTNESS)
    if prompt is None:
        prompt = build_classify_structure_prompt(strictness=strictness)

    response_text = ''
    try:
        def _parse_candidate_response(text):
            parsed = parse_structure_classification_payload(
                text,
                'classify_structure_candidate',
                'classify_structure_candidate',
            )
            parsed['raw_response'] = text or ''
            return parsed

        payload = run_vision_json_task(
            task_name='classify_structure_candidate',
            image_file=image_file,
            prompt=prompt,
            parser=_parse_candidate_response,
            metadata={'strictness': strictness},
        )
    except Exception as e:
        logger.warning("classify_structure_candidate failed for %s: %s", image_file, e)
        return {
            'structure_type': 'uncertain',
            'is_complete_compound': False,
            'confidence': 'low',
            'reason': f'model_call_failed: {e}',
            'raw_response': '',
            'border_contact': {'suspicious': False, 'sides': [], 'ratios': {}},
            'strictness': strictness,
            'MODEL_CALL_OK': False,
            'ERROR_TYPE': classify_exception(e),
            'OUTPUT_CONTRACT_ERROR': str(e),
        }

    structure_type = payload['structure_type']
    is_complete_compound = payload['is_complete_compound']
    response_text = payload.get('raw_response', '')

    border_contact = analyze_border_contact(image_file)
    should_run_crop_check = structure_type == 'complete_compound' and border_contact.get('suspicious')
    if strictness == 'permissive':
        should_run_crop_check = False

    crop_check_confirmed_not_cropped = False
    visual_review_errors = []
    if should_run_crop_check:
        crop_check_prompt = build_crop_check_prompt(', '.join(border_contact.get('sides', [])) or 'none', strictness=strictness)
        try:
            def _parse_crop_response(text):
                parsed = parse_crop_check_payload(text)
                parsed['raw_response'] = text or ''
                return parsed

            crop_check_payload = run_vision_json_task(
                task_name='crop_check',
                image_file=image_file,
                prompt=crop_check_prompt,
                parser=_parse_crop_response,
                metadata={'strictness': strictness, 'border_sides': border_contact.get('sides', [])},
            )
            crop_check_response_text = crop_check_payload.get('raw_response', '')
        except Exception as e:
            logger.warning("crop-check visual model failed for %s: %s", image_file, e)
            crop_check_response_text = ''
            visual_review_errors.append(f'crop-check model error: {e}')
        if crop_check_response_text:
            crop_status = crop_check_payload['crop_status']
            is_cropped = crop_check_payload['is_cropped']
            if is_cropped or crop_status == 'fragment':
                structure_type = 'fragment'
                is_complete_compound = False
                response_text = crop_check_response_text
                payload = {
                    'structure_type': 'fragment',
                    'is_complete_compound': False,
                    'confidence': crop_check_payload['confidence'],
                    'reason': crop_check_payload['reason'],
                }
            elif crop_status == 'not_cropped':
                crop_check_confirmed_not_cropped = True

    should_run_border_review = structure_type == 'complete_compound' and border_contact.get('suspicious')
    if strictness == 'permissive':
        should_run_border_review = False

    border_review_confirmed_complete = False
    if should_run_border_review:
        review_prompt = build_border_review_prompt(', '.join(border_contact.get('sides', [])) or 'none', strictness=strictness)
        try:
            def _parse_border_response(text):
                parsed = parse_structure_classification_payload(
                    text,
                    'border_review',
                    'border_review',
                )
                parsed['raw_response'] = text or ''
                return parsed

            review_payload = run_vision_json_task(
                task_name='border_review',
                image_file=image_file,
                prompt=review_prompt,
                parser=_parse_border_response,
                metadata={'strictness': strictness, 'border_sides': border_contact.get('sides', [])},
            )
            review_response_text = review_payload.get('raw_response', '')
        except Exception as e:
            logger.warning("border-review visual model failed for %s: %s", image_file, e)
            review_response_text = ''
            visual_review_errors.append(f'border-review model error: {e}')
        if review_response_text:
            review_type = review_payload['structure_type']
            review_complete = review_payload['is_complete_compound']
            if review_type != 'complete_compound':
                structure_type = review_type
                is_complete_compound = bool(review_complete)
                response_text = review_response_text
                payload = review_payload
            else:
                border_review_confirmed_complete = True

    if (
        structure_type == 'complete_compound'
        and border_contact.get('suspicious')
        and visual_review_errors
        and not (crop_check_confirmed_not_cropped or border_review_confirmed_complete)
    ):
        structure_type = 'uncertain'
        is_complete_compound = False
        payload['confidence'] = 'low'
        payload['reason'] = '; '.join(visual_review_errors[:2])

    if strictness == 'strict' and structure_type == 'complete_compound' and border_contact.get('suspicious'):
        suspicious_sides = set(border_contact.get('sides') or [])
        visual_review_confirmed_complete = crop_check_confirmed_not_cropped or border_review_confirmed_complete
        if suspicious_sides.intersection({'left', 'right', 'top', 'bottom'}) and not visual_review_confirmed_complete:
            structure_type = 'uncertain'
            is_complete_compound = False
            if not isinstance(payload, dict):
                payload = {}
            payload['reason'] = (
                f"Border-contact review: drawing touches image border "
                f"on {', '.join(sorted(suspicious_sides))}, and visual completeness was not confirmed."
            )

    reason = payload.get('reason') if isinstance(payload.get('reason'), str) else ''
    confidence = payload.get('confidence') if isinstance(payload.get('confidence'), str) else ''
    return {
        'structure_type': structure_type,
        'is_complete_compound': bool(is_complete_compound),
        'confidence': confidence,
        'reason': (reason or '').strip(),
        'raw_response': response_text or '',
        'border_contact': border_contact,
        'strictness': strictness,
        'MODEL_CALL_OK': True,
    }


@proxy_decorator
# @cost_time
def structure_to_id(image_file, prompt=None):
    """
    Extracts the compound ID from a chemical structure image using the visual model
    configured by VISUAL_MODEL_NAME, VISUAL_MODEL_URL, and VISUAL_MODEL_KEY.
    """

    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image file for structure_to_id not found: {image_file}")

    if prompt is None:
        prompt = build_structure_to_id_prompt()
    response_text = ''
    try:
        def _parse_structure_to_id_response(text):
            parsed = parse_visual_structure_id_payload(text)
            parsed['RAW_RESPONSE'] = text or ''
            return parsed

        payload = run_vision_json_task(
            task_name='structure_to_id',
            image_file=image_file,
            prompt=prompt,
            parser=_parse_structure_to_id_response,
        )
        payload['MODEL_CALL_OK'] = True
        return payload
    except Exception as e:
        logger.warning("structure_to_id failed for %s: %s", image_file, e)
        return {
            'COMPOUND_ID': 'None',
            'VISUAL_ROLE': 'unknown',
            'ID_SOURCE': 'none',
            'EVIDENCE': f'model_call_failed: {e}',
            'CONFIDENCE': 'low',
            'RAW_RESPONSE': '',
            'MODEL_CALL_OK': False,
            'ERROR_TYPE': classify_exception(e),
            'OUTPUT_CONTRACT_ERROR': str(e),
        }

@proxy_decorator
def get_compound_id_from_description(description, audit_path=None, metadata=None):
    """
    Extracts a compound ID from a description string using an OpenAI-compatible text model.
    """

    prompt = build_description_to_id_prompt(description)
    retry = 3

    def _parser(response_text):
        return parse_compound_id_payload(
            response_text,
            'get_compound_id_from_description',
            'get_compound_id_from_description',
            none_value="None",
        )

    return run_text_json_task(
        task_name='get_compound_id_from_description',
        prompt=prompt,
        parser=_parser,
        retry=retry,
        audit_path=audit_path,
        metadata={
            'description_chars': len(str(description or '')),
            **(metadata or {}),
        },
    )


@proxy_decorator
def resolve_compound_id_alias(raw_id, compound_id_list, context='', audit_path=None, metadata=None):
    """
    Resolve a raw/aliased compound ID to one canonical ID from the provided allowlist.
    Returns None when resolution is ambiguous or unsupported.
    """
    prompt = build_resolve_compound_id_alias_prompt(raw_id, compound_id_list, context=context)
    retry = 3

    def _parser(response_text):
        return parse_compound_id_payload(
            response_text,
            'resolve_compound_id_alias',
            'resolve_compound_id_alias',
            allowed_ids=compound_id_list,
            none_value=None,
        )

    return run_text_json_task(
        task_name='resolve_compound_id_alias',
        prompt=prompt,
        parser=_parser,
        retry=retry,
        audit_path=audit_path,
        metadata={
            'raw_id': str(raw_id or ''),
            'compound_id_count': len(compound_id_list or []),
            'context_chars': len(str(context or '')),
            **(metadata or {}),
        },
    )
