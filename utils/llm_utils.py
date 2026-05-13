import os
import warnings
warnings.filterwarnings("ignore")

import time
import requests
from functools import wraps
import json
import base64
import re
import sys
import signal
import threading

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import PIL.Image
    import PIL.ImageDraw
    import PIL.ImageFont
except ImportError:
    print("Error: Pillow (PIL) package not found. Please install it: pip install Pillow")
    print("Testing of structure_to_id with dummy image creation will be skipped.")
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
    print("Error: constants.py not found. Please ensure it's in the same directory, parent directory, or your PYTHONPATH.")
    sys.exit(1)

from utils.skill_prompt_loader import load_merged_skill_json, render_skill_prompt_with_examples


GEMINI_API_KEY_FOR_GEMINI_MODELS = getattr(constants, 'GEMINI_API_KEY', None)
GEMINI_MODEL_NAME = getattr(constants, 'GEMINI_MODEL_NAME', 'gemma-3-27b-it')
DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT = GEMINI_MODEL_NAME

# For get_compound_id_from_description (OpenAI-compatible text model)
LLM_TEXT_MODEL_NAME = getattr(constants, 'LLM_OPENAI_COMPATIBLE_MODEL_NAME', None)
LLM_TEXT_MODEL_URL = getattr(constants, 'LLM_OPENAI_COMPATIBLE_MODEL_URL', None)
LLM_TEXT_MODEL_KEY = getattr(constants, 'LLM_OPENAI_COMPATIBLE_MODEL_KEY', None)

# Visual Model Configuration
VISUAL_MODEL_NAME = getattr(constants, 'VISUAL_MODEL_NAME', None)
VISUAL_MODEL_URL = getattr(constants, 'VISUAL_MODEL_URL', None)
VISUAL_MODEL_KEY = getattr(constants, 'VISUAL_MODEL_KEY', GEMINI_API_KEY_FOR_GEMINI_MODELS if GEMINI_API_KEY_FOR_GEMINI_MODELS else None)
STRUCTURE_FILTER_STRICTNESS = getattr(constants, 'STRUCTURE_FILTER_STRICTNESS', 'strict')
VISION_MODEL_TIMEOUT_SECONDS = int(getattr(constants, 'VISION_MODEL_TIMEOUT_SECONDS', 120))
VISION_MODEL_MAX_RETRIES = max(1, int(getattr(constants, 'VISION_MODEL_MAX_RETRIES', 1)))
VISION_MODEL_OUTER_TIMEOUT_PADDING_SECONDS = max(1, int(getattr(constants, 'VISION_MODEL_OUTER_TIMEOUT_PADDING_SECONDS', 10)))
VISION_MODEL_CONCURRENCY = max(1, int(getattr(constants, 'VISION_MODEL_CONCURRENCY', 2)))
LLM_MODEL_TIMEOUT_SECONDS = int(getattr(constants, 'LLM_MODEL_TIMEOUT_SECONDS', 180))

HTTP_PROXY = getattr(constants, 'HTTP_PROXY', '')
HTTPS_PROXY = getattr(constants, 'HTTPS_PROXY', '')
_visual_model_semaphore = threading.BoundedSemaphore(VISION_MODEL_CONCURRENCY)

# OpenAI-compatible model
if LLM_TEXT_MODEL_NAME and LLM_TEXT_MODEL_URL and LLM_TEXT_MODEL_KEY:
    LLM_MODEL_TYPE = 'openai'
# Gemini model
elif not LLM_TEXT_MODEL_KEY or not LLM_TEXT_MODEL_URL or not LLM_TEXT_MODEL_NAME \
        and (GEMINI_API_KEY_FOR_GEMINI_MODELS and GEMINI_MODEL_NAME):
    print(f"LLM OpenAI-compatible model not configured, using Gemini model instead.")
    LLM_MODEL_TYPE = 'gemini'
# 如果都没有
else:
    raise ValueError("No LLM model configured for get_compound_id_from_description. Please set LLM_OPENAI_COMPATIBLE_MODEL_NAME, LLM_OPENAI_COMPATIBLE_MODEL_URL, and LLM_OPENAI_COMPATIBLE_MODEL_KEY in constants.py.")

if VISUAL_MODEL_KEY and VISUAL_MODEL_URL and VISUAL_MODEL_NAME:
    VISUAL_MODEL_TYPE = 'openai'
    print(f"Info: Using OpenAI-compatible visual model: {VISUAL_MODEL_NAME}")
elif not VISUAL_MODEL_KEY or not VISUAL_MODEL_URL or not VISUAL_MODEL_NAME \
        and (GEMINI_API_KEY_FOR_GEMINI_MODELS and GEMINI_MODEL_NAME):
    print(f"VISUAL_MODEL_NAME not configured, using Gemini model instead.")
    VISUAL_MODEL_TYPE = 'gemini'
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

_genai_configured_with_key = None


def require_genai():
    if genai is None:
        raise ImportError("google.generativeai package not found. Please install it: pip install google-generativeai")
    return genai


def require_openai():
    if OpenAI is None:
        raise ImportError("openai package not found. Please install it: pip install openai")
    return OpenAI


def configure_genai(api_key):
    """
    Configures the Google Generative AI client.
    Ensures configuration happens only once or if the key changes.
    """
    global _genai_configured_with_key
    current_genai = require_genai()
    if _genai_configured_with_key == api_key and _genai_configured_with_key is not None:
        return

    if api_key:
        current_genai.configure(api_key=api_key)
        _genai_configured_with_key = api_key
    elif os.getenv('GOOGLE_API_KEY'):
        try:
            current_genai.configure()
            _genai_configured_with_key = os.getenv('GOOGLE_API_KEY')
            print("Info: Configured GenAI using GOOGLE_API_KEY environment variable.")
        except Exception as e:
            print(f"Warning: Failed to configure GenAI with GOOGLE_API_KEY env var: {e}")
    else:
        print("Warning: GEMINI_API_KEY not provided for configure_genai and GOOGLE_API_KEY env var not set or failed to configure.")


def sanitize_model_response_text(response_text):
    if not isinstance(response_text, str):
        return ''
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
    response_text = re.sub(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', r'\1', response_text, flags=re.DOTALL)
    return response_text.strip()


def extract_json_content(text):
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    if '```json' in text:
        json_match = text.split('```json', 1)
        if len(json_match) > 1:
            return json_match[1].split('```', 1)[0].strip()
    if '```' in text and text.count('```') >= 2:
        parts = text.split('```', 2)
        if len(parts) >= 2:
            return parts[1].strip()
    if '“json' in text:
        json_match = text.split('“json', 1)
        if len(json_match) > 1:
            temp_content = json_match[1].strip()
            if temp_content.endswith('”'):
                temp_content = temp_content[:-1].strip()
            return temp_content

    start_brace = text.find('{')
    end_brace = text.rfind('}')
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        return text[start_brace:end_brace + 1].strip()
    return None


def build_content_to_dict_prompt(content, assay_name, compound_id_list=None):
    if compound_id_list is None:
        compound_id_list_block = '开始提取数据...\n\n'
    else:
        compound_id_list = list(dict.fromkeys(compound_id_list))
        compounds = ', '.join([f'"{cid}"' for cid in compound_id_list])
        compound_id_list_block = '化合物ID列表如下，解析时请不要超出此列表范围：\n'
        compound_id_list_block += f"{compounds}\n\n"
        compound_id_list_block += '\n开始提取数据: \n'

    return render_skill_prompt_with_examples(
        'biocheminsight-text-models',
        'references/content_to_dict_prompt.md',
        'references/examples/content_to_dict_examples.md',
        {
            'ASSAY_NAME': assay_name,
            'MARKDOWN_TEXT': content,
            'COMPOUND_ID_LIST_BLOCK': compound_id_list_block,
        },
    )


def build_content_to_multi_assay_dict_prompt(content, assay_names, compound_id_list=None):
    assay_names = [str(name).strip() for name in (assay_names or []) if str(name).strip()]
    if compound_id_list is None:
        compound_id_list_block = '开始提取数据...\n\n'
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


@proxy_decorator
# @cost_time
def content_to_dict(content, assay_name, compound_id_list=None, retry=3):
    """
    Converts the content of a Markdown text to a dictionary using Google Generative AI.
    """
    LLM_MODEL_TYPE = 'openai'
    if not LLM_TEXT_MODEL_KEY or not LLM_TEXT_MODEL_URL or not LLM_TEXT_MODEL_NAME:
        print(f"LLM OpenAI-compatible model not configured, using Gemini model instead.")
        LLM_MODEL_TYPE = 'gemini'

    print(f"Info: Converting content to dict using model type: {LLM_MODEL_TYPE}")

    prompt = build_content_to_dict_prompt(content, assay_name, compound_id_list)
    retry_delays = get_retry_delays(TEXT_MODEL_RUNTIME, 'content_to_dict', channel='text')
    json_system_prompt = get_system_prompt(
        TEXT_MODEL_RUNTIME,
        'text',
        'json_extraction',
        "You are a helpful assistant designed to output JSON.",
    )
    temperature = get_task_temperature(TEXT_MODEL_RUNTIME, 'content_to_dict', channel='text', default=0.0)


    if LLM_MODEL_TYPE == 'gemini':
        configure_genai(GEMINI_API_KEY_FOR_GEMINI_MODELS)
        model = require_genai().GenerativeModel(DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT)
        response_text_for_error = "N/A" 

        for attempt in range(retry):
            try:
                response = model.generate_content(prompt)
                if not response.candidates or not response.candidates[0].content.parts:
                    response_text_for_error = str(response)
                    raise ValueError(f"Model '{DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT}' returned no content/candidates.")

                result_text = response.candidates[0].content.parts[0].text
                response_text_for_error = result_text
                result_text = result_text.replace('null', 'None')
                json_content = extract_json_content(result_text) or result_text.strip()

                if not json_content:
                    raise ValueError("Could not extract JSON content from the model's response.")

                assay_dict = json.loads(json_content)
                return assay_dict
            except json.JSONDecodeError as json_e:
                print(f"Attempt {attempt + 1}/{retry} (JSONDecodeError): {json_e} in model '{DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT}'")
                print(f"Problematic JSON content: {json_content[:500] if json_content else 'None'}{'...' if json_content and len(json_content) > 500 else ''}")
                if attempt < retry - 1: sleep_before_retry(attempt, retry, retry_delays); continue
                print(f"Final attempt failed. Prompt:\n{prompt[:500]}...\nResponse:\n{response_text_for_error}"); raise
            except Exception as e:
                print(f"Attempt {attempt + 1}/{retry} (Exception): {e} in model '{DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT}'")
                if attempt < retry - 1: sleep_before_retry(attempt, retry, retry_delays); continue
                print(f"Final attempt failed. Prompt:\n{prompt[:500]}...\nResponse:\n{response_text_for_error}"); raise e
    elif LLM_MODEL_TYPE == 'openai':
        client = require_openai()(api_key=LLM_TEXT_MODEL_KEY, base_url=LLM_TEXT_MODEL_URL, timeout=LLM_MODEL_TIMEOUT_SECONDS)
        response_text_for_error = "N/A"
        for attempt in range(retry):
            try:
                print(f"Info: Calling LLM '{LLM_TEXT_MODEL_NAME}' at '{LLM_TEXT_MODEL_URL}' for content_to_dict.")
                response = client.chat.completions.create(
                    model=LLM_TEXT_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": json_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
                response_text_for_error = sanitize_model_response_text(response.choices[0].message.content or '')
                response_text_for_error = response_text_for_error.replace('null', 'None')
                json_content = extract_json_content(response_text_for_error)
                if not json_content:
                    raise ValueError("Could not extract JSON content from the model's response.")
                assay_dict = json.loads(json_content)
                return assay_dict
            except json.JSONDecodeError as json_e:
                print(f"Attempt {attempt + 1}/{retry} (JSONDecodeError): {json_e} in model '{LLM_TEXT_MODEL_URL}'")
                print(f"Problematic JSON content: {json_content[:500] if json_content else 'None'}{'...' if json_content and len(json_content) > 500 else ''}")
                if attempt < retry - 1: sleep_before_retry(attempt, retry, retry_delays); continue
                print(f"Final attempt failed. Prompt:\n{prompt[:500]}...\nResponse:\n{response_text_for_error}"); raise
            except Exception as e:
                print(f"Attempt {attempt + 1}/{retry} (Exception): {e} in model '{LLM_TEXT_MODEL_URL}'")
                if attempt < retry - 1: sleep_before_retry(attempt, retry, retry_delays); continue
                print(f"Final attempt failed. Prompt:\n{prompt[:500]}...\nResponse:\n{response_text_for_error}"); raise e
    else:
        print(f"Error: Unsupported LLM model type '{LLM_MODEL_TYPE}' for content_to_dict.")
    return None


@proxy_decorator
def identify_assay_visual_review_requests(ocr_context, assay_dicts, parsed_tables=None, retry=2):
    """
    Ask the text model to decide which extracted assay cells need visual reread.
    The decision is intentionally model-owned and context-based: the model sees
    the whole OCR chunk, parsed table grids, assay names, and extracted values.
    """
    if not isinstance(assay_dicts, dict) or not assay_dicts:
        return {}

    LLM_MODEL_TYPE = 'openai'
    if not LLM_TEXT_MODEL_KEY or not LLM_TEXT_MODEL_URL or not LLM_TEXT_MODEL_NAME:
        LLM_MODEL_TYPE = 'gemini'

    prompt = build_identify_assay_visual_review_requests_prompt(ocr_context, assay_dicts, parsed_tables)
    retry_delays = get_retry_delays(TEXT_MODEL_RUNTIME, 'identify_assay_visual_review_requests', channel='text')
    json_system_prompt = get_system_prompt(
        TEXT_MODEL_RUNTIME,
        'text',
        'json_extraction',
        "You are a helpful assistant designed to output JSON.",
    )
    temperature = get_task_temperature(TEXT_MODEL_RUNTIME, 'identify_assay_visual_review_requests', channel='text', default=0.0)

    if LLM_MODEL_TYPE == 'gemini':
        configure_genai(GEMINI_API_KEY_FOR_GEMINI_MODELS)
        model = require_genai().GenerativeModel(DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT)
        for attempt in range(retry):
            try:
                response = model.generate_content(prompt)
                result_text = response.candidates[0].content.parts[0].text if response.candidates else ''
                json_content = extract_json_content(result_text) or result_text.strip()
                payload = json.loads(json_content)
                return payload if isinstance(payload, dict) else {}
            except Exception as e:
                if attempt < retry - 1:
                    sleep_before_retry(attempt, retry, retry_delays)
                    continue
                print(f"Warning: identify_assay_visual_review_requests failed: {e}")
                return {}

    if LLM_MODEL_TYPE == 'openai':
        client = require_openai()(api_key=LLM_TEXT_MODEL_KEY, base_url=LLM_TEXT_MODEL_URL, timeout=LLM_MODEL_TIMEOUT_SECONDS)
        for attempt in range(retry):
            try:
                response = client.chat.completions.create(
                    model=LLM_TEXT_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": json_system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )
                response_text = sanitize_model_response_text(response.choices[0].message.content or '')
                json_content = extract_json_content(response_text) or response_text.strip()
                payload = json.loads(json_content)
                return payload if isinstance(payload, dict) else {}
            except Exception as e:
                if attempt < retry - 1:
                    sleep_before_retry(attempt, retry, retry_delays)
                    continue
                print(f"Warning: identify_assay_visual_review_requests failed: {e}")
                return {}

    return {}


@proxy_decorator
def reconcile_assay_values_with_visual_report(ocr_context, assay_dicts, visual_report, retry=2):
    """
    Let the text model reconcile the original OCR/text extraction with the
    visual audit report. The vision model reports observations; the text model
    owns the final structured assay dictionary.
    """
    if not isinstance(assay_dicts, dict) or not assay_dicts:
        return assay_dicts
    if not isinstance(visual_report, dict) or not visual_report:
        return assay_dicts

    LLM_MODEL_TYPE = 'openai'
    if not LLM_TEXT_MODEL_KEY or not LLM_TEXT_MODEL_URL or not LLM_TEXT_MODEL_NAME:
        LLM_MODEL_TYPE = 'gemini'

    prompt = build_reconcile_assay_values_with_visual_report_prompt(ocr_context, assay_dicts, visual_report)
    retry_delays = get_retry_delays(TEXT_MODEL_RUNTIME, 'reconcile_assay_values_with_visual_report', channel='text')
    json_system_prompt = get_system_prompt(
        TEXT_MODEL_RUNTIME,
        'text',
        'json_extraction',
        "You are a helpful assistant designed to output JSON.",
    )
    temperature = get_task_temperature(TEXT_MODEL_RUNTIME, 'reconcile_assay_values_with_visual_report', channel='text', default=0.0)

    if LLM_MODEL_TYPE == 'gemini':
        configure_genai(GEMINI_API_KEY_FOR_GEMINI_MODELS)
        model = require_genai().GenerativeModel(DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT)
        for attempt in range(retry):
            try:
                response = model.generate_content(prompt)
                result_text = response.candidates[0].content.parts[0].text if response.candidates else ''
                json_content = extract_json_content(result_text) or result_text.strip()
                payload = json.loads(json_content)
                return payload if isinstance(payload, dict) else assay_dicts
            except Exception as e:
                if attempt < retry - 1:
                    sleep_before_retry(attempt, retry, retry_delays)
                    continue
                print(f"Warning: reconcile_assay_values_with_visual_report failed: {e}")
                return assay_dicts

    if LLM_MODEL_TYPE == 'openai':
        client = require_openai()(api_key=LLM_TEXT_MODEL_KEY, base_url=LLM_TEXT_MODEL_URL, timeout=LLM_MODEL_TIMEOUT_SECONDS)
        for attempt in range(retry):
            try:
                response = client.chat.completions.create(
                    model=LLM_TEXT_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": json_system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )
                response_text = sanitize_model_response_text(response.choices[0].message.content or '')
                json_content = extract_json_content(response_text) or response_text.strip()
                payload = json.loads(json_content)
                return payload if isinstance(payload, dict) else assay_dicts
            except Exception as e:
                if attempt < retry - 1:
                    sleep_before_retry(attempt, retry, retry_delays)
                    continue
                print(f"Warning: reconcile_assay_values_with_visual_report failed: {e}")
                return assay_dicts

    return assay_dicts


@proxy_decorator
def content_to_multi_assay_dict(content, assay_names, compound_id_list=None, retry=3):
    """
    Converts Markdown/OCR content into a nested dictionary for multiple requested assay names.
    """
    LLM_MODEL_TYPE = 'openai'
    if not LLM_TEXT_MODEL_KEY or not LLM_TEXT_MODEL_URL or not LLM_TEXT_MODEL_NAME:
        print("LLM OpenAI-compatible model not configured, using Gemini model instead.")
        LLM_MODEL_TYPE = 'gemini'

    print(f"Info: Converting content to multi-assay dict using model type: {LLM_MODEL_TYPE}")

    prompt = build_content_to_multi_assay_dict_prompt(content, assay_names, compound_id_list)
    retry_delays = get_retry_delays(TEXT_MODEL_RUNTIME, 'content_to_multi_assay_dict', channel='text')
    json_system_prompt = get_system_prompt(
        TEXT_MODEL_RUNTIME,
        'text',
        'json_extraction',
        "You are a helpful assistant designed to output JSON.",
    )
    temperature = get_task_temperature(TEXT_MODEL_RUNTIME, 'content_to_multi_assay_dict', channel='text', default=0.0)

    if LLM_MODEL_TYPE == 'gemini':
        configure_genai(GEMINI_API_KEY_FOR_GEMINI_MODELS)
        model = require_genai().GenerativeModel(DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT)
        response_text_for_error = "N/A"

        for attempt in range(retry):
            try:
                response = model.generate_content(prompt)
                if not response.candidates or not response.candidates[0].content.parts:
                    response_text_for_error = str(response)
                    raise ValueError(f"Model '{DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT}' returned no content/candidates.")

                result_text = response.candidates[0].content.parts[0].text
                response_text_for_error = result_text
                result_text = result_text.replace('null', 'None')
                json_content = extract_json_content(result_text) or result_text.strip()

                if not json_content:
                    raise ValueError("Could not extract JSON content from the model's response.")

                assay_dict = json.loads(json_content)
                return assay_dict
            except json.JSONDecodeError as json_e:
                print(f"Attempt {attempt + 1}/{retry} (JSONDecodeError): {json_e} in model '{DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT}'")
                print(f"Problematic JSON content: {json_content[:500] if json_content else 'None'}{'...' if json_content and len(json_content) > 500 else ''}")
                if attempt < retry - 1:
                    sleep_before_retry(attempt, retry, retry_delays)
                    continue
                print(f"Final attempt failed. Prompt:\n{prompt[:500]}...\nResponse:\n{response_text_for_error}")
                raise
            except Exception as e:
                print(f"Attempt {attempt + 1}/{retry} (Exception): {e} in model '{DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT}'")
                if attempt < retry - 1:
                    sleep_before_retry(attempt, retry, retry_delays)
                    continue
                print(f"Final attempt failed. Prompt:\n{prompt[:500]}...\nResponse:\n{response_text_for_error}")
                raise e
    elif LLM_MODEL_TYPE == 'openai':
        client = require_openai()(api_key=LLM_TEXT_MODEL_KEY, base_url=LLM_TEXT_MODEL_URL, timeout=LLM_MODEL_TIMEOUT_SECONDS)
        response_text_for_error = "N/A"
        for attempt in range(retry):
            try:
                print(f"Info: Calling LLM '{LLM_TEXT_MODEL_NAME}' at '{LLM_TEXT_MODEL_URL}' for content_to_multi_assay_dict.")
                response = client.chat.completions.create(
                    model=LLM_TEXT_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": json_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
                response_text_for_error = sanitize_model_response_text(response.choices[0].message.content or '')
                response_text_for_error = response_text_for_error.replace('null', 'None')
                json_content = extract_json_content(response_text_for_error)
                if not json_content:
                    raise ValueError("Could not extract JSON content from the model's response.")
                assay_dict = json.loads(json_content)
                return assay_dict
            except json.JSONDecodeError as json_e:
                print(f"Attempt {attempt + 1}/{retry} (JSONDecodeError): {json_e} in model '{LLM_TEXT_MODEL_URL}'")
                print(f"Problematic JSON content: {json_content[:500] if json_content else 'None'}{'...' if json_content and len(json_content) > 500 else ''}")
                if attempt < retry - 1:
                    sleep_before_retry(attempt, retry, retry_delays)
                    continue
                print(f"Final attempt failed. Prompt:\n{prompt[:500]}...\nResponse:\n{response_text_for_error}")
                raise
            except Exception as e:
                print(f"Attempt {attempt + 1}/{retry} (Exception): {e} in model '{LLM_TEXT_MODEL_URL}'")
                if attempt < retry - 1:
                    sleep_before_retry(attempt, retry, retry_delays)
                    continue
                print(f"Final attempt failed. Prompt:\n{prompt[:500]}...\nResponse:\n{response_text_for_error}")
                raise e
    else:
        print(f"Error: Unsupported LLM model type '{LLM_MODEL_TYPE}' for content_to_multi_assay_dict.")
    return None


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
            print(f"Warning: Unknown MIME type for {image_path}, using default {mime_type}.")
        return f"data:{mime_type};base64,{encoded_image_text}"
    except FileNotFoundError: print(f"Error: Image file not found at {image_path}"); raise
    except Exception as e: print(f"Error encoding image {image_path}: {e}"); raise


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
            print(f"Warning: {last_exc}")
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
            print(f"Warning: {last_exc}")
        elif exc[0] is not None:
            last_exc = exc[0]
            print(f"Warning: Visual model error on attempt {attempt}/{retries}: {last_exc}")
        else:
            return result[0]

        # Back-off before next retry
        if attempt < retries:
            wait = min(5 * attempt, 15)
            print(f"Retrying visual model call in {wait}s ...")
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

    print(f"Info: Using visual model type: {VISUAL_MODEL_TYPE}")

    if VISUAL_MODEL_TYPE == 'gemini':
        if not GEMINI_API_KEY_FOR_GEMINI_MODELS:
            raise ValueError("GEMINI_API_KEY not configured in constants.py for Gemini visual model.")
        if not actual_model_name:
            raise ValueError("VISUAL_MODEL_NAME is required for Gemini visual model.")

        configure_genai(GEMINI_API_KEY_FOR_GEMINI_MODELS)
        model = require_genai().GenerativeModel(actual_model_name)
        print(f"Info: Using Gemini visual model: {actual_model_name}")
        try:
            if PIL is None:
                 raise ImportError("Pillow (PIL) library is required for Gemini image processing but not found.")
            img = PIL.Image.open(image_file)
            mime_type = PIL.Image.MIME.get(img.format.upper())
            if not mime_type:
                 raise ValueError(f"Unsupported image format '{img.format}' for Gemini. Supported: PNG, JPEG, WEBP, HEIC, HEIF.")

            with open(image_file, 'rb') as f_bytes:
                image_bytes = f_bytes.read()
            image_part = {"mime_type": mime_type, "data": image_bytes}

            print(f"Info: Sending prompt and image ({mime_type}) to Gemini model '{actual_model_name}'.")
            response = model.generate_content([prompt, image_part], request_options={'timeout': VISION_MODEL_TIMEOUT_SECONDS})

            if not response.candidates or not response.candidates[0].content.parts:
                 response_text = f"Error: Gemini model '{actual_model_name}' returned no content or candidates."
                 print(f"Warning: {response_text}. Full response: {response}")
            else:
                response_text = response.text
        except Exception as e:
            print(f"Error with Gemini visual model '{actual_model_name}': {e}")
            raise

    elif VISUAL_MODEL_TYPE == 'openai':
        if not VISUAL_MODEL_KEY:
            raise ValueError("VISUAL_MODEL_KEY (for OpenAI API key) is not configured in constants.py.")
        if not actual_model_name:
            raise ValueError("VISUAL_MODEL_NAME is required for OpenAI visual model.")

        print(f"Info: Using OpenAI visual model: {actual_model_name}")
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
            print(f"Info: Sending prompt and image to OpenAI model '{actual_model_name}'.")
            completion = client.chat.completions.create(model=actual_model_name, messages=messages, temperature=temperature)
            response_text = completion.choices[0].message.content
        except Exception as e:
            print(f"Error with OpenAI visual model '{actual_model_name}': {e}")
            raise

    return sanitize_model_response_text(response_text)


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
        print(f"Warning: Failed to analyze border contact for {image_file}: {e}")
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

    try:
        response_text = call_visual_model(image_file, prompt)
    except Exception as e:
        print(f"Warning: classify_structure_candidate failed after retries for {image_file}: {e}")
        return {
            'structure_type': 'uncertain',
            'is_complete_compound': False,
            'reason': f'Visual model error: {e}',
            'raw_response': '',
            'border_contact': {'suspicious': False, 'sides': [], 'ratios': {}},
            'strictness': strictness,
        }
    json_content = extract_json_content(response_text)
    payload = {}
    if json_content:
        try:
            payload = json.loads(json_content)
        except Exception as e:
            print(f"Warning: Failed to parse classification JSON: {e}; raw response: {response_text}")
    schema = VISION_MODEL_OUTPUT_SCHEMAS.get('classify_structure_candidate', {})
    if payload and not validate_required_keys(payload, schema):
        print(f"Warning: Classification payload missing required keys; raw response: {response_text}")

    structure_type = normalize_structure_type(
        payload.get('structure_type') or payload.get('type') or payload.get('label') or response_text
    )
    is_complete_compound = coerce_bool(payload.get('is_complete_compound'))
    if is_complete_compound is None:
        is_complete_compound = structure_type == 'complete_compound'

    if is_complete_compound:
        structure_type = 'complete_compound'
    elif structure_type == 'complete_compound':
        structure_type = 'uncertain'

    border_contact = analyze_border_contact(image_file)
    should_run_crop_check = structure_type == 'complete_compound' and border_contact.get('suspicious')
    if strictness == 'permissive':
        should_run_crop_check = False

    crop_check_confirmed_not_cropped = False
    if should_run_crop_check:
        crop_check_prompt = build_crop_check_prompt(', '.join(border_contact.get('sides', [])) or 'none', strictness=strictness)
        try:
            crop_check_response_text = call_visual_model(image_file, crop_check_prompt)
        except Exception as e:
            print(f"Warning: crop-check visual model failed for {image_file}: {e}")
            crop_check_response_text = ''
        crop_check_json_content = extract_json_content(crop_check_response_text)
        if crop_check_json_content:
            try:
                crop_check_payload = json.loads(crop_check_json_content)
                crop_status = str(crop_check_payload.get('crop_status', '')).strip().lower()
                is_cropped = coerce_bool(crop_check_payload.get('is_cropped'))
                if is_cropped is None:
                    is_cropped = crop_status == 'fragment'
                if is_cropped or crop_status == 'fragment':
                    structure_type = 'fragment'
                    is_complete_compound = False
                    response_text = crop_check_response_text
                    payload = {'reason': crop_check_payload.get('reason', crop_check_response_text)}
                elif crop_status == 'not_cropped':
                    crop_check_confirmed_not_cropped = True
            except Exception as e:
                print(f"Warning: Failed to parse crop-check JSON: {e}; raw response: {crop_check_response_text}")

    should_run_border_review = structure_type == 'complete_compound' and border_contact.get('suspicious')
    if strictness == 'permissive':
        should_run_border_review = False

    border_review_confirmed_complete = False
    if should_run_border_review:
        review_prompt = build_border_review_prompt(', '.join(border_contact.get('sides', [])) or 'none', strictness=strictness)
        try:
            review_response_text = call_visual_model(image_file, review_prompt)
        except Exception as e:
            print(f"Warning: border-review visual model failed for {image_file}: {e}")
            review_response_text = ''
        review_json_content = extract_json_content(review_response_text)
        if review_json_content:
            try:
                review_payload = json.loads(review_json_content)
                review_type = normalize_structure_type(
                    review_payload.get('structure_type') or review_payload.get('type') or review_payload.get('label')
                )
                review_complete = coerce_bool(review_payload.get('is_complete_compound'))
                if review_complete is None:
                    review_complete = review_type == 'complete_compound'
                if review_complete:
                    review_type = 'complete_compound'
                elif review_type == 'complete_compound':
                    review_type = 'uncertain'

                if review_type != 'complete_compound':
                    structure_type = review_type
                    is_complete_compound = bool(review_complete)
                    response_text = review_response_text
                    payload = review_payload
                else:
                    border_review_confirmed_complete = True
            except Exception as e:
                print(f"Warning: Failed to parse border review JSON: {e}; raw response: {review_response_text}")

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

    reason = payload.get('reason') if isinstance(payload.get('reason'), str) else response_text
    return {
        'structure_type': structure_type,
        'is_complete_compound': bool(is_complete_compound),
        'reason': (reason or '').strip(),
        'raw_response': response_text or '',
        'border_contact': border_contact,
        'strictness': strictness,
    }


@proxy_decorator
# @cost_time
def structure_to_id(image_file, prompt=None):
    """
    Extracts the compound ID from a chemical structure image using the visual model
    specified by VISUAL_MODEL_TYPE and its associated VISUAL_MODEL_* constants.
    """

    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image file for structure_to_id not found: {image_file}")

    if prompt is None:
        prompt = build_structure_to_id_prompt()
    try:
        response_text = call_visual_model(image_file, prompt)
    except Exception as e:
        print(f"Warning: structure_to_id failed after retries for {image_file}: {e}")
        return ''
    print(f'Info: Received response from visual model: {response_text}')
    return response_text

@proxy_decorator
def get_compound_id_from_description(description):
    """
    Extracts a compound ID from a description string using an OpenAI-compatible text model.
    """

    prompt = build_description_to_id_prompt(description)
    retry = 3
    retry_delays = get_retry_delays(TEXT_MODEL_RUNTIME, 'get_compound_id_from_description', channel='text')
    json_system_prompt = get_system_prompt(
        TEXT_MODEL_RUNTIME,
        'text',
        'json_extraction',
        "You are a helpful assistant designed to output JSON.",
    )
    temperature = get_task_temperature(TEXT_MODEL_RUNTIME, 'get_compound_id_from_description', channel='text', default=0.0)

    content = ''
    for attempt in range(retry):
        try:
            if LLM_MODEL_TYPE == 'gemini':
                if not GEMINI_API_KEY_FOR_GEMINI_MODELS:
                    raise ValueError("GEMINI_API_KEY not configured in constants.py for Gemini text model.")
                configure_genai(GEMINI_API_KEY_FOR_GEMINI_MODELS)
                model = require_genai().GenerativeModel(DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT)
                print(f"Info: Calling Gemini model '{DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT}' for description to ID.")
                response = model.generate_content(prompt)
                content = response.candidates[0].content.parts[0].text
            else:
                client = require_openai()(api_key=LLM_TEXT_MODEL_KEY, base_url=LLM_TEXT_MODEL_URL, timeout=LLM_MODEL_TIMEOUT_SECONDS)
                print(f"Info: Calling LLM '{LLM_TEXT_MODEL_NAME}' at '{LLM_TEXT_MODEL_URL}' for description to ID.")
                response = client.chat.completions.create(
                    model=LLM_TEXT_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": json_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
                content = sanitize_model_response_text(response.choices[0].message.content or '')
            break
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retry} failed for get_compound_id_from_description: {e}")
            if attempt < retry - 1:
                sleep_before_retry(attempt, retry, retry_delays)
                continue
            print(f"Error calling LLM for get_compound_id_from_description: {e}")
            return f"Error: Could not get ID due to API error - {e}"

    json_str = extract_json_content(content) or content.strip()

    if not json_str:
        print(f"Warning: Could not extract JSON string (get_compound_id_from_description). Raw: '{content}'")
        return content

    try:
        data = json.loads(json_str)
        schema = TEXT_MODEL_OUTPUT_SCHEMAS.get('get_compound_id_from_description', {})
        if not validate_required_keys(data, schema):
            print(f"Warning: Description-to-ID payload missing required keys. Raw: '{content}'")
        return data.get("COMPOUND_ID", content)
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse JSON (get_compound_id_from_description). JSON string: '{json_str}'. Raw: '{content}'")
        return content


@proxy_decorator
def resolve_compound_id_alias(raw_id, compound_id_list, context=''):
    """
    Resolve a raw/aliased compound ID to one canonical ID from the provided allowlist.
    Returns None when resolution is ambiguous or unsupported.
    """
    prompt = build_resolve_compound_id_alias_prompt(raw_id, compound_id_list, context=context)
    retry = 3
    retry_delays = get_retry_delays(TEXT_MODEL_RUNTIME, 'resolve_compound_id_alias', channel='text')
    json_system_prompt = get_system_prompt(
        TEXT_MODEL_RUNTIME,
        'text',
        'json_extraction',
        "You are a helpful assistant designed to output JSON.",
    )
    temperature = get_task_temperature(TEXT_MODEL_RUNTIME, 'resolve_compound_id_alias', channel='text', default=0.0)

    content = ''
    for attempt in range(retry):
        try:
            if LLM_MODEL_TYPE == 'gemini':
                if not GEMINI_API_KEY_FOR_GEMINI_MODELS:
                    raise ValueError("GEMINI_API_KEY not configured in constants.py for Gemini text model.")
                configure_genai(GEMINI_API_KEY_FOR_GEMINI_MODELS)
                model = require_genai().GenerativeModel(DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT)
                print(f"Info: Calling Gemini model '{DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT}' for compound ID alias resolution.")
                response = model.generate_content(prompt)
                content = response.candidates[0].content.parts[0].text
            else:
                client = require_openai()(api_key=LLM_TEXT_MODEL_KEY, base_url=LLM_TEXT_MODEL_URL, timeout=LLM_MODEL_TIMEOUT_SECONDS)
                print(f"Info: Calling LLM '{LLM_TEXT_MODEL_NAME}' at '{LLM_TEXT_MODEL_URL}' for compound ID alias resolution.")
                response = client.chat.completions.create(
                    model=LLM_TEXT_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": json_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
                content = sanitize_model_response_text(response.choices[0].message.content or '')
            break
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retry} failed for resolve_compound_id_alias: {e}")
            if attempt < retry - 1:
                sleep_before_retry(attempt, retry, retry_delays)
                continue
            print(f"Error calling LLM for resolve_compound_id_alias: {e}")
            return None

    json_str = extract_json_content(content) or content.strip()
    if not json_str:
        print(f"Warning: Could not extract JSON string (resolve_compound_id_alias). Raw: '{content}'")
        return None

    try:
        data = json.loads(json_str)
        schema = TEXT_MODEL_OUTPUT_SCHEMAS.get('resolve_compound_id_alias', {})
        if not validate_required_keys(data, schema):
            print(f"Warning: Alias-resolver payload missing required keys. Raw: '{content}'")
            return None
        resolved = data.get("COMPOUND_ID")
        if resolved is None:
            return None
        resolved_text = str(resolved).strip()
        if not resolved_text or resolved_text.lower() == 'none':
            return None
        return resolved_text
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse JSON (resolve_compound_id_alias). JSON string: '{json_str}'. Raw: '{content}'")
        return None
