import os
import warnings
warnings.filterwarnings("ignore")

import time
import requests
from functools import wraps
import json
import base64
import sys

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google.generativeai package not found. Please install it: pip install google-generativeai")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not found. Please install it: pip install openai")
    sys.exit(1)

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

HTTP_PROXY = getattr(constants, 'HTTP_PROXY', '')
HTTPS_PROXY = getattr(constants, 'HTTPS_PROXY', '')

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
def configure_genai(api_key):
    """
    Configures the Google Generative AI client.
    Ensures configuration happens only once or if the key changes.
    """
    global _genai_configured_with_key
    if _genai_configured_with_key == api_key and _genai_configured_with_key is not None:
        return

    if api_key:
        genai.configure(api_key=api_key)
        _genai_configured_with_key = api_key
    elif os.getenv('GOOGLE_API_KEY'):
        try:
            genai.configure()
            _genai_configured_with_key = os.getenv('GOOGLE_API_KEY')
            print("Info: Configured GenAI using GOOGLE_API_KEY environment variable.")
        except Exception as e:
            print(f"Warning: Failed to configure GenAI with GOOGLE_API_KEY env var: {e}")
    else:
        print("Warning: GEMINI_API_KEY not provided for configure_genai and GOOGLE_API_KEY env var not set or failed to configure.")


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

    if compound_id_list is None:
        compound_id_list_str = '开始提取数据...\n\n'
    else:
        compound_id_list_str = 'compound_id_list如下，解析时请不要超出此列表范围：\n'
        compounds = ', '.join([f'"{cid}"' for cid in compound_id_list])
        compound_id_list_str += f"化合物ID列表: {compounds}\n\n"
        compound_id_list_str += '\n开始提取数据...\n\n'

    prompt = f'''从提供的Markdown文本中，提取以下化合物ID及其对应的"{assay_name}"测定值。

<MARKDOWN_TEXT>
{content}
</MARKDOWN_TEXT>

你的任务是将提取的数据转换为字典格式，其中键__COMPOUND_ID__是提供给你的化合物列表中的化合物ID，值__ASSAY_VALUE__从Markdown中提取的对应实验值。
1. 如果表格只有两列，第一列通常是化合物编号，第二列是活性。如果表格有多列，你需要根据表头中的化合物编号和活性列来提取数据，但通常奇数列是化合物编号，偶数列是活性。
2. 有时，提供的化合物ID和Markdown中可能略有不同，例如"Example 1"可能在Markdown中为"1"，或者"Compound 1"可能在Markdown中为"1"。在这种情况下，你需要自动判断并确认它们是否为同一个化合物。并将"__COMPOUND_ID__"记录为提供的化合物ID，例如"Example 1"或"Compound 1"，而不是Markdown中的"1"。
3. 请按以下格式输出转换后的字典：
```json
{{
    "__COMPOUND_ID__": "__ASSAY_VALUE__",
    "__COMPOUND_ID__": "__ASSAY_VALUE__",
    ...
}}
```
{compound_id_list_str}'''


    if LLM_MODEL_TYPE == 'gemini':
        configure_genai(GEMINI_API_KEY_FOR_GEMINI_MODELS)
        model = genai.GenerativeModel(DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT)
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

                json_content = None
                if '```json' in result_text:
                    json_match = result_text.split('```json', 1)
                    if len(json_match) > 1:
                        json_content = json_match[1].split('```', 1)[0].strip()
                elif '```' in result_text and result_text.count('```') >= 2 and json_content is None:
                    parts = result_text.split('```', 2)
                    if len(parts) >= 2: json_content = parts[1].strip()
                elif '“json' in result_text and json_content is None: # Check for new format
                    json_match = result_text.split('“json', 1)
                    if len(json_match) > 1:
                        temp_content = json_match[1].strip()
                        if temp_content.endswith('”'): temp_content = temp_content[:-1].strip()
                        json_content = temp_content
                
                if json_content is None:
                    start_brace = result_text.find('{')
                    end_brace = result_text.rfind('}')
                    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                        json_content = result_text[start_brace : end_brace+1].strip()
                    else:
                        json_content = result_text.strip()

                if not json_content:
                    raise ValueError("Could not extract JSON content from the model's response.")

                assay_dict = json.loads(json_content)
                return assay_dict
            except json.JSONDecodeError as json_e:
                print(f"Attempt {attempt + 1}/{retry} (JSONDecodeError): {json_e} in model '{DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT}'")
                print(f"Problematic JSON content: {json_content[:500] if json_content else 'None'}{'...' if json_content and len(json_content) > 500 else ''}")
                if attempt < retry - 1: time.sleep(1 + attempt); continue
                print(f"Final attempt failed. Prompt:\n{prompt[:500]}...\nResponse:\n{response_text_for_error}"); raise
            except Exception as e:
                print(f"Attempt {attempt + 1}/{retry} (Exception): {e} in model '{DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT}'")
                if attempt < retry - 1: time.sleep(1 + attempt); continue
                print(f"Final attempt failed. Prompt:\n{prompt[:500]}...\nResponse:\n{response_text_for_error}"); raise e
    elif LLM_MODEL_TYPE == 'openai':
        client = OpenAI(api_key=LLM_TEXT_MODEL_KEY, base_url=LLM_TEXT_MODEL_URL)
        response_text_for_error = "N/A"
        for attempt in range(retry):
            try:
                print(f"Info: Calling LLM '{LLM_TEXT_MODEL_NAME}' at '{LLM_TEXT_MODEL_URL}' for content_to_dict.")
                response = client.chat.completions.create(
                    model=LLM_TEXT_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                response_text_for_error = response.choices[0].message.content
                response_text_for_error = response_text_for_error.replace('null', 'None')
                # 去除<think>和</think>之间的所有内容
                if '<think>' in response_text_for_error and '</think>' in response_text_for_error:
                    think_start = response_text_for_error.index('<think>') + len('<think>')
                    think_end = response_text_for_error.index('</think>')
                    response_text_for_error = response_text_for_error[:think_start] + response_text_for_error[think_end + len('</think>'):]

                json_content = None
                if '```json' in response_text_for_error:
                    json_match = response_text_for_error.split('```json', 1)
                    if len(json_match) > 1:
                        json_content = json_match[1].split('```', 1)[0].strip()
                elif '```' in response_text_for_error and response_text_for_error.count('```') >= 2 and json_content is None:
                    parts = response_text_for_error.split('```', 2)
                    if len(parts) >= 2: json_content = parts[1].strip()
                elif '“json' in response_text_for_error and json_content is None: # Check for new format
                    json_match = response_text_for_error.split('“json', 1)
                    if len(json_match) > 1:
                        temp_content = json_match[1].strip()
                        if temp_content.endswith('”'): temp_content = temp_content[:-1].strip()
                        json_content = temp_content
                
                if json_content is None:
                    start_brace = response_text_for_error.find('{')
                    end_brace = response
                    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                        json_content = response_text_for_error[start_brace : end_brace+1].strip()
                else:
                    json_content = json_content.strip()
                if not json_content:
                    raise ValueError("Could not extract JSON content from the model's response.")
                assay_dict = json.loads(json_content)
                return assay_dict
            except json.JSONDecodeError as json_e:
                print(f"Attempt {attempt + 1}/{retry} (JSONDecodeError): {json_e} in model '{LLM_TEXT_MODEL_URL}'")
                print(f"Problematic JSON content: {json_content[:500] if json_content else 'None'}{'...' if json_content and len(json_content) > 500 else ''}")
                if attempt < retry - 1: time.sleep(1 + attempt); continue
                print(f"Final attempt failed. Prompt:\n{prompt[:500]}...\nResponse:\n{response_text_for_error}"); raise
            except Exception as e:
                print(f"Attempt {attempt + 1}/{retry} (Exception): {e} in model '{LLM_TEXT_MODEL_URL}'")
                if attempt < retry - 1: time.sleep(1 + attempt); continue
                print(f"Final attempt failed. Prompt:\n{prompt[:500]}...\nResponse:\n{response_text_for_error}"); raise e
    else:
        print(f"Error: Unsupported LLM model type '{LLM_MODEL_TYPE}' for content_to_dict.")
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
            print(f"Warning: Unknown MIME type for {image_path}, using fallback {mime_type}.")
        return f"data:{mime_type};base64,{encoded_image_text}"
    except FileNotFoundError: print(f"Error: Image file not found at {image_path}"); raise
    except Exception as e: print(f"Error encoding image {image_path}: {e}"); raise


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
        # prompt = "What is the ID of the compound inside the red dashed box? If not found, please answer 'None'."
        prompt = '''Return the ID for the red boxed structure.
Accept: Example/Compound/Embodiment/Intermediate/Formula/实施例/化合物, or standalone numeric IDs (12, (12), No.12, 编号12, IIa, I).
Reject page/line numbers, Figure/Table/Scheme, and values with units (mg, mL, MHz, ppm, m/z, δ, %).
Output the ID text only; else None.'''
#         prompt = '''任务：在整页中找到与红框“目标结构”对应的化合物编号。
# 只输出编号原文；找不到输出“None”。

# 规则：
# - 排除：图(figure)/表(table)/方案(scheme)等编号
# - 多个候选→选与目标结构直接指向/连线/最近邻者；不唯一→“None”'''
#         prompt = '''任务：在整页中找到与红框“目标结构”对应的化合物编号。你需要根据上下文和空间关系，仔细判别化合编号的位置，不要误判。
# 只输出编号原文；找不到输出“None”。'''

    response_text = None
    actual_model_name = VISUAL_MODEL_NAME

    print(f"Info: Using visual model type: {VISUAL_MODEL_TYPE}")

    if VISUAL_MODEL_TYPE == 'gemini':
        if not GEMINI_API_KEY_FOR_GEMINI_MODELS:
            raise ValueError("GEMINI_API_KEY not configured in constants.py for Gemini visual model.")
        if not actual_model_name:
            # actual_model_name = 'gemini-2.0-flash'
            actual_model_name = GEMINI_MODEL_NAME
            print(f"Info: VISUAL_MODEL_NAME for Gemini not set in constants.py, defaulting to '{actual_model_name}'.")

        configure_genai(GEMINI_API_KEY_FOR_GEMINI_MODELS)
        model = genai.GenerativeModel(actual_model_name)
        print(f"Info: Using Gemini visual model: {actual_model_name}")
        try:
            if PIL is None:
                 raise ImportError("Pillow (PIL) library is required for Gemini image processing but not found.")
            img = PIL.Image.open(image_file)
            mime_type = PIL.Image.MIME.get(img.format.upper())
            if not mime_type:
                 raise ValueError(f"Unsupported image format '{img.format}' for Gemini. Supported: PNG, JPEG, WEBP, HEIC, HEIF.")

            with open(image_file, 'rb') as f_bytes: image_bytes = f_bytes.read()
            image_part = {"mime_type": mime_type, "data": image_bytes}

            print(f"Info: Sending prompt and image ({mime_type}) to Gemini model '{actual_model_name}'.")
            response = model.generate_content([prompt, image_part])

            if not response.candidates or not response.candidates[0].content.parts:
                 response_text = f"Error: Gemini model '{actual_model_name}' returned no content or candidates."
                 print(f"Warning: {response_text}. Full response: {response}")
            else:
                response_text = response.text
        except Exception as e: print(f"Error with Gemini visual model '{actual_model_name}': {e}"); raise

    elif VISUAL_MODEL_TYPE == 'openai':
        if not VISUAL_MODEL_KEY:
            raise ValueError("VISUAL_MODEL_KEY (for OpenAI API key) is not configured in constants.py.")
        if not actual_model_name:
            actual_model_name = 'gpt-4o'
            print(f"Info: VISUAL_MODEL_NAME for OpenAI not set in constants.py, defaulting to '{actual_model_name}'.")

        print(f"Info: Using OpenAI visual model: {actual_model_name}")
        try:
            client = OpenAI(api_key=VISUAL_MODEL_KEY, base_url=VISUAL_MODEL_URL)
            image_base64_uri = encode_image_to_base64_data_uri(image_file)
            messages = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_base64_uri}},
            ]}]
            print(f"Info: Sending prompt and image to OpenAI model '{actual_model_name}'.")
            completion = client.chat.completions.create(model=actual_model_name, messages=messages, max_tokens=4096, temperature=0.2)
            response_text = completion.choices[0].message.content
            # 去掉前后的\n
            if response_text.startswith('\n'):
                response_text = response_text[1:]
            if response_text.endswith('\n'):
                response_text = response_text[:-1]

            # 去掉<|begin_of_box|>1<|end_of_box|>
            # if response_text.startswith('<|begin_of_box|>') and response_text.endswith('<|end_of_box|>'):
            if '<|begin_of_box|>' in response_text and '<|end_of_box|>' in response_text:
                # response_text = response_text[len('<|begin_of_box|>'):-len('<|end_of_box|>')].strip()
                response_text = response_text.split('<|begin_of_box|>', 1)[-1].split('<|end_of_box|>', 1)[0].strip()

            print(f'Info: Received response from {actual_model_name} model: {response_text}')
        except Exception as e: print(f"Error with OpenAI visual model '{actual_model_name}': {e}"); raise

    return response_text


@proxy_decorator
def get_compound_id_from_description(description):
    """
    Extracts a compound ID from a description string using an OpenAI-compatible text model.
    """

    prompt = f"""任务：从下方文本中抽取该化合物编号。
输入：
{description}

输出要求：
- 仅输出一行合法 JSON，键固定为 COMPOUND_ID。
- 若无法确定或不存在，返回 "None"。
- 禁止输出占位符、空值、解释性文字、代码块或多余字符；**绝不能输出 "__ID__"**。

请自检：若提取结果为空、为占位符、或含“不确定/未知”等词，则改为 "None"。

```json
{{"COMPOUND_ID": "__ID__"}}
```
"""

    try:
        if LLM_MODEL_TYPE == 'gemini':
            if not GEMINI_API_KEY_FOR_GEMINI_MODELS:
                raise ValueError("GEMINI_API_KEY not configured in constants.py for Gemini text model.")
            configure_genai(GEMINI_API_KEY_FOR_GEMINI_MODELS)
            model = genai.GenerativeModel(DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT)
            print(f"Info: Calling Gemini model '{DEFAULT_GEMINI_TEXT_MODEL_FOR_CONTENT_DICT}' for description to ID.")
            response = model.generate_content(prompt)
            content = response.candidates[0].content.parts[0].text
        else:
            client = OpenAI(api_key=LLM_TEXT_MODEL_KEY, base_url=LLM_TEXT_MODEL_URL)
            print(f"Info: Calling LLM '{LLM_TEXT_MODEL_NAME}' at '{LLM_TEXT_MODEL_URL}' for description to ID.")
            response = client.chat.completions.create(
                model=LLM_TEXT_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            content = response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM for get_compound_id_from_description: {e}")
        return f"Error: Could not get ID due to API error - {e}"

    json_str = None
    if "```json" in content:
        json_str = content.split("```json", 1)[-1].split("```", 1)[0].strip()
    elif "“json" in content and json_str is None:
        temp_content = content.split("“json", 1)[-1].strip()
        if temp_content.endswith("”"): temp_content = temp_content[:-1].strip()
        json_str = temp_content
    elif json_str is None:
        start_brace = content.find('{'); end_brace = content.rfind('}')
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            json_str = content[start_brace : end_brace+1].strip()
        else: json_str = content.strip()

    if not json_str:
        print(f"Warning: Could not extract JSON string (get_compound_id_from_description). Raw: '{content}'")
        return content

    try:
        data = json.loads(json_str)
        return data.get("COMPOUND_ID", content)
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse JSON (get_compound_id_from_description). JSON string: '{json_str}'. Raw: '{content}'")
        return content

if __name__ == '__main__':
    content = '''|   Example | TR-FRET EC5o (M)   |   Example.1 | TR-FRET EC5o (M).1   |   Examp le | TR-FRET EC5o (M).2   |   Example.2 | TR-FRET EC5o (M).3   |

|----------:|:-------------------|------------:|:---------------------|-----------:|:---------------------|------------:|:---------------------|

|         1 | ****               |          20 | ****                 |         39 | ***                  |          58 | ****                 |

|         2 | ****               |          21 | ****                 |         40 | ****                 |          59 | ***                  |

|         3 | ****               |          22 | ***                  |         41 | ****                 |          60 | ****                 |

|         4 | ****               |          23 | ****                 |         42 | ****                 |          61 | ***                  |

|         5 | ***                |          24 | ****                 |         43 | ***                  |          62 | ***                  |

|         6 | ****               |          25 | ****                 |         44 | ****                 |          63 | ****                 |

|         7 | ****               |          26 | **                   |         45 | ****                 |          64 | ****                 |

|         8 | >10 M              |          27 | ****                 |         46 | ****                 |          65 | ****                 |

|         9 | >10 M              |          28 | **                   |         47 | ****                 |          66 | ****                 |

|        10 | >10 M              |          29 | ***                  |         48 | ****                 |          67 | ***                  |

|        11 | ****               |          30 | ****                 |         49 | ****                 |          68 | **                   |

|        12 | ****               |          31 | ****                 |         50 | ****                 |          69 | ****                 |

|        13 | ****               |          32 | ***                  |         51 | ***                  |          70 | ****                 |

|        14 | *                  |          33 | ****                 |         52 | **                   |          71 | ****                 |

|        15 | ***                |          34 | ****                 |         53 | ****                 |          72 | ****                 |

|        16 | *                  |          35 | ****                 |         54 | ****                 |          73 | ***                  |

|        17 | *                  |          36 | ****                 |         55 | ****                 |          74 | ****                 |

|        18 | ****               |          37 | ***                  |         56 | ***                  |          75 | ****                 |

|        19 | ****               |          38 | ****                 |         57 | ****                 |          76 | ****                 |

|        77 | ****               |         105 | ****                 |        133 | ****                 |         161 | ***                  |

|        78 | ****               |         106 | ***                  |        134 | **                   |         162 | ****                 |

|        79 | ****               |         107 | ****                 |        135 | ****                 |         163 | ****                 |

|        80 | ****               |         108 | ***                  |        136 | ****                 |         164 | ****                 |

|        81 | ****               |         109 | ****                 |        137 | ****                 |         165 | ****                 |

|        82 | ****               |         110 | ****                 |        138 | ****                 |         166 | ****                 |

|        83 | ****               |         111 | ****                 |        139 | ****                 |         167 | ****                 |

|        84 | ****               |         112 | ****                 |        140 | ****                 |         168 | ***                  |

|        85 | ****               |         113 | ****                 |        141 | **                   |         169 | ****                 |


|   Example | TR-FRET EC5o (M)   |   Example.1 | TR-FRET EC5o (M).1   |   Examp le | TR-FRET EC5o (M).2   | Example.2   | TR-FRET EC5o (M).3   |

|----------:|:-------------------|------------:|:---------------------|-----------:|:---------------------|:------------|:---------------------|

|        86 | ****               |         114 | ****                 |        142 | ****                 | 170.0       | ****                 |

|        87 | ****               |         115 | ****                 |        143 | ****                 | 171.0       | ****                 |

|        88 | **                 |         116 | ***                  |        144 | ****                 | 172.0       | ****                 |

|        89 | ****               |         117 | ****                 |        145 | **                   | 173.0       | ****                 |

|        90 | ****               |         118 | ****                 |        146 | **                   | 174.0       | ****                 |

|        91 | ****               |         119 | ****                 |        147 | ****                 | 175.0       | ****                 |

|        92 | ****               |         120 | ****                 |        148 | ***                  | 176.0       | ****                 |

|        93 | ****               |         121 | ****                 |        149 | ****                 | 177.0       | ****                 |

|        94 | ****               |         122 | ****                 |        150 | ****                 | 178.0       | ***                  |

|        95 | ****               |         123 | ****                 |        151 | ****                 | 179.0       | ****                 |

|        96 | ****               |         124 | ***                  |        152 | ****                 | 180.0       | ****                 |

|        97 | ****               |         125 | ****                 |        153 | ****                 | 181.0       | ****                 |

|        98 | ****               |         126 | ****                 |        154 | ****                 | 182.0       | ****                 |

|        99 | ****               |         127 | ****                 |        155 | ****                 | 183.0       | ****                 |

|       100 | ****               |         128 | ****                 |        156 | ****                 | 184.0       | ****                 |

|       101 | ****               |         129 | ****                 |        157 | *                    | 185.0       | ****                 |

|       102 | ****               |         130 | ****                 |        158 | ****                 | 186.0       | ****                 |

|       103 | ****               |         131 | ***                  |        159 | ****                 | 187.0       | ***                  |

|       104 | ****               |         132 | ****                 |        160 | **                   | 188.0       | ****                 |

|       189 | ****               |         217 | ****                 |        247 | ****                 |             |                      |

|       190 | ****               |         218 | ****                 |        248 | ****                 |             |                      |

|       191 | ****               |         219 | ****                 |        249 | ****                 |             |                      |

|       192 | ****               |         220 | ****                 |        250 | ***                  |             |                      |

|       193 | ****               |         221 | ****                 |        251 | ***                  |             |                      |

|       194 | ****               |         222 | ****                 |        252 | ****                 |             |                      |

|       195 | ****               |         223 | ****                 |        253 | ***                  |             |                      |

|       196 | ****               |         224 | ****                 |        254 | ***                  |             |                      |

|       197 | ****               |         225 | ****                 |        255 | *                    |             |                      |


| Example   | TR-FRET   | Example.1   | TR-FRET.1   | Examp    | TR-FRET.2   | Example.2   | TR-FRET.3   |

|:----------|:----------|:------------|:------------|:---------|:------------|:------------|:------------|

| Example   | ECso (M)  | Example     | EC5o (M)    | ECso (M) |             | Example     | EC5o (M)    |

| 198       | ****      | 226         | ****        | le 256   | ****        |             |             |

| 199       | ****      | 227         | ****        | 257      | ****        |             |             |

| 200       | ****      | 228         | ****        | 258      | ****        |             |             |

| 201       | **        | 229         | ****        |          |             |             |             |

| 202       | ****      | 230         | ****        | 260      | ****        |             |             |

| 203       | ***       | 231         | ****        | 261      | ****        |             |             |

| 204       | ****      | 232         | ****        | 262      | ****        |             |             |

| 205       | ***       | 233         | ****        | 263      | ****        |             |             |

| 206       | ****      | 234         | ****        | 264      | ****        |             |             |

| 207       | ***       | 235         | ****        | 265      | ****        |             |             |

| 208       | ****      | 236         | NT          | 266      | ****        |             |             |

| 209       | ****      | 237         | ****        | 267      | ****        |             |             |

| 210       | **        | 238         | ****        |          |             |             |             |

| 211       | ****      | 239         | ****        |          |             |             |             |

| 212       | **        | 240         | ****        |          |             |             |             |

| 213       | ***       | 241         | ****        |          |             |             |             |

| 214       | ****      | 242         | ****        |          |             |             |             |

| 215       | ****      | 243         | ****        |          |             |             |             |

| 216       | ****      | 244         | ****        |          |             |             |             |

| 245       | ***       | 246         | **          |          |             |             |             |
'''
    assay_name = 'TR-FRET EC50 (M)'
    compound_id_list = ['114', '246', '245']
    assay_dict = content_to_dict(content, assay_name, compound_id_list)
    print(assay_dict)