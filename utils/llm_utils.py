import os
import warnings
warnings.filterwarnings("ignore")

import time
from functools import wraps
import google.generativeai as genai
import json

import sys
sys.path.append('..')
from constants import VISUAL_MODEL_URL, VISUAL_MODEL_KEY, HTTP_PROXY, HTTPS_PROXY
from constants import SECONDARY_MODEL_NAME, SECONDARY_MODEL_KEY, SECONDARY_MODEL_URL

from openai import OpenAI

def proxy_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        os.environ['http_proxy'] = HTTP_PROXY
        os.environ['https_proxy'] = HTTPS_PROXY
        result = func(*args, **kwargs)
        os.environ['http_proxy'] = ''
        os.environ['https_proxy'] = ''
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

def configure_genai(api_key):
    """
    Configures the Google Generative AI client.
    """
    genai.configure(api_key=api_key)

@proxy_decorator
def content_to_dict(content, assay_name, compound_id_list=None, retry=3, model_name='gemini-1.5-flash'):
    """
    Converts the content of a Markdown text to a dictionary using Google Generative AI.
    """
    if compound_id_list is None:
        compound_id_list_str = '开始提取数据...\n\n'
    else:
        compound_id_list_str = 'compound_id_list如下，解析时请不要超出此列表范围：\n'
        compound_id_list_str += ', '.join(compound_id_list)
        compound_id_list_str += '\n开始提取数据...\n\n'

    prompt = f'''从提供的Markdown文本中，提取以下化合物ID及其对应的"{assay_name}"测定值。

<MARKDOWN_TEXT>
{content}
</MARKDOWN_TEXT>

你的任务是将提取的数据转换为字典格式，其中键__COMPOUND_ID__是提供给你的化合物列表中的化合物ID，值__ASSAY_VALUE__从Markdown中提取的对应实验值。
1. 如果表格只有两列，第一列通常是化合物编号，第二列是活性。如果表格有多列，你需要根据表头中的化合物编号和活性列来提取数据，但通常奇数列是化合物编号，偶数列是活性。
2. 有时，提供的化合物ID和Markdown中可能略有不同，例如"Example 1"可能在Markdown中为"1"，或者"Compound 1"可能在Markdown中为"1"。在这种情况下，你需要自动判断并确认它们是否为同一个化合物。并将"__COMPOUND_ID__"记录为提供的化合物ID，例如"Example 1"或"Compound 1"，而不是Markdown中的"1"。
3. 请按以下格式输出转换后的字典：
json
{{
    __COMPOUND_ID__: __ASSAY_VALUE__,
    __COMPOUND_ID__: __ASSAY_VALUE__,
    ...
}}
{compound_id_list_str}'''

    model = genai.GenerativeModel(model_name)

    for attempt in range(retry):
        try:
            response = model.generate_content(prompt)
            result = response.candidates[0].content.parts[0].text
            result = result.replace('null', 'None')
            # Extract JSON content
            json_content = result.split('```json')[-1].split('```')[0].strip()
            assay_dict = json.loads(json_content)
            return assay_dict
        except Exception as e:
            if attempt < retry - 1:
                continue
            else:
                raise e
    return None

def structure_to_id(image_file):
    """
    Extracts the compound ID from a highlighted chemical structure image.
    """

    import base64

    def encode_image_to_base64(image_path):
        with open(image_path, 'rb') as image_file:
            return str(base64.b64encode(image_file.read()).decode('utf-8'))
        
    image_base64 = encode_image_to_base64(image_file)

    client = OpenAI(
        api_key=VISUAL_MODEL_KEY,
        base_url=VISUAL_MODEL_URL,
    )
    model_type = client.models.list().data[0].id
    prompt = "红色高亮化合物结构对应的编号是什么？"
    messages = [{
        'role': 'user',
        'content': prompt
    }]

    response = client.chat.completions.create(
        model=model_type,
        messages=messages,
        seed=42,
        extra_body={'images': [image_base64]}
    )
    response = response.choices[0].message.content
    return response


def get_compound_id_from_description(description):

    prompt = f"""根据下面的内容描述一个化合物，请找出它的编号。

```markdown
{description}
```
注意：只输出化合物ID的JSON格式。

```json
{{"COMPOUND_ID": "__ID__"}}
```
"""

    client = OpenAI(
        api_key=SECONDARY_MODEL_KEY,
        base_url=SECONDARY_MODEL_URL,
    )

    response = client.chat.completions.create(
        model=SECONDARY_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content