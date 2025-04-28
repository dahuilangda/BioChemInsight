import os
import warnings
warnings.filterwarnings("ignore")

import time
from functools import wraps
import google.generativeai as genai
import json

import sys
sys.path.append('..')
from constants import VISUAL_MODEL_URL, VISUAL_MODEL_KEY, VISUAL_MODEL_NAME, HTTP_PROXY, HTTPS_PROXY
from constants import LLM_MODEL_NAME, LLM_MODEL_KEY, LLM_MODEL_URL, GEMINI_API_KEY

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
def content_to_dict(content, assay_name, compound_id_list=None, retry=3, model_name='gemini-2.0-flash', api_key=GEMINI_API_KEY):
    configure_genai(api_key)
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

def structure_to_id(image_file, prompt=None):
    """
    Extracts the compound ID from a highlighted chemical structure image.
    """

    import base64

    def encode_image_to_base64(image_path):
        with open(image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read())
        encoded_image_text = encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

            # return str(base64.b64encode(image_file.read()).decode('utf-8'))
        
    image_base64 = encode_image_to_base64(image_file)

    client = OpenAI(
        api_key=VISUAL_MODEL_KEY,
        base_url=VISUAL_MODEL_URL,
    )
    if VISUAL_MODEL_NAME:
        model_type = VISUAL_MODEL_NAME
    else:
        model_type = client.models.list().data[0].id

    if prompt is None:
        # prompt = "红色虚线框中化合物对应的编号或名称是什么？"
        # prompt = "红色虚线框中化合物对应的编号或名称是什么？如果没找到，请回答“没有”。"
        prompt = "What is the ID or name of the red highlight compound in the red dashed box? If not found, please answer 'None'."

    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_base64
                    },
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    # response = client.chat.completions.create(
    #     model=model_type,
    #     messages=messages,
    #     seed=42,
    #     extra_body={'images': [image_base64]}
    # )

    response = client.chat.completions.create(
        model = model_type,
        messages = messages,
    )

    response = response.choices[0].message.content
    return response


def get_compound_id_from_description(description):

    prompt = f"""根据下面的内容描述一个化合物，请找出它的编号(名称)。

```markdown
化合物:“{description}”
```
注意：只输出化合物ID的JSON格式。

```json
{{"COMPOUND_ID": "__ID__"}}
```
"""

    client = OpenAI(
        api_key=LLM_MODEL_KEY,
        base_url=LLM_MODEL_URL,
    )

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
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
    assay_name = 'TR-FRET EC5o (M)'
    compound_id_list = ['114', '246', '245']
    assay_dict = content_to_dict(content, assay_name, compound_id_list)
    print(assay_dict)