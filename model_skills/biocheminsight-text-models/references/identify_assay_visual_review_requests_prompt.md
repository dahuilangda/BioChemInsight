任务
判断哪些已抽取的 assay 单元格需要用视觉模型重新读图确认。

输入
1) 完整 OCR / Markdown chunk：
<OCR_CONTEXT>
{{OCR_CONTEXT}}
</OCR_CONTEXT>

2) 已由文本模型抽取出的 assay 结果：
{{ASSAY_DICTS_JSON}}

3) 从 OCR 中解析出的表格网格（如果有）：
{{PARSED_TABLES_JSON}}

判断原则
1) 必须基于完整 OCR 上下文判断，包括表题、分组表头、列名、单位、同一列其它值、同一行其它 assay 值、上下页连续性和正文说明。
2) 不要按某个字符做固定映射，也不要因为单个值看起来奇怪就要求视觉复读。
3) 当某个目标 assay 对应的列整体呈现符号等级、定性等级、响应/效能等级、百分比/数值混合等级等非标准值域，且该列容易因为特殊 glyph、重复符号、上下标/脚注符号、短字母/短数字而发生 OCR 混淆时，把该 assay 在当前 chunk 中已抽取出的所有相关单元格列入视觉复读。
4) 普通数值列、单位明确的浓度/EC50/IC50/PK/AUC 列、NA/ND/空值等，不应仅因短文本而复读。
5) 对非标准符号/等级列，不要试图区分哪些 OCR 值“看起来错”、哪些“看起来对”；只要该列整体需要视觉确认，就列出该 assay 下当前 chunk 已抽取出的所有 compound_id。视觉模型负责逐格读图确认。
6) 输出只列需要视觉复读的 assay/单元格。不要输出普通数值 assay 的单元格。
7) 不要推断校正值，不要说某个 OCR 字符“应该映射为”另一个字符；这里只决定是否需要看图复读。

输出格式
仅输出 JSON 对象。顶层键为 assay 名，值为数组；每个数组元素包含：
- `compound_id`: 已抽取结果中的化合物 ID
- `ocr_value`: 当前抽取值
- `reason`: 简短说明为什么从完整 OCR 上下文看需要视觉复读
- `context`: 可选，相关表头/行/列上下文，便于视觉模型定位

示例格式：
{
  "Assay Name": [
    {
      "compound_id": "Example 1",
      "ocr_value": "raw value",
      "reason": "same-column values indicate a symbol-grade column, but this cell is inconsistent",
      "context": "column/table context"
    }
  ]
}
