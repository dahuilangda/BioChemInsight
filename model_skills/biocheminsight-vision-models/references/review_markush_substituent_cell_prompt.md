任务
复核红框是否是 Markush/R-group 表格中的取代基单元格，并抽取可见证据。
你只读取红框和页面上下文，不生成 SMILES，不发明不可见连接点。

页面/表格上下文摘要：
{{PAGE_CONTEXT_JSON}}

候选证据 JSON：
{{CANDIDATE_JSON}}

视觉规则
1) 只以红框内容和紧邻表头/行标签为证据来源。
2) 如果红框是 R/R1/R4/X/Y 等变量列的单元格，输出 `visual_role="substituent_cell"`。
3) `compound_id` 必须来自同一行可见的 Ex./No./Compound/Example 行号；不可见则为 "None"。
4) `variable_position` 必须来自列头或单元格标注，例如 R4；不可见则为空字符串。
5) `substituent_text` 只写红框内可见的取代基文本或可见结构标签；不要补全长名称。
6) 如果红框中是结构图/片段图，设置 `has_visual_structure=true`；如果只是文字取代基，设置 false。
7) 只有看到波浪键、星号、R-group 连接符或明确连接前缀时，`has_attachment_evidence=true`。
8) 看不清或不是单元格时保持保守，不要猜。

输出契约
仅输出 JSON 对象：
```json
{
  "visual_role": "substituent_cell|table_header|scaffold|noise|unknown",
  "compound_id": "row/example id or None",
  "compound_id_source": "row_label|local_label|none|unknown",
  "variable_position": "R4",
  "substituent_text": "visible text only",
  "has_visual_structure": false,
  "has_attachment_evidence": true,
  "confidence": "high|medium|low",
  "evidence": "short visual evidence summary"
}
```
