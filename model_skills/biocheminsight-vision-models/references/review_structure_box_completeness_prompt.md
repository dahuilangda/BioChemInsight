任务
复核红框是否完整、单独地框住一个可送 MolNexTR 的 2D 化学结构或 fragment。
你只判断 box 质量，不生成 SMILES，不补全结构，不建议新 box。

候选 JSON：
{{CANDIDATE_JSON}}

视觉规则
1) `complete_single_structure`：红框内完整包含一个 2D 化学结构/fragment，键、原子标签、wavy/attachment 端点没有被边界截断。
2) `partial_or_truncated`：红框只框住结构的一半、一小段、wavy/attachment 局部，或任何键/环/原子标签明显越过红框边界。
3) `multiple_structures`：红框内包含多个彼此独立的结构图，MolNexTR 会被混淆。
4) `non_structure`：红框主要是文字、表格线、页眉页脚、谱图文字、空白或噪声。
5) `uncertain`：看不清是否完整；不确定时不要放行。
6) 只基于红框和紧邻边界可见内容判断；不要因为表格上下文猜测完整结构。

输出契约
仅输出 JSON 对象：
```json
{
  "box_status": "complete_single_structure|partial_or_truncated|multiple_structures|non_structure|uncertain",
  "is_single_structure": true,
  "is_complete_box": true,
  "confidence": "high|medium|low",
  "evidence": "short visual evidence summary"
}
```
