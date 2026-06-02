任务
复核红框中的 Markush 片段/取代基结构候选，并抽取它能否作为某个 R-group 行的结构证据。
你只做视觉证据核验，不生成 SMILES，不发明不可见结构。

MolNexTR 候选 JSON：
{{FRAGMENT_CANDIDATE_JSON}}

页面/表格上下文摘要：
{{PAGE_CONTEXT_JSON}}

视觉规则
1) 红框必须包含可见结构图、片段图或带 attachment/wavy bond 的取代基图，才可作为 `fragment` 或 `substituent`。
2) 如果红框只是文字、名称、表格线、空白或噪声，输出 `visual_role="noise"` 或 `unknown`。
3) `compound_id` 必须来自同一行可见的 ID 值；如果当前页是续表，可用 `inherited_markush_context.active_compound_id_header`
   解释该列是 Ex./No./Compound/Example 列，但不能凭继承上下文发明当前行 ID 值。当前行 ID 不可见则为 "None"。
4) `variable_position` 必须来自同一列表头、红框附近可见标注，或同一 `active_scope_id` 内继承的变量表头。
   使用继承变量表头时，必须确认当前红框仍在同一续表/同一列范围内，并在 evidence 中写明继承来源页。
5) 只有看到波浪键、星号、开放键、R-group 连接符或明确 attachment 标记时，`has_attachment_evidence=true`。
6) 不能根据视觉判断 attachment 接在哪个原子；视觉模型在原子级连接位点上不可靠。
7) `molnextr_has_attachment_atom=true` 只有在 MolNexTR 候选 JSON/MOLBLOCK 本身包含明确 attachment atom、dummy atom、星号 atom、开放连接原子或等价结构位点时才可给出。
8) `attachment_site_consistent=true` 只表示 MolNexTR 已给出的 attachment 位点与红框中可见 wavy/open bond 的位置一致；如果 MolNexTR 没有明确位点，必须为 false。
9) MolNexTR 的 MOLBLOCK/结构证据必须与红框中可见结构一致；SMILES 不能作为 pose 或连接位点证据。
10) 如果当前页出现新的独立表格、不同表头、不同母核/片段范围，或 `inherited_markush_context` 显示 scope 已重置，
   不要沿用旧表头；行号、变量位点或 attachment 任何一项看不清，保持低/中置信，不要猜。

输出契约
仅输出 JSON 对象：
```json
{
  "visual_role": "fragment|substituent|scaffold|text_cell|noise|unknown",
  "compound_id": "row/example id or None",
  "compound_id_source": "row_label|local_label|none|unknown",
  "variable_position": "R4",
  "molnextr_consistent": true,
  "has_attachment_evidence": true,
  "molnextr_has_attachment_atom": true,
  "attachment_site_consistent": true,
  "confidence": "high|medium|low",
  "evidence": "short visual evidence summary"
}
```
