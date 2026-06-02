任务
为一组候选结构页面制定 Markush 母核、变量位点、片段/取代基和跨页延续关系计划。
此任务只规划关系和组装候选，不生成最终 SMILES，不发明不可见结构。

输入
页面上下文摘要 JSON：
{{PAGE_CONTEXTS_JSON}}

结构候选摘要 JSON：
{{STRUCTURE_CANDIDATES_JSON}}

规划规则
1) 对每个输入页面输出一个且仅一个页面决策。
2) 判断页面角色：
   - `markush_scaffold`: 页面包含带 R/X/Y/Ar/Het 等变量位点的母核或通式。
   - `fragment_table`: 页面主要包含片段、取代基、R-group 表格行或文本片段定义。
   - `continuation`: 页面延续前序 Markush 表或片段表，需要继承母核/表头/变量位点。
   - `complete_compound`: 页面主要是单个或多个完整确定化合物。
   - `non_structure`: 无可用结构/片段关系。
   - `unknown`: 无法判断。
3) 如果当前页的片段表缺少母核、变量列、表头或记录上下文，但前序页面能在同一 `active_scope_id`
   的 Markush/R-group 表格范围内解释这些关系，
   设置 `use_prior_markush_context=true` 并选择最合适的早于当前页的 `context_source_page`。
4) 上下文来源可以跨多页延续；不要只看相邻上一页。但继承范围不能跨到另一个表、另一个母核、
   不同变量表头、非连续 Markush 记录块或页面上下文中标明已重置的 scope。
5) `relationships` 只连接输入中可见或候选摘要中已有的证据：
   - `scaffold_ref` 指向母核候选、页面母核描述或 null。
   - `fragment_refs` 指向片段候选、表格行、文本片段或空数组。
   - `variable_positions` 只写可见变量位点名，例如 R4、X、Y；无法确认则为空数组。
   - `compound_id` 只使用可见行号、Example/Compound/No.、caption/heading 或同一 table scope 中当前行的记录 ID；
     无法确认则为 "None"。
   - `compound_id_source` 必须说明 ID 来源：row_label、local_label、heading、caption、
     inherited_context、none 或 unknown。
6) 姿态一致性必须保守：
   - 如果片段连接方向、attachment/wavy bond、变量位点和母核标注一致，使用 `consistent`。
   - 如果片段明显不能连接到该变量位点，使用 `inconsistent`。
   - 如果该关系不需要姿态判断，使用 `not_applicable`。
   - 看不清或缺少连接信息，使用 `unknown`。
7) `assembly_status` 必须反映能否进入后续组装：
   - `ready`: 母核、变量位点、片段和记录 ID 均有足够证据，且姿态明确一致。
   - `needs_context`: 需要继承或补充上下文才能组装。
   - `not_applicable`: 非 Markush/片段关系。
   - `uncertain`: 关系存在但证据不足。
8) 如果没有明确母核引用或同一 table scope 内可继承的母核上下文，不要标记为 `ready`；使用 `needs_context`。
9) 如果片段姿态、attachment/wavy bond 或变量位点无法判断，不要标记为 `ready`；使用 `unknown` pose 和 `uncertain` 或 `needs_context`。
10) `fragment` 候选只有在 `fragment_visual_review` 明确满足以下条件时才可作为可装配片段：
    `compound_id` 与关系记录一致、`variable_position` 与关系变量一致、`molnextr_consistent=true`、`has_attachment_evidence=true`、
    `molnextr_has_attachment_atom=true`、`attachment_site_consistent=true`。
11) `text_substituent` 候选只是红框表格文本证据；它不能作为 MolNexTR 结构片段，不能把关系标记为 `ready`，也不能声称 pose consistent。
12) 使用 `inherited_markush_context` 时，`scaffold_ref` 必须指向该 scope 的 `active_scaffold_ref`；
    `variable_positions` 必须落在该 scope 的 active/inherited 变量表头内。若不匹配，使用 `needs_context` 或 `uncertain`。
13) 不要输出 assay 值，不要根据专利页码或具体化合物名写死判断。
14) 不要把只在上下文中出现的记录当作当前页面新记录；关系必须列出 `source_pages`。

输出契约
仅输出 JSON 对象，顶层必须包含 `pages` 和 `relationships`。
`pages` 数组必须覆盖每个输入页面。
每个页面决策格式：
```json
{
  "page": 1,
  "role": "markush_scaffold|fragment_table|continuation|complete_compound|non_structure|unknown",
  "use_prior_markush_context": false,
  "context_source_page": null,
  "confidence": "high|medium|low",
  "reason": "short generic reason"
}
```

每个关系格式：
```json
{
  "record_id": "stable local relationship id",
  "compound_id": "Example/Compound/row ID or None",
  "compound_id_source": "row_label|local_label|heading|caption|inherited_context|none|unknown",
  "source_pages": [1],
  "scaffold_ref": "candidate/page reference or null",
  "fragment_refs": ["candidate/page/table-row reference"],
  "variable_positions": ["R1"],
  "assembly_status": "ready|needs_context|not_applicable|uncertain",
  "pose_consistency": "consistent|inconsistent|not_applicable|unknown",
  "confidence": "high|medium|low",
  "reason": "short evidence summary"
}
```
