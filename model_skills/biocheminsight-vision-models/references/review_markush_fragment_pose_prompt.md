任务
复核红框中的 Markush 片段/取代基是否能支持一个候选 Markush 关系的组装计划。
你只做视觉证据核验，不生成最终 SMILES，不发明不可见母核或连接位点。

输入关系 JSON：
{{RELATIONSHIP_JSON}}

MolNexTR 片段证据 JSON：
{{FRAGMENT_CANDIDATE_JSON}}

MolNexTR 母核证据 JSON：
{{SCAFFOLD_CANDIDATE_JSON}}

页面/表格上下文摘要：
{{PAGE_CONTEXT_JSON}}

视觉规则
1) 当前图像红框通常是片段/取代基/表格局部；母核候选通过 `SCAFFOLD_CANDIDATE_JSON` 提供。
2) 关系必须同时有母核候选、片段候选、变量位点和记录 ID；缺任一项都不能 ready。
3) 变量位点必须一致：例如关系写 R4，则母核候选或页面上下文必须支持 R4，不能只看到泛化的 R 就批准。
4) 如果红框中可见 wavy bond、attachment point、R/X/Y/Ar/Het 等变量位点或明确取代基连接方向，记录为可见证据。
5) MolNexTR 的 MOLBLOCK/坐标图证据可作为结构识别辅助；SMILES 不能作为 pose 或连接位点证据。
6) 如果视觉与 MolNexTR MOLBLOCK/红框证据明显不一致，不能标记 ready。
7) 如果只有表格文本或片段名称，没有可见连接/姿态证据，不能确认 pose consistent。
8) 如果红框内容是噪声、空白、被严重裁切、或不是当前关系的片段/母核，输出 `assembly_status="uncertain"`。
9) 只有视觉上能确认母核变量位点和片段/变量列不冲突，并且 MolNexTR MOLBLOCK/红框证据相符，才输出 `assembly_status="ready"` 和 `pose_consistency="consistent"`。
10) 保守处理：看不清时用 `pose_consistency="unknown"`，不要猜测。

输出契约
仅输出 JSON 对象：
```json
{
  "visual_role": "scaffold|fragment|substituent|table_cell|noise|unknown",
  "molnextr_consistent": true,
  "has_attachment_evidence": true,
  "variable_position_visible": true,
  "pose_consistency": "consistent|inconsistent|not_applicable|unknown",
  "assembly_status": "ready|needs_context|not_applicable|uncertain",
  "confidence": "high|medium|low",
  "evidence": "short visual evidence summary"
}
```
