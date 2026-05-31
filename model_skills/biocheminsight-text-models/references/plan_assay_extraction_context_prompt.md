任务
为一组候选 bioactivity 页面制定通用抽取上下文计划。此任务只规划上下文和实体锚点，不抽取 assay 数值。

输入
目标测定字段列表：
{{ASSAY_NAMES_JSON}}

页面上下文摘要 JSON：
{{PAGE_CONTEXTS_JSON}}

规划规则
1) 对每个输入页面输出一个且仅一个决策。
2) 判断当前页面是新记录块、延续记录块、独立记录块、非 assay 页面或不确定页面。
   记录块可能是表格、段落、图注、结构旁标注、列表、专利实施例段落或它们的组合。
3) 如果当前页是延续记录且缺少必要上下文，应选择最合适的前序页面作为 `context_source_page`。
4) 上下文来源可以跨多页延续；不要只看相邻上一页。
5) 只有当前序页面能解释当前页记录的化合物锚点、assay 值、单位、方法、endpoint、
   图注或结构指向关系时，才设置 `use_prior_context=true`。
6) 如果当前页有自己的完整上下文，应优先视为新记录块或独立记录块，不要继承旧上下文。
7) 规划当前页的实体锚点策略：
   - `text_compound_id`: assay 记录能通过可见文本 ID / Example / Compound / No. 锚定。
   - `visual_structure_anchor`: assay 记录主要通过同页结构图、红框结构、箭头或空间邻近关系锚定。
   - `mixed`: 文本 ID 和结构图锚点都可能需要使用。
   - `unknown`: 无法判断。
8) 当页面摘要包含 same-page structure anchors，且 assay 表格/标注缺少文本 Compound ID，
   但视觉结构明显是记录主体时，应选择 `visual_structure_anchor` 或 `mixed`；这只是规划，
   不得生成任何活性值。
9) 不要输出具体化合物活性值，不要改写 assay 名称。
10) 计划必须保守：无法确认延续关系或锚点关系时使用 `unknown`，并让 `use_prior_context=false`。

输出契约
仅输出 JSON 对象，顶层必须包含 `pages`。`pages` 数组必须覆盖每个输入页面。
每个页面决策格式：
```json
{
  "pages": [
    {
      "page": 1,
      "role": "new_record|continuation|standalone|non_assay|unknown",
      "use_prior_context": false,
      "context_source_page": null,
      "entity_anchor_strategy": "text_compound_id|visual_structure_anchor|mixed|unknown",
      "confidence": "high|medium|low",
      "reason": "short generic reason"
    }
  ]
}
```

字段要求
- `page`: 输入页面号。
- `role`: 页面在 assay 记录流中的角色。
- `use_prior_context`: 只有当前记录需要继承前序上下文时才为 true。
- `context_source_page`: 使用前序上下文时必须是输入页面中早于当前页的页码，否则为 null。
- `entity_anchor_strategy`: 当前页 assay 记录应如何锚定到化合物主体。
- `confidence`: `high` / `medium` / `low`。
- `reason`: 简短说明依据，只说明表头/续表/列结构/实体锚点线索。
