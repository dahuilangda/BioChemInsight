任务
复核 assay 抽取结果中每个候选值最应该归属哪个 requested assay，并输出逐项判定。

输入
当前输出顶层 assay：
{{ASSAY_NAME}}

本次任务请求的所有 assay 名称（只能从这些原始字符串中选择 best_requested_assay）：
{{ASSAY_NAMES_JSON}}

OCR / Markdown 上下文：
<OCR_CONTEXT>
{{OCR_CONTEXT}}
</OCR_CONTEXT>

待复核的当前 assay 抽取结果：
{{ASSAY_PAYLOAD_JSON}}

复核规则
1) 对每个 compound_id 独立判断候选值/列/记录最应该归属哪个 requested assay。
2) `best_requested_assay` 表示候选列/记录的最佳归属，不表示“是否也可宽泛兼容”。必须选择最具体、最直接、最能解释候选 method/platform/endpoint/unit/context 的 requested assay。
3) 如果候选列/记录的 method/platform/endpoint/unit 与某个 sibling assay 更具体、更直接或近似逐字匹配，则 `best_requested_assay` 必须选择该 sibling，而不是选择当前较宽泛的 assay。
4) 宽泛 assay（例如只描述 target 或 assay family）不能夺取已经由更具体 sibling assay 命名的方法/平台/终点列；除非 OCR 上下文明示该列就是宽泛 assay 的唯一结果。
5) requested assay 名称本身是任务上下文的一部分。若当前内容是续表/分块表格，OCR 只保留 method/endpoint 而没有重复 target，但 requested assays 中存在同 method/endpoint 的 target-specific sibling，且 OCR 没有出现冲突 target，则 `best_requested_assay` 必须选择 target-specific sibling，而不是泛化 assay。
6) exact header string 不自动优先。若表头写的是泛化 method/endpoint，而 requested assays 同时包含 target-specific sibling 和泛化 sibling，且文档/任务上下文说明这些 assay pages 属于该 target-specific context，则 target-specific sibling 是最佳归属。
7) 对同一候选值，不要用泛化 sibling 抢占 target-specific sibling；只有 OCR/文档上下文明示不同 target、无 target 或泛化 assay 是唯一 requested match 时，才选择泛化 sibling。
8) 如果候选列/记录和当前 assay 在 method/platform、endpoint、unit 或 biology context 上冲突，或更应该归属其它 requested assay，则 `compatible_current=false`。
9) 仅根据输入上下文、requested assay 任务上下文和候选对象复核，不新增 compound_id，不改写 value。

输出契约
仅输出 JSON 对象；顶层键必须是输入中的 compound_id。每项格式：
```json
{
  "__COMPOUND_ID__": {
    "compatible_current": true,
    "best_requested_assay": "{{ASSAY_NAME}}",
    "confidence": "high",
    "reason": "short reason"
  }
}
```

字段要求
- `compatible_current`: 布尔值；只有候选最佳归属就是当前 assay 且语义兼容时才为 true。
- `best_requested_assay`: 必须严格等于 requested assay 名称列表中的一个原始字符串。
- `confidence`: `high` / `medium` / `low`。
- `reason`: 简短说明选择依据，重点说明 method/platform/endpoint/unit/context。
