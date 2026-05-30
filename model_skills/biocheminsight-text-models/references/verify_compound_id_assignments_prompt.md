任务
复核 assay 抽取结果中的 compound_id 是否来自同一行/同一记录的完整最终化合物 ID，并在必要时映射到 allowlist 中的规范 ID。

输入
允许输出的规范 compound_id 列表：
{{COMPOUND_ID_LIST_JSON}}

OCR / Markdown 上下文：
<OCR_CONTEXT>
{{OCR_CONTEXT}}
</OCR_CONTEXT>

待复核的 assay 抽取结果：
{{ASSAY_PAYLOAD_JSON}}

复核规则
1) 对每个当前 compound_id 独立判断。必须确认该 ID 是 assay 值所在同一行、同一表格记录或明确连续记录中的完整最终化合物编号。
2) 不要把表格行号、页码、图表编号、脚注数字、单元格序号、数值的一部分、或多位数 ID 的片段当成 compound_id。
3) 如果 OCR 显示当前 key 只是完整 ID 的片段，必须返回 `valid_current_id=false`；若 allowlist 中有可由上下文确认的正确完整 ID，可把 `canonical_compound_id` 设为该 allowlist 原始字符串。
4) 如果当前 key 是带前缀、括号、标点或格式差异的别名形式，且上下文确认同一最终化合物，返回 allowlist 中对应的规范 ID。
5) 如果无法从 OCR 上下文确认当前项的完整 ID 与 assay 值同行/同记录，返回 `valid_current_id=false`，`canonical_compound_id="None"`。
6) 只能选择 allowlist 中的原始字符串作为 `canonical_compound_id`；无法确认时使用字符串 `"None"`。
7) 不得把行序号、候选序号或“第 N 个化合物行”映射为 compound ID。若当前 key 不是同一行真实 ID 单元格内容的别名，必须返回 `valid_current_id=false`，不能把它映射到该行真实 ID。

输出契约
仅输出 JSON 对象；顶层键必须是输入中的当前 compound_id。每项格式：
```json
{
  "__CURRENT_COMPOUND_ID__": {
    "valid_current_id": true,
    "canonical_compound_id": "__ALLOWLIST_ID__",
    "confidence": "high",
    "reason": "same row has complete compound ID and assay value"
  }
}
```

字段要求
- `valid_current_id`: 布尔值。只有当前项可确认映射到完整最终化合物 ID 时才为 true。
- `canonical_compound_id`: allowlist 中的一个原始字符串，或 `"None"`。
- `confidence`: `high` / `medium` / `low`。
- `reason`: 简短说明证据，特别指出完整 ID、行/记录关系，以及为什么不是截断片段。
