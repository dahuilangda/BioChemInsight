任务
独立复核 assay 抽取结果中的 `value` 是否真的是目标 assay 的测量值，而不是 compound ID、行号、页码、表号、脚注、相邻列或其它 assay 的值。必要时基于同一行/同一记录的 OCR 证据纠正 `value` 和 `unit`。

目标 assay：
{{ASSAY_NAME}}

本次任务请求的所有 assay 名称：
{{ASSAY_NAMES_JSON}}

OCR / Markdown 上下文：
<OCR_CONTEXT>
{{OCR_CONTEXT}}
</OCR_CONTEXT>

待复核的 assay 抽取结果：
{{ASSAY_PAYLOAD_JSON}}

复核规则
1) 对每个 compound_id 独立判断。必须在 OCR 上下文中定位同一行、同一表格记录或明确连续记录里的 compound ID 与目标 assay 测量值。
2) `value` 必须来自 assay 数值/等级单元格，不得来自 Example/Compound/No./ID 单元格、行号、表号、页码、脚注、浓度条件、化学量、NMR/MS 数据或相邻 assay 列。
3) 如果当前 `value` 实际来自 compound ID 单元格，必须先检查同一行/记录是否有明确的目标 assay 测量值。
4) 如果同一行/记录中能明确看到正确的目标 assay 测量值，即使当前 `value` 是错误的 ID/行号，也必须返回 `valid_assay_value=true`，并在 `corrected_value` / `corrected_unit` 中给出纠正后的值和单位；不得在已能纠正时返回 `valid_assay_value=false`。单位可来自表头、脚注或单元格。
5) 如果当前值正确，也返回 `valid_assay_value=true`，`corrected_value` 填当前值，`corrected_unit` 填当前单位或从 OCR 表头确认的单位。
6) 如果目标 assay 与候选列在 method/platform/endpoint/单位/上下文上冲突，或候选值更应属于 sibling assay，返回 `valid_assay_value=false`，`corrected_value="None"`。
7) 不要猜测、不要用常识补值。只有 OCR 文本中能确认同一行/记录关系时才保留或纠正。
8) OCR 表格的表头行可能出现在表格顶部、底部、分页续表处或重复行中。若同一表格内某行明显包含化合物编号列标签与 assay 终点/单位/方法表头，应把它作为该表格列标签使用；不要仅因为表头行在数据行之后就拒绝同表格数据。

输出契约
仅输出 JSON 对象；顶层键必须是输入中的 compound_id。每项格式：
```json
{
  "__COMPOUND_ID__": {
    "valid_assay_value": true,
    "corrected_value": "__MEASURED_VALUE_OR_None__",
    "corrected_unit": "__UNIT_OR_EMPTY__",
    "confidence": "high",
    "reason": "same row has compound ID and target assay measurement; value is not the ID cell"
  }
}
```

字段要求
- `valid_assay_value`: 布尔值。只有能确认目标 assay 测量值时才为 true。
- `corrected_value`: 目标 assay 的测量值；无法确认时使用字符串 `"None"`。
- `corrected_unit`: 单位；未知但值有效时可为空字符串。
- `confidence`: `high` / `medium` / `low`。
- `reason`: 简短说明证据，必须指出值来自哪个 assay 值单元/表头，以及为什么不是 compound ID/行号/相邻列。
