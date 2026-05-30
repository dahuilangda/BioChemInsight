任务
将 OCR+文本模型的 assay 抽取初稿，与视觉模型的逐格审查报告合并，输出最终 assay 字典。

输入
1) 完整 OCR / Markdown chunk：
<OCR_CONTEXT>
{{OCR_CONTEXT}}
</OCR_CONTEXT>

2) 当前 assay 抽取初稿：
{{ASSAY_DICTS_JSON}}

3) 视觉模型审查报告：
{{VISUAL_REPORT_JSON}}

合并规则
1) 输出必须保持与“当前 assay 抽取初稿”相同的顶层 assay 名结构。
2) 对视觉报告中 `action="replace"`、`visual_value` 非空、且 `confidence="high"` 的项，只替换对应对象的 `value`；若视觉报告提供 `unit` 或 `description`，同步更新这些字段。
3) 对 `action="replace"` 但 `confidence` 不是 `high` 的项，默认保留初稿原值，除非 OCR 上下文能非常明确地支持同一单元格同一值。
4) 对 `action="keep"` 或 `action="uncertain"` 的项，保留初稿原值。
5) 不要新增 OCR 初稿中不存在的 compound_id，除非视觉报告明确指出它是同一行同一 assay 的纠错且 compound_id 与初稿一致。
6) 不要删除初稿中已有的 compound_id。
7) 不要自行推断字符映射；视觉报告是唯一可以改变值的依据。OCR 上下文只用于确认 assay 名、compound_id 和表格范围是否一致。
8) 若视觉报告与 OCR 初稿或表格上下文冲突且无法确定，保留初稿原值。
9) 每个最终单元格都保持 rich assay object 结构：`value/unit/method/description/confidence/reason/assay_match`。
10) `assay_match` 是 assay 兼容性与最佳请求归属自检，合并时必须从 OCR 初稿保留。只有当视觉报告明确证明初稿匹配了错误 assay 列/记录时，才可将 `assay_match.compatible` 改为 `false`；若改为 `false`，该项不应出现在最终输出中。`assay_match.best_requested_assay` 必须保留并严格等于当前顶层 assay 名。

输出格式
仅输出 JSON 对象，格式与当前 assay 抽取初稿相同：
{
  "Assay Name": {
    "Compound ID": {
      "value": "final value",
      "unit": "unit or empty",
      "method": "method/context or empty",
      "description": "symbol/value explanation or empty",
      "confidence": "high|medium|low",
      "reason": "short reason",
      "assay_match": {
        "target": "target/method/platform/endpoint/unit from requested assay",
        "candidate": "target/method/platform/endpoint/unit from selected column",
        "compatible": true,
        "best_requested_assay": "Assay Name",
        "reason": "candidate column matches requested assay concept"
      }
    }
  }
}
