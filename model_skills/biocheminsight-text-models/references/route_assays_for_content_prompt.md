任务
判断当前 OCR / Markdown 内容中，哪些 requested assay 应该进入正式 bioactivity 抽取。

输入
本次任务请求的所有 assay 名称（输出键必须严格使用这些原始字符串）：
{{ASSAY_NAMES_JSON}}

OCR / Markdown 上下文：
<OCR_CONTEXT>
{{OCR_CONTEXT}}
</OCR_CONTEXT>

路由规则
1) 逐个 requested assay 判断当前内容是否包含应该归属该 assay 的可抽取 bioactivity 记录。
2) 如果同一列/表/记录同时像多个 requested assay，必须选择最具体、最直接、最能解释 method/platform/endpoint/unit/biology context 的 assay。
3) 宽泛 assay 不能夺取已经由更具体 sibling assay 命名的方法、平台或终点列；除非上下文明示该列就是宽泛 assay 的唯一结果。
4) requested assay 名称本身是任务上下文的一部分。若当前内容是续表/分块表格，OCR 只保留 method/endpoint 而没有重复 target，但 requested assays 中存在同 method/endpoint 的 target-specific sibling，且 OCR 没有出现冲突 target，则必须选择 target-specific sibling，而不是选择更泛化的 assay 名称。
5) exact header string 不自动优先。若表头写的是泛化 method/endpoint，而 requested assays 同时包含 target-specific sibling 和泛化 sibling，且文档/任务上下文说明这些 assay pages 属于该 target-specific context，则 target-specific sibling 是最佳归属，泛化 sibling 的 `extract` 必须为 false。
6) 对同一批记录，不要同时将 target-specific sibling 和泛化 sibling 标记为 true；只能保留最具体的一个。
7) 只根据当前 OCR 内容和 requested assay 任务上下文判断；不要因为任务请求了某个 assay 就把无关内容标记为需要抽取。
8) 如果当前内容只有结构、合成步骤、引用、图例说明或无数值/等级 bioactivity 记录，则所有 assay 的 extract 都为 false。
9) 当表头、脚注或相邻正文说明 assay/method/endpoint 时，应结合整页/文档上下文判断，不要只看单个单元格。

输出契约
仅输出 JSON 对象，顶层必须包含 `assays`。`assays` 必须包含每个 requested assay 名称作为键。
每项格式：
```json
{
  "assays": {
    "__REQUESTED_ASSAY_NAME__": {
      "extract": true,
      "confidence": "high",
      "reason": "short reason"
    }
  }
}
```

字段要求
- `extract`: 布尔值；只有当前内容存在应归属该 requested assay 的 bioactivity 记录时才为 true。
- `confidence`: `high` / `medium` / `low`。
- `reason`: 简短说明依据，重点说明 method/platform/endpoint/unit/context，以及为什么不是 sibling assay。
