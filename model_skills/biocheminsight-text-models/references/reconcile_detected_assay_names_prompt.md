任务
复核自动扫描得到的候选 bioactivity / assay 名称，输出正式抽取阶段应保留的最小 assay 名称集合。

输入
候选 assay 名称（只能从这些原始字符串中选择保留项，不要创造新名称）：
{{ASSAY_NAMES_JSON}}

页面级检测证据：
{{PAGE_DECISIONS_JSON}}

OCR / Markdown 证据片段：
<OCR_CONTEXT>
{{OCR_CONTEXT}}
</OCR_CONTEXT>

判断原则
1) 保留真正对应可抽取 compound-level bioactivity 结果列、结果表或结果记录的 assay 名称。每个保留项都必须能在 OCR 证据中对应到独立的结果列/结果值域/单位/记录集合。
2) 如果多个候选名称描述同一个结果列/表/记录，保留最具体、最能同时解释 target / biological context / method-platform / endpoint / unit 的候选。
3) 不要把同一结果重复拆成一个具体名称和一个泛化 method/endpoint 名称；泛化名称只有在它代表独立结果列/表/记录时才保留。
4) 不要把 assay protocol、实验方法段落、背景描述、检测原理、target-only 描述、或宽泛方法类别当成独立 bioactivity 名称，除非它明确对应独立于其它候选的 compound-level 结果列/记录集合。
5) 如果两个候选代表不同 endpoint、不同 target、不同 assay modality、不同单位或不同结果列，应同时保留。
6) 不要依据某个词是否常见来删除候选；必须结合 OCR 证据和页面级检测证据判断候选是否指向独立结果。
7) 当证据不足以证明两个候选是同一结果时，必须保留两者，并在 reason 中说明不确定性。
8) `keep=false` 是强合并决策，只能在高置信确认该候选不是独立 assay、且可由某个保留候选覆盖同一结果时使用。
9) 对每个 `keep=true` 的候选，reason 必须说明它对应的独立结果证据；如果只能说明 protocol/context，而不能指出独立结果列/记录，则该候选不应保留。

输出契约
仅输出 JSON 对象：
```json
{
  "assay_names": ["__KEPT_ORIGINAL_CANDIDATE_NAME__"],
  "decisions": {
    "__ORIGINAL_CANDIDATE_NAME__": {
      "keep": true,
      "canonical_assay_name": "__KEPT_ORIGINAL_CANDIDATE_NAME__",
      "confidence": "high",
      "reason": "short reason"
    }
  }
}
```

字段要求
- `assay_names`: 最终保留的名称列表；每一项必须是输入候选中的原始字符串。
- `decisions`: 必须包含每个输入候选的决策。
- `keep`: 当前候选是否作为独立 assay 保留；不确定时必须为 `true`。
- `canonical_assay_name`: 如果 `keep=true`，必须等于当前候选；如果 `keep=false`，必须是 `assay_names` 中最能覆盖它的保留候选，或字符串 `"None"`。
- `confidence`: `high` / `medium` / `low`；`keep=false` 时必须为 `high`。
- `reason`: 简短说明，重点解释该候选是否代表独立 compound-level bioactivity 结果。
