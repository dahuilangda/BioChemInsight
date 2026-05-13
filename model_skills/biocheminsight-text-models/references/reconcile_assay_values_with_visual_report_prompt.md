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
2) 对视觉报告中 `action="replace"` 且 `visual_value` 非空的项，将对应 assay / compound_id 的值替换为 `visual_value`。
3) 对 `action="keep"` 或 `action="uncertain"` 的项，保留初稿原值。
4) 不要新增 OCR 初稿中不存在的 compound_id，除非视觉报告明确指出它是同一行同一 assay 的纠错且 compound_id 与初稿一致。
5) 不要删除初稿中已有的 compound_id。
6) 不要自行推断字符映射；视觉报告是唯一可以改变值的依据。OCR 上下文只用于确认 assay 名、compound_id 和表格范围是否一致。
7) 若视觉报告与 OCR 初稿或表格上下文冲突且无法确定，保留初稿原值。

输出格式
仅输出 JSON 对象，格式与当前 assay 抽取初稿相同：
{
  "Assay Name": {
    "Compound ID": "final value"
  }
}
