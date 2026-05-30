任务：将一个原始化合物编号别名，映射到给定 allowlist 中唯一的规范 Compound ID。

输入：
<RAW_ID>
{{RAW_ID}}
</RAW_ID>

<OPTIONAL_CONTEXT>
{{OPTIONAL_CONTEXT}}
</OPTIONAL_CONTEXT>

<ALLOWLIST>
{{ALLOWLIST}}
</ALLOWLIST>

规则：
1) 只能从 ALLOWLIST 中选择一个最终 COMPOUND_ID；如果无法确定，返回 "None"。
2) 允许语义等价与目标化合物前缀等价；核心判断是原始 ID 与 allowlist 项是否在上下文中指向同一个最终/目标化合物。
3) 允许大小写、空白、标点、括号、常见编号前缀和轻微 OCR 变体；只在非常有把握时归并。不要把非化合物编号字段误当编号。
4) 若 allowlist 中存在多个可能冲突项，且上下文不足以唯一判断，则返回 "None"。
5) Intermediate / Int. / Preparation 代表合成中间体；Embodiment 也不是本项目需要的目标最终化合物。不要把任何原始别名解析成 allowlist 中的 Intermediate / Int. / Preparation / Embodiment 项；若只有这些候选，返回 "None"。
6) Example/Ex./No./Compound/Formula 与 Intermediate/Int./Preparation/Embodiment 不属于可互换前缀；不得仅按数字把 Intermediate 1、Int. 1、Preparation 1 或 Embodiment 1 归并为 Example 1，除非上下文明确说明该最终目标化合物就是 Example 1。
7) 先自检再输出：
   - `CONFIDENCE` 只能是 `high` / `medium` / `low`。
   - 若仍有明显歧义、上下文不足、或你只能猜测，则将 `COMPOUND_ID` 设为 `"None"`，`CONFIDENCE` 设为 `low`。
   - `REASON` 用 20 个词以内说明依据。
8) 不要输出任何解释文字，只输出 JSON。

输出格式（只输出这一行 JSON，不要代码块或解释文字）：
{"COMPOUND_ID":"<ALLOWLIST中的某一项或None>","CONFIDENCE":"high|medium|low","REASON":"short reason"}
