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
2) 允许语义等价与目标化合物前缀等价；以下只是常见示例，不是封闭列表，核心判断是二者是否指向同一个最终/目标化合物：
   - Example 1 ≈ Ex.1 ≈ 1 ≈ (1) ≈ No.1 ≈ 编号1
   - 若 allowlist 中只有 "Example 1"，而原始别名是 "Ex.1"、"Compound 1"、"1"、"(1)"、"No.1"，应输出 "Example 1"。
3) 允许轻微 OCR 变体，只在非常有把握时归并：
   - 12a ≈ 12A
   - CDKl 不适用于化合物编号；不要把非编号字段误当编号。
4) 若 allowlist 中存在多个可能冲突项（例如同时有 "Example 1" 与 "Compound 1"），且上下文不足以唯一判断，则返回 "None"。
5) Intermediate / Int. / Preparation 代表合成中间体；Embodiment 也不是本项目需要的目标最终化合物。不要把任何原始别名解析成 allowlist 中的 Intermediate / Int. / Preparation / Embodiment 项；若只有这些候选，返回 "None"。
6) Example/Ex./No./Compound/Formula 与 Intermediate/Int./Preparation/Embodiment 不属于可互换前缀；不得仅按数字把 Intermediate 1、Int. 1、Preparation 1 或 Embodiment 1 归并为 Example 1，除非上下文明确说明该最终目标化合物就是 Example 1。
7) 先自检再输出：
   - `CONFIDENCE` 只能是 `high` / `medium` / `low`。
   - 若仍有明显歧义、上下文不足、或你只能猜测，则将 `COMPOUND_ID` 设为 `"None"`，`CONFIDENCE` 设为 `low`。
   - `REASON` 用 20 个词以内说明依据，如 “unique allowlist match for Example-style alias” 或 “ambiguous between Example 1 and Compound 1”。
8) 不要输出任何解释文字，只输出 JSON。

输出格式：
```json
{"COMPOUND_ID":"<ALLOWLIST中的某一项或None>","CONFIDENCE":"high|medium|low","REASON":"short reason"}
```
