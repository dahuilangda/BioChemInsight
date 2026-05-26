任务
从提供的 Markdown / OCR整理文本中，抽取化合物 ID 与其对应的目标测定字段“{{ASSAY_NAME}}”，并输出为字典。

输入
<MARKDOWN_TEXT>
{{MARKDOWN_TEXT}}
</MARKDOWN_TEXT>

通用抽取规则
1) 只在“提供的化合物ID列表”范围内匹配与输出；不要生成列表之外的ID。
2) ID 等价匹配（不区分大小写，忽略空格与标点）：
   - 常见但不限于这些前缀/表头：Example / Ex. / No. / Compound / Formula / 实施例 / 化合物；只要上下文表明它是最终/目标化合物编号，也应参与匹配。
   - 常见但不限于这些形式：数字（1）、(1)、Ex.1、No.1、编号1、罗马数字（I，IIa 等）；核心判断是其是否对应提供列表中的最终化合物。
   - 当 Markdown 中出现“1”“(1)”等别名时，需与提供的化合物ID列表做等价判断；输出的键使用“提供列表中的规范ID”（如"Example 1"或"Compound 1"），而不是 Markdown 中的别名。
   - 若提供列表中只有 "Example 1"，而 Markdown 写成 "1"、"(1)"、"Ex.1"、"No.1"、甚至 "Compound 1"，也应视为同一化合物，并输出键 "Example 1"。
   - 若 OCR/Markdown 来自连续页面，上页末尾的最终化合物标题可延续到下页开头的结果表格或结构说明；但仍必须映射到“提供的化合物ID列表”中的某一项。
   - Intermediate / Int. / Preparation 是合成中间体，Embodiment 也不是本项目需要的目标最终化合物；不要输出这些编号，也不要把它们的活性值匹配给 Example。
3) 表格解析优先级：
   a) 若恰好两列：第1列=化合物编号，第2列=“{{ASSAY_NAME}}”。
   b) 若多于两列：优先使用表头包含“{{ASSAY_NAME}}”的列作为取值列；ID 列使用表头含“ID/编号/Example/Ex./No./Compound/Formula/实施例/化合物”等字样的列。
   c) 若无表头或表头含糊：按列对成对解析（奇数列为ID、其后一列为该ID的“{{ASSAY_NAME}}”）。
   d) 化合物 ID 单元格必须按“整格/整行标签”读取。不要从多位数 ID、带前缀/后缀 ID、脚注、序号或相邻单元格中截取单个数字作为 ID。
   e) 如果同一行可见 `Example 31`、`Ex. 31`、`Compound 31`、`31`、`31a`、`A-2`、`ENANT-2` 等完整标签，输出键必须对应完整标签的 allowlist 映射；不得只输出 `1`、`2` 或丢掉 `A-`/`ENANT-` 等有意义前缀。
4) 同一ID出现多次时：优先取与“{{ASSAY_NAME}}”表头最直接对应的那一行；若等同，取首次出现。
5) 提取值保留原始文本（如“<0.1”“ND”“1.2×10^3”），不要改动单位或数值格式。
6) 忽略与图/表/方案编号相关的数字，以及带单位但并非“{{ASSAY_NAME}}”单元格的数字（如 mg, mL, MHz, ppm, m/z, δ, % 等）。
7) 仅输出找到的键值对；若某ID未找到对应数值，则不写入结果。
8) 每个抽取项输出为带自检信息的对象：
   - `value`: 原始测定值文本。
   - `confidence`: `high` / `medium` / `low`，表示该 ID 行与 assay 列匹配的可靠性。
   - `reason`: 20 个词以内，说明使用了哪个 ID 单元格/行和哪个 assay 列。
   - 只有当 ID 与目标 assay 值能在同一行/同一记录中对应时才输出；若只能猜测，宁可不输出该项。

输出契约
仅输出 JSON 对象；键为规范化后的化合物 ID，值为抽取对象。格式如下：
```json
{
  "__COMPOUND_ID__": {
    "value": "__ASSAY_VALUE__",
    "confidence": "high|medium|low",
    "reason": "same row ID cell and target assay column"
  }
}
```

{{COMPOUND_ID_LIST_BLOCK}}
