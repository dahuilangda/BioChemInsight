任务
从提供的 Markdown / OCR整理文本中，同时抽取多个目标测定字段，并输出为嵌套字典。

输入
目标测定字段列表（必须严格使用这些名字作为顶层键）：
{{ASSAY_NAMES_JSON}}

<MARKDOWN_TEXT>
{{MARKDOWN_TEXT}}
</MARKDOWN_TEXT>

通用抽取规则
1) 只在“提供的化合物ID列表”范围内匹配与输出；不要生成列表之外的ID。
   - 若未提供化合物ID列表，则从文本中的最终化合物/Example/Compound ID 行中抽取，不要抽取中间体、制备例或表格序号。
   - 未提供化合物ID列表时，仍必须抽取页面内可见 assay 表格的 Example / Compound / No. / ID 单元格作为 JSON key；不要因为缺少 allowlist 就返回空对象。后续系统会再做规范 ID 校验。
2) ID 等价匹配（不区分大小写，忽略空格与标点）：
   - 常见但不限于这些前缀/表头：Example / Ex. / No. / Compound / Formula / 实施例 / 化合物；只要上下文表明它是最终/目标化合物编号，也应参与匹配。
   - 常见但不限于这些形式：数字（1）、(1)、Ex.1、No.1、编号1、罗马数字（I，IIa 等）；核心判断是其是否对应提供列表中的最终化合物。
   - 当 Markdown 中出现“1”“(1)”等别名时，需与提供的化合物ID列表做等价判断；输出的键使用“提供列表中的规范ID”。
   - 若 OCR/Markdown 来自连续页面，上页末尾的最终化合物标题可延续到下页开头的结果表格或结构说明；但仍必须映射到“提供的化合物ID列表”中的某一项。
   - Intermediate / Int. / Preparation 是合成中间体，Embodiment 也不是本项目需要的目标最终化合物；不要输出这些编号，也不要把它们的活性值匹配给 Example。
   - 不得按行序号生成 `Example 1`、`Example 2` 等 key。只有当单元格本身写着 `Example 1` 时才可输出该 key；如果 `Example` 表头下的单元格是 `58`，key 就是 `58` 或 allowlist 中与 `58` 等价的规范 ID。
3) Assay 名称匹配：
   - 目标测定字段列表是用户要抽取的 assay 概念，不要求表头逐字相同。
   - 对每个目标 assay，分别拆分靶点/体系/方法/平台/终点/单位等语义要素来匹配表题、表头、脚注和正文；不要依赖某个固定关键词。
   - 若目标名较宽泛，应结合页面上下文和同一表格内的候选列判断最接近的实际测量结果，而不是因为表头未逐字写出完整目标名就返回空对象。
   - 若两个目标 assay 名相近，也必须逐个独立判断并保留所有请求的顶层键。不要因为一个 assay 已经抽取了某列，就省略另一个目标 assay。
   - 若两个目标 assay 经判断确实对应同一列，可以在两个 assay 顶层键下分别输出相同 ID/value，但 `method` 和 `reason` 需说明各自匹配依据。
   - 对每个目标 assay 独立做兼容性自检：把目标 assay 名和候选表题/表头/脚注分别拆成 target biology、method/platform、endpoint、unit、context。只有候选列/记录与该目标 assay 概念兼容时才写入该 assay。
   - 若某个目标 assay 和候选列/记录在 method/platform、endpoint 或 biological context 上明确冲突，则该 assay 输出 `{}`；不要为了填满所有 assay 而把同一列复制到不匹配的 assay 下。
   - 若目标 assay 缺少某个要素，允许使用同一表题、脚注、页面标题或用户提供 assay 名上下文补全；但补全后的候选仍必须和该目标 assay 兼容。
   - 对每个候选列/记录，必须先在目标测定字段列表中选择一个 `best_requested_assay`，它表示该候选最应该归属的请求 assay。只有当 `best_requested_assay` 与当前输出顶层 assay 名完全相同，且兼容性自检为 `true` 时，才允许把该项写入该顶层 assay。
4) 若表格中同一行同时含有多个目标测定字段，则应分别写入对应 assay 的子字典。
5) 多列表格时：
   - ID 列优先识别表头或上下文明确表示化合物编号的列。
   - 对于每个目标测定字段，优先匹配与该 assay 名最接近/最一致的表头列。
   - 若表头不逐字包含目标 assay 名，按第 3 条的语义要素选择最匹配列。
   - OCR 表格的表头行可能出现在表格顶部、底部、分页续表处或重复行中。若同一表格内某行明显包含化合物编号列标签与 assay 终点/单位/方法表头，应把它作为该表格列标签使用，不要因为表头在数据行之后就拒绝同表格数据。
   - 化合物 ID 单元格必须按“整格/整行标签”读取。不要从一个多位数 ID、带前缀/后缀 ID、行号、脚注或相邻单元格中截取单个数字作为 ID。
   - 如果同一行可见 `Example 31`、`Ex. 31`、`Compound 31`、`31`、`31a`、`A-2`、`ENANT-2` 等完整标签，输出键必须对应完整标签的 allowlist 映射；不得只输出 `1`、`2`、`31` 的一部分或丢掉 `A-`/`ENANT-` 等有意义前缀。
   - 对简单表格也要用表头、行边界、同列上下文做自检：ID 必须来自 ID 列，assay 值必须来自目标 assay 列；不要把 assay 数值、序号或脚注当作 ID。
   - `value` 必须来自 assay 测量值单元格，不得来自 Example/Compound/No./ID 单元格。若同一数字同时可能是 ID 和测量值，必须用表头和同一行位置确认它来自测量值列；不能确认时不要输出。
   - `Example` 是列名时，不能把行位置解释成 `Example 1/2/3`；必须读取该列当前行的真实单元格内容作为 compound ID。
6) 如果同一化合物在同一 assay 下出现多次：
   - 优先取与该 assay 表头最直接对应的单元格；
   - 若等同，则取首次出现。
7) 提取值通常保留原始文本（如“<0.1”“ND”“1.2×10^3”），不要改动单位或数值格式。
   - 但不要机械保留明显不符合该列值域的 OCR 噪声。先根据表头、单位、同列其它单元格、同一行其它 assay 值和正文说明判断该列的值域。
   - 对定性等级、符号等级、百分比、活性/效能/响应等级等列，如果某个单元格是孤立异常 glyph 或明显 OCR 误读，只有在上下文足以还原该单元格原意时才输出校正后的值；若无法可靠还原，保留原始 OCR 文本。
   - 不要把 OCR 异常字符当作新的 assay 语义；输出应符合该列在表格中实际使用的值域。
8) 忽略与图/表/方案编号相关的数字，以及带单位但并非 assay 单元格的数字（如 mg, mL, MHz, ppm, m/z, δ, % 等）。
9) 顶层必须包含所有目标测定字段；若某 assay 在当前文本中未找到任何结果，则该 assay 的值输出为空对象 `{}`。不得遗漏、改名或合并目标 assay 顶层键。
10) 每个抽取项输出为带自检信息的对象：
   - `value`: 核心测定值；数字值尽量与单位分开，符号/等级值保留原文。
   - `unit`: 单位；从表头、脚注、正文或单元格提取，未知则空字符串。
   - `method`: 实验方法/体系/靶点描述；优先来自表题、表头或目标 assay 名。
   - `description`: 符号/等级/非数字值的含义说明；无说明则空字符串。
   - `confidence`: `high` / `medium` / `low`，表示该 ID 行与 assay 列匹配的可靠性。
   - `reason`: 20 个词以内，说明使用了哪个 ID 单元格/行和哪个 assay 列。
   - `assay_match`: 自检对象，必须包含：
     - `target`: 简述目标 assay 的 target/method/platform/endpoint/unit。
     - `candidate`: 简述被抽取列/记录的 target/method/platform/endpoint/unit。
     - `compatible`: 布尔值；只有为 `true` 时才允许输出该项。
     - `best_requested_assay`: 必须严格等于目标测定字段列表中的一个原始字符串，表示候选列/记录最应该归属的请求 assay；只有它等于当前输出顶层 assay 名时才允许输出。
     - `reason`: 说明为什么候选 assay 与目标 assay 兼容。
   - 只有当 ID 与目标 assay 值能在同一行/同一记录中对应时才输出；若只能猜测，宁可不输出该项。
11) 仅输出 JSON，不要解释。

输出契约
仅输出 JSON 对象；顶层键为目标 assay 名，值为“规范化化合物 ID -> 抽取对象”的字典。格式如下：
```json
{
  "__ASSAY_NAME__": {
    "__COMPOUND_ID__": {
      "value": "__ASSAY_VALUE__",
      "unit": "__UNIT_OR_EMPTY__",
      "method": "__ASSAY_METHOD_OR_CONTEXT__",
      "description": "__SYMBOL_OR_VALUE_DESCRIPTION_OR_EMPTY__",
      "confidence": "high|medium|low",
      "reason": "same row ID cell and target assay column",
      "assay_match": {
        "target": "target/method/platform/endpoint/unit from requested assay",
        "candidate": "target/method/platform/endpoint/unit from selected column",
        "compatible": true,
        "best_requested_assay": "__ASSAY_NAME__",
        "reason": "candidate column matches requested assay concept"
      }
    }
  },
  "__ASSAY_NAME__": {}
}
```

{{COMPOUND_ID_LIST_BLOCK}}
