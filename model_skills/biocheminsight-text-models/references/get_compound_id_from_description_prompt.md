任务：从下方自由文本中抽取最终化合物编号（Compound ID）。输入可能来自 OCR、模型推理、人工说明或混合文本；请严格按规则只返回一个最终 ID。

输入：
{{DESCRIPTION}}

输出要求（必须同时满足）：
- 仅输出一行合法 JSON：键固定为 COMPOUND_ID。
- 同时输出 `CONFIDENCE`（`high|medium|low`）和 `REASON`（20 个词以内）。
- 若无法确定或不存在，返回 "None"（字符串）。
- 禁止输出占位符、空值、解释文字、前后空白、代码块或多余字符；绝不能输出 "__ID__"。
- 结果自检：若结果为空、为占位符、或包含“不确定/未知/unknown/maybe/possible/疑似”等词，改为 "None"。
- 若只能猜测最终 ID，则把 `COMPOUND_ID` 改为 `"None"`，`CONFIDENCE` 设为 `low`。

候选定义（先判非法，后选合法）：
【合法ID（正向模式，区分大小写与空格保持原文）】
1) 含最终/目标化合物关键词的形式（优先级高于纯数字类；以下只是常见示例，不是封闭列表）：
   - 英文：Example 12 / Ex. 12 / No. 12 / Compound 12 / Formula 12 等
   - 中文：实施例12 / 化合物12 等
   - 标题带说明：如 “Compound 3 (Hydrochloride Salts of Compound 1)” —— 仅取核心ID“Compound 3”
2) 局部/行首短标签（仅在表格行首或结构近旁作为最终/目标化合物本地标签时视为合法；以下只是常见示例）：
   - 12 / (12) / Ex.12 / No.12 / 编号12 / 12a / 12A / I / II / IIa 等

【非法（硬性排除，命中则绝不作为ID）】
- 段落/页码/行号等：如 “[0159]”“[0001]”“1/21”“Page 3”
- 图表编号：Figure/图、Table/表、Scheme/反应式 等
- 非目标编号：Intermediate 12 / Int. 12 / Preparation 12 / Embodiment 12 等不作为最终 Compound ID 输出；若只有这类候选，返回 "None"
- 含单位或分析上下文：mg、mL、MHz、ppm、m/z、δ、% 及各类谱图/条件描述
- 普通有序/无序列表编号（非表格行首标签）
- 任何仅为占位符或模板（如 “__ID__”）

选择与消歧（按顺序执行，命中即停）：
A. 若存在显式答案行（优先识别这些前缀，不区分大小写）：“Answer:”“Final answer:”“答案：”“输出：”
   - 取该行（或其后两行内）出现的首个【合法ID】。
B. 若无显式答案行：在全文中抽取全部【合法ID】，按以下优先级择一：
   1) 含关键词形式（Example/Ex./No./Compound/Formula/实施例/化合物）优先于纯数字/(n)/字母数字标签；
   2) 在同一优先级内，选择文末出现的最后一个（更可能是结论）。
   3) 如果文本来自跨页/连续页面，上页末尾的最终化合物标题（如 Example/Ex./No./Compound/Formula）可延续到下页开头的结构或结果描述；但若下页/当前局部有更近且有效的最终化合物编号，则优先更近编号。
   4) 如果同一数字同时出现多个关键词前缀（如 “Example 7” 与 “Intermediate 7”/“Embodiment 7”），只可选择明确作为最终/答案的 Example/Ex./No./Compound/Formula；不得输出 Intermediate 或 Embodiment。若上下文不能唯一确定，返回 "None"。
C. 若仅出现非法候选或无候选，则返回 "None"。

归一化与格式：
- 若命中“标题+说明”，仅保留核心ID（如 “Compound 3 (… )”→“Compound 3”）。
- 保留原文中明确出现的有效前缀类别（包括 Ex. 与 No.）；不得把 Intermediate/Embodiment 改写成 Example，也不得输出 Intermediate/Embodiment。
- 去除ID前后的标点与多余空白；其余大小写与内部空格保持原文。
- 仅输出：{"COMPOUND_ID":"<最终ID或None>","CONFIDENCE":"high|medium|low","REASON":"short reason"}

示例（仅作理解，不要在输出中复现）：
- “Answer: Compound 2” → 输出：{"COMPOUND_ID":"Compound 2","CONFIDENCE":"high","REASON":"explicit answer line"}
- 文末独立一行 “Compound 3” 且上文出现 “[0159]” → 输出：{"COMPOUND_ID":"Compound 3","CONFIDENCE":"medium","REASON":"valid final ID in local context"}
- 全文只有 “[0007]”“Figure 5” 等 → 输出：{"COMPOUND_ID":"None","CONFIDENCE":"low","REASON":"no valid compound ID found"}

现在请基于以上规则给出最终结果；除目标 JSON 外不要输出任何多余字符。
