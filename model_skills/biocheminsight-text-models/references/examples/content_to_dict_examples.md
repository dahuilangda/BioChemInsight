示例1：标准两列表格
输入片段：
| Compound | {{ASSAY_NAME}} |
| Compound 1 | 3.2 nM |
输出：
```json
{"Compound 1":{"value":"3.2","unit":"nM","method":"target assay","description":"","confidence":"high","reason":"Compound 1 row, target assay column"}}
```

示例2：别名需要映射到给定规范 ID
给定化合物ID列表：["Example 1","Example 2"]
输入片段：
| No. | {{ASSAY_NAME}} |
| (1) | 54 nM |
输出：
```json
{"Example 1":{"value":"54","unit":"nM","method":"target assay","description":"","confidence":"high","reason":"No. (1) row maps to Example 1"}}
```

示例3：跨前缀别名也要映射到给定规范 ID
给定化合物ID列表：["Example 1"]
输入片段：
| Compound | {{ASSAY_NAME}} |
| Compound 1 | 18 nM |
输出：
```json
{"Example 1":{"value":"18","unit":"nM","method":"target assay","description":"","confidence":"high","reason":"Compound 1 row maps to Example 1"}}
```

示例4：多列表格，只抽目标列
输入片段：
| Compound ID | {{ASSAY_NAME}} | Non-target property |
| Compound 7 | 0.18 uM | 12 mg/mL |
若目标字段是第一项 assay 列，则输出：
```json
{"Compound 7":{"value":"0.18","unit":"uM","method":"target assay","description":"","confidence":"high","reason":"Compound 7 row, target assay column"}}
```

示例5：未命中 allowlist 时不要输出
给定化合物ID列表：["Compound 1"]
输入片段：
| Compound | {{ASSAY_NAME}} |
| Compound 9 | 10 nM |
输出：
```json
{}
```

示例6：保留原始值文本
输入片段：
| Compound | {{ASSAY_NAME}} |
| Compound 3 | <0.1 |
输出：
```json
{"Compound 3":{"value":"<0.1","unit":"","method":"target assay","description":"","confidence":"high","reason":"Compound 3 row, target assay column"}}
```

示例7：不能截断多位数 ID
给定化合物ID列表：["Example 2","Example 31"]
输入片段：
| Example | {{ASSAY_NAME}} |
| Example 31 | 8.30 |
输出：
```json
{"Example 31":{"value":"8.30","unit":"","method":"target assay","description":"","confidence":"high","reason":"Example 31 full ID row"}}
```
