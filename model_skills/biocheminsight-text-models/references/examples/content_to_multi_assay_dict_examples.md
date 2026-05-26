示例1：同一表同时抽两个 assay
目标测定字段列表：["CDK2/cyclin E IC50 (nM)","CDK1/Cyclin B1 IC50 (nM)"]
输入片段：
| Compound | CDK2/cyclin E IC50 (nM) | CDK1/Cyclin B1 IC50 (nM) |
| Compound 1 | 3.2 | 118 |
输出：
```json
{
  "CDK2/cyclin E IC50 (nM)": {
    "Compound 1": {"value":"3.2","confidence":"high","reason":"Compound 1 row, CDK2 column"}
  },
  "CDK1/Cyclin B1 IC50 (nM)": {
    "Compound 1": {"value":"118","confidence":"high","reason":"Compound 1 row, CDK1 column"}
  }
}
```

示例2：别名要映射到给定规范 ID
目标测定字段列表：["NanoBRET CDK2/Cyclin E1 IC50 (nM)"]
给定化合物ID列表：["Example 1","Example 2"]
输入片段：
| No. | NanoBRET CDK2/Cyclin E1 IC50 (nM) |
| (1) | 54 nM |
输出：
```json
{
  "NanoBRET CDK2/Cyclin E1 IC50 (nM)": {
    "Example 1": {"value":"54 nM","confidence":"high","reason":"No. (1) row maps to Example 1"}
  }
}
```

示例3：某 assay 未出现则输出空对象
目标测定字段列表：["IC50","Kd"]
输入片段：
| Compound | IC50 |
| Compound 7 | 0.18 uM |
输出：
```json
{
  "IC50": {
    "Compound 7": {"value":"0.18 uM","confidence":"high","reason":"Compound 7 row, IC50 column"}
  },
  "Kd": {}
}
```

示例4：不能截断多位数 ID
目标测定字段列表：["Calcitonin Receptor Assay 1 EC50"]
给定化合物ID列表：["Example 2","Example 31"]
输入片段：
| Example | Calcitonin Receptor Assay 1 EC50 |
| Example 31 | 8.30 |
输出：
```json
{
  "Calcitonin Receptor Assay 1 EC50": {
    "Example 31": {"value":"8.30","confidence":"high","reason":"Example 31 full ID row"}
  }
}
```
