示例1：同一表同时抽两个 assay
目标测定字段列表：["Target A assay value (nM)","Target B assay value (nM)"]
输入片段：
| Compound | Target A assay value (nM) | Target B assay value (nM) |
| Compound 1 | 3.2 | 118 |
输出：
```json
{
  "Target A assay value (nM)": {
    "Compound 1": {"value":"3.2","unit":"nM","method":"Target A assay value","description":"","confidence":"high","reason":"Compound 1 row, Target A column"}
  },
  "Target B assay value (nM)": {
    "Compound 1": {"value":"118","unit":"nM","method":"Target B assay value","description":"","confidence":"high","reason":"Compound 1 row, Target B column"}
  }
}
```

示例2：别名要映射到给定规范 ID
目标测定字段列表：["Cell-based target assay value (nM)"]
给定化合物ID列表：["Example 1","Example 2"]
输入片段：
| No. | Cell-based target assay value (nM) |
| (1) | 54 nM |
输出：
```json
{
  "Cell-based target assay value (nM)": {
    "Example 1": {"value":"54","unit":"nM","method":"Cell-based target assay value","description":"","confidence":"high","reason":"No. (1) row maps to Example 1"}
  }
}
```

示例3：某 assay 未出现则输出空对象
目标测定字段列表：["Primary assay value","Secondary assay value"]
输入片段：
| Compound | Primary assay value |
| Compound 7 | 0.18 uM |
输出：
```json
{
  "Primary assay value": {
    "Compound 7": {"value":"0.18","unit":"uM","method":"Primary assay value","description":"","confidence":"high","reason":"Compound 7 row, primary assay column"}
  },
  "Secondary assay value": {}
}
```

示例4：不能截断多位数 ID
目标测定字段列表：["Target assay value"]
给定化合物ID列表：["Example 2","Example 31"]
输入片段：
| Example | Target assay value |
| Example 31 | 8.30 |
输出：
```json
{
  "Target assay value": {
    "Example 31": {"value":"8.30","unit":"","method":"Target assay value","description":"","confidence":"high","reason":"Example 31 full ID row"}
  }
}
```
