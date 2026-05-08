示例1：
ALLOWLIST：["Example 1","Example 2"]
RAW_ID：1
输出：
```json
{"COMPOUND_ID":"Example 1"}
```

示例2：
ALLOWLIST：["Example 1"]
RAW_ID：Compound 1
输出：
```json
{"COMPOUND_ID":"Example 1"}
```

示例3：
ALLOWLIST：["Compound 12A"]
RAW_ID：(12a)
输出：
```json
{"COMPOUND_ID":"Compound 12A"}
```

示例4：
ALLOWLIST：["Example 1","Compound 1"]
RAW_ID：1
输出：
```json
{"COMPOUND_ID":"None"}
```

示例5：
ALLOWLIST：["Example 8"]
RAW_ID：[0008]
输出：
```json
{"COMPOUND_ID":"None"}
```
