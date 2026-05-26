示例1：
ALLOWLIST：["Example 1","Example 2"]
RAW_ID：1
输出：
```json
{"COMPOUND_ID":"Example 1","CONFIDENCE":"high","REASON":"unique Example-style alias match"}
```

示例2：
ALLOWLIST：["Example 1"]
RAW_ID：Compound 1
输出：
```json
{"COMPOUND_ID":"Example 1","CONFIDENCE":"high","REASON":"only allowlist target matches alias"}
```

示例3：
ALLOWLIST：["Compound 12A"]
RAW_ID：(12a)
输出：
```json
{"COMPOUND_ID":"Compound 12A","CONFIDENCE":"high","REASON":"case-insensitive OCR variant matches allowlist"}
```

示例4：
ALLOWLIST：["Example 1","Compound 1"]
RAW_ID：1
输出：
```json
{"COMPOUND_ID":"None","CONFIDENCE":"low","REASON":"ambiguous between multiple allowlist entries"}
```

示例5：
ALLOWLIST：["Example 8"]
RAW_ID：[0008]
输出：
```json
{"COMPOUND_ID":"None","CONFIDENCE":"low","REASON":"paragraph marker is not a compound alias"}
```
