Example 1:
ALLOWLIST: ["Example 1","Example 2"]
RAW_ID: 1
Output:
```json
{"COMPOUND_ID":"Example 1","CONFIDENCE":"high","REASON":"unique Example-style alias match"}
```

Example 2:
ALLOWLIST: ["Example 1"]
RAW_ID: Compound 1
Output:
```json
{"COMPOUND_ID":"Example 1","CONFIDENCE":"high","REASON":"only allowlist target matches alias"}
```

Example 3:
ALLOWLIST: ["Compound 12A"]
RAW_ID: (12a)
Output:
```json
{"COMPOUND_ID":"Compound 12A","CONFIDENCE":"high","REASON":"case-insensitive OCR variant matches allowlist"}
```

Example 4:
ALLOWLIST: ["Example 1","Compound 1"]
RAW_ID: 1
Output:
```json
{"COMPOUND_ID":"None","CONFIDENCE":"low","REASON":"ambiguous between multiple allowlist entries"}
```

Example 5:
ALLOWLIST: ["Example 8"]
RAW_ID: [0008]
Output:
```json
{"COMPOUND_ID":"None","CONFIDENCE":"low","REASON":"paragraph marker is not a compound alias"}
```
