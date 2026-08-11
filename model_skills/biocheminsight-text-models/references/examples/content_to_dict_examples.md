Example 1: standard single-assay row
Input fragment:
| Compound | {{ASSAY_NAME}} |
| Compound 1 | 3.2 nM |
Output:
```json
{
  "Compound 1": {
    "value": "3.2",
    "unit": "nM",
    "method": "target assay",
    "description": "",
    "confidence": "high",
    "reason": "same row ID and selected assay column",
    "assay_match": {
      "target": "{{ASSAY_NAME}}",
      "candidate": "{{ASSAY_NAME}}",
      "compatible": true,
      "best_requested_assay": "{{ASSAY_NAME}}",
      "reason": "selected column matches requested assay"
    }
  }
}
```

Example 2: alias maps to allowlist
Given compound ID list: ["Example 1"]
Input fragment:
| No. | {{ASSAY_NAME}} |
| (1) | 54 nM |
Output:
```json
{
  "Example 1": {
    "value": "54",
    "unit": "nM",
    "method": "target assay",
    "description": "",
    "confidence": "high",
    "reason": "alias row maps to allowlist ID",
    "assay_match": {
      "target": "{{ASSAY_NAME}}",
      "candidate": "{{ASSAY_NAME}}",
      "compatible": true,
      "best_requested_assay": "{{ASSAY_NAME}}",
      "reason": "selected column matches requested assay"
    }
  }
}
```

Example 3: no output when allowlist does not match
Given compound ID list: ["Compound 1"]
Input fragment:
| Compound | {{ASSAY_NAME}} |
| Compound 9 | 10 nM |
Output:
```json
{}
```
