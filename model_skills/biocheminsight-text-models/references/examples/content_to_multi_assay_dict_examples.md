Example 1: extract two assays from one table
Target assay field list: ["Target A assay value (nM)","Target B assay value (nM)"]
Input fragment:
| Compound | Target A assay value (nM) | Target B assay value (nM) |
| Compound 1 | 3.2 | 118 |
Output:
```json
{
  "Target A assay value (nM)": {
    "Compound 1": {
      "value": "3.2",
      "unit": "nM",
      "method": "Target A assay value",
      "description": "",
      "confidence": "high",
      "reason": "same row ID and Target A column",
      "assay_match": {
        "target": "Target A assay value (nM)",
        "candidate": "Target A assay value (nM)",
        "compatible": true,
        "best_requested_assay": "Target A assay value (nM)",
        "reason": "selected column matches requested assay"
      }
    }
  },
  "Target B assay value (nM)": {
    "Compound 1": {
      "value": "118",
      "unit": "nM",
      "method": "Target B assay value",
      "description": "",
      "confidence": "high",
      "reason": "same row ID and Target B column",
      "assay_match": {
        "target": "Target B assay value (nM)",
        "candidate": "Target B assay value (nM)",
        "compatible": true,
        "best_requested_assay": "Target B assay value (nM)",
        "reason": "selected column matches requested assay"
      }
    }
  }
}
```

Example 2: output an empty object when an assay is absent
Target assay field list: ["Primary assay value","Secondary assay value"]
Input fragment:
| Compound | Primary assay value |
| Compound 7 | 0.18 uM |
Output:
```json
{
  "Primary assay value": {
    "Compound 7": {
      "value": "0.18",
      "unit": "uM",
      "method": "Primary assay value",
      "description": "",
      "confidence": "high",
      "reason": "same row ID and primary assay column",
      "assay_match": {
        "target": "Primary assay value",
        "candidate": "Primary assay value",
        "compatible": true,
        "best_requested_assay": "Primary assay value",
        "reason": "selected column matches requested assay"
      }
    }
  },
  "Secondary assay value": {}
}
```
