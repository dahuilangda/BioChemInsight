Task
Translate a compound name into its English form for an exact external database lookup.
This is a translation task only: never describe, draw, or invent a chemical structure.

Compound name
{{COMPOUND_NAME}}

Rules
1) Return the standard English chemical name for the same compound, in the nomenclature an authoritative database would index.
2) Preserve locants, stereo-descriptors, salt and counterion text exactly as the source states them.
3) If the source does not name a specific definite compound, return an empty `name` — refuse rather than guess.

Output contract
Return only JSON:
```json
{
  "name": "__ENGLISH_COMPOUND_NAME_OR_EMPTY__",
  "confidence": "high|medium|low",
  "reason": "short reason"
}
```
