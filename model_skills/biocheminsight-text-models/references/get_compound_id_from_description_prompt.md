Task
Extract one final/target Compound ID from free text.
Return only one JSON object. If it cannot be determined, return `"None"`.

Input
{{DESCRIPTION}}

Valid ID evidence
- Explicit identifier governing the final or target compound.
- Local label in the same table row or attached to the target structure.
- Explicit answer lines such as `Answer:`, `Final answer:`, or `答案：` may be preferred.

Invalid sources
- Paragraph numbers, page numbers, row numbers, table/figure/scheme numbers, generic list numbers
- Intermediate / Int. / Preparation / Embodiment, unless context explicitly says it is also the final target ID
- Experimental values such as mg, mL, MHz, ppm, m/z, δ, %, conditions, spectra, or yield
- Placeholders, templates, or uncertain text such as unknown/maybe/possible

Selection rules
1) When an explicit answer line exists, choose the first valid ID on the answer line or within the following two lines.
2) When no answer line exists, prefer a final target ID with Example/Ex./No./Compound/Formula/实施例/化合物.
3) Among peer candidates, choose the one closer to the conclusion and clearer in context.
4) Do not rewrite Intermediate/Embodiment into Example by number.
5) If evidence is insufficient to uniquely support a final/target ID, return `"None"` and `CONFIDENCE="low"`.

Output contract
Return only JSON:
```json
{
  "COMPOUND_ID": "source identifier or None",
  "CONFIDENCE": "high|medium|low",
  "REASON": "short reason"
}
```
