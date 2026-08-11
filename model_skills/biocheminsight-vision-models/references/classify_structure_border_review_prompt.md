Task
Review the same candidate image with stricter boundary criteria.

Boundary rules
1) If any bond, atom, ring, label, substituent, or attachment is clipped by the boundary or clearly outside the frame, classify as `fragment`.
2) Explicit R-group, wildcard atom, wavy/variable bond, or variable placeholder is `markush`.
3) Mostly text, arrows, tables, or layout elements are `noise`.
4) Classify as `complete_compound` only when the full complete and definite molecule is visible.
5) Do not mistake compound numbers, local titles, salts/counterions, fixed substituent text, or atom symbols for Markush placeholders.
6) Border contact alone is not enough to call a fragment. You must see actual chemical content clipped, or be unable to confirm completeness.

Local hint
- Border-contact heuristic detected drawing content touching these sides: {{BORDER_SIDES}}.
- This is only a review hint, not an automatic rejection condition.

Output contract
Return only JSON:
```json
{
  "structure_type": "complete_compound|markush|fragment|noise|uncertain",
  "is_complete_compound": true,
  "confidence": "high|medium|low",
  "reason": "short reason"
}
```
