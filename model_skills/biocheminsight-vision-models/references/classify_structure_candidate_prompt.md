Task
Classify the candidate chemical image as exactly one type:
- `complete_compound`: one complete, definite molecule
- `markush`: explicit variable site or variable attachment structure
- `fragment`: incomplete, cropped, only a fragment, or only a substituent
- `noise`: not a single target structure
- `uncertain`: unclear or not stable to judge

Decision rules
1) Only `complete_compound` may set `is_complete_compound=true`.
2) Only explicit variability counts as `markush`: R/R1/R2/X/Y/Z, wildcard atom, wavy/variable bond, Ar/Het, or a variable definition such as `R1 = alkyl`.
3) Fixed substituent text, local numbering, salt names, atom symbols, stereochemical marks, or explanatory text alone do not make `markush`.
4) If any bond, atom, ring, substituent, or attachment is visibly clipped by the boundary, outside the frame, or only partial, classify as `fragment`.
5) A side chain, linker, isolated ring, partial scaffold, cut fragment, or local substituent is `fragment` even when it is clearly drawn.
6) Classify as `complete_compound` only when there is one complete and definite target molecule and the main chemical content is visible.
7) Multiple unrelated structures, reaction arrows, tables, pure text, legends, headers/footers, or noise are `noise`.
8) If completeness is uncertain, do not allow `complete_compound`.

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
