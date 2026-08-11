Task
Decide whether the candidate structure is cropped or truncated by the image boundary.

Rules
1) If any bond, atom, ring, substituent, wedge bond, label, or attachment is clipped by the boundary or clearly outside the frame, output `fragment`.
2) When the entire definite molecule is visible, output `not_cropped`, even if it is very close to the boundary.
3) Being near an edge, inside a dense patent panel, or next to numbers/text is not enough to make `fragment`.
4) If you cannot confirm, output `uncertain`.

Local hint
- Border-contact heuristic detected drawing content touching these sides: {{BORDER_SIDES}}.
- This only tells you where to inspect the boundary carefully.

Output contract
Return only JSON:
```json
{
  "crop_status": "fragment|not_cropped|uncertain",
  "is_cropped": true,
  "confidence": "high|medium|low",
  "reason": "short reason"
}
```
