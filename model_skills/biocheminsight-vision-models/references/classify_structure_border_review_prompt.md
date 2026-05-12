Re-check the same candidate image with a stricter cropping rule.

Hard rule:
- If any bond, atom, ring, label, or substituent is cut off by the image boundary or visibly exits the frame, this is fragment, not complete_compound.
- If any R-group or variable placeholder is present, this is markush.
- If the image is mostly text/arrows/layout artifacts, this is noise.
- Only choose complete_compound when the whole exact molecule is fully visible.
- Do not confuse compound numbers, example labels, salts/counterions, atom symbols, or nearby local captions with Markush placeholders.
- In dense patent panels, one dominant exact molecule with nearby labels can still be complete_compound if no chemistry content is cut off.

Local cue:
- Border-contact heuristic detected drawing content touching these sides: {{BORDER_SIDES}}.
- This cue is not proof of truncation. Use it to inspect the edge carefully, then accept complete_compound when the whole molecule is still visible in a tight crop.

Return JSON only:
{"structure_type":"complete_compound|markush|fragment|noise|uncertain","is_complete_compound":true,"reason":"short reason"}
