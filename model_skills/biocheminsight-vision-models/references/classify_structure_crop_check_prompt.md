Decide whether the candidate chemical structure is cropped or truncated by the image boundary.

Answer fragment if any bond, atom, ring, substituent, wedge bond, label, or attachment point is visibly cut off by the image border or exits the frame.
Answer not_cropped if the entire exact molecule is fully visible, even when it sits close to the frame.
Do not call it cropped just because the structure is near an edge, inside a dense patent panel, or accompanied by nearby numbering/text, unless chemistry content is actually cut off.

Border-contact heuristic detected drawing content touching these sides: {{BORDER_SIDES}}.
Treat that heuristic only as a warning to inspect the border carefully. If the crop boundary is tight but all chemistry is fully visible, answer not_cropped.

Return JSON only:
{"crop_status":"fragment|not_cropped|uncertain","is_cropped":true,"reason":"short reason"}
