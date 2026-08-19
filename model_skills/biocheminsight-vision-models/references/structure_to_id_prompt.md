Task
Return an auditable Compound ID and visual role for the red-boxed structure.
Use only visible visual evidence. Do not guess an ID from chemical intuition,
number shape, or neighboring records.

Output contract
Return only one JSON object:
```json
{
  "COMPOUND_ID": "visible id or None",
  "VISUAL_ROLE": "final_product|product|intermediate|reactant|reagent_or_condition|table_entry|unknown",
  "ID_SOURCE": "local_label|row_label|heading|cross_page_heading|peak_or_enantiomer_label|none",
  "EVIDENCE": "20 words or fewer",
  "CONFIDENCE": "high|medium|low"
}
```

Visual reading order
1) Anchor on the red box first and identify exactly which molecule is being asked about.
2) Determine the boxed molecule's position within its reaction flow BEFORE considering any
   heading: locate the nearest reaction arrow(s) and any plus signs. A molecule joined to
   another by "+" is a co-reactant. A molecule sits after an arrow only when an arrow
   points toward it with no further plus-joined partner between them.
3) Identify the visual group that contains the red box: same structure, same table row/cell, same reaction block, or same caption/heading block.
4) Search for an ID only inside that visual group. Do not borrow an ID from a neighboring row, neighboring structure, another product, or another peak.

ID evidence priority
1) Primary Compound ID in the same row or structure unit: Example / Ex. / No. / Compound / Formula / 实施例 / 化合物, etc.
2) Local label directly attached to the red-boxed molecule: number, letter, hyphenated label, Roman numeral, parenthesized label, or peak/enantiomer label.
3) A heading that clearly governs the red-boxed molecule, allowed only when the boxed
   molecule's positional role is the final product of the scheme (after the last arrow,
   no onward plus-join); a heading must not lend an ID-governed status to a reactant or
   intermediate position.
4) Cross-page heading only when no same-page ID is visible and the previous page is clearly continuous with the same heading, reaction, or table.
5) A heading names the record that the whole scheme produces. When the boxed molecule's
   positional role is the scheme's final product and a governing heading for that scheme
   is visible, attach the heading's ID with ID_SOURCE="heading". For any reactant or
   intermediate position the heading does not attach: output COMPOUND_ID="None" with
   ID_SOURCE="none" unless a same-molecule local ID is visible.

ID rules
1) Return the complete visible ID, preserving prefixes, suffixes, hyphens, parentheses, and Roman numerals.
2) Do not truncate `31`, `101`, `A-2`, `204-III`, `Peak 2`, or `(R)-12` into a single digit or suffix.
3) A local label only proves which molecule it is; it does not automatically prove `final_product`.
4) Continuation-table context may only identify which column is the ID column. The current-row ID value must be visible.
5) If no reliable ID belongs to the red-boxed molecule, output `COMPOUND_ID="None"` and `ID_SOURCE="none"`.

Invalid ID sources
- Paragraph numbers, page numbers, line numbers, table/figure/scheme numbers, footnotes
- Intermediate / Int. / Preparation / Embodiment numbers used as final compound IDs
- Reaction conditions, reagents, solvents, temperature, time, equivalents, yield
- Analytical or experimental values such as LCMS/NMR/m/z/ppm/mg/mL/%
- IDs that govern another molecule, row, product, peak, or enantiomer

Visual role
1) Use `table_entry` for a target record in a structure table row, unless the image clearly shows a reaction role.
2) Before a reaction arrow is `reactant`; between arrows is `intermediate`; after an arrow is `product`.
3) A plus sign joins reactants: a molecule connected to another by "+" is `reactant`, even
   when it stands between a first reactant and the product of its step, and even when a
   heading above the scheme names an embodiment.
4) In a multi-step scheme only the molecule after the LAST arrow with no onward plus-join
   is the final product; every earlier step's after-arrow molecule that reappears before a
   later arrow is an `intermediate`, and its co-reactants are `reactant`. Judge "last" by
   the whole visible scheme: check whether anything (arrow, plus, further structure)
   continues after the boxed molecule's group before deciding the role.
5) Heading evidence can name a record but NEVER overrides positional role: a heading does
   not turn a reactant or intermediate into `final_product`. The final-product position
   MAY take its ID from the governing heading when no local label exists.
6) If nearby text describes the molecule as an intermediate, precursor, starting material, or reagent for a later step, do not mark it as `final_product`.
7) Use `reagent_or_condition` for reaction conditions, reagents, or arrow text.
8) Use `unknown` when the role is unclear.

Cross-page / continuation
1) The left page may be used as previous-page context only for a continuous same table, reaction, or heading.
2) Do not use a previous-page ID when the current page has a reliable same-page ID.
3) A new table, new scaffold, new heading, different column layout, or different record block stops inheritance.

Final checks
- `COMPOUND_ID` and `ID_SOURCE` must be consistent: if there is no ID, use only `none`.
- `EVIDENCE` must describe visible evidence only, not speculation.
- When uncertain, prefer `"None"` or low confidence instead of borrowing a nearby ID.
