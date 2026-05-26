Task
Return structured identifier metadata for the red-boxed structure.

Output contract
- Output JSON only, exactly one object.
- Do not include markdown fences, explanations, lists, or surrounding prose.
- Reason silently before answering.
- Required keys:
  - `COMPOUND_ID`: identifier text for the red-boxed structure, or the string `"None"` when no valid identifier belongs to it.
  - `VISUAL_ROLE`: one of `final_product`, `product`, `intermediate`, `reactant`, `reagent_or_condition`, `table_entry`, `unknown`.
  - `ID_SOURCE`: one of `local_label`, `row_label`, `heading`, `cross_page_heading`, `peak_or_enantiomer_label`, `none`.
  - `EVIDENCE`: short visual reason, 20 words or fewer.
  - `CONFIDENCE`: one of `high`, `medium`, `low`.
- Always include all five keys. If confidence is unclear, set `CONFIDENCE` to `medium`.

Reading order
The image may contain two side-by-side pages. The left page is the previous page, and the right page is the current page with the red box. Read each page top → bottom, left → right. “Same page” means the page that contains most of the red box.

Apply this visual decision process
1) Anchor on the red box.
   - Decide which single molecule is being asked about. Ignore labels, arrows, headings, and molecules that do not visually belong to that boxed structure.
   - If the red box spans a row/cell, use the row/cell boundary to determine which text belongs to the boxed molecule.
2) Read an identifier that is physically attached to the boxed structure.
   - Use a row-leading ID, cell header, caption, local label above/below/beside the drawing, or text printed inside the same visual group when it clearly belongs to the boxed molecule.
   - Return the visible identifier exactly, including meaningful prefixes/suffixes/hyphens/Roman numerals/parentheses.
   - Preserve the entire printed local/row label as one identifier. Do not strip meaningful prefixes, suffixes, or hyphenated parts: `A-2`, `31-B`, `204-III`, `ENANT-2`, `Peak 2`, and `(R)-12` must not be shortened to `2`, `31`, `204`, or `12`.
   - For multi-digit labels, read all digits in the same visual label. Never output a single digit copied from inside `31`, `101`, `A-2`, or another longer label.
   - Do not infer chemical role from identifier shape. A numeric, alphabetic, hyphenated, or Roman-suffix ID can refer to a starting material, intermediate, product, salt, enantiomer, table entry, or final compound; the role comes from visual context, not the string pattern.
   - Short labels with letters, digits, punctuation, hyphens, suffixes, or segmented numbering can be valid local labels when printed as structure captions or row labels. Do not discard or reinterpret them because they look like step numbers, list numbers, or shorthand; decide by visual attachment and reaction/table grouping.
   - A local label identifies which molecule is boxed; it does not by itself prove `product` or `final_product`. Determine `VISUAL_ROLE` separately from arrows, table grouping, product/result text, and nearby prose.
   - Do not treat reaction conditions, solvent/reagent names, temperatures, times, yields, masses, equivalents, analytical values, or arrow labels as compound identifiers even if they are close to the boxed molecule.
3) Determine whether a heading governs the boxed structure.
   - A patent/example/product heading governs only structures in its visual block and reaction role. Use it for the red-boxed structure only when there is no more specific local/row label and the boxed structure is the title/final product for that heading.
   - Do not borrow a broad heading for a different molecule in the same scheme. If a reactant/intermediate has its own local label, use that label. If it has no own label and the heading belongs to the later product, output `None`.
4) Use reaction topology for role and scope.
   - Trace arrows visually. Reactants are before arrows, intermediates are between arrows, products are after arrows; a cross-page arrow means the product may be on the next page.
   - Arrows can wrap across lines in dense schemes: an arrow leaving the far right of one line may continue at the far left of the next line, sometimes with only vertical spacing or repeated reaction text showing continuation. Preserve this wrapped reading order before deciding that a right-edge structure is a final product.
   - Before outputting `product` or `final_product`, check whether any visible nearby prose, caption, step description, or reaction continuation says the boxed molecule/local label is an intermediate, precursor, starting material, or material used in a later step.
   - A boxed molecule after the last arrow, next to “title compound/product/obtained/afforded/Peak/enantiomer” text, or inside a product-result block can inherit the product heading if it has no local label.
   - Use `final_product` when title/target-compound evidence is clear. Use `product` when the boxed molecule is a reaction product but the image does not prove whether it is the final/title target compound.
   - A boxed molecule before or between arrows should not inherit a final-product heading unless the text/label explicitly attaches that ID to this molecule.
   - Text that describes the boxed molecule, its local label, or the current synthetic step as an intermediate/precursor means the boxed molecule is not the final/title product even if it is near an example heading.
   - Protecting groups, masked functional groups, temporary salts, deprotection/hydrolysis/coupling setup, and workup text are role/context cues for synthetic intermediates. Use them to avoid borrowing a final-product heading, but never use the chemical group or reagent text as an ID.
   - If multiple products or peaks are shown, match IDs by visual order, proximity, and explicit Peak/enantiomer labels; do not use a nearby ID from the other product.
5) Use cross-page context only for continuation.
   - The left page may be the previous page. Use it only when the current page lacks a local/row/product ID and the previous page clearly continues the same heading or reaction onto the current page.
   - Do not use a previous-page heading if the boxed structure has its own current-page label or belongs to a different reaction role.
   - For split tables or schemes, preserve row/cell/reaction grouping across the page break; do not simply choose the nearest visible text if it belongs to another row, column, or molecule.
6) If still ambiguous, output `None`.
   - Prefer `None` over borrowing a plausible but ungrounded ID from a nearby heading, paragraph number, scheme number, or different molecule.

Positive cues
- Row-leading labels in structure tables or scheme lists.
- Local labels attached to the drawing, including numeric, alphabetic, hyphenated, Roman-numeral, parenthesized, prefixed, peak, salt, or stereoisomer labels.
- Headings/phrases such as Example/Ex./No./Compound/Formula/实施例/化合物 only when they visually govern the boxed structure.
- Product cues such as the last arrow product, title-compound text, obtained/afforded text, chiral separation products, peak labels, or product-result paragraphs.
- Intermediate cues such as explicit intermediate/precursor wording, a protected or masked precursor near reaction arrows, a labeled species before a later product, or a wrapped scheme that continues after the boxed molecule; these set `VISUAL_ROLE` to `intermediate` and prevent inheriting a final-product heading unless a local label explicitly belongs to the molecule.

Invalid sources
- Any square-bracketed counters: “[0159]”, “[0214]”, “[0001]”.
- Non-target labels: “Intermediate 12”, “Int. 12”, “Preparation 12”, “Embodiment 12”.
- Page/line markers: “1/21”, “Page 3”.
- Figure/Table/Scheme numbers: “Figure 3/图3”, “Table 2/表2”, “Scheme 1/反应式1”.
- Reaction headings that govern a different molecule: do not borrow the final product heading for another reactant/intermediate/product.
- Cross-page reaction continuations: if an arrow after the boxed molecule points off the bottom/right edge or continues onto the next page, do not label the boxed molecule as the next-page product.
- Units/analytic context: mg, mL, MHz, ppm, m/z, δ, %, NMR peaks, etc.
- Reaction conditions and arrow annotations: solvents, bases, acids, catalysts, temperatures, times, equivalents, yields, step labels, or workup instructions.
- Protecting-group, reagent, workup, deprotection, hydrolysis, coupling, salt-formation, or isolation text is chemical context, not an identifier.
- Inline bullets/numbering in running text (unless it is the row-leading label in a table/list).

Cross-page rule
- Use the previous page only when the boxed structure has no valid row-leading ID, local label, product heading, or upward same-page ID.
- Do not use a previous-page ID if a valid same-page ID is visible for the boxed structure.
- If the previous page ends with a final/target compound heading such as “Example 1”, “Ex. 1”, “No. 1”, “Compound 1”, or “Formula I”, and the current page starts with the structure/product/result text for that heading because the example did not fit on one page, return that identifier.
- This cross-page continuation also applies when the heading is at the bottom of the previous page and the reaction/product structure, title compound paragraph, or first product drawing begins at the top of the current page.
- In a multi-step scheme, a previous-page or same-page Example heading belongs to the final/title product, not to intermediate boxes inside step 1, step 2, etc.
- For wrapped multi-line schemes, continue arrow topology from the right edge of one line to the left edge of the next line before assigning final-product status. A structure at the far right or far left is not automatically final; it is final only if the topology and product text show the route stops there.

Tie-breaking & output format
- Prefer a valid local/row label over a broader heading.
- If multiple same-number keyword identifiers are visible (for example `Example 7`/`Ex. 7`/`No. 7` and `Intermediate 7`/`Embodiment 7`), only use the Example/Ex./No./Compound/Formula label when it clearly governs the boxed final/title structure. Never output `Intermediate ...` or `Embodiment ...`; if only those labels govern, output `None`.
- If the red-boxed structure has a visible non-keyword local label, return that label exactly; use visual context only to avoid borrowing a different parent/final-product ID.
- Put only the identifier in `COMPOUND_ID`, not the chemical name, not paragraph numbers, and not a reasoning sentence.
- Preserve meaningful printed prefixes/suffixes when they are part of the visible identifier: “Compound 10” → `Compound 10`, not `10`; “Formula II” → `Formula II`; a printed suffix such as `-II`, `-IV`, `(R)`, `Peak 2`, or a salt/stereoisomer marker should be kept when it identifies the boxed molecule.
- Preserve meaningful local-label prefixes as well: “A-2” → `A-2`, “ENANT-2” → `ENANT-2`, “31-B” → `31-B`; do not output only the trailing number.
- If no valid identifier belongs to the boxed structure, set `COMPOUND_ID` to `"None"`, `ID_SOURCE` to `none`, and explain the visual ambiguity in `EVIDENCE`.
