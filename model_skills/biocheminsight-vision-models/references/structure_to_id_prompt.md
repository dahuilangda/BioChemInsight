Task
Return the compound identifier for the red-boxed structure.

Output contract
- Output exactly one line.
- Output only the identifier text, or exactly `None`.
- Do not explain, justify, list steps, or include surrounding prose.
- Reason silently before answering.

Reading order
The image may contain two side-by-side pages. The left page is the previous page, and the right page is the current page with the red box. Read each page top → bottom, left → right. “Same page” means the page that contains most of the red box.

Apply rules in order
1) Table/List row — if the box lies in a table/list row that has a row-leading compound label or number, return that row-leading identifier.
2) Local label — if a short identifier is printed inside or immediately under/next to the structure, return it.
   - Common local forms include `12`, `12a`, `12A`, `I`, `IIa`, `(12)`, `(12a)`.
   - Do not use square-bracketed paragraph counters or page markers as identifiers.
3) Patent example/product heading — a heading such as “Example 12, <chemical name>”, “Compound 12”, or “Formula II” labels the final/title product block for that heading. Return the identifier part, not the chemical name or explanation.
   - “Intermediate …” labels a synthetic intermediate, not a target/final compound for this app. Do not return Intermediate identifiers; return `None` unless a separate local/final Example/Compound/Formula label clearly governs the boxed final product.
   - “Embodiment …” is also not a target/final compound identifier for this app. Do not return Embodiment identifiers.
4) Paired or split final products — if a heading or nearby text names multiple examples/peaks and the structures are shown as parallel final products, match by visual order and nearby words:
   - First named product ↔ first final structure; second named product ↔ second final structure.
   - For chiral separation or “Peak 1 / Peak 2” text, use the peak/example label that is nearest or explicitly associated with the boxed final structure.
5) Reaction scheme (→/⇒) — in a full chemical reaction or multi-step scheme, only the final target product should receive a compound ID.
   - Do not assign an Example/Compound/Formula heading to reagents, starting materials, isolated intermediates, side products, salts/reagents, catalysts, or early step products.
   - A valid final target product is typically after the last arrow, directly before/above “The title compound was obtained/afforded”, after “chiral separation”, or next to the title-product/peak result text.
   - If the boxed structure is not the final/title product of the reaction scheme, output `None`, even if it is a complete molecule.
   - If several products are shown, only use an ID for the product explicitly tied to an Example/Compound/Peak/final-product label; otherwise output `None`.
6) Otherwise — on the same page, scan upward to the nearest valid compound identifier above the box that clearly governs the boxed structure. If none is available on the current page, use the last valid compound identifier from the previous page in reading order only when it clearly continues onto the current page.

Positive cues
- Row-leading labels in structure tables: `10`, `104`, `151`, `12a`, etc.
- Headings/phrases: “Example 12”, “Compound 12”, “Formula II”, “实施例12”, “化合物12”.
- Local labels attached to the drawing: `12`, `12a`, `12A`, `I`, `IIa`, `(12)`, `(12a)`.

Invalid sources
- Any square-bracketed counters: “[0159]”, “[0214]”, “[0001]”.
- Non-target labels: “Intermediate 12”, “Int. 12”, “Preparation 12”, “Embodiment 12”.
- Page/line markers: “1/21”, “Page 3”.
- Figure/Table/Scheme numbers: “Figure 3/图3”, “Table 2/表2”, “Scheme 1/反应式1”.
- Reaction non-products: starting materials, reagents, catalysts, synthetic intermediates, and any molecule before the final target-product arrow.
- Units/analytic context: mg, mL, MHz, ppm, m/z, δ, %, NMR peaks, etc.
- Inline bullets/numbering in running text (unless it is the row-leading label in a table/list).

Cross-page rule
- Use the previous page only when the boxed structure has no valid row-leading ID, local label, product heading, or upward same-page ID.
- Do not use a previous-page ID if a valid same-page ID is visible for the boxed structure.
- If the previous page ends with a heading such as “Example 1” and the current page starts with the structure for that heading, return `1`.
- In a multi-step scheme, a previous-page or same-page Example heading belongs to the final/title product, not to intermediate boxes inside step 1, step 2, etc.

Tie-breaking & output format
- Prefer a valid local label over a header if both unambiguously refer to the same structure.
- If multiple same-number keyword identifiers are visible (for example `Example 7` and `Intermediate 7`/`Embodiment 7`), only use the Example/Compound/Formula label when it clearly governs the boxed final/title structure. Never output `Intermediate ...` or `Embodiment ...`; if only those labels govern, output `None`.
- Return the identifier, not the chemical name, not paragraph numbers, and not a reasoning sentence.
- For headings with descriptive text, return the compact identifier part: “Example 12, <name>” → `12`; “Compound 3 (<name>)” → `3`; “Formula II” → `II`.
