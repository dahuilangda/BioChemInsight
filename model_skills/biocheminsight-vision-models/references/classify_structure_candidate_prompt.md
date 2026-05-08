Task
Classify the candidate chemistry image into exactly one type:
- complete_compound: one complete, specific molecule suitable for exact structure recognition
- markush: variable groups such as R/R1/R2/X/Y/Z, wildcard positions, wavy/variable bonds, Ar/Het placeholders, or unspecified substituents
- fragment: only part of a molecule, truncated scaffold, substituent, linker, isolated ring, or cut-off structure
- noise: not a single complete molecule, such as arrows, tables, text, legends, page artifacts, or multiple unrelated items
- uncertain: ambiguous or too low quality to decide

General decision rules
1) Set is_complete_compound=true only for complete_compound.
2) If any Markush-style variability appears, including labels like R, R1, R2, X, Y, Z, Ar, Het, alkyl, halo, or generic variable substituent definitions, classify as markush.
3) Distinguish variable placeholders from ordinary chemistry labels. Compound numbers, example numbers, salt names, stereochemistry wedges, charge labels, atom symbols, and fixed substituent text do not by themselves make the image Markush.
4) A dense patent crop can still be complete_compound if there is one dominant exact molecule and nearby numbering/text is only local annotation rather than a second unrelated object.
5) If the structure is incomplete, cropped, truncated, or any bond/atom/ring exits or is clipped by the image boundary, classify as fragment.
6) If the image mainly contains layout/text/reaction artifacts, multiple unrelated molecules, or a reaction scheme rather than one target molecule, classify as noise.
7) Be conservative: if unsure whether it is a full exact molecule, do not mark it complete.

Output contract
Return JSON only:
{"structure_type":"complete_compound|markush|fragment|noise|uncertain","is_complete_compound":true,"reason":"short reason"}
