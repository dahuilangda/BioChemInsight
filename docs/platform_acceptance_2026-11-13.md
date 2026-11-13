# Retained-document acceptance, 2026-11-13

This audit uses actual model/API calls and source PDFs, rather than HTTP fixtures, to check extraction. Browser fixtures separately check interactions. Values in the audit gold files are not included in runtime prompts or branching logic.

## Retained jobs

Only two jobs remain visible. Twenty older jobs were archived (`deleted` status); their evidence files were retained because newer jobs reference them. The pre-cleanup snapshot and manifest are in `output/platform_acceptance_20261113/`.

| Document | Canonical task | PDF SHA256 |
|---|---|---|
| sample.pdf, WO2024/173453, 296 pages | 9aa5ad989bf04195b13edc0ba9e83a11 | fca64b948225c6217ff6cc6b6a753426b40eda64da18cd6218b382ab147f8428 |
| Scientific Reports, DOI 10.1038/s41598-023-31080-2, 14 pages | afb8501f571246d0ba89dddccd58e2dd | 7703bb0583a279e37a144a7a5855161ea9abbf81af32e3ac05e4d6ad4e40cc8a |

## Results and limits

| Capability | Actual result | Interpretation |
|---|---|---|
| SR Markush assembly | 18/18 graphs agree with independently parsed complete names; 17/18 pass final blinded visual review | The remaining member, 6o, stays pending. Graph agreement does not override a rejected visual review. |
| SR experimental table | 19/19 table values agree with manually transcribed source gold | Includes the control. Three ambiguous comparison-prose readings are retained as quarantined evidence rather than assigned by mention order; this is not a claim that all existing observations are correct. |
| SR properties | 18 rows × 5 calculated properties; previously recovered ADME evidence retained | Experimental, predicted and calculated observations remain separate. |
| Patent repeated-block tables | 266 distinct source IDs over pages 270–272, versus 231 previously extracted | Source includes a blank ID 259; no missing row was invented from a numeric sequence. |
| Patent categorical values | 28/28 sampled source values agree with manual gold | Stars and NT remain source categories, not fabricated point estimates. Source bin definitions are retained. |
| Patent ID ownership | Eight sampled source rows verified; four shifted IDs corrected | Eight corrected ID–graph pairs agree with independent printed-name graphs. This does not validate all existing patent structures. |
| Live MolSight perturbation | Four crops × two variants; 6/8 exact isomeric matches | One compound has a stereo error in both variants. Stability is not accuracy. |
| External name lookup | Actual source control: local OPSIN miss followed by PubChem resolution | Kept as a name-derived candidate requiring review, not automatically accepted as image recognition. |
| Actual browser | Continuous PDF, linked location, source images, molecular editor and visible Jobs actions exercised | Initial PDF paint measured 682 ms in one local warm run; not a cold-network performance guarantee. |

The patent contains small molecules, not a peptide benchmark. Neither retained PDF establishes accuracy on modified peptides or cyclic peptides. The custom MolSight checkpoint and peptide-visible policy were preserved. A separate independently annotated peptide/cyclic-peptide cohort is still required for that claim. Two documents cannot establish statistical generalization.

## Display failures found during live inspection

The assembled structures had not been merged into the retained result rows. Accepted members now carry SMILES, molblocks, scaffold evidence and fragment evidence in those same records. Evidence initially lived outside the artifact endpoint's allowed directory; files were copied into the canonical task evidence directory and references updated, without broadening file access permissions.

JSME draws frames around unused SVG toolbar areas. Only those frame geometries are suppressed; molecular atoms, bonds and interaction surfaces remain intact. Source and extracted images use flat styling.

Measurement columns now prioritize experimental observations and coverage. Uncertainty notation in a header (such as ± SD) does not create another endpoint. Raw header evidence remains available. Different readings of the same endpoint retain page references and receive a warning treatment; corrections are explicit user edits and support version restoration.

## Reproduction artifacts

`output/platform_acceptance_20261113/` contains source page images, manual gold, name-derived graph gold, ownership reviews, assembly outputs, table outputs, perturbation outputs and model call audit logs. Logs include unsuccessful diagnostic runs; the final reviewed JSON files define the reported final outcomes.

The Python acceptance-related suite passed 82 tests. Browser navigation coverage includes actual PDF.js rendering plus fixture-backed editing, crop extraction, undo, column visibility, rename and sorting controls. Production build passed. These checks do not certify every chemical prediction.
