# Peptide representation, conversion, and depiction

Architecture and implementation plan, researched on 2026-11-08. Current implementations are listed below; the remaining sections describe the target design.

## Scope

Provide one chemical representation for sequence-to-structure conversion, structure-to-sequence analysis, and two-dimensional depiction. Cover linear, cyclic, branched, and crosslinked peptides through explicit monomer connections. Preserve atom identity, stereochemistry, terminal groups, and unidentified components.

The atomic structure and the monomer graph serve different purposes. The atomic structure determines chemical identity. The monomer graph records residue boundaries, connection sites, and sequence conventions. Neither a sequence string nor a drawing replaces the atomic structure.

## Current behavior

`peptidecore` provides 322 verified HELMCoreLibrary peptide definitions, immutable monomer connection graphs, explicit attachment assembly, defined HELM/BILN subsets, ordinary sequence/FASTA import, and loss-checked sequence export. Molecular records preserve source text, library identity, atom ownership, connection endpoints, isotopes, charge, and stereochemistry.

SMILES decomposition now performs exact monomer matching and a bounded search over non-overlapping atom covers. Returned graphs pass complete mapped reassembly checks. Unknown fragments, ambiguous assignments, aliases, and search limits remain explicit. All 322 source monomers and 30 new modified-residue combinations have verified reconstructions in the controlled tests. A declared ten-monomer patent extension also reconstructs the three reviewed reference structures. Broad, independent chemical-space coverage remains to be measured.

`cyclopeptide2d` accepts SMILES and registered chemical records. Sequence-derived layouts use exact attachment atoms, whole-monomer ownership, separate chain paths, and explicit closure edges. Raw SMILES use directed amide paths and bounded cycle search. Local bridge detection preserves flexible side chains inside crosslink-induced cycles. All connection endpoints participate in angular validation. See [layout research and implementation](PEPTIDE_LAYOUT_RESEARCH.md) for algorithms and current limits.

The fixed review matrix accepts 96/96 layouts, a new-seed synthetic matrix accepts 24/24, and the earlier eight-case long-ring diagnostic now accepts 8/8. The three reviewed patent structures pass at seed 0. These controlled results do not establish general chemical-space coverage or real-image recognition accuracy.

Repository sequence assembly also exists in `utils/sequence_parser.py` and `training/molsight_markush/patent_reference.py`. The former contains hand-written side-chain definitions; the latter implements a named monomer set and two closure types for a particular patent family. Neither is a general notation implementation or a suitable source of standard monomer identifiers.

## Notation

| Format | Intended role | Conditions |
| --- | --- | --- |
| HELM 2 | Primary exchange format for complex peptides | Include the monomer-library version and definitions needed to resolve all symbols |
| BILN | Readable peptide input and output | Resolve symbols and attachment points through the same monomer library |
| FASTA | Ordinary protein sequence import and export | Do not imply that the letters encode caps, crosslinks, or nonstandard residues |
| ProForma | Proteomics import and export | Preserve modification identifiers and uncertainty; materialize a molecule only when the structure is determined |
| WIPO ST.26 | Patent sequence-listing import | Preserve feature annotations and links to the source listing |
| Isomeric SMILES / MOL / SDF | Atomic structure exchange | Preserve stereochemistry, isotopes, charge, and component identity |

HELM explicitly separates monomer definitions, polymer chains, and connections. Its specification includes ambiguous structures as well as fully specified molecules. The public specification page identifies version 2.04; specification revisions and the notation's version suffix must be handled separately. See the [HELM specification and test resources](https://pistoiaalliance.atlassian.net/wiki/spaces/HELM/pages/2535522305/HELM+Notation).

Use HELMCoreLibrary identifiers when their structures and attachments match. A local name such as a patent abbreviation is an alias with a source, not a new universal identifier. Pistoia recommends HELMCoreLibrary and publishes a monomer JSON schema. See [HELM monomer definitions](https://github.com/PistoiaHELM/HELMMonomerSets).

BILN is a published peptide notation with explicit connection labels. pyPept provides BILN parsing, monomer-based assembly, and HELM conversion. It is an integration candidate and an independent comparison implementation. See the [BILN paper](https://pubmed.ncbi.nlm.nih.gov/35984937/) and [pyPept repository](https://github.com/Boehringer-Ingelheim/pyPept).

ProForma represents modifications, crosslinks, and uncertain assignments. A mass shift or an unresolved modification position does not uniquely determine an atomic structure. See the [HUPO-PSI specification](https://github.com/HUPO-PSI/ProForma). ST.26 serves patent sequence listings and should remain a separate import format; see [WIPO Sequence](https://www.wipo.int/en/web/standards/sequence/faq).

### Examples

```text
Linear peptide:
PEPTIDE1{A.G.K}$$$$V2.0

Head-to-tail closure:
PEPTIDE1{A.G.K}$PEPTIDE1,PEPTIDE1,3:R2-1:R1$$$V2.0

Specified D-alanine:
PEPTIDE1{A.[dA].K}$$$$V2.0

Disulfide connection:
PEPTIDE1{C.A.A.A.C}$PEPTIDE1,PEPTIDE1,1:R3-5:R3$$$V2.0
```

The symbols in these examples require the corresponding monomer definitions. A plain FASTA export of the disulfide example loses its connection. Such an export must report the loss rather than present the letters as an equivalent chemical representation.

## Chemical records

Use a versioned record containing:

| Field | Contents |
| --- | --- |
| Atomic structure | Atoms, bonds, isotopes, charge, and specified stereochemistry |
| Monomer instances | Stable instance IDs and references to exact monomer definitions |
| Atom ownership | Input atom indices assigned to each monomer or unidentified component |
| Connections | Endpoint instances, attachment labels, endpoint atoms, and bond types |
| Chains | Ordered residue instances and the direction used for sequence export |
| Uncertainty | Alternative decompositions or unresolved structures |
| Source | Original text, format version, library digest, and normalization operations |

A monomer definition includes its polymer type, identifier, structure, labeled attachments, default caps, aliases, origin, and content digest. Attachment definitions must identify the retained atom and the atoms or caps replaced during connection. Validate valence, attachment uniqueness, and the stereochemical effect of assembly.

Keep the original structure alongside any normalized structure. Protonation, salt removal, tautomer selection, and isotope removal change information and require explicit policies. Atom-index maps must survive these operations.

A residue's absolute configuration cannot be checked solely by requiring an unchanged CIP letter before and after assembly: changing its substituents can change CIP priorities without inverting the center. Verify mapped stereochemical relationships in the assembled graph. Distinguish D/L naming, absolute configuration, and the atom-order-dependent `@`/`@@` syntax.

## Sequence to SMILES

1. Parse the declared format with a grammar-aware implementation. Keep source locations for errors.
2. Resolve every monomer against a pinned library. Resolve aliases without changing stereochemistry or inferring a different positional isomer.
3. Construct the monomer graph, including ordinary backbone bonds, explicit crosslinks, terminal groups, and chemical linkers.
4. Validate attachment occupancy and bond compatibility before editing atoms.
5. Assemble through mapped attachments and retain the atom-to-monomer correspondence.
6. Sanitize and verify the resulting atomic graph and mapped stereochemistry.
7. Export isomeric SMILES and a conversion report that records the library and any unresolved information.

Unknown symbols, ambiguous positions, and mixtures need distinct results. A valid ambiguous HELM expression can be preserved as a document without being converted into one invented molecule. Missing attachment definitions must not be silently inferred in the strict conversion API.

Evaluate helmkit as the Python assembly backend. It exposes monomer mappings and custom monomer data. Its published throughput is an author measurement, not a performance claim for this project. Reference HELM tools remain useful for format conformance. See [helmkit](https://github.com/adaliaramon/helmkit) and the [HELM reference toolkit](https://github.com/PistoiaHELM/HELM2NotationToolkit).

Do not adopt an implementation solely because it parses a few simple examples. Test attachment caps, terminal monomers, inline structures, nested syntax, isotope labels, and stereo at connection atoms. Pin the source version, monomer data, and configuration used in those tests.

## SMILES to sequence

This operation is structural analysis, not string conversion. RDKit's `MolToSequence` and `MolToHELM` require monomer metadata. Ordinary SMILES does not carry that metadata. The behavior was reproduced locally by converting a HELM-derived molecule through SMILES and obtaining an empty sequence. See the [RDKit file-format API](https://rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html).

When provenance is available, use the retained monomer graph after verifying that it still matches the molecule. For an unrelated SMILES:

1. Find plausible inter-monomer bonds using defined chemical rules, including amide, ester, disulfide, and supported linker connections.
2. Consider alternative cuts where backbone amides and side-chain amides are indistinguishable locally. Retain the original atoms and bond endpoints.
3. Restore the defined caps for fragment identification and search an indexed monomer library using stereochemical graph matches.
4. Resolve overlapping matches as a constrained graph-cover problem. Each retained atom belongs to one component; connections must agree with the original bonds.
5. Keep unmatched fragments as structures with explicit connections. Report competing decompositions rather than assigning a familiar residue by mass or approximate similarity.
6. Reassemble each proposed monomer graph and compare it with the original atomic graph before exporting a resolved sequence.

rBAN provides an established fragmentation-and-identification approach; Smiles2Monomers provides a complementary graph-cover approach. rBAN's paper also documents dependence on fragmentation rules. Its published software license is noncommercial, so it should be evaluated independently rather than assumed to be a distributable dependency. See [rBAN](https://link.springer.com/article/10.1186/s13321-019-0335-x).

Reassembly equivalence proves that the proposed representation reconstructs the molecule; it does not prove that the monomer decomposition is unique or matches its biosynthetic origin. A modified residue and a residue plus an attached modification may be chemically equivalent decompositions.

A cyclic chain has no intrinsic first residue. Export a deterministic starting point while retaining the mapping to source atom and residue indices. Rotate the sequence only under a policy that preserves directed connections; reversing a cyclic string is not automatically equivalent. Compare reconstructed graphs rather than literal sequence strings.

## Sequence to 2D

Sequence input should assemble into the same chemical record used by SMILES input. The drawing code then receives atoms, monomer ownership, connections, and any trusted local coordinates. It should not parse sequence syntax or reconstruct chemistry independently.

Choose the drawing method from the connection graph:

| Structure | Layout approach |
| --- | --- |
| Linear chain | Residue coordinates connected in an open chain |
| Single cyclic chain | Existing joint residue-pose optimization |
| Branched chain | Joint residue and linker poses with every branch connection enforced |
| Multiple cycles | Rigid local fragments joined by a constraint graph with all closures enforced |
| Fused residue or ring system | Preserve the fused component as a larger local fragment |
| Unidentified chemical component | Use its atomic graph and ordinary chemical coordinates without assigning a false residue |

The registered monomer graph provides decomposition for sequence input. Directed amide analysis supplies a backbone for unannotated SMILES. Ring perception remains a geometry tool rather than a source of peptide connectivity.

Separate fixed small-ring geometry from adjustable acyclic connections. Give difficult residues several valid local coordinate alternatives with identical connection definitions. Search these alternatives together with residue poses, using interference between residues to choose which regions to reconsider. For intersecting backbones, change the initial connection path or the affected group of residue poses instead of repeatedly optimizing one arrangement.

Use RDKit and CoordGen as measured comparisons and as possible sources of local coordinates. RDKit supports constrained coordinates and ring templates; CoordGen includes macrocycle templates. Candidate selection should use the same independent chemical and geometric checks for every implementation. See [RDKit depiction](https://www.rdkit.org/docs/source/rdkit.Chem.rdDepictor.html) and [CoordGen](https://github.com/schrodinger/coordgenlibs).

## Performance

Measure conversion, decomposition, initial-coordinate generation, optimization, validation, and rendering separately. Report initialization and file/library loading separately from repeated calls.

For conversion, cache parsed monomers and attachment data by library digest. Index monomers by inexpensive graph features before exact matching. Bound alternative graph decompositions and return incomplete results when the search budget is reached. Repeated molecules can reuse verified decompositions; their cache key must include the exact chemistry and library version.

For drawing, retain exact spatial contact queries. Replace quadratic contact lookup tables with compact adjacency structures when larger inputs justify the change. Reuse local coordinates by monomer identity and connection state. Cache local candidate evaluations only with explicit invalidation when coordinates or nearby residues change.

Move expensive numeric operations into compiled code only after profiling. Sparse constrained optimization can exploit the fact that each connection touches a small number of residue poses. Compare it with the existing dense SLSQP implementation using both accepted-layout rate and wall time.

GPU work should target batches of candidate layouts with the same approximate dimensions. Keep coordinates, contact data, derivatives, and optimization state on the device. Use the CPU for independent final validation. A GPU version needs measured end-to-end improvement at the same acceptance criteria, not just a faster distance kernel. See [CUDA performance guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).

## Validation

Maintain separate results for syntax, chemical conversion, structural decomposition, and drawing. Successful conversion does not imply that the drawing is acceptable, and a readable drawing does not validate a sequence assignment.

The corpus should include:

- Standard residues, D residues, glycine, proline, and residues with several stereocenters.
- N-methyl residues, peptoids, beta/gamma amino acids, and local monomer extensions.
- Head-to-tail closure, disulfides, lactams, thioethers, branches, and multiple chains.
- Acetylation, amidation, fatty-acid attachments, and defined chemical linkers.
- Ester-containing peptides and structures with multiple macrocycles.
- Unknown residues, ambiguous monomer assignments, isotopes, charge states, and invalid attachment use.

Use the HELM conformance examples for notation coverage, public peptide structures for breadth, and independently checked difficult structures for chemical accuracy. Keep a test split with unseen monomers and unseen connection patterns; splitting only by random molecule ID allows closely related sequences to appear on both sides.

Conversion tests must check atoms, bonds, charge, isotopes, terminal groups, specified/unspecified stereochemistry, and reconstructed graph identity. Test randomized SMILES atom orders, equivalent HELM polymer ordering, cyclic rotations, and serialization through files. Record ambiguous and unsupported cases separately from incorrect conversions.

Drawing measurements include bond and angle deviations, atom/label clearance, crossings, small-ring intersections, stereo rendering, runtime percentiles, peak memory, and accepted layouts per second. Include failed and timed-out inputs in the denominator. Test glyph positions at the final drawing scale rather than treating atom-center distances as complete text clearance.

## Delivery sequence

| Step | Deliverable | Completion condition |
| --- | --- | --- |
| 1 | Versioned monomer library and chemical record | Standard definitions imported, aliases separated, attachments validated, data origins retained |
| 2 | HELM/BILN assembly and ordinary sequence import | Independent reference comparisons and chemical round trips pass for the declared subset |
| 3 | SMILES decomposition | Reassembly verified; unknown components and competing assignments reported; unseen-monomer tests included |
| 4 | Shared sequence/SMILES depiction interface | Identical chemical inputs reach the same layout pipeline with atom ownership preserved |
| 5 | General connection-graph layout | Branched and multiply cyclic test families satisfy the unchanged validation rules |
| 6 | CPU optimization and optional GPU batch backend | Repeatable throughput and latency improvement at matched chemical and drawing accuracy |

Keep `cyclopeptide2d` installable as the current drawing component. Add a separate chemistry/conversion package when its records and dependency choices are implemented; do not change the existing import paths during this work. New APIs should declare their supported notation features and chemical structures, with explicit outcomes for unresolved inputs.

## MolSight training

Generator development and validation precede a synthetic training release, supervised fine-tuning, and reinforcement learning. Preserve complete-molecule recognition while evaluating Markush, attachment-bearing fragments, peptides, and cyclic peptides separately. Audit the existing objectives, data handling, rewards, and evaluation before training. See the [data and training plan](../training/molsight_markush/PLAN.md) for release requirements and reproduced trainer defects.
