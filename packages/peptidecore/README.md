# PeptideCore

Peptide monomer definitions, explicit connections, molecular assembly, and structure decomposition. Requires Python 3.10+ and RDKit.

The bundled library contains 322 peptide monomers from HELMCoreLibrary. It includes standard and modified residues, D residues, and terminal groups. Custom libraries can add peptide monomers and chemical linkers.

## Installation

```bash
python -m pip install ./packages/peptidecore
```

See [Peptide notation](NOTATION.md) for the format profile, complete examples, and exchange requirements.

## Conversion

```python
from peptidecore import from_helm, from_sequence, to_helm

linear = from_sequence("AGK")
print(linear.smiles)

cyclic = from_helm("PEPTIDE1{A.G.K}$PEPTIDE1,PEPTIDE1,3:R2-1:R1$$$V2.0")
print(cyclic.smiles)
print(to_helm(cyclic.graph))
print(cyclic.as_dict())
```

`from_sequence` accepts the 22 unambiguous proteinogenic amino-acid codes, ignores whitespace, and converts lowercase letters to uppercase. `U` maps to `seC` and `O` to `Pyl` in HELMCoreLibrary. Residues use the structures and stereochemistry in the bundled library. `from_fasta` accepts one FASTA record. Ambiguous symbols, gaps, and stop symbols are rejected.

### BILN

```python
from peptidecore import from_biln, to_biln

record = from_biln("C(1,3)-A-A-A-C(1,3)")
print(record.smiles)
print(to_biln(record.graph))
```

BILN uses `-` between monomers, `.` between chains, and `(bond, attachment)` for explicit connections. Each bond label must occur twice. Brackets protect names containing delimiters, such as `[D-Ala-ol]`. Connection suffixes are accepted inside or after a bracketed name. Multiple connections on one residue are supported. Symbols are resolved against the selected library without case folding or guessed aliases.

BILN export numbers chains by their output order and assigns new bond labels. It preserves chemical connections rather than the original text or chain identifiers. Chemical polymer types require HELM and are rejected by the BILN exporter. See the [BILN publication](https://pubmed.ncbi.nlm.nih.gov/35984937/) and [pyPept reference implementation](https://github.com/Boehringer-Ingelheim/pyPept).

`from_helm` supports the following explicit HELM features:

| Feature | Support |
| --- | --- |
| Peptide chains and bracketed monomer names | Supported |
| Ordinary R2–R1 backbone connections | Supported |
| Explicit intra-chain and inter-chain connections | Supported |
| Head-to-tail closure and disulfide connections | Supported |
| Side-chain connections, including lactams | Supported when the monomers define the required attachments |
| Terminal groups | Supported as registered monomers |
| CHEM components | Supported with a custom monomer library |
| Multiple unconnected chains | Preserved as separate molecular components |
| Inline structural monomers, repeats, and ambiguous expressions | Not yet supported |
| Polymer groups and annotations | Not yet supported |
| RNA and DNA polymers | Not supported |

Accepted version suffixes are `V2.0` and an empty legacy suffix. Unsupported expressions raise `PeptideError` with a machine-readable `code`; they are not simplified or repaired.

`to_sequence` exports a single, unmodified linear peptide. It rejects cyclic, crosslinked, multichain, modified, and capped inputs. It also verifies that the structure matches the standard sequence, including when the record was built with a custom library.

### SMILES to monomer graphs

```python
from peptidecore import decompose_smiles, to_helm, to_sequence

result = decompose_smiles("C[C@H](N)C(=O)NCC(=O)N[C@@H](CCCCN)C(=O)O")
if result.unique:
    record = result.records[0]
    print(to_sequence(record))
    print(to_helm(record.graph))
else:
    print(result.as_dict())
```

The recognizer matches exact monomer structures, unused caps, and used attachment boundaries. Non-overlapping matches must cover the input atoms. Every returned graph is reassembled, then checked against the input with a complete atom map, including stereochemistry. It does not require residue metadata in the SMILES and retains the input atom order used by RDKit's SMILES reader.

| Status | Meaning |
| --- | --- |
| `resolved` | One verified graph after an exhaustive search within the selected library |
| `ambiguous` | Multiple verified graphs; inspect `search_exhausted` for search completeness |
| `partial` | No complete verified graph; inspect matches, unidentified fragments, and verification failures |
| `budget_exceeded` | Search limits prevent a conclusion; verified graphs may still be available |

Chemical identity does not guarantee a unique sequence. Glutamic acid and gamma-glutamic acid have the same isolated atomic structure but different attachment definitions. Modified residues can also have alternative decompositions into residues and attached groups. Such alternatives remain explicit. Chemically identical library symbols with identical attachment definitions are grouped as aliases on a match.

Unidentified fragments retain their atom indices, structures with boundary attachment atoms, and original connection endpoints. No residue is assigned by mass or approximate similarity. Unspecified atom or bond stereochemistry is not filled in from a library. Protonation, isotope labels, and tautomers are matched as supplied. Enhanced stereo groups are currently rejected.

`timeout`, `max_states`, `max_matches`, and `max_solutions` bound the search. Timeouts are cooperative; index construction and individual RDKit calls can finish after the requested time. Search limits are recorded and never establish uniqueness. The current library index supports up to eight attachments per monomer. Atom coverage alone does not prove a valid decomposition; only `records` contain verified assemblies.

## Structure records

`ChemicalRecord` contains the assembled molecule, monomer graph, library identity, source text, atom ownership, and atomic connection endpoints. `molecule` returns a copy. `smiles` is canonical isomeric SMILES produced by the installed RDKit version.

Sequence positions are one-based. Atomic connection indices and `atom_origins` are zero-based. An atom origin refers to its monomer's source SMILES parsed with explicit hydrogens retained. Atoms restored by an unused attachment cap inherit the corresponding cap position. Implicit hydrogens are not separate mapped atoms.

Assembly replaces labeled cap atoms with temporary attachment atoms, joins matching pairs with RDKit `molzip`, and restores unused caps. Connection occupancy is checked before assembly. Every library monomer must reproduce its source structure after all caps are restored. Assembly preserves specified stereochemistry, isotope labels, and formal charges.

## Custom monomers

```python
from peptidecore import Attachment, Monomer, core_library, from_helm

linker = Monomer(
    symbol="ethylene",
    polymer_type="CHEM",
    name="Ethylene linker",
    smiles="[H:1]CC[H:2]",
    attachments=(
        Attachment("R1", "[*:1][H]"),
        Attachment("R2", "[*:2][H]"),
    ),
)
library = core_library().extend((linker,), name="linkers")
record = from_helm(
    "PEPTIDE1{C.A.C}|CHEM1{[ethylene]}$PEPTIDE1,CHEM1,1:R3-1:R1|PEPTIDE1,CHEM1,3:R3-1:R2$$$V2.0",
    library=library,
)
```

Attachment maps must be unique and match the declared labels. This implementation supports a single capped atom at each attachment and single bonds between monomers. Invalid cap definitions, duplicate identifiers, unavailable attachments, and reused attachments are rejected. A library extension cannot replace an existing identifier.

`MonomerLibrary.to_json()` exports a complete library snapshot. `MonomerLibrary.from_json()` reloads it, validates every definition, and checks its content digest. Store this snapshot with records that use custom monomers.

## Peptide drawing

Install both local packages:

```bash
python -m pip install ./packages/peptidecore './packages/cyclopeptide2d[sequence]'
```

```python
from cyclopeptide2d import depict_sequence, render_svg

result = depict_sequence("PEPTIDE1{G.G.G.G.G.G.G.G}$PEPTIDE1,PEPTIDE1,8:R2-1:R1$$$V2.0")
if result.accepted:
    svg = render_svg(result)
```

This interface uses registered monomer boundaries for linear, cyclic, branched, and multiply closed peptides. Drawing has separate geometry and image-quality checks; see [Cyclopeptide2D](../cyclopeptide2d/README.md) for measured coverage and current limits.

`depict_sequence(..., input_format='BILN')` accepts BILN. A recovered SMILES graph can be passed to `depict_record` with its original atom mappings.

## Library provenance

The data is a peptide-only selection from [Pistoia's HELMMonomerSets](https://github.com/PistoiaHELM/HELMMonomerSets), source Git blob `91361a1099df74e08c9cd53e95c708025b321244`. Original records and their fields are retained. The bundled manifest records the original file hash, selection, and bundled file hash. Loading verifies the bundled hash and entry count.

The labeled `smiles` field determines each monomer's structure. Some original `molfile` fields use a different tautomer or charge representation. They remain in the data for inspection and are not substituted during assembly. In particular, the bundled histidine tautomer differs from RDKit's sequence constructor. No tautomer normalization is applied to conversion output.

Monomer data is distributed under the upstream MIT license in `THIRD_PARTY_LICENSES/HELMMonomerSets.txt`. Library digests include structures, attachment definitions, monomer types, and natural analogs. Dependencies and library versions should be recorded when comparing generated SMILES across environments.

## Development

From this directory:

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Tests check every bundled monomer, independent RDKit assembly comparisons, stereochemistry, isotope and charge preservation, connection ownership, terminal groups, custom linkers, input errors, and export restrictions.

Decomposition tests include all 322 source monomers, new combinations of modified residues, randomized SMILES atom orders, multiple rings and chains, shared external attachment atoms, unknown isotope-bearing residues, ambiguous assignments, and interrupted searches. These are controlled tests rather than a claim of general recognition accuracy.

`examples/patent_extension.json` contains ten source-specific monomer definitions and three previously reviewed structures from US20240287138A1. Its aliases map patent abbreviations to existing HELM identifiers or source-qualified identifiers; aliases are not applied automatically. The file records source definitions and provenance. Load its definitions explicitly:

```python
import json
from pathlib import Path
from peptidecore import Monomer, core_library, decompose_smiles

example = json.loads(Path("examples/patent_extension.json").read_text())
library = core_library().extend(
    (Monomer.from_helm(item) for item in example["monomers"]),
    name=example["namespace"],
)
result = decompose_smiles(example["references"]["68"]["smiles"], library=library)
```

The core library alone leaves unknown regions in these structures. With the declared extension, all three have complete, reassembly-verified graphs. Multiple equivalent residue assignments remain visible.

Profile a JSON list or mapping containing SMILES:

```bash
PYTHONPATH=src python examples/profile_decomposition.py \
  --input src/peptidecore/data/helm_core_peptide.json \
  --out /tmp/decomposition_profile.json --repeats 3
```

The report separates setup from measured calls and records source hashes, library identity, search outcomes, limits, coverage, median latency, and the 95th percentile. Failed and incomplete runs remain in the report.
