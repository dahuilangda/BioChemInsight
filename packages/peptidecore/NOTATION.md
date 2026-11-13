# Peptide notation

PeptideCore uses existing sequence formats. Chemical identity is the assembled isomeric atomic graph together with explicit monomer definitions and attachment endpoints.

## Formats

| Format | Accepted representation | Export behavior |
| --- | --- | --- |
| HELM 2, `V2.0` | Defined PEPTIDE and CHEM monomers, ordered chains, explicit connections | Preserves chain IDs, monomers, connection positions, and attachment labels |
| BILN | Defined peptide monomers, multiple chains, paired connection labels | Renumbers chains and connection labels while retaining chemistry |
| Protein sequence / single-record FASTA | The 22 unambiguous proteinogenic codes | Available only for an unmodified, uncapped, unconnected single peptide chain |
| Isomeric SMILES | Atomic structure, including specified stereo, charge, and isotopes | Sequence export requires a verified monomer decomposition |

HELM is the primary exchange format for modified, cyclic, branched, and crosslinked peptides. Each release includes the monomer library snapshot and its content digest. A name alone does not identify an unfamiliar residue's chemistry.

The current HELM profile supports fully specified chains and single inter-monomer bonds. Repeats, ambiguous expressions, inline structures, polymer groups, and annotations are rejected explicitly. This is a defined subset, not full HELM 2 conformance. Unsupported input remains available in the caller's original source record; it does not become a guessed molecule.

## Examples

| Structure | HELM |
| --- | --- |
| Linear | `PEPTIDE1{A.G.K}$$$$V2.0` |
| Capped and modified | `PEPTIDE1{[ac].A.[dA].[meL].[Aib].[am]}$$$$V2.0` |
| Head-to-tail ring | `PEPTIDE1{A.G.K}$PEPTIDE1,PEPTIDE1,3:R2-1:R1$$$V2.0` |
| Disulfide | `PEPTIDE1{C.A.G.G.A.C}$PEPTIDE1,PEPTIDE1,1:R3-6:R3$$$V2.0` |
| Two closures | `PEPTIDE1{C.A.G.G.A.C}$PEPTIDE1,PEPTIDE1,1:R3-6:R3|PEPTIDE1,PEPTIDE1,6:R2-1:R1$$$V2.0` |
| Branch | `PEPTIDE1{A.K.G}|PEPTIDE2{A.G}$PEPTIDE1,PEPTIDE2,2:R3-2:R2$$$V2.0` |
| Selenocysteine and pyrrolysine | `PEPTIDE1{A.[seC].[Pyl].G}$$$$V2.0` |

Positions are one-based. `R1`, `R2`, and `R3` refer to the selected monomer definition; they are not universal atom names. Ordinary PEPTIDE chain adjacency connects the preceding monomer's `R2` to the following monomer's `R1`. Explicit connections add edges rather than replacing these backbone bonds. Attachment occupancy is validated before assembly.

The disulfide example has the BILN representation `C(1,3)-A-G-G-A-C(1,3)`. D residues use registered identifiers such as `[dA]`; lowercase protein sequence letters are case-normalized and do not indicate D configuration. `U` maps explicitly to HELMCoreLibrary `seC`; `O` maps to `Pyl`. Ambiguous protein codes `B`, `J`, `X`, and `Z`, gaps, and stops do not specify a unique atomic structure and are rejected by atomic assembly.

## Exchange and verification

Store the source format and text, normalized HELM, the complete monomer library with its digest, isomeric SMILES, atom ownership, and connection endpoints. Custom monomers retain their source-qualified identifiers and definitions; aliases are not substituted automatically.

Every accepted conversion is checked by reassembly. HELM/BILN round trips compare atomic identity rather than text. BILN connection numbers and chain names can change without changing the molecule. Cyclic peptides have no intrinsic first residue, and different verified monomer decompositions can describe the same atomic graph. Neither an arbitrary sequence rotation nor a partial SMILES match establishes a unique sequence label.

For training, molecular recognition targets and sequence annotations remain separate. A molecular image can have a determined atomic structure while its monomer assignment is ambiguous. Such a sample can supervise atomic recognition without asserting one unique sequence.

## References

- [Pistoia HELM 2 reference toolkit](https://github.com/PistoiaHELM/HELM2NotationToolkit)
- [Pistoia HELM monomer definitions](https://github.com/PistoiaHELM/HELMMonomerSets)
- [BILN and pyPept reference implementation](https://github.com/Boehringer-Ingelheim/pyPept)
- [INSDC amino-acid codes](https://www.ddbj.nig.ac.jp/ddbj/code-e.html)
