"""Fixed single-port group vocabulary; no unresolved R groups or valence repair."""
import hashlib,json
from functools import lru_cache
from rdkit import Chem
from rdkit.Chem import rdAbbreviations,rdDepictor
from molsight.abbreviations import REGISTRY

TEMPLATES=[]
LABEL_TO_ID={}
for label,smiles in REGISTRY.items():
    if smiles not in [item[1] for item in TEMPLATES]:TEMPLATES.append((label,smiles))
    LABEL_TO_ID[label]=1+[item[1] for item in TEMPLATES].index(smiles)
REGISTRY_SHA256=hashlib.sha256(json.dumps(TEMPLATES,separators=(',',':')).encode()).hexdigest()


@lru_cache(maxsize=None)
def template(identifier):
    if not 1<=identifier<=len(TEMPLATES):raise ValueError('Unknown fixed group')
    mol=Chem.MolFromSmiles(TEMPLATES[identifier-1][1])
    ports=[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==0]
    if len(ports)!=1 or mol.GetAtomWithIdx(ports[0]).GetDegree()!=1:raise ValueError('Invalid group port')
    if any(a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()):raise ValueError('Unsupported internal group stereo')
    rdDepictor.Compute2DCoords(mol)
    return mol,ports[0],mol.GetAtomWithIdx(ports[0]).GetNeighbors()[0].GetIdx()


@lru_cache(maxsize=1)
def depiction_abbreviations():
    # Generate exact queries from the same registry used for expansion. Do not
    # assume a third-party depiction label has the same rooted chemistry.
    displays={item.label:(item.displayLabel or item.label,item.displayLabelW or item.label) for item in rdAbbreviations.GetDefaultAbbreviations()}
    definitions=[]
    for identifier,(label,smiles) in sorted(enumerate(TEMPLATES,1),key=lambda x:-Chem.MolFromSmiles(x[1][1]).GetNumAtoms()):
        molecule=Chem.MolFromSmiles(smiles);query=Chem.RWMol(molecule)
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum()==0:replacement=Chem.AtomFromSmarts('*')
            else:
                aromatic='a' if atom.GetIsAromatic() else 'A'
                charge=f'+{atom.GetFormalCharge()}' if atom.GetFormalCharge()>=0 else str(atom.GetFormalCharge())
                replacement=Chem.AtomFromSmarts(f'[0#{atom.GetAtomicNum()};{aromatic};D{atom.GetDegree()};H{atom.GetTotalNumHs()};{charge}]')
            query.ReplaceAtom(atom.GetIdx(),replacement)
        display,western=displays.get(label,(label,label))
        text=f'{label} {Chem.MolToSmarts(query)} {display} {western}'
        definitions.extend(rdAbbreviations.ParseAbbreviations(text))
    return definitions


def condense_visible(original,max_coverage=.7):
    labeled=rdAbbreviations.LabelMolAbbreviations(original,depiction_abbreviations(),maxCoverage=max_coverage)
    # A plain fixed label cannot encode specified stereo hidden inside a group.
    groups=[group for group in Chem.GetMolSubstanceGroups(labeled)
        if not any(original.GetAtomWithIdx(i).GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED for i in group.GetAtoms())]
    filtered=Chem.Mol(labeled);Chem.ClearMolSubstanceGroups(filtered)
    for group in groups:Chem.AddMolSubstanceGroup(filtered,group)
    return rdAbbreviations.CondenseAbbreviationSubstanceGroups(filtered)
