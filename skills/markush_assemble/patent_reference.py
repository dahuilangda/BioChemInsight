"""Independent reconstruction of US20240287138A1 examples under visual review.

Uses explicit named building blocks, NOT the mass-selected residue lexicon.
The returned structures are candidates until the accompanying review records
resolve diagram/table and stereochemical evidence. No gold files are overwritten.
"""
from __future__ import annotations
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# :2 identifies the backbone carboxyl carbon (side-chain acids stay unmarked).
BLOCKS = {
    'bhcLeu': ('N[C@H]1CC[C@@H]([C:2](=O)O)C1', ['S','R']),
    'Nle': ('N[C@@H](CCCC)[C:2](=O)O', ['S']),
    '4Pal': ('N[C@@H](Cc1ccncc1)[C:2](=O)O', ['S']),
    'Pipt5OH': ('N1[C@H]([C:2](=O)O)CC[C@@H](O)C1', ['S','R']),
    'W': ('N[C@@H](Cc1c[nH]c2ccccc12)[C:2](=O)O', ['S']),
    'Trp7az': ('N[C@@H](Cc1c[nH]c2ncccc12)[C:2](=O)O', ['S']),
    'dProt4OH': ('N1[C@@H]([C:2](=O)O)C[C@H](O)C1', ['R','S']),
    'dProt4OAcOH': ('N1[C@@H]([C:2](=O)O)C[C@H](OCC(=O)O)C1', ['R','S']),
    'N': ('N[C@@H](CC(N)=O)[C:2](=O)O', ['S']),
    'Trp7Cl': ('N[C@@H](Cc1c[nH]c2c(Cl)cccc12)[C:2](=O)O', ['S']),
    'Trp7CN': ('N[C@@H](Cc1c[nH]c2c(C#N)cccc12)[C:2](=O)O', ['S']),
    'aMeS': ('N[C@@](C)(CO)[C:2](=O)O', ['S']),
    'Phe4CONH2': ('N[C@@H](Cc1ccc(C(N)=O)cc1)[C:2](=O)O', ['S']),
    'Phe4COOH': ('N[C@@H](Cc1ccc(C(=O)O)cc1)[C:2](=O)O', ['S']),
    'Y': ('N[C@@H](Cc1ccc(O)cc1)[C:2](=O)O', ['S']),
    'Sar': ('N(C)C[C:2](=O)O', []),
    'SbMeW7OMe': ('N[C@@H]([C@@H](C)c1c[nH]c2c(OC)cccc12)[C:2](=O)O', ['S','S']),
    'Pip2c': ('N1[C@H]([C:2](=O)O)CNCC1', ['S']),
    'NMeE': ('N(C)[C@@H](CCC(=O)O)[C:2](=O)O', ['S']),
    'C': ('N[C@@H](CS)[C:2](=O)O', ['R']),
}
# Extended from the patent abbreviation table, with explicit absolute
# configurations. Missing codes are rejected; no mass-selected substitutes.
BLOCKS.update({
    'S': ('N[C@@H](CO)[C:2](=O)O', ['S']),
    'Q': ('N[C@@H](CCC(N)=O)[C:2](=O)O', ['S']),
    'H': ('N[C@@H](Cc1cnc[nH]1)[C:2](=O)O', ['S']),
    'K': ('N[C@@H](CCCCN)[C:2](=O)O', ['S']),
    'D': ('N[C@@H](CC(=O)O)[C:2](=O)O', ['S']),
    'E': ('N[C@@H](CCC(=O)O)[C:2](=O)O', ['S']),
    'T': ('N[C@@H]([C@H](O)C)[C:2](=O)O', ['S','R']),
    'Pip': ('N1[C@H]([C:2](=O)O)CCCC1', ['S']),
    'dP': ('N1[C@@H]([C:2](=O)O)CCC1', ['R']),
    'NMeA': ('N(C)[C@@H](C)[C:2](=O)O', ['S']),
    'NMeS': ('N(C)[C@@H](CO)[C:2](=O)O', ['S']),
    'Phe4CH2NH2': ('N[C@@H](Cc1ccc(CN)cc1)[C:2](=O)O', ['S']),
    'Phe3CONH2': ('N[C@@H](Cc1cccc(C(N)=O)c1)[C:2](=O)O', ['S']),
    'TyrOcm': ('N[C@@H](Cc1ccc(OCC(=O)O)cc1)[C:2](=O)O', ['S']),
    'TyrOMe': ('N[C@@H](Cc1ccc(OC)cc1)[C:2](=O)O', ['S']),
    'SbMeW': ('N[C@@H]([C@@H](C)c1c[nH]c2ccccc12)[C:2](=O)O', ['S','S']),
    'SbMe4Pal': ('N[C@@H]([C@@H](C)c1ccncc1)[C:2](=O)O', ['S','S']),
    'PyrimAla': ('N[C@@H](Cc1cncnc1)[C:2](=O)O', ['S']),
    'NxG29': ('N(CCOC)C[C:2](=O)O', []),
    'NxG21': ('N(CCO)C[C:2](=O)O', []),
    'Trp7Br': ('N[C@@H](Cc1c[nH]c2c(Br)cccc12)[C:2](=O)O', ['S']),
})
BLOCKS.update({
    'Ahp': ('N[C@@H](CCCCC)[C:2](=O)O', ['S']),
    'AhpF3': ('N[C@@H](CCCCC(F)(F)F)[C:2](=O)O', ['S']),
    'hF': ('N[C@@H](CCc1ccccc1)[C:2](=O)O', ['S']),
    'hhL': ('N[C@@H](CCCC(C)C)[C:2](=O)O', ['S']),
    'hS': ('N[C@@H](CCO)[C:2](=O)O', ['S']),
    'hSOMe': ('N[C@@H](CCOC)[C:2](=O)O', ['S']),
    'hE': ('N[C@@H](CCCC(=O)O)[C:2](=O)O', ['S']),
    'Cit': ('N[C@@H](CCCNC(N)=O)[C:2](=O)O', ['S']),
    'Dab': ('N[C@@H](CCN)[C:2](=O)O', ['S']),
    'DapMe': ('N[C@@H](CNC)[C:2](=O)O', ['S']),
    'NMehSMe': ('N(C)[C@@H](CCOC)[C:2](=O)O', ['S']),
    'NMeT': ('N(C)[C@@H]([C@H](O)C)[C:2](=O)O', ['S','R']),
    'NEtAla': ('N(CC)[C@@H](C)[C:2](=O)O', ['S']),
    'NEtG': ('N(CC)C[C:2](=O)O', []),
    'dNMeA': ('N(C)[C@H](C)[C:2](=O)O', ['R']),
    'LysMe': ('N[C@@H](CCCCNC)[C:2](=O)O', ['S']),
    'LysMe2': ('N[C@@H](CCCCN(C)C)[C:2](=O)O', ['S']),
    'NxG38': ('N(CCCO)C[C:2](=O)O', []),
    'aMeW': ('N[C@@](C)(Cc1c[nH]c2ccccc12)[C:2](=O)O', ['S']),
    'Trp1Me': ('N[C@@H](Cc1cn(C)c2ccccc12)[C:2](=O)O', ['S']),
    'TrpCH2COOH': ('N[C@@H](Cc1cn(CC(=O)O)c2ccccc12)[C:2](=O)O', ['S']),
    'SbMeW2Me': ('N[C@@H]([C@@H](C)c1c(C)[nH]c2ccccc12)[C:2](=O)O', ['S','S']),
    'SbMeW7Me': ('N[C@@H]([C@@H](C)c1c[nH]c2c(C)cccc12)[C:2](=O)O', ['S','S']),
    'SbPrW': ('N[C@@H]([C@@H](CCC)c1c[nH]c2ccccc12)[C:2](=O)O', ['S','S']),
    '3Pal': ('N[C@@H](Cc1cccnc1)[C:2](=O)O', ['S']),
    'Bhv': ('N[C@@H](C(O)(C)C)[C:2](=O)O', ['S']),
    'F4SO2NH2': ('N[C@@H](Cc1ccc(S(N)(=O)=O)cc1)[C:2](=O)O', ['S']),
    'F4mSO3H': ('N[C@@H](Cc1ccc(CS(=O)(=O)O)cc1)[C:2](=O)O', ['S']),
    'F4SO3H': ('N[C@@H](Cc1ccc(S(=O)(=O)O)cc1)[C:2](=O)O', ['S']),
    'F4MCO2H': ('N[C@@H](Cc1ccc(CC(=O)O)cc1)[C:2](=O)O', ['S']),
    'Tyr2F': ('N[C@@H](Cc1c(F)cc(O)cc1)[C:2](=O)O', ['S']),
    'TyrOCF3': ('N[C@@H](Cc1ccc(OC(F)(F)F)cc1)[C:2](=O)O', ['S']),
    'SerOCH2cPr': ('N[C@@H](COCC1CC1)[C:2](=O)O', ['S']),
    'dProt4NH2': ('N1[C@@H]([C:2](=O)O)C[C@H](N)C1', ['R','S']),
    'dProt4CH2NH2': ('N1[C@@H]([C:2](=O)O)C[C@H](CN)C1', ['R','R']),
    'Pipc4OH': ('N1[C@H]([C:2](=O)O)C[C@H](O)CC1', ['S','R']),
    'RbMeY': ('N[C@@H]([C@H](C)c1ccc(O)cc1)[C:2](=O)O', ['S','R']),
})

CODE_ALIASES = {'trp7c1':'Trp7Cl', 'sbmew70me':'SbMeW7OMe',
                'dprot40acoh':'dProt4OAcOH'}

SEQUENCES = {
    53: ['bhcLeu','Nle','4Pal','Pipt5OH','W','dProt4OH','N','4Pal',
         'Trp7Cl','aMeS','Phe4CONH2','Phe4COOH','Sar','SbMeW7OMe'],
    68: ['Nle','4Pal','Pip2c','Trp7az','NMeE','N','4Pal','Trp7Cl',
         'aMeS','Phe4COOH','Y','Sar','SbMeW7OMe','C'],
    83: ['bhcLeu','Nle','4Pal','Pip2c','W','dProt4OH','N','4Pal',
         'Trp7Cl','aMeS','Phe4CONH2','Phe4COOH','Sar','SbMeW7OMe'],
    77: ['bhcLeu','Nle','4Pal','Pipt5OH','Trp7az','dProt4OAcOH','N',
         'Phe4COOH','Trp7CN','aMeS','Phe4CONH2','Phe4CONH2','Sar','SbMeW7OMe'],
}


def build(seq_id: int, interpretation='table'):
    codes = list(SEQUENCES[seq_id])
    if interpretation == 'figure' and seq_id == 77:
        # Original PDF p.25 (printed p.24), right-side aromatic substituent
        # explicitly terminates COOH; TABLE 1 says CONH2 at this position.
        codes[11] = 'Phe4COOH'
    return assemble_codes(codes, closure='clac_thioether' if seq_id == 68 else 'head_to_tail',
                          terminal='amide' if seq_id == 68 else None,
                          seq_id=seq_id, interpretation=interpretation)


def assemble_codes(codes, closure='head_to_tail', terminal=None,
                   seq_id=None, interpretation='sequence'):
    """Typed-port graph assembly with complete, rigid residue ownership.

    The chemistry is fixed by the notation before checking mass. No topology,
    ring size, terminal group, positional isomer or stereo is selected by mass.
    """
    canonical = {k.casefold():k for k in BLOCKS}
    normalized = []
    for code in codes:
        key = CODE_ALIASES.get(code.casefold(), canonical.get(code.casefold()))
        if key is None:
            raise ValueError(f'unsupported_named_residue:{code}')
        normalized.append(key)
    codes = normalized
    if closure not in ('head_to_tail', 'clac_thioether'):
        raise ValueError(f'unsupported_closure:{closure}')
    if closure == 'clac_thioether' and (not codes or codes[-1] != 'C' or terminal != 'amide'):
        raise ValueError('ClAc family requires explicitly terminal Cys-CONH2')
    if closure == 'head_to_tail' and terminal:
        raise ValueError('head-to-tail ring cannot also have a terminal group')
    if len(codes) < 3:
        raise ValueError('insufficient_residues')
    combined = Chem.RWMol()
    starts, ends, roles = [], [], []
    for k, code in enumerate(codes):
        smiles, expected = BLOCKS[code]
        frag = Chem.MolFromSmiles(smiles)
        c = next(a.GetIdx() for a in frag.GetAtoms() if a.GetAtomMapNum() == 2)
        frag.GetAtomWithIdx(c).SetBoolProp('_backbone_carboxyl', True)
        # Atom maps are bookkeeping and must not participate in RDKit's
        # legacy CIP ranking (mapped carboxyl carbon can outrank CH2-S).
        for atom in frag.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.AssignStereochemistry(frag, cleanIt=True, force=True)
        centres = Chem.FindMolChiralCenters(frag, includeUnassigned=True)
        if [label for _,label in centres] != expected:
            raise ValueError(f'curated_block_configuration_mismatch:{code}:{centres}')
        hydroxyl = next(n.GetIdx() for n in frag.GetAtomWithIdx(c).GetNeighbors()
                         if n.GetAtomicNum() == 8 and
                         frag.GetBondBetweenAtoms(c, n.GetIdx()).GetBondType() == Chem.BondType.SINGLE)
        edit = Chem.RWMol(frag)
        if closure == 'clac_thioether' and k == len(codes)-1:
            edit.ReplaceAtom(hydroxyl, Chem.Atom('N'))  # terminal Cys-CONH2
        else:
            edit.RemoveAtom(hydroxyl)
        piece = edit.GetMol()
        n = 0
        c = next(a.GetIdx() for a in piece.GetAtoms() if a.HasProp('_backbone_carboxyl'))
        offset = combined.GetNumAtoms()
        for a in piece.GetAtoms():
            atom = Chem.Atom(a)
            atom.SetAtomMapNum(0)
            atom.SetIntProp('_review_residue', k)
            combined.AddAtom(atom)
        for b in piece.GetBonds():
            combined.AddBond(offset+b.GetBeginAtomIdx(), offset+b.GetEndAtomIdx(), b.GetBondType())
        starts.append(offset+n); ends.append(offset+c)
        roles.append({'position': k+1, 'code': code, 'free_block_cip': expected,
                      'atoms': list(range(offset, combined.GetNumAtoms()))})
    for k in range(len(codes)-1):
        combined.AddBond(ends[k], starts[k+1], Chem.BondType.SINGLE)
    if closure == 'head_to_tail':
        combined.AddBond(ends[-1], starts[0], Chem.BondType.SINGLE)
    else:
        # ClAc cap: Nle-N-C(=O)-CH2-S-Cys; no head-to-tail peptide bond.
        co = combined.AddAtom(Chem.Atom('C'))
        ox = combined.AddAtom(Chem.Atom('O'))
        ch2 = combined.AddAtom(Chem.Atom('C'))
        sulfur = next(a.GetIdx() for a in combined.GetAtoms() if a.GetAtomicNum() == 16)
        combined.AddBond(co, ox, Chem.BondType.DOUBLE)
        combined.AddBond(co, ch2, Chem.BondType.SINGLE)
        combined.AddBond(co, starts[0], Chem.BondType.SINGLE)
        combined.AddBond(ch2, sulfur, Chem.BondType.SINGLE)
    mol = combined.GetMol()
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    for role in roles:
        role['assembled_cip'] = [mol.GetAtomWithIdx(a).GetProp('_CIPCode')
            for a in role['atoms'] if mol.GetAtomWithIdx(a).HasProp('_CIPCode')]
        if role['assembled_cip'] != role['free_block_cip']:
            raise ValueError(f'assembly_configuration_changed:{role}')
    return mol, {'seq_id': seq_id, 'interpretation': interpretation, 'codes': codes,
                 'smiles': Chem.MolToSmiles(mol),
                 'formula': rdMolDescriptors.CalcMolFormula(mol),
                 'exact_mass': rdMolDescriptors.CalcExactMolWt(mol),
                 'residue_stereo': roles, 'closure':closure, 'terminal':terminal,
                 'status': 'candidate_pending_full_visual_stereo_audit'}


def export_reviewed(out_path, source_pdf='/data/wo_patents/US20240287138A1.pdf'):
    """Completed diagram/table reviews; 77 stays conflicted."""
    import hashlib, json
    from pathlib import Path
    pdf_hash = hashlib.sha256(Path(source_pdf).read_bytes()).hexdigest()
    crops = {53:'US20240287138A1/page_024_box_0.png',
             68:'US20240287138A1/page_024_box_1.png',
             83:'US20240287138A1/page_025_box_1.png'}
    records = {}
    for seq, crop in crops.items():
        _, record = build(seq)
        record.pop('status', None)
        record['review'] = {
            'status':'figure_reviewed', 'crop':crop,
            'crop_sha256':hashlib.sha256((Path(__file__).resolve().parents[2]/'data'/'real_crops'/crop).read_bytes()).hexdigest(),
            'pdf_page':25 if seq == 83 else 24, 'printed_page':24 if seq == 83 else 23,
            'source_url':'https://patents.google.com/patent/US20240287138A1/en',
            'source_pdf_sha256':pdf_hash,
            'smiles_sha256':hashlib.sha256(record['smiles'].encode()).hexdigest(),
            'checks':['original PDF full-page sequence-label placement',
                      'residue sequence and abbreviation definitions',
                      'ring size and substituent positions',
                      'per-residue absolute configuration and figure wedge directions',
                      'closure chemistry and terminal groups',
                      'exact molecular formula and rounded exact mass'],
            'corrections': ([
                '53 is a separate compound from legacy key 3',
                'two 4-pyridyl side chains; OCR 3Pal is not the figure',
                'Pipt5OH is (2S,5R)-5-hydroxypiperidine, not 3-hydroxy',
                'dProt4OH is (2R,4S)',
                'SbMeW7OMe is (2S,3S)',
            ] if seq == 53 else [
                'actual crop is box_1; box_2 is SEQ 60',
                'N-terminal ClAc cap closes onto terminal cysteine sulfur',
                'cysteine terminus is CONH2, not a head-to-tail amide',
                '7-chloroindole, not the 6-chloro positional isomer',
                'Phe4COOH alpha centre is S',
                'SbMeW7OMe is (2S,3S)',
            ] if seq == 68 else [
                'original PDF page 25, lower figure explicitly labelled SEQ 83',
                'Pip2c is (S)-piperazine-2-carboxylic acid with a second ring NH',
                'two 4-pyridyl side chains and a 7-chloroindole',
                'dProt4OH is (2R,4S); SbMeW7OMe is (2S,3S)',
            ])}
        records[str(seq)] = record
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({'schema':'patent-figure-gold-v1','rows':records}, indent=2)+'\n')
    return records


def reviewed_smiles(seq_id):
    """Reviewed figure references only; never silently use mass-only gold."""
    import json, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from skills.patent_gold_review import validate_reference
    path = root/'data'/'patent_gold'/'reviewed_gold.json'
    data = json.loads(path.read_text())
    record = data.get('rows', {}).get(str(seq_id))
    ok, why = validate_reference(record, record.get('review',{}).get('crop') if record else None, seq_id=seq_id)
    if not ok:
        raise ValueError(f'SEQ {seq_id} is not an audited figure reference: {why}')
    return record['smiles']


if __name__ == '__main__':
    import argparse, json
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='output/gold_audit/rebuilt_candidates.json')
    ap.add_argument('--reviewed-out', help='export the completed reviews (53, 68 and 83)')
    args = ap.parse_args()
    result = {}
    for seq in SEQUENCES:
        for interpretation in (('table','figure') if seq == 77 else ('table',)):
            mol, report = build(seq, interpretation)
            result[f'{seq}_{interpretation}'] = report
            print(seq, interpretation, report['formula'], round(report['exact_mass'],4))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')

    if args.reviewed_out:
        export_reviewed(args.reviewed_out)
