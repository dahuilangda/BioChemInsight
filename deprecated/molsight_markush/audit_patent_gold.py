"""Audit legacy mass-selected gold without promoting any row automatically.

A successful check is only a necessary condition. `pending_visual_review` is
not verified. `rejected` means a concrete input/code/chemical contradiction.
The original gold and OCR sources remain unchanged.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from patent_reference import BLOCKS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block_query(smiles):
    mol = Chem.MolFromSmiles(smiles)
    c = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 2)
    hydroxyl = next(n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors()
        if n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(c,n.GetIdx()).GetBondTypeAsDouble() == 1)
    edit = Chem.RWMol(mol); edit.RemoveAtom(hydroxyl)
    query = edit.GetMol()
    for atom in query.GetAtoms():
        atom.SetAtomMapNum(0)
    return query


def audit_row(seq, row, table=None):
    mol = Chem.MolFromSmiles(row.get('smiles',''))
    result = {'seq_id': int(seq), 'status': 'pending_visual_review', 'issues': []}
    if mol is None:
        result.update(status='rejected', issues=[{'kind':'invalid_smiles'}])
        return result
    result['canonical_smiles'] = Chem.MolToSmiles(mol)
    result['formula'] = rdMolDescriptors.CalcMolFormula(mol)
    result['exact_mass'] = rdMolDescriptors.CalcExactMolWt(mol)
    declared = row.get('declared_mass')
    if declared is not None:
        delta = result['exact_mass'] - declared
        result['mass_delta'] = delta
        # Patent table reports hundredths of a Da. This is a consistency
        # check, never an identity/stereochemistry acceptance condition.
        if abs(delta) > .025:
            result['issues'].append({'kind':'mass_mismatch', 'delta':delta})
    codes = row.get('codes', [])
    if any(c.casefold() == 'clac' for c in codes):
        sulfur = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
        if not any(a.GetDegree() == 2 and a.IsInRing() for a in sulfur):
            result['issues'].append({'kind':'missing_cap_thioether_closure',
                'observed_sulfur': [{'index':a.GetIdx(), 'degree':a.GetDegree(),
                                     'in_ring':a.IsInRing()} for a in sulfur]})
    # These aliases only repair glyph confusions attested in the source
    # abbreviation table; no chemical near-name guessing is performed.
    aliases = {'trp7c1':'trp7cl', 'sbmew70me':'sbmew7ome', 'dprot40acoh':'dprot4oacoh'}
    wanted = Counter(aliases.get(c.casefold(), c.casefold()) for c in codes)
    checked = {}
    for code, (smiles, _) in BLOCKS.items():
        expected = wanted.get(code.casefold(), 0)
        if not expected:
            continue
        query = block_query(smiles)
        graph = len(mol.GetSubstructMatches(query, useChirality=False))
        stereo = len(mol.GetSubstructMatches(query, useChirality=True))
        checked[code] = {'expected_at_least':expected, 'graph_matches':graph,
                         'stereo_matches':stereo}
        if graph < expected:
            result['issues'].append({'kind':'named_residue_graph_mismatch',
                                      'code':code, 'expected':expected, 'matches':graph})
        elif stereo < expected:
            result['issues'].append({'kind':'named_residue_stereo_mismatch',
                                      'code':code, 'expected':expected, 'matches':stereo})
    result['block_checks'] = checked
    unspecified = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    result['unspecified_stereocentres'] = [a for a,s in unspecified if s == '?']
    if table:
        result['ocr_table_sequence'] = table.get('sequence')
        result['source_note'] = 'OCR table is advisory; original page/diagram overrides it.'
    if result['issues']:
        result['status'] = 'rejected'
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gold', default='data/patent_gold/legacy_mass_selected_100.json')
    ap.add_argument('--table', default='/tmp/table1_rows.json')
    ap.add_argument('--out', default='output/gold_audit/legacy_audit.json')
    args = ap.parse_args()
    gold = json.loads(Path(args.gold).read_text())
    table = json.loads(Path(args.table).read_text()) if Path(args.table).exists() else {}
    rows = {str(k):audit_row(k,v,table.get(str(k))) for k,v in gold.items()}
    issues = Counter(i['kind'] for r in rows.values() for i in r['issues'])
    counts = Counter(r['status'] for r in rows.values())
    report = {'source':{'path':args.gold,'sha256':sha256(args.gold)},
              'scope':'legacy file only; filename 289 is not coverage evidence',
              'rows_total':len(rows), 'status_counts':dict(counts),
              'issue_counts':dict(issues), 'fully_verified_rows':0,
              'mass_pass_is_not_identity_proof':True, 'rows':rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2))

if __name__ == '__main__':
    main()
