"""Rebuild patent table rows without any mass-driven chemical substitutions.

Compilation is NOT gold approval. All candidates retain pending source review;
unsupported notation, unknown monomers and source mass conflicts are separate.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
try:
    from .patent_reference import assemble_codes, build, SEQUENCES
except ImportError:
    from patent_reference import assemble_codes, build, SEQUENCES


def parse_notation(text):
    text = ''.join(str(text).split())
    if not text.startswith('cyclo('):
        raise ValueError('unsupported_noncyclic_notation')
    depth, end = 1, None
    for i, ch in enumerate(text[6:], start=6):
        if ch == '(': depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        raise ValueError('unbalanced_cyclic_notation')
    inner = text[6:end]
    depth, start, codes = 0, 0, []
    for i, ch in enumerate(inner):
        if ch == '(': depth += 1
        elif ch == ')': depth -= 1
        elif ch == '-' and depth == 0:
            codes.append(inner[start:i]); start = i+1
    codes.append(inner[start:])
    if not all(codes) or depth:
        raise ValueError('invalid_residue_tokens')
    suffix = text[end+1:]
    if suffix not in ('','-NH2'):
        raise ValueError(f'unsupported_terminal_notation:{suffix}')
    closure, terminal = 'head_to_tail', None
    if codes[0].casefold() == 'clac':
        if suffix != '-NH2' or codes[-1].casefold() != 'c':
            raise ValueError('ClAc closure requires explicit terminal Cys-CONH2')
        codes = codes[1:]
        closure, terminal = 'clac_thioether', 'amide'
    elif suffix:
        raise ValueError('cyclic_terminus_without_supported_cap')
    return codes, closure, terminal


def rebuild_row(row):
    seq = int(row['seq_id'])
    try:
        codes, closure, terminal = parse_notation(row['sequence'])
        correction = None
        expected53 = list(SEQUENCES[53])
        ocr53 = list(expected53);ocr53[7] = '3Pal'
        if seq == 53 and [c.casefold() for c in codes] == [c.casefold() for c in ocr53]:
            codes = expected53
            correction = {'position':8,'ocr':'3Pal','original':'4Pal',
                          'evidence':'original US PDF p24 figure and TABLE 1; reviewed SEQ53'}
        _, record = assemble_codes(codes, closure, terminal, seq_id=seq)
    except (ValueError, AssertionError) as e:
        return {'seq_id':seq, 'status':'unsupported_or_invalid', 'error':str(e),
                'source_notation':row.get('sequence')}
    record['source_notation'] = row['sequence']
    if correction:
        record['source_notation_correction'] = correction
    record['declared_mass'] = row.get('exact_mass')
    record['source_formula_ocr_unverified'] = row.get('formula')
    declared = record['declared_mass']
    record['mass_delta'] = record['exact_mass'] - declared if declared else None
    record['status'] = ('chemical_consistency_pending_source_review'
                        if record['mass_delta'] is not None and abs(record['mass_delta']) <= .025
                        else 'source_mass_conflict')
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--table',default='/tmp/table1_rows.json')
    ap.add_argument('--out',default='output/gold_audit/rebuilt_table_candidates.json')
    args=ap.parse_args()
    source=Path(args.table);table=json.loads(source.read_text())
    rows={key:rebuild_row(row) for key,row in table.items()}
    counts=Counter(r['status'] for r in rows.values())
    errors=Counter(r['error'] for r in rows.values() if r.get('error'))
    report={'source':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                      'trust':'OCR table; original page review required'},
            'rows_total':len(rows),'status_counts':dict(counts),
            'fully_verified_by_this_compiler':0,'errors':dict(errors), 'rows':rows}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','errors')},indent=2))
    print('top unresolved',errors.most_common(12))

if __name__ == '__main__':main()
