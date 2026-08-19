"""Literature extraction accuracy eval.

Deterministic judges over the two Sci Rep papers: assay values compared
exactly against ground truth extracted from the PDFs, emitted structures
compared against OPSIN references (contamination fails the run), repeated
rolls measure stability. Artifacts: evals/artifacts/runs.jsonl.
"""

import json
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem

from utils.series_member_synthesis import name_to_molecule

ROOT = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH = json.load(open(os.path.join(ROOT, 'literature_ground_truth.json'), encoding='utf-8'))
ARTIFACTS = os.path.join(ROOT, 'artifacts')


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol else ''


def strip_spaces(value):
    return re.sub(r'\s+', '', str(value or ''))


def page_contexts_from_pdf(pdf_path, pages, cache_path):
    import fitz
    cached = {}
    if os.path.exists(cache_path):
        cached = json.load(open(cache_path, encoding='utf-8'))
    contexts = []
    doc = fitz.open(pdf_path)
    for page in pages:
        key = str(page)
        if key not in cached:
            cached[key] = doc[page - 1].get_text()
            json.dump(cached, open(cache_path, 'w', encoding='utf-8'), ensure_ascii=False)
        contexts.append({'page': page, 'markdown': cached[key]})
    return contexts


def judge_assay(extracted, expected_members):
    """Deterministic value judge: exact match after whitespace stripping."""
    correct, wrong, missing = [], [], []
    for member_id, truth in expected_members.items():
        entry = extracted.get(member_id)
        value = entry.get('value') if isinstance(entry, dict) else entry
        if value is None or str(value).strip() == '':
            missing.append(member_id)
        elif strip_spaces(value) == strip_spaces(truth['value']):
            correct.append(member_id)
        else:
            wrong.append({'id': member_id, 'got': str(value), 'want': truth['value']})
    unexpected = [k for k in extracted if k not in expected_members]
    return {
        'expected': len(expected_members),
        'correct': len(correct),
        'missing': missing,
        'wrong': wrong,
        'unexpected_keys': unexpected,
    }


def judge_structure_benzimidazole(member_rows, truth):
    """Contamination gate: emitted structure must equal OPSIN reference."""
    reference = {}
    for member_id, data in truth['members'].items():
        if not data.get('name'):
            continue
        mol, _ = name_to_molecule(data['name'])
        if mol is not None:
            reference[member_id] = Chem.MolToSmiles(mol, isomericSmiles=False)
    emitted = {}
    for row in member_rows:
        compound_id = str(row.get('COMPOUND_ID') or '')
        smiles = canonical(str(row.get('SMILES') or ''))
        if compound_id and smiles:
            emitted[compound_id] = smiles
    matched, contaminated = [], []
    for member_id, smiles in emitted.items():
        if member_id in reference and smiles == reference[member_id]:
            matched.append(member_id)
        else:
            contaminated.append({
                'id': member_id,
                'got': smiles,
                'want': reference.get(member_id, '(no reference)'),
            })
    return {
        'reference_members': len(reference),
        'emitted': len(emitted),
        'matched': len(matched),
        'contaminated': contaminated,
        'coverage': round(len(matched) / max(1, len(reference)), 3),
    }


def judge_structure_quinoline(member_rows, truth):
    """Core-containment judge: every emitted member must embed the
    quinoline-benzimidazole core (catches the document's corrupted names)."""
    core = Chem.MolFromSmarts(truth['core_smarts'])
    embedded, bad = [], []
    for row in member_rows:
        compound_id = str(row.get('COMPOUND_ID') or '')
        mol = Chem.MolFromSmiles(str(row.get('SMILES') or ''))
        if mol is not None and mol.HasSubstructMatch(core):
            embedded.append(compound_id)
        else:
            bad.append({'id': compound_id, 'smiles': str(row.get('SMILES') or '')[:80]})
    expected_members = [k for k in truth['members'] if k.startswith('9')]
    return {
        'expected_members': len(expected_members),
        'emitted': len(member_rows),
        'embedded_core': len(embedded),
        'violations': bad,
        'coverage': round(len(embedded) / max(1, len(expected_members)), 3),
    }


def run_assay_eval(paper, pdf_path, cache_path, out_dir):
    from activity_parser import extract_activity_data_multi
    truth = GROUND_TRUTH[paper]
    result = extract_activity_data_multi(
        pdf_file=pdf_path,
        assay_page_start=min(truth['assay_pages']),
        assay_page_end=max(truth['assay_pages']),
        assay_names=[truth['assay_name']],
        pages_per_chunk=2,
        # Production passes the structure-side series identifier; page-level
        # series expansion then supplies the member allowlist.
        compound_id_list=[truth['series_id']],
        output_dir=out_dir,
        structure_records=None,
    )
    extracted = result.get(truth['assay_name'], {})
    return judge_assay(extracted, truth['members'])


def run_structure_eval(paper, pdf_path, cache_path, out_dir):
    from utils.series_member_synthesis import synthesize_series_members
    truth = GROUND_TRUTH[paper]
    contexts = page_contexts_from_pdf(pdf_path, truth['assay_pages'], cache_path)
    import fitz
    total_pages = fitz.open(pdf_path).page_count
    pages_all = page_contexts_from_pdf(pdf_path, list(range(1, total_pages + 1)), cache_path)
    member_rows, report = synthesize_series_members([], pages_all, audit_path=os.path.join(out_dir, 'model_calls.jsonl'))
    rows = [r for r in member_rows if str(r.get('SERIES_COMPOUND_ID') or '') == truth['series_id']]
    if paper == 'benzimidazole':
        verdict = judge_structure_benzimidazole(rows, truth)
    else:
        verdict = judge_structure_quinoline(rows, truth)
    verdict['report'] = [
        {
            'series_id': s.get('series_id'),
            'defined': s.get('defined_members'),
            'emitted': s.get('emitted'),
            'quarantined': s.get('quarantined', 0),
            'consistency': s.get('consistency'),
        }
        for s in report.get('series', [])
    ]
    return verdict


def main():
    rolls = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    papers = sys.argv[2].split(',') if len(sys.argv) > 2 else ['benzimidazole', 'quinoline']
    os.makedirs(ARTIFACTS, exist_ok=True)
    pdfs = {
        'benzimidazole': os.path.join(ROOT, '..', 'data', 'literature_test', 'scirep_benzimidazole.pdf'),
        'quinoline': os.path.join(ROOT, '..', 'data', 'literature_test', 'scirep_quinoline.pdf'),
    }
    overall = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'rolls': rolls, 'papers': {}}
    for paper in papers:
        paper_result = {'assay': None, 'structure_rolls': [], 'pass': True}
        out_dir = os.path.join(ARTIFACTS, f'{paper}_{int(time.time())}')
        os.makedirs(out_dir, exist_ok=True)
        cache_path = os.path.join(ARTIFACTS, f'{paper}_pages.json')
        try:
            paper_result['assay'] = run_assay_eval(paper, pdfs[paper], cache_path, out_dir)
            if paper_result['assay']['wrong'] or paper_result['assay']['unexpected_keys']:
                paper_result['pass'] = False
        except Exception:
            paper_result['assay'] = {'error': traceback.format_exc()[-800:]}
            paper_result['pass'] = False
        for roll in range(1, rolls + 1):
            try:
                verdict = run_structure_eval(paper, pdfs[paper], cache_path, out_dir)
                contamination_key = 'contaminated' if paper == 'benzimidazole' else 'violations'
                if verdict.get(contamination_key):
                    verdict['roll_pass'] = False
                    paper_result['pass'] = False
                else:
                    verdict['roll_pass'] = True
                paper_result['structure_rolls'].append(verdict)
                print(f"[{paper}] roll {roll}: coverage={verdict.get('coverage')} "
                      f"{'CONTAMINATED: ' + str(verdict.get(contamination_key)) if verdict.get(contamination_key) else 'clean'}")
            except Exception:
                paper_result['structure_rolls'].append({'error': traceback.format_exc()[-800:], 'roll_pass': False})
                paper_result['pass'] = False
        if paper_result['assay'] and 'wrong' in paper_result['assay']:
            a = paper_result['assay']
            print(f"[{paper}] assay: {a['correct']}/{a['expected']} correct, "
                  f"missing={len(a['missing'])}, wrong={len(a['wrong'])}, unexpected={a['unexpected_keys']}")
        overall['papers'][paper] = paper_result
    with open(os.path.join(ARTIFACTS, 'runs.jsonl'), 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(overall, ensure_ascii=False) + '\n')
    print('PASS' if all(p.get('pass') for p in overall['papers'].values()) else 'FAIL')
    return 0 if all(p.get('pass') for p in overall['papers'].values()) else 1


if __name__ == '__main__':
    sys.exit(main())
