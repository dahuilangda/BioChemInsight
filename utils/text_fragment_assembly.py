"""Turn grounded table-cell names into attachment fragments without a drawn crop.

Every member retains its own compound-row assignment. Local OPSIN (or the
shared, conservative PubChem fallback) supplies the fragment graph. Chemical
assembly and the pipeline's final visual review remain independent gates.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def build_text_fragment_assemblies(contexts, candidates, output_dir, audit_path=None):
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from PIL import Image, ImageDraw, ImageFont
    from utils.series_member_synthesis import substituent_text_to_attachment_fragment, assemble_member_on_scaffold
    from utils.markush_assembly import _load_mol, _dummy_atoms, _dummy_variable_label, _combine_labeled_substituents
    from utils.markush_text_substituent import normalize_variable_position

    folder = Path(output_dir) / 'text_fragments'; folder.mkdir(parents=True, exist_ok=True)
    assemblies, fragments, gaps = [], [], []
    seen = set()
    for context in contexts or []:
        page = context.get('page')
        scaffolds = [candidate for candidate in candidates if candidate.get('page') == page and candidate.get('structure_type') == 'markush']
        if len(scaffolds) != 1:
            if context.get('text_assignments'):
                gaps.append({'page': page, 'reason': 'No unique scaffold in the table scope'})
            continue
        scaffold = scaffolds[0]
        block = scaffold.get('molblock_full') or scaffold.get('molblock') or ''
        molecule = _load_mol(block)
        if molecule is None:
            gaps.append({'page': page, 'reason': 'Scaffold molblock unavailable'}); continue
        sites = _dummy_atoms(molecule)
        if not sites:
            gaps.append({'page': page, 'reason': 'Scaffold has no explicit attachment sites'}); continue
        groups = {}
        for assignment in context.get('text_assignments') or []:
            cid = str(assignment.get('compound_id') or '').strip()
            variable = normalize_variable_position(assignment.get('variable_position') or '')
            text = str(assignment.get('substituent_text') or '').strip()
            if cid and variable and text:
                groups.setdefault(cid, {}).setdefault(variable, set()).add(text)
        for cid, assignments in groups.items():
            if any(len(texts) != 1 for texts in assignments.values()):
                gaps.append({'page': page, 'compound_id': cid, 'reason': 'Conflicting text for the same row and variable'}); continue
            if len(assignments) != len(sites):
                gaps.append({'page': page, 'compound_id': cid, 'reason': 'Not every scaffold site has a row-owned substituent'}); continue
            labels = [_dummy_variable_label(molecule.GetAtomWithIdx(site)) for site in sites]
            if len(sites) > 1 and set(labels) != set(assignments):
                gaps.append({'page': page, 'compound_id': cid, 'reason': 'Multi-site variable mapping is ambiguous'}); continue
            members, refs, variables, evidence = [], [], [], []
            for position, (variable, texts) in enumerate(sorted(assignments.items())):
                text = next(iter(texts))
                fragment = substituent_text_to_attachment_fragment(text, audit_path=audit_path)
                if fragment is None:
                    gaps.append({'page': page, 'compound_id': cid, 'variable': variable, 'reason': 'No unambiguous name-derived attachment fragment'}); break
                mapped = labels[0] or variable if len(sites) == 1 else variable
                AllChem.Compute2DCoords(fragment)
                fragment_smiles = Chem.MolToSmiles(fragment)
                digest = hashlib.sha256(json.dumps([page, cid, variable, text], ensure_ascii=False).encode()).hexdigest()[:20]
                ref = f'text_fragment:{digest}'
                image_path = folder / f'{digest}.png'
                drawing = Draw.MolToImage(fragment, size=(560, 260))
                panel = Image.new('RGB', (560, 340), 'white'); panel.paste(drawing, (0, 70))
                try: font = ImageFont.truetype('DejaVuSans.ttf', 15)
                except OSError: font = ImageFont.load_default()
                ImageDraw.Draw(panel).text((16, 10), f'Compound {cid} | {variable} = {text}', fill='#334155', font=font)
                ImageDraw.Draw(panel).text((16, 38), 'Source: table text; graph: name parser', fill='#64748b', font=font)
                panel.save(image_path)
                fragments.append({'ref': ref, 'page': page, 'structure_type': 'text_substituent',
                    'candidate_source': fragment.GetProp('_LookupSource') if fragment.HasProp('_LookupSource') else 'name_parser',
                    'smiles': fragment_smiles, 'molblock': Chem.MolToMolBlock(fragment), 'molblock_full': Chem.MolToMolBlock(fragment),
                    'image_file': str(image_path), 'segment_file': str(image_path),
                    'text_evidence': {'compound_id': cid, 'variable': variable, 'substituent_text': text, 'page': page}})
                members.append((mapped, fragment)); refs.append(ref); variables.append(mapped)
                evidence.append({'compound_id': cid, 'variable': variable, 'mapped_site': mapped, 'text': text, 'page': page})
            else:
                if len(members) == 1:
                    product, reason = assemble_member_on_scaffold(block, Chem.MolToSmiles(members[0][1]), members[0][0])
                else:
                    smiles, molblock, reason, notes = _combine_labeled_substituents(molecule, members)
                    product = Chem.MolFromMolBlock(molblock) if molblock else None
                if product is None or any(atom.GetAtomicNum() == 0 for atom in product.GetAtoms()):
                    gaps.append({'page': page, 'compound_id': cid, 'reason': reason or 'Incomplete assembly'}); continue
                Chem.SanitizeMol(product)
                key = (cid, scaffold['ref'], tuple(variables))
                if key in seen:
                    gaps.append({'page': page, 'compound_id': cid, 'reason': 'Duplicate member assignment'}); continue
                seen.add(key)
                assemblies.append({'record_id': f'text_assembly:{cid}:{page}', 'compound_id': cid,
                    'scaffold_ref': scaffold['ref'], 'fragment_refs': refs, 'variable_positions': variables,
                    'source_pages': [page], 'assembly_status': 'assembled', 'blocked_reasons': [],
                    'assembled_smiles': Chem.MolToSmiles(product), 'assembled_molblock': Chem.MolToMolBlock(product),
                    'method': 'name_parser_explicit_attachment', 'normalization_notes': [],
                    'text_assignment_evidence': evidence})
    return assemblies, fragments, gaps


def crosscheck_assembly_names(assemblies, contexts):
    """An independently stated full name can veto a visually plausible graph.

    No name means no independent confirmation; it never licenses inventing a
    name or replacing the source graph with a guessed product.
    """
    from rdkit import Chem
    from utils.opsin_local import name_to_smiles as py2opsin
    from utils.series_member_synthesis import harvest_full_names_from_text
    names = harvest_full_names_from_text([item.get('compound_id') for item in assemblies], contexts)
    for assembly in assemblies:
        name = names.get(str(assembly.get('compound_id') or ''))
        review = {'status': 'unavailable', 'method': 'source_full_name_local_parse'}
        assembly['independent_name_review'] = review
        if not name: continue
        expected = Chem.MolFromSmiles(py2opsin(name))
        actual = Chem.MolFromSmiles(str(assembly.get('assembled_smiles') or ''))
        if expected is None or actual is None:
            review.update(status='unresolved', source_name=name)
            continue
        expected_smiles = Chem.MolToSmiles(expected, isomericSmiles=True)
        actual_smiles = Chem.MolToSmiles(actual, isomericSmiles=True)
        review.update(status='exact_match' if expected_smiles == actual_smiles else 'conflict', source_name=name, source_name_smiles=expected_smiles)
        if review['status'] == 'conflict':
            assembly['assembly_status'] = 'blocked'
            assembly['blocked_reasons'] = list(dict.fromkeys([*(assembly.get('blocked_reasons') or []), 'independent_source_name_graph_conflict']))
    return assemblies


def attach_source_name_candidates(records, pdf_file):
    """Supplement result-only rows from unique printed full names, as proposals.

    Source-derived candidates remain separate from the accepted image graph.
    This local operation does not call external name services.
    """
    import fitz
    from rdkit import Chem
    from utils.opsin_local import name_to_smiles as py2opsin
    from utils.series_member_synthesis import harvest_full_names_from_text
    pending = [row for row in records if row.get('COMPOUND_ID') and row.get('STRUCTURE_REVIEW_SOURCE') != 'user' and not row.get('NAME_STRUCTURE_CANDIDATE')]
    if not pending: return
    with fitz.open(pdf_file) as document:
        contexts = [{'page': index+1, 'markdown': page.get_text()} for index, page in enumerate(document)]
    names = harvest_full_names_from_text([row['COMPOUND_ID'] for row in pending], contexts)
    for row in pending:
        name = names.get(str(row['COMPOUND_ID']))
        if not name: continue
        molecule = Chem.MolFromSmiles(py2opsin(name))
        if molecule is None or any(atom.GetAtomicNum() == 0 or atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()): continue
        proposal = Chem.MolToSmiles(molecule, isomericSmiles=True)
        current = str(row.get('SMILES') or '')
        if current:
            decoded = Chem.MolFromSmiles(current)
            exact = bool(decoded and Chem.MolToSmiles(decoded, isomericSmiles=True) == proposal)
            row['INDEPENDENT_NAME_REVIEW'] = json.dumps({'status': 'exact_match' if exact else 'conflict', 'source_name': name}, ensure_ascii=False)
            if exact: continue
            # Preserve the native decoded graph; a source conflict cannot be
            # published as a resolved molecule or silently replaced by a name.
            row['MODEL_STRUCTURE_CANDIDATE'] = current
            row['SMILES'] = ''
            row['MOLBLOCK'] = ''
            row['STRUCTURE_IMAGE_VERIFIED'] = False
            row['STRUCTURE_RECOVERY_STATUS'] = 'source_name_graph_conflict'
        row.update(COMPOUND_NAME=name, NAME_STRUCTURE_CANDIDATE=proposal,
            NAME_STRUCTURE_SOURCE='source_full_name_local_parse', SOURCE_NAME_EVIDENCE=name)
        if not current: row['STRUCTURE_RECOVERY_STATUS'] = 'name_proposal_requires_review'
