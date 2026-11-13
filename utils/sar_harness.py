"""Document-wide SAR evidence accounting and bounded, gap-driven tool selection.

A successful API call is not a correctness verdict. Geometry, molecular identity,
subject labels and measurements are separate evidence domains. External name
lookup is eligible only for a source-supported name and a recorded need.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path

OPERATIONS = {
    'discover_measurements': {'purpose': 'Inspect unaccounted document pages for biological, ADME, safety and property result records', 'cost': 'vision'},
    'locate_boxes': {'purpose': 'Acquire missing geometry before image decoding', 'cost': 'detector'},
    'read_region_evidence': {'purpose': 'Resolve local subject and measurement ownership using tight/expanded evidence', 'cost': 'vision'},
    'recognize_smiles': {'purpose': 'Decode a grounded crop and independently verify the graph', 'cost': 'inference'},
    'opsin_resolve': {'purpose': 'Parse a source-supported full name or attachment fragment locally', 'cost': 'local'},
    'pubchem_lookup': {'purpose': 'Resolve a failed local name or cross-check a documented identity conflict', 'cost': 'external'},
    'assemble_fragments': {'purpose': 'Bind row-owned substituents to explicit template sites, then verify the assembly', 'cost': 'chemistry+vision'},
    'read_result_table': {'purpose': 'Recover unaccounted compound/result rows and grouped headers from a discovered result region', 'cost': 'vision'},
    'peptide_graph_validate': {'purpose': 'Account for typed residues, noncanonical modifications, stereochemistry and ring connections without replacing the native model graph', 'cost': 'local_chemistry'},
    'human_review': {'purpose': 'Resolve missing or contradictory evidence after bounded attempts', 'cost': 'human'},
}


def _json(value, fallback):
    try: return json.loads(value) if isinstance(value, str) else fallback
    except (TypeError, ValueError): return fallback


def unit_id(record, index):
    anchor = str(record.get('RECORD_ID') or record.get('BOX_COORDS_FILE') or record.get('SEGMENT_FILE') or f'row:{index}')
    return 'sar_' + hashlib.sha256(anchor.encode()).hexdigest()[:16]


def _graph(smiles):
    from rdkit import Chem
    try: mol = Chem.MolFromSmiles(str(smiles or ''))
    except Exception: mol = None
    if mol is None or mol.GetNumAtoms() == 0: return None, False
    return Chem.MolToSmiles(mol), any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())


def assess_sar(records, *, page_entities=None, result_pages=None, assay_records=None, previous=None):
    previous = previous or {}
    attempts = list(previous.get('attempts') or [])
    units, gaps, identities = [], [], {}
    for index, row in enumerate(records):
        identifier = str(row.get('COMPOUND_ID') or '').strip()
        key = unit_id(row, index)
        graph, variable = _graph(row.get('SMILES'))
        typ = str(row.get('STRUCTURE_TYPE') or '')
        is_template = typ in {'markush', 'fragment', 'text_substituent'} or variable
        user_owned = row.get('STRUCTURE_REVIEW_SOURCE') == 'user'
        locked_fields = _json(row.get('USER_LOCKED_FIELDS'), [])
        local_status = row.get('LOCAL_EVIDENCE_STATUS')
        observations = _json(row.get('BIOACTIVITY_OBSERVATIONS'), [])
        edits = _json(row.get('USER_MEASUREMENT_EDITS'), {})
        if not isinstance(observations, list): observations = []
        if not isinstance(edits, dict): edits = {}
        peptide_notation = str(row.get('PEPTIDE_NOTATION') or row.get('SEQUENCE_TEXT') or '')
        has_peptide = bool(peptide_notation or row.get('SEQUENCE_REF'))
        if graph and not has_peptide:
            from rdkit import Chem
            molecule = Chem.MolFromSmiles(graph)
            has_peptide = len(molecule.GetSubstructMatches(Chem.MolFromSmarts('[N]-[C;X4]-[C](=O)-[N]'))) >= 2
        page = row.get('PAGE_NUM') or row.get('page_num') or row.get('page')
        unit = {'unit_id': key, 'row_index': index, 'page': page, 'compound_id': identifier,
            'result_only': str(row.get('RESULT_ONLY') or '').lower() == 'true' or typ == 'measurement_record',
            'kind': 'template' if is_template else 'peptide' if has_peptide else 'compound', 'peptide_notation': peptide_notation, 'graph_valid': graph is not None,
            'source_crop': str(row.get('SEGMENT_FILE') or ''), 'source_box': str(row.get('BOX_COORDS_FILE') or ''),
            'name': str(row.get('COMPOUND_NAME') or ''), 'has_name_candidate': bool(row.get('NAME_STRUCTURE_CANDIDATE')), 'user_owned': user_owned, 'locked_fields': locked_fields,
            'observation_count': len(observations), 'manual_measurement_count': len(edits), 'gaps': []}
        def gap(field, reason):
            entry = {'gap_id': f'{key}:{field}', 'unit_id': key, 'row_index': index, 'page': page,
                     'field': field, 'reason': reason, 'status': 'open'}
            gaps.append(entry); unit['gaps'].append(entry['gap_id'])
        if has_peptide and graph and _json(row.get('PEPTIDE_GRAPH_VALIDATION'), {}).get('status') != 'resolved':
            gap('peptide_structure', 'Typed residues, modifications and cyclization connections require complete graph accounting')
        if is_template:
            if row.get('MARKUSH_ASSEMBLY_STATUS') != 'assembled': gap('assembly', 'Template/fragment still needs explicit site and row-owned substituent evidence')
        else:
            if not graph: gap('smiles', 'No chemically parseable complete structure')
            if not identifier: gap('compound_id', 'No source-supported compound identifier')
            if local_status in {'failed', 'needs_review'} and not user_owned and row.get('ID_SOURCE') != 'manual':
                gap('ownership', 'Local identifier/value ownership is unresolved')
        for observation in observations:
            if not identifier or str(observation.get('subject_id') or '') != identifier:
                gap('measurement_ownership', 'Measurement subject conflicts with the current compound identifier'); break
        cells = {}
        for observation in observations:
            if not isinstance(observation, dict): continue
            source = (observation.get('page'), observation.get('table_index'), observation.get('column'), observation.get('source_region'))
            if source[0] is not None and source[1] is not None:
                cells.setdefault(source, set()).add(str(observation.get('value') or ''))
        if any(len(values) > 1 and source[2] not in edits for source, values in cells.items()):
            gap('measurement_conflict', 'Different reads disagree about the same source cell; review the source before choosing a value')
        if any(item.get('association_status') == 'needs_review' for item in observations):
            gap('measurement_ownership', 'Comparison prose needs individual subject/value verification; quarantined readings are retained')
        if row.get('PREVIOUS_ANNOTATION_TEXT') and not observations and local_status in {'failed', 'needs_review'}:
            gap('measurement', 'Previous annotation could not be bound to the local subject')
        if identifier and graph and not is_template:
            identities.setdefault(identifier, {}).setdefault(graph, []).append(key)
        units.append(unit)
    by_unit = {item['unit_id']: item for item in units}
    for identifier, graphs in identities.items():
        if len(graphs) > 1:
            for matching in graphs.values():
                for key in matching:
                    entry = {'gap_id': f'{key}:identity_conflict', 'unit_id': key, 'row_index': by_unit[key]['row_index'],
                        'page': by_unit[key]['page'], 'field': 'identity_conflict', 'reason': f'Identifier {identifier} has multiple distinct graphs; do not merge assay values by ID alone', 'status': 'open'}
                    gaps.append(entry); by_unit[key]['gaps'].append(entry['gap_id'])
    inventory = []
    if result_pages is None:
        gaps.append({'gap_id': 'document:measurement_discovery', 'unit_id': None, 'page': None, 'field': 'measurement_discovery', 'reason': 'Document-wide measurement coverage has not been inspected', 'status': 'open'})
    for page in page_entities or []:
        number = page.get('page_num')
        observed = {item['compound_id'] for item in units if str(item['page']) == str(number)}
        expected = {str(item.get('label_in_page') or '').strip() for item in page.get('compounds', []) if item.get('label_in_page') and item.get('evidence_kind') not in {'result_row', 'text_name', 'sequence'}}
        # Member structures can be assembled elsewhere in the document. A
        # result-row mention never requires detecting a molecule on that page.
        observed.update(item['compound_id'] for item in units if item['graph_valid'] and item['kind'] != 'template' and any(str(entry.get('label_in_page') or '') == item['compound_id'] and entry.get('evidence_kind') == 'template_member' for entry in page.get('compounds', [])))
        missing = sorted(expected - observed)
        inventory.append({'page': number, 'expected_subjects': len(expected), 'represented_subjects': len(expected & observed), 'missing_subjects': missing, 'page_read_ok': page.get('ok', True)})
        for label in missing:
            gaps.append({'gap_id': f'page:{number}:subject:{label}', 'unit_id': None, 'page': number,
                'field': 'coverage', 'reason': 'Page inventory subject is not represented by an extracted unit', 'expected_subject': label, 'status': 'open'})
    # A discovered result page must have accounted result records, not only structures.
    represented_result_pages = {str(row.get('PAGE_NUM') or row.get('page_num') or '') for row in records if _json(row.get('BIOACTIVITY_OBSERVATIONS'), [])}
    for row in records:
        represented_result_pages.update(str(page) for page in _json(row.get('RESULT_SOURCE_PAGES'), []))
        represented_result_pages.update(str(item.get('page')) for item in _json(row.get('BIOACTIVITY_OBSERVATIONS'), []) if item.get('page'))
    represented_result_pages.update(str(row.get('PAGE_NUM') or row.get('page_num') or '') for row in assay_records or [] if row.get('PAGE_NUM') or row.get('page_num'))
    for page in result_pages or []:
        if str(page) not in represented_result_pages:
            gaps.append({'gap_id': f'page:{page}:results', 'unit_id': None, 'page': page, 'field': 'result_records',
                'reason': 'Result page discovered but compound-level cells have not been accounted for', 'status': 'open'})
    return {'goal': 'Recover traceable compound–structure–measurement relationships from the document',
        'units': units, 'gaps': gaps, 'result_pages': result_pages, 'measurement_discovery': previous.get('measurement_discovery'), 'page_inventory': inventory, 'attempts': attempts, 'recovery_actions_used': int(previous.get('recovery_actions_used') or 0),
        'coverage': {'units': len(units), 'open_gaps': len(gaps), 'inventory_pages': len(inventory),
            'inventory_checked': bool(inventory), 'correctness_confirmed': bool(units) and all(row.get('STRUCTURE_IMAGE_VERIFIED') in {True, 'True', 'true'} or row.get('STRUCTURE_REVIEW_SOURCE') == 'user' for row in records) and not gaps, 'complete': bool(units) and bool(inventory) and not gaps},
        'operations': OPERATIONS, 'planning_failures': list(previous.get('planning_failures') or [])}


def plan_recovery(report, max_actions=8):
    by_id = {item['unit_id']: item for item in report['units']}
    attempts = report.get('attempts') or []
    actions, seen = [], set()
    priority = {'measurement_discovery': 0, 'coverage': 1, 'identity_conflict': 1, 'result_records': 2}
    for gap in sorted(report['gaps'], key=lambda item: priority.get(item['field'], 3)):
        unit = by_id.get(gap.get('unit_id'), {})
        if unit.get('user_owned') or (gap['field'] == 'compound_id' and 'COMPOUND_ID' in unit.get('locked_fields', [])):
            continue
        previous = [item for item in attempts if item.get('gap_id') == gap['gap_id']]
        field = gap['field']
        if len(previous) >= 3:
            operation = 'human_review'
        elif field == 'measurement_discovery': operation = 'discover_measurements'
        elif field == 'coverage': operation = 'locate_boxes'
        elif field == 'result_records': operation = 'read_result_table'
        elif field == 'assembly': operation = 'assemble_fragments'
        elif field == 'peptide_structure': operation = 'peptide_graph_validate' if not previous else 'human_review'
        elif field in {'compound_id', 'ownership', 'measurement', 'measurement_ownership', 'identity_conflict'}:
            operation = 'human_review' if any(item.get('operation') == 'read_region_evidence' for item in previous) else ('read_region_evidence' if unit.get('source_crop') or unit.get('source_box') else 'locate_boxes')
        elif field == 'smiles':
            tried = {item['operation'] for item in previous}
            if unit.get('has_name_candidate'): operation = 'human_review'
            elif unit.get('source_crop') and not unit.get('result_only') and 'recognize_smiles' not in tried: operation = 'recognize_smiles'
            elif unit.get('name') and 'opsin_resolve' not in tried: operation = 'opsin_resolve'
            elif unit.get('name') and any(item['operation'] == 'opsin_resolve' and item.get('status') == 'unresolved' for item in previous): operation = 'pubchem_lookup'
            else: operation = 'human_review'
        else: operation = 'human_review'
        if operation in {'locate_boxes', 'assemble_fragments', 'read_result_table'}:
            scope_attempts = [item for item in attempts if item.get('operation') == operation and (str(item.get('page')) == str(gap.get('page')) or str(item.get('gap_id', '')).startswith(f"page:{gap.get('page')}:"))]
            # One expensive page pass addresses every subject on that page.
            # Repeating it for each missing label wastes budget and can never
            # substitute for a different source of identity evidence.
            if scope_attempts:
                operation = 'human_review'
        fingerprint = (gap.get('page') if operation in {'locate_boxes', 'assemble_fragments', 'read_result_table'} else gap.get('unit_id') or gap['page'], operation)
        if fingerprint in seen: continue
        seen.add(fingerprint)
        actions.append({'operation': operation, 'gap_id': gap['gap_id'], 'unit_id': gap.get('unit_id'),
            'row_index': unit.get('row_index'), 'page': gap.get('page'), 'reason': gap['reason'],
            'evidence': {'source_crop': unit.get('source_crop'), 'source_box': unit.get('source_box'), 'name': unit.get('name')},
            'requires_review': operation == 'human_review'})
        if len(actions) >= max_actions: break
    return actions


def save_sar_report(output_dir, records, **context):
    path = Path(output_dir) / 'sar_harness.json'
    try: previous = json.loads(path.read_text())
    except (OSError, ValueError): previous = {}
    report = assess_sar(records, previous=previous, **context)
    report['next_actions'] = plan_recovery(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.json.tmp')
    try:
        with os.fdopen(fd, 'w') as stream: json.dump(report, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return report


async def resolve_gap_name(report, action):
    """Execute one eligible name operation with registry audit and no implicit PubChem fallback."""
    from skills.registry import get, load_skills
    if action['operation'] not in {'opsin_resolve', 'pubchem_lookup'}:
        raise ValueError('Action is not a name-resolution operation')
    if action['operation'] == 'pubchem_lookup' and not any(item.get('gap_id') == action['gap_id'] and item.get('operation') == 'opsin_resolve' and item.get('status') == 'unresolved' for item in report.get('attempts', [])):
        raise ValueError('External lookup requires a recorded failed local name parse')
    name = action.get('evidence', {}).get('name')
    if not name: raise ValueError('No source-supported chemical name')
    load_skills()
    result = await get(action['operation']).handler({'name': name, 'intent': 'compound', 'evidence': action['reason'], 'purpose': 'sar_gap_recovery'})
    report.setdefault('attempts', []).append({'gap_id': action['gap_id'], 'operation': action['operation'], 'status': result.get('status'), 'result': result})
    return result


def reserve_result_action(selected, eligible, report, budget):
    """Reserve progress on source data while geometry/identity recovery continues.

    This is a document-wide coverage constraint, independent of endpoint names,
    page numbers, chemistry families or a model's preferred reading order.
    """
    if not budget or any(item['operation'] == 'read_result_table' for item in selected): return selected
    tables = [item for item in eligible if item['operation'] == 'read_result_table']
    if not tables: return selected
    decisions = (report.get('measurement_discovery') or {}).get('decisions') or {}
    def evidence_density(item):
        evidence = decisions.get(str(item.get('page'))) or decisions.get(item.get('page')) or {}
        return len(evidence.get('endpoint_descriptors') or evidence.get('assay_names') or [])
    required = max(tables, key=evidence_density)
    return [{**required, 'planning_reason': 'Host reserves document-result coverage; prefer richer measured endpoint evidence'}, *selected[:max(0, budget-1)]]


def select_recovery_actions(report, max_actions=8, audit_path=None):
    """A semantic planner ranks only host-eligible actions; it cannot fabricate inputs."""
    eligible = plan_recovery(report, max_actions=64)
    if not eligible: return []
    from utils.llm_utils import LLM_TEXT_MODEL_KEY, LLM_TEXT_MODEL_NAME, LLM_TEXT_MODEL_URL, require_openai
    from utils.model_harness import run_json_task, extract_json_content
    from utils.skill_prompt_loader import render_skill_prompt_with_examples
    projection = {'goal': report['goal'], 'coverage': report['coverage'], 'eligible_actions': eligible,
        'operations': OPERATIONS, 'measurement_inventory': report.get('measurement_discovery'), 'recent_attempts': [{key: item.get(key) for key in ('gap_id', 'operation', 'status', 'error')} for item in report.get('attempts', [])[-12:]]}
    prompt = render_skill_prompt_with_examples('biocheminsight-text-models', 'references/plan_sar_recovery_prompt.md', None,
        {'STATE_JSON': json.dumps(projection, ensure_ascii=False), 'MAX_ACTIONS': str(max_actions)})
    by_key = {(item['gap_id'], item['operation']): item for item in eligible}
    def parse(text):
        payload = json.loads(extract_json_content(text))
        actions = payload.get('actions')
        if not isinstance(actions, list) or len(actions) > max_actions:
            raise ValueError('Invalid recovery action budget')
        selected, seen = [], set()
        for item in actions:
            key = (item.get('gap_id'), item.get('operation'))
            if key not in by_key or key in seen:
                raise ValueError('Planner chose an ineligible or repeated operation')
            seen.add(key); selected.append({**by_key[key], 'planning_reason': str(item.get('reason') or '')})
        return reserve_result_action(selected, eligible, report, max_actions)
    client = require_openai()(api_key=LLM_TEXT_MODEL_KEY, base_url=LLM_TEXT_MODEL_URL, timeout=90, max_retries=0)
    def call():
        return client.chat.completions.create(model=LLM_TEXT_MODEL_NAME, messages=[{'role': 'user', 'content': prompt}], temperature=0).choices[0].message.content
    return run_json_task(task_name='plan_sar_recovery', channel='text', operation=call, parser=parse,
        retry=2, audit_path=audit_path, metadata={'scope': 'document_sar_recovery', 'eligible_actions': len(eligible)})


def _append_observations(records, extracted):
    """Only a unique ID/graph pair can claim table values. Ambiguity remains a separate row."""
    from utils.measurement_evidence import quarantine_ambiguous_prose_ownership
    quarantine_ambiguous_prose_ownership(records)
    for result in extracted['records']:
        subject = result['compound_id']
        matches = [row for row in records if str(row.get('COMPOUND_ID') or '') == subject and str(row.get('STRUCTURE_TYPE') or '') not in {'markush','fragment','text_substituent'}]
        valid = [row for row in matches if _graph(row.get('SMILES'))[0] and not _graph(row.get('SMILES'))[1] and row.get('MEASUREMENT_LINK_STATUS') != 'identifier_unverified']
        graphs = {_graph(row.get('SMILES'))[0] for row in valid}
        results = [row for row in matches if str(row.get('RESULT_ONLY') or '').lower() == 'true']
        target = valid[0] if valid and len(graphs) == 1 else results[0] if len(results) == 1 else matches[0] if len(matches) == 1 and not matches[0].get('SMILES') else None
        if target is None:
            target = {'RECORD_ID': f"result:{result['page']}:{result['table_index']}:{subject}",
                'COMPOUND_ID': subject, 'PAGE_NUM': result['page'], 'SMILES': '', 'RESULT_ONLY': 'true',
                'MEASUREMENT_LINK_STATUS': 'needs_structure_identity'}
            records.append(target)
        existing = _json(target.get('BIOACTIVITY_OBSERVATIONS'), [])
        incoming = result['observations']
        by_source = {(item.get('page'), item.get('table_index'), item.get('column'), item.get('source_region'), item.get('value')): item for item in existing}
        for observation in incoming:
            by_source[(observation['page'], observation['table_index'], observation['column'], observation.get('source_region'), observation.get('value'))] = observation
        target['BIOACTIVITY_OBSERVATIONS'] = json.dumps(list(by_source.values()), ensure_ascii=False)
        notes=result.get('notes') or []
        if isinstance(notes,list) and notes: target['MEASUREMENT_NOTES'] = '\n'.join(str(note) for note in notes)
        from utils.measurement_evidence import observation_values, user_measurement_edits
        target.update(observation_values(target))
        target.update(user_measurement_edits(target))
        pages = _json(target.get('RESULT_SOURCE_PAGES'), [])
        target['RESULT_SOURCE_PAGES'] = json.dumps(sorted(set(pages + [result['page']])))


def new_recovery_session(parent_report):
    """A user-requested new task gets a fresh budget; evidence history survives.

    Checkpoint resume subsequently restores that task's own budget usage.
    """
    return {**parent_report, 'attempts': list(parent_report.get('attempts') or []),
        'parent_actions_used': int(parent_report.get('recovery_actions_used') or 0),
        'recovery_actions_used': 0, 'planning_failures': []}


def recover_sar_rows(records, pdf_file, output_dir, *, page_entities=None, result_pages=None, max_actions=8, audit_path=None, progress=None):
    """Bounded observe→plan→act→verify→replan loop. Recovery produces reviewable rows;
    only independent verification can promote a decoded structure to resolved."""
    from utils.measurement_evidence import review_structure_regions, read_measurement_page
    from utils.api_harness import APIHarness
    import constants
    import copy
    records = copy.deepcopy(records)
    input_digest = hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode()).hexdigest()
    checkpoint_path = Path(output_dir) / 'recovery_checkpoint.json'
    report = save_sar_report(output_dir, records, page_entities=page_entities, result_pages=result_pages)
    try:
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get('input_digest') == input_digest:
            records, report = checkpoint['records'], checkpoint['report']
            result_pages = report.get('result_pages')
    except (OSError, ValueError, KeyError): pass
    def checkpoint():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=checkpoint_path.parent, suffix='.checkpoint.tmp')
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump({'input_digest': input_digest, 'records': records, 'report': report}, stream, ensure_ascii=False)
            os.replace(temporary, checkpoint_path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)

    used = int(report.get('recovery_actions_used') or 0)
    planner_available = not bool(report.get('planning_failures'))
    initial_used = used
    for round_index in range(3):
        report = assess_sar(records, page_entities=page_entities, result_pages=result_pages, previous=report)
        if not report['gaps'] or used >= max_actions: break
        try:
            if not planner_available: raise RuntimeError('Planner unavailable earlier in this task; use bounded host fallback')
            actions = select_recovery_actions(report, max_actions=min(4, max_actions-used), audit_path=audit_path)
        except Exception as exc:
            if planner_available:
                report.setdefault('planning_failures', []).append(f'{type(exc).__name__}: {str(exc)[:180]}')
            planner_available = False
            budget = min(2, max_actions-used)
            eligible = [action for action in plan_recovery(report, max_actions=64) if action['operation'] != 'human_review']
            actions = reserve_result_action(eligible[:budget], eligible, report, budget)
        if not actions or all(action['operation'] == 'human_review' for action in actions): break
        changed = False
        for action in actions:
            if action['operation'] == 'human_review': continue
            used += 1
            if progress: progress(used, max_actions, action)
            operation, row_index = action['operation'], action.get('row_index')
            row = records[row_index] if row_index is not None and row_index < len(records) else None
            attempt = {'gap_id': action['gap_id'], 'operation': operation, 'page': action.get('page'), 'round': round_index, 'status': 'unresolved'}
            before = json.dumps(records, sort_keys=True, default=str)
            try:
                if operation == 'discover_measurements':
                    import fitz
                    from utils.measurement_evidence import discover_pdf_measurements
                    with fitz.open(pdf_file) as document: pages = list(range(1, document.page_count + 1))
                    result_pages, names, decisions, texts = discover_pdf_measurements(pdf_file, pages, audit_path=audit_path)
                    report['result_pages'] = result_pages
                    report['measurement_discovery'] = {'pages': result_pages, 'names': names, 'decisions': decisions}
                    attempt['status'] = 'reviewed'
                    changed = True
                elif operation == 'read_region_evidence' and row is not None:
                    review_structure_regions([{'row': row, 'page_num': action.get('page'), 'box_json_path': row.get('BOX_COORDS_FILE'), 'page_image_path': row.get('PAGE_IMAGE_FILE')}], audit_path=audit_path)
                    attempt['status'] = row.get('LOCAL_EVIDENCE_STATUS', 'unresolved')
                elif operation in {'opsin_resolve', 'pubchem_lookup'}:
                    result = asyncio.run(resolve_gap_name({**report, 'attempts': list(report.get('attempts', []))}, action))
                    attempt.update(status=result.get('status'), result=result)
                    candidate = result.get('smiles') or (result.get('record') or {}).get('isomeric_smiles')
                    if row is not None and candidate and _graph(candidate)[0]:
                        # A name-derived candidate is not an image-verified graph.
                        row['NAME_STRUCTURE_CANDIDATE'] = candidate
                        row['NAME_STRUCTURE_SOURCE'] = operation
                        row['STRUCTURE_RECOVERY_STATUS'] = 'needs_visual_identity_check'
                elif operation == 'recognize_smiles' and row is not None:
                    base = os.environ.get('STRUCTURE_API_BASE_URL', 'http://harness-api:8011')
                    client = APIHarness(base, timeout=240); client.session.trust_env = False
                    result = client.image('resolve-figure', Path(row['SEGMENT_FILE']), {'document_id': Path(pdf_file).stem, **({'sequence_text': row.get('PEPTIDE_NOTATION') or row.get('SEQUENCE_TEXT')} if row.get('PEPTIDE_NOTATION') or row.get('SEQUENCE_TEXT') else {})}, {})
                    attempt.update(status=result.get('state'), result=result)
                    selected = result.get('selected') or {}
                    if result.get('state') == 'RESOLVED' and selected.get('verdict') == 'verified' and _graph(selected.get('smiles'))[0]:
                        row['SMILES'] = selected['smiles']; row['STRUCTURE_RECOVERY_STATUS'] = 'verified'
                    else:
                        row['STRUCTURE_RECOVERY_STATUS'] = 'needs_review'
                        row['STRUCTURE_RECOVERY_CANDIDATE'] = selected.get('smiles') or ''
                elif operation == 'peptide_graph_validate' and row is not None:
                    from skills.name_resolution.skill import peptide_graph_validate
                    result = asyncio.run(peptide_graph_validate({'smiles': row.get('SMILES')}))
                    row['PEPTIDE_GRAPH_VALIDATION'] = json.dumps(result, ensure_ascii=False)
                    row['PEPTIDE_GRAPH_STATUS'] = str(result.get('status') or 'unresolved')
                    attempt.update(status=result.get('status'), result=result)
                elif operation == 'read_result_table':
                    result = read_measurement_page(pdf_file, int(action['page']), audit_path=audit_path)
                    _append_observations(records, result)
                    from utils.text_fragment_assembly import attach_source_name_candidates
                    attach_source_name_candidates(records, pdf_file)
                    attempt.update(status='recovered' if result['records'] else 'no_extractable_records', gaps=result['gaps'])
                elif operation == 'locate_boxes':
                    from utils.sar_regions import locate_structure_regions
                    additions = locate_structure_regions(pdf_file, int(action['page']),
                        str(Path(output_dir) / f"region_recovery_{action['page']}"), records)
                    records.extend(additions)
                    attempt.update(status='geometry_acquired' if additions else 'no_new_regions', detector_confidence=.15)
                elif operation == 'assemble_fragments':
                    from pipeline import plan_markush_relationships_for_group, _build_markush_assembled_structure_records
                    page = int(action['page'])
                    payload = plan_markush_relationships_for_group(pdf_file, [page], records, [], str(Path(output_dir) / f'assembly_recovery_{page}'), audit_path=audit_path)
                    additions = _build_markush_assembled_structure_records(payload or {})
                    known = {(str(item.get('COMPOUND_ID') or ''), str(item.get('SMILES') or '')) for item in records}
                    for addition in additions:
                        key = (str(addition.get('COMPOUND_ID') or ''), str(addition.get('SMILES') or ''))
                        if key not in known: records.append(addition); known.add(key)
                    attempt['status'] = 'verified_members' if additions else 'assembly_needs_evidence'
                else: attempt['status'] = 'not_executable'
            except Exception as exc:
                attempt.update(status='failed', error=f'{type(exc).__name__}: {str(exc)[:200]}')
            report.setdefault('attempts', []).append(attempt)
            report['recovery_actions_used'] = used
            checkpoint()
            if before != json.dumps(records, sort_keys=True, default=str): changed = True
        # Attempt/discovery state can unlock a different operation even when
        # rows did not change. The round/action budgets bound replanning.
    # Preserve the attempts across the final independent assessment.
    report = assess_sar(records, page_entities=page_entities, result_pages=result_pages, previous=report)
    report['next_actions'] = plan_recovery(report)
    report['recovery_actions_used'] = used
    recent_attempts = report.get('attempts', [])[-max(1, used-initial_used):] if used > initial_used else []
    report['execution_status'] = 'blocked_provider' if report.get('planning_failures') and (not recent_attempts or all(item.get('status') == 'failed' for item in recent_attempts)) else 'partial' if report['gaps'] else 'finished'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / 'sar_harness.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return records, report
