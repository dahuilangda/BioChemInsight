"""Grounded measurement evidence shared by page discovery, local reviews and exports.

No document labels or expected result values live here. Geometry belongs to the host;
models transcribe the selected subject and classify endpoint/unit/provenance.
"""
from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from pathlib import Path

import constants

KINDS = {'experimental', 'predicted', 'calculated', 'unknown'}
MEASUREMENT_DISCOVERY_DPI = int(getattr(constants, 'MEASUREMENT_DISCOVERY_DPI', 150))


def normalize_unit(value):
    return str(value or '').strip().replace('μ', 'µ').replace('uM', 'µM')


def normalize_assay_units(name):
    return re.sub(r'\(\s*[uμµ]M\s*\)', '(µM)', name)


def unique_assay_names(names):
    unique = {}
    for name in names:
        unique.setdefault(normalize_assay_units(name), name)
    return list(unique.values())


def assay_column_with_unit(name, unit):
    column = normalize_assay_units(name)
    unit = normalize_unit(unit)
    if unit and f'({unit})' not in column:
        return f'{column} ({unit})'
    return column


def measurement_column(observation):
    endpoint = unicodedata.normalize('NFKC', str(observation.get('endpoint') or '').strip())
    if not endpoint:
        raise ValueError('Measurement endpoint missing')
    unit = normalize_unit(observation.get('unit'))
    header_unit = re.search(r'\(([^()]*)\)$', endpoint)
    if unit and header_unit and normalize_unit(header_unit[1]) == unit:
        endpoint = endpoint[:header_unit.start()].rstrip()
    # Uncertainty describes the reported value, not a different assay endpoint.
    endpoint = re.sub(r'\s*(?:±|\+/-)\s*(?:SD|SEM|SE|standard deviation|standard error)\s*$', '', endpoint, flags=re.IGNORECASE).strip()
    target = str(observation.get('target') or '').strip()
    target_in_endpoint = bool(target and re.search(r'(?<!\w)' + re.escape(target) + r'(?!\w)', endpoint, re.IGNORECASE))
    name = f'{target} · {endpoint}' if target and not target_in_endpoint else endpoint
    if unit:
        name += f' ({unit})'
    conditions = str(observation.get('conditions') or '').strip()
    if conditions:
        name += f' · {conditions}'
    kind = str(observation.get('source_kind') or 'unknown')
    if kind in {'predicted', 'calculated'}:
        name += f' · {kind}'
    return name


def parse_local_evidence(text):
    from utils.model_harness import extract_json_content
    payload = json.loads(extract_json_content(text))
    for key in ('compound_id', 'annotation_text', 'confidence', 'gap'):
        if not isinstance(payload.get(key), str):
            raise ValueError(f'Local evidence missing {key}')
    if payload['confidence'] not in {'high', 'medium', 'low'} or not isinstance(payload.get('observations'), list):
        raise ValueError('Invalid local evidence contract')
    subject = payload['compound_id'].strip()
    accepted, rejected = [], []
    for item in payload['observations']:
        if not isinstance(item, dict):
            raise ValueError('Observation must be an object')
        for key in ('subject_id', 'endpoint', 'target', 'unit', 'value', 'source_kind', 'evidence'):
            if not isinstance(item.get(key), str):
                raise ValueError(f'Observation missing {key}')
        if item['source_kind'] not in KINDS:
            raise ValueError('Unknown measurement provenance')
        if not subject or item['subject_id'].strip() != subject or not item['endpoint'].strip() or not item['value'].strip() or not item['evidence'].strip():
            rejected.append(item)
            continue
        item['column'] = measurement_column(item)
        accepted.append(item)
    payload['observations'] = accepted
    payload['rejected_observations'] = rejected
    return payload


def parse_page_evidence(text):
    from utils.model_harness import extract_json_content
    payload = json.loads(extract_json_content(text))
    if type(payload.get('has_result_records')) is not bool or payload.get('confidence') not in {'high', 'medium', 'low'}:
        raise ValueError('Invalid measurement-page decision')
    if not isinstance(payload.get('endpoints'), list) or not isinstance(payload.get('evidence'), str):
        raise ValueError('Missing measurement-page evidence')
    for item in payload['endpoints']:
        if not isinstance(item, dict) or any(not isinstance(item.get(key), str) for key in ('endpoint', 'target', 'unit', 'source_kind')):
            raise ValueError('Invalid endpoint descriptor')
        if item['source_kind'] not in KINDS:
            raise ValueError('Unknown endpoint provenance')
        item['column'] = measurement_column(item)
    if payload['has_result_records'] and (not payload['endpoints'] or not payload['evidence'].strip()):
        raise ValueError('Positive discovery requires result evidence and endpoints')
    if not payload['has_result_records'] and payload['endpoints']:
        raise ValueError('Negative discovery cannot contain result endpoints')
    return payload


def read_evidence(image_file, task, *, audit_path=None, metadata=None):
    from utils.llm_utils import run_vision_json_task
    from utils.skill_prompt_loader import render_skill_reference
    local = task == 'local_structure_evidence'
    reference = 'local_structure_evidence_prompt.md' if local else 'discover_measurement_page_prompt.md'
    return run_vision_json_task(task_name=task, image_file=str(image_file),
        prompt=render_skill_reference('biocheminsight-vision-models', f'references/{reference}', {}),
        parser=parse_local_evidence if local else parse_page_evidence, audit_path=audit_path, metadata=metadata)


def apply_local_evidence(record, evidence):
    """Commit ownership as a unit; never retain observations from a previous subject."""
    record['LOCAL_EVIDENCE_STATUS'] = 'verified' if evidence['confidence'] == 'high' and not evidence['gap'] and evidence['compound_id'].strip() else 'needs_review'
    record['LOCAL_EVIDENCE_JSON'] = json.dumps(evidence, ensure_ascii=False)
    if record['LOCAL_EVIDENCE_STATUS'] != 'verified':
        record['PREVIOUS_ANNOTATION_TEXT'] = record.get('ANNOTATION_TEXT', '')
        record['ANNOTATION_TEXT'] = ''
        record['BIOACTIVITY_OBSERVATIONS'] = '[]'
        return
    previous = str(record.get('COMPOUND_ID') or '')
    subject = evidence['compound_id'].strip()
    if subject:
        if previous and previous != subject:
            record['PREVIOUS_COMPOUND_ID'] = previous
            record['OWNERSHIP_CORRECTION'] = 'local region evidence conflicts with previous assignment'
        record['COMPOUND_ID'] = subject
        record['ID_SOURCE'] = 'local_region_evidence'
        record['ID_CONFIDENCE'] = evidence['confidence']
    record['PREVIOUS_ANNOTATION_TEXT'] = record.get('ANNOTATION_TEXT', '')
    record['ANNOTATION_TEXT'] = evidence['annotation_text']
    record['BIOACTIVITY_OBSERVATIONS'] = json.dumps(evidence['observations'], ensure_ascii=False)
    for observation in evidence['observations']:
        column = observation['column']
        record[column] = observation['value']
        record[f'{column}__source_kind'] = observation['source_kind']


def review_structure_regions(jobs, *, audit_path=None):
    from PIL import Image, ImageDraw
    import tempfile
    from utils.model_harness import ModelCapacityError, ModelCircuitOpen
    reviewed = 0
    loaded_page = None
    image = None
    for job in jobs:
        row = job.get('row')
        if row is None or row.get('ID_SOURCE') == 'manual':
            continue
        try:
            box_path = job.get('box_json_path') or row.get('BOX_COORDS_FILE')
            image_path = job.get('page_image_path') or row.get('PAGE_IMAGE_FILE')
            if not box_path or not image_path:
                continue
            box = json.loads(Path(box_path).read_text())
            if image_path != loaded_page:
                with Image.open(image_path) as source:
                    image = source.convert('RGB')
                loaded_page = image_path
            x1, y1, x2, y2 = box['box']
            sx = image.width / box['page_width']; sy = image.height / box['page_height']
            core = (max(0, int(x1*sx)), max(0, int(y1*sy)), min(image.width, int(x2*sx)), min(image.height, int(y2*sy)))
            if core[2] - core[0] < 2 or core[3] - core[1] < 2:
                raise ValueError('Empty evidence crop')
            attempts = []
            evidence = None
            last_error = None
            with tempfile.TemporaryDirectory(prefix='bci-local-evidence-') as temporary:
                for attempt in (1, 2, 0):
                    padding = 0 if attempt == 0 else max(36, int(max(core[2]-core[0], core[3]-core[1]) * .25))
                    region = (max(0, core[0]-padding), max(0, core[1]-padding), min(image.width, core[2]+padding), min(image.height, core[3]+padding))
                    if attempt == 2:
                        vertical=max(16,int((core[3]-core[1])*.12))
                        region=(0,max(0,core[1]-vertical),image.width,min(image.height,core[3]+vertical))
                    crop = image.crop(region)
                    if attempt:
                        ImageDraw.Draw(crop).rectangle((core[0]-region[0], core[1]-region[1], core[2]-region[0]-1, core[3]-region[1]-1), outline='#e64a3d', width=2)
                    if crop.width < 900:
                        crop = crop.resize((900, max(1, round(crop.height * 900 / crop.width))))
                    path = Path(temporary) / f'region_{attempt}.png'; crop.save(path)
                    try:
                        reading = read_evidence(
                            path, 'local_structure_evidence', audit_path=audit_path,
                            metadata={'page': job.get('page_num'), 'box_file': str(box_path),
                                      'scope': 'local_ownership', 'attempt': attempt, 'context_padding': padding},
                        )
                    except Exception as exc:
                        last_error = exc
                        if isinstance(exc, (ModelCapacityError, ModelCircuitOpen)):
                            break
                        continue
                    attempts.append(reading)
                    verified = reading['confidence'] == 'high' and not reading['gap'] and reading['compound_id']
                    if verified or attempt != 0 or evidence is None:
                        evidence = reading
                    if verified:
                        break
            if evidence is None:
                raise last_error
            if last_error is not None:
                row['LOCAL_EVIDENCE_ERROR'] = f'{type(last_error).__name__}: {str(last_error)[:180]}'
            else:
                row.pop('LOCAL_EVIDENCE_ERROR', None)
            apply_local_evidence(row, evidence)
            row['LOCAL_EVIDENCE_ATTEMPTS'] = json.dumps(attempts, ensure_ascii=False)
            reviewed += 1
        except Exception as exc:
            row['LOCAL_EVIDENCE_STATUS'] = 'failed'
            row['LOCAL_EVIDENCE_ERROR'] = f'{type(exc).__name__}: {str(exc)[:180]}'
            row['BIOACTIVITY_OBSERVATIONS'] = '[]'
    return reviewed


def discover_pdf_measurements(pdf_file, pages, *, audit_path=None, progress_callback=None, check_cancel=None):
    import fitz
    import tempfile
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    from utils.model_harness import ModelCapacityError, is_retryable_exception
    decisions = {}
    retry_pages = []

    def read(page_number):
        with fitz.open(pdf_file) as document:
            page = document[page_number - 1]
            pixmap = page.get_pixmap(dpi=MEASUREMENT_DISCOVERY_DPI, alpha=False)
            text = page.get_text()
        with tempfile.TemporaryDirectory(prefix='bci-measurement-page-') as temporary:
            image = Path(temporary) / 'page.png'; pixmap.save(image)
            result = read_evidence(image, 'discover_measurement_page', audit_path=audit_path,
                metadata={'page': page_number, 'scope': 'measurement_page_discovery'})
        endpoints = result['endpoints']
        return {
            'has_assay_data': result['has_result_records'],
            'confidence': result['confidence'],
            'assay_names': [item['column'] for item in endpoints],
            'reason': result['evidence'],
            'endpoint_descriptors': endpoints,
            'gap': result.get('gap', ''),
            'status': 'reviewed',
        }, text

    markdowns = {}
    workers = max(1, min(4, int(getattr(constants, 'VISION_MODEL_CONCURRENCY', 2))))
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        remaining = iter(pages)
        futures = {}
        completed = 0
        while True:
            if check_cancel:
                check_cancel()
            for _ in range(workers - len(futures)):
                number = next(remaining, None)
                if number is None:
                    break
                futures[executor.submit(read, number)] = number
            if not futures:
                break
            ready, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in ready:
                number = futures.pop(future)
                try:
                    decisions[number], markdowns[number] = future.result()
                except ModelCapacityError:
                    raise
                except Exception as exc:
                    if is_retryable_exception(exc):
                        retry_pages.append(number)
                    decisions[number] = {'has_assay_data': None, 'confidence': 'low', 'assay_names': [],
                        'reason': f'{type(exc).__name__}: {str(exc)[:180]}', 'status': 'failed'}
                completed += 1
                if progress_callback:
                    progress_callback({'stage': 'llm', 'current': completed, 'total': len(pages),
                        'message': f'Reviewed measurement evidence on {completed}/{len(pages)} pages'})
    finally:
        executor.shutdown(cancel_futures=True)
    if retry_pages:
        deadline = time.monotonic() + getattr(constants, 'MEASUREMENT_RETRY_COOLDOWN_SECONDS', 20)
        while (remaining_seconds := deadline - time.monotonic()) > 0:
            if check_cancel:
                check_cancel()
            time.sleep(min(remaining_seconds, 0.25) if check_cancel else remaining_seconds)
        for number in sorted(retry_pages):
            if check_cancel:
                check_cancel()
            if progress_callback:
                progress_callback({
                    'stage': 'llm', 'current': len(pages), 'total': len(pages),
                    'message': f'Retrying measurement evidence on page {number}',
                })
            try:
                decisions[number], markdowns[number] = read(number)
            except ModelCapacityError:
                raise
            except Exception:
                continue
    failed = [page for page, result in decisions.items() if result['status'] == 'failed']
    if failed:
        raise RuntimeError(f'Measurement page review incomplete on pages {sorted(failed)}; retry discovery')
    selected = sorted(page for page, decision in decisions.items() if decision['has_assay_data'])
    names = [name for page in selected for name in decisions[page]['assay_names']]
    decisions = {page: decisions[page] for page in sorted(decisions)}
    markdowns = {page: markdowns[page] for page in sorted(markdowns)}
    return selected, list(dict.fromkeys(names)), decisions, markdowns


def user_measurement_edits(record):
    value = record.get('USER_MEASUREMENT_EDITS')
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return {}
    return {str(key): str(value) for key, value in parsed.items()} if isinstance(parsed, dict) else {}


def parse_measurement_table(text):
    from utils.model_harness import extract_json_content
    result = json.loads(extract_json_content(text))
    if not isinstance(result, dict) or not isinstance(result.get('tables'), list) or not isinstance(result.get('gaps'), list):
        raise ValueError('Missing table coverage contract')
    for table in result['tables']:
        if not isinstance(table, dict):
            raise ValueError('Invalid table')
        columns, rows = table.get('columns'), table.get('rows')
        if not isinstance(columns, list) or not columns or not isinstance(rows, list):
            raise ValueError('Invalid table shape')
        labels = set()
        for column in columns:
            if not isinstance(column, dict):
                raise ValueError('Invalid table column')
            header = column.get('header_evidence')
            kind = column.get('source_kind')
            if not isinstance(kind, str) or kind not in KINDS or not isinstance(header, str) or not header.strip():
                raise ValueError('Endpoint lacks header/provenance evidence')
            if 'conditions' in column and not isinstance(column['conditions'], str):
                raise ValueError('Invalid result conditions')
            label = measurement_column(column)
            if label in labels:
                raise ValueError('Indistinguishable result columns require better header context')
            labels.add(label)
            column['column'] = label
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('Invalid table row')
            subject = row.get('compound_id')
            evidence = row.get('row_evidence')
            if not isinstance(subject, str) or not subject.strip() or not isinstance(evidence, str) or not evidence.strip():
                raise ValueError('Missing source-owned row identifier')
            subject_key = unicodedata.normalize('NFKC', subject).strip()
            if subject_key in seen:
                raise ValueError('Duplicate subject within table; preserve replicate context explicitly')
            seen.add(subject_key)
            values = row.get('values')
            if not isinstance(values, list) or len(values) != len(columns):
                raise ValueError('Row does not account for every result column')
            for value in values:
                if value is None or type(value) in (str, int):
                    continue
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError('Invalid measurement cell')
    return result


def nearby_result_notes(document, page_number):
    """Bounded source text for printed category legends; never infer thresholds."""
    notes=[]
    for index in range(max(0,page_number-2),min(len(document),page_number)):
        for line in document[index].get_text().splitlines():
            if re.search(r'[*†‡].*[<>=≤≥]|\b(?:not tested|untested)\b',line,re.IGNORECASE):
                notes.append(f"PDF page {index+1}: {line.strip()}")
    return list(dict.fromkeys(notes))[:12]


def read_measurement_page(pdf_file, page_number, *, audit_path=None):
    import fitz
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    from utils.llm_utils import VISION_MODEL_CONCURRENCY, run_vision_json_task
    from utils.model_harness import ModelCapacityError, classify_exception
    from utils.skill_prompt_loader import render_skill_reference
    from utils.table_regions import table_windows
    observations, all_tables, all_gaps = [], [], []
    window_errors = []
    last_error = None
    with tempfile.TemporaryDirectory(prefix='bci-result-table-') as temporary:
        image=Path(temporary)/'page.png'
        with fitz.open(pdf_file) as document:
            page=document[page_number-1]
            page.get_pixmap(dpi=180,alpha=False).save(image)
            windows=table_windows(page,image,Path(temporary)/'windows')
            source_notes=nearby_result_notes(document,page_number)
        selections=windows or [{'image':str(image),'table_index':None,'source_region':None,'context':''}]
        prompt=render_skill_reference('biocheminsight-vision-models','references/read_measurement_table_prompt.md',{})
        def read_window(selection):
            instruction = prompt
            if windows:
                instruction+='\nThis image is a host-selected table window with its original header. Read only the visible subject rows. Page captions/footnotes (context only; do not import result rows):\n'+selection['context']
            if source_notes: instruction+='\nPrinted source notes (use only when they apply to this table; preserve category symbols in cells):\n'+'\n'.join(source_notes)
            try:
                return run_vision_json_task(
                    task_name='read_measurement_table', image_file=selection['image'], prompt=instruction,
                    parser=parse_measurement_table, audit_path=audit_path,
                    metadata={'page': page_number, 'scope': 'result_record_recovery',
                              'source_region': selection['source_region']},
                )
            except Exception as exc:
                return exc

        workers = min(len(selections), VISION_MODEL_CONCURRENCY)
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for offset in range(0, len(selections), workers):
                batch = list(executor.map(read_window, selections[offset:offset + workers]))
                results.extend(batch)
                exhausted = next((item for item in batch if isinstance(item, ModelCapacityError)), None)
                if exhausted is not None:
                    break

        for index, selection in enumerate(selections):
            attempted = index < len(results)
            payload = results[index] if attempted else exhausted
            if isinstance(payload, Exception):
                last_error = payload
                window_errors.append({
                    'source_region': selection['source_region'],
                    'error_type': classify_exception(payload),
                    'error': str(payload),
                    'attempted': attempted,
                })
                continue
            all_tables.extend(payload['tables'])
            all_gaps.extend(payload['gaps'])
            for local_index,table in enumerate(payload['tables']):
                table_index=selection['table_index'] if selection['table_index'] is not None else local_index
                for row in table['rows']:
                    cells=[{**column,'subject_id':row['compound_id'],'value':str(value),'evidence':row['row_evidence'],
                        'page':page_number,'table_index':table_index,'table_caption':table.get('caption',''),
                        **({'source_region':selection['source_region'],'source_bbox':selection['bbox']} if windows else {})}
                        for column,value in zip(table['columns'],row['values']) if value is not None]
                    observations.append({'compound_id':row['compound_id'],'page':page_number,'table_index':table_index,'observations':cells,'notes':table.get('notes') or []})
    if len(window_errors) == len(selections):
        raise last_error
    all_gaps.extend(f"Table region {item['source_region']}: {item['error']}" for item in window_errors)
    return {
        'records': observations, 'tables': all_tables, 'gaps': all_gaps,
        'window_count': len(selections), 'window_errors': window_errors,
    }


def removed_measurements(records):
    removed = set()
    for record in records:
        try: values = json.loads(str(record.get('USER_REMOVED_MEASUREMENTS') or '[]'))
        except (ValueError, TypeError): continue
        if isinstance(values, list): removed.update(value for value in values if isinstance(value, str))
    return removed


def measurement_link_status(smiles):
    from rdkit import Chem

    graphs = set()
    missing = False
    for text in {str(value).strip() for value in smiles}:
        molecule = Chem.MolFromSmiles(text) if text else None
        if molecule is None or not molecule.GetNumAtoms():
            missing = True
        else:
            graphs.add(Chem.MolToSmiles(molecule, isomericSmiles=True))
    if not graphs:
        return "structure_not_found"
    if missing or len(graphs) > 1:
        return "ambiguous_identity"
    return ""


def split_unverified_measurements(values):
    suffix = '__structure_id_supported'
    unverified_columns = {
        key.removesuffix(suffix) for key, value in values.items()
        if key.endswith(suffix) and str(value).strip().casefold() == 'false'
    }
    linked, unverified = {}, {}
    for key, value in values.items():
        column = key.rpartition('__')[0] if '__' in key else key
        target = unverified if column in unverified_columns or key in unverified_columns else linked
        target[key] = value
    return linked, unverified


def measurement_location(cell, *, page_prefix='p.'):
    page = cell.get('page')
    location = f"{page_prefix} {page}" if page else 'source'
    if cell.get('table_index') is not None:
        location += f", table {int(cell['table_index']) + 1}"
    return location


def visual_measurement_reason(cell):
    location = measurement_location(cell, page_prefix='PDF page')
    return f"Visual table on {location}: {cell['header_evidence']} / {cell['evidence']}"


def assay_source_columns(column, item):
    match = item.get('assay_match') if isinstance(item, dict) else None
    if not isinstance(match, dict):
        return {}
    values = {}
    if match.get('visual_page'):
        values[f'{column}__source_page'] = str(match['visual_page'])
    if match.get('structure_id_supported') is False:
        values[f'{column}__structure_id_supported'] = 'false'
    return values


def condition_measurement_values(item):
    if not isinstance(item, dict):
        return {}
    by_column = {}
    for cell in item.get('condition_observations') or ():
        by_column.setdefault(cell['column'], []).append(cell)
    values = {}
    for column, cells in by_column.items():
        distinct_values = {cell['value'] for cell in cells}
        if len(distinct_values) == 1:
            values[column] = cells[0]['value']
        else:
            values[column] = '; '.join(dict.fromkeys(
                f"{cell['value']} [{measurement_location(cell)}]" for cell in cells
            ))
        values[f'{column}__reason'] = '; '.join(dict.fromkeys(
            visual_measurement_reason(cell)
            for cell in cells
        ))
        values.update(assay_source_columns(column, item))
        pages = sorted({cell['page'] for cell in cells if type(cell.get('page')) is int})
        if pages:
            values[f'{column}__source_page'] = ', '.join(map(str, pages))
    return values


def flatten_assay_records(assay_name, values):
    """Convert rich observations into scientific columns without dropping units."""
    records = {}
    for subject, item in (values or {}).items():
        qualified = condition_measurement_values(item)
        if qualified:
            records[str(subject)] = qualified
            continue
        if isinstance(item, dict):
            raw = item.get('value', item.get('VALUE', ''))
            unit = str(item.get('unit') or item.get('UNIT') or '').strip()
            if item.get('endpoint'):
                column = measurement_column(item)
            else:
                column = assay_column_with_unit(str(assay_name), unit)
            record = {column: '' if raw is None else str(raw)}
            record.update(assay_source_columns(column, item))
            for field in ('method', 'description', 'source_kind', 'confidence', 'reason'):
                if item.get(field): record[f'{column}__{field}'] = str(item[field])
        else:
            record = {str(assay_name): '' if item is None else str(item)}
        records[str(subject)] = record
    return records


def observation_values(record):
    """Flatten owned observations without silently discarding repeat-source values."""
    try: observations = json.loads(str(record.get('BIOACTIVITY_OBSERVATIONS') or '[]'))
    except (ValueError, TypeError): return {}
    if not isinstance(observations, list): return {}
    groups = {}
    subject = str(record.get('COMPOUND_ID') or '')
    for item in observations:
        if isinstance(item, dict) and item.get('association_status') != 'needs_review' and item.get('subject_id') == subject and isinstance(item.get('column'), str) and isinstance(item.get('value'), str):
            groups.setdefault(item['column'], []).append(item)
    result = {}
    for column, items in groups.items():
        if len({item['value'] for item in items}) == 1:
            result[column] = items[0]['value']
        else:
            result[column] = '; '.join(
                f"{item['value']} [{measurement_location(item)}]" for item in items
            )
    return result


def quarantine_ambiguous_prose_ownership(records):
    """Comparison prose with several known subjects is not row-scoped evidence.

    Preserve the reading for review; do not assign a value by mention order.
    This deliberately prefers an unresolved observation over an invented pairing.
    """
    subjects = {str(row.get('COMPOUND_ID') or '').strip() for row in records} - {''}
    patterns = [re.compile(r'(?<!\w)' + re.escape(subject) + r'(?!\w)', re.I) for subject in subjects]
    changed = 0
    for row in records:
        if not row.get('BIOACTIVITY_OBSERVATIONS'): continue
        try: observations = json.loads(str(row.get('BIOACTIVITY_OBSERVATIONS') or '[]'))
        except (ValueError, TypeError): continue
        if not isinstance(observations, list): continue
        for item in observations:
            if not isinstance(item, dict) or item.get('association_status') == 'user_verified': continue
            evidence = str(item.get('evidence') or '')
            comparison = re.search(r'\b(respectively|compared|replacement|versus)\b', evidence, re.I)
            if comparison and sum(bool(pattern.search(evidence)) for pattern in patterns) > 1:
                if item.get('association_status') != 'needs_review': changed += 1
                item['association_status'] = 'needs_review'
                item['association_reason'] = 'Comparison prose mentions multiple subjects; individual value ownership requires verification'
        row['BIOACTIVITY_OBSERVATIONS'] = json.dumps(observations, ensure_ascii=False)
    return changed
