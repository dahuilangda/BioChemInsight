"""Extraction ledger: document-level state for structure mining runs.

One ledger per extraction task. Every detected segment becomes a unit that
must reach a terminal state (accepted | filtered | failed), so nothing can
vanish silently: the coverage report at the end enumerates any unit still
non-terminal. Corrections (retry strategies) are recorded on the unit.

The ledger is the harness's context carrier: it answers "what did this run
see, what happened to each thing, what is still missing".
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

TERMINAL_STATES = {'accepted', 'filtered', 'failed'}
KNOWN_STATES = {
    'detected', 'classified', 'gate_passed', 'gate_rejected',
    'recognizing', 'recognized', 'identifying', 'identified',
    'correcting', 'accepted', 'filtered', 'failed',
}


class ExtractionLedger:
    """Append-friendly, atomically persisted unit ledger for one run."""

    def __init__(self, output_dir, goal=None):
        self.output_dir = Path(output_dir)
        self.ledger_path = self.output_dir / 'extraction_ledger.json'
        self.goal = dict(goal or {})
        self.units = {}
        self.pages = {}
        self.failed_pages = []
        self.corrections = []
        self.started_at = time.strftime('%Y-%m-%dT%H:%M:%S')

    # -- unit lifecycle -------------------------------------------------
    def open_unit(self, page_num, segment_idx, bbox, segment_file=None):
        unit_id = f'p{page_num}s{segment_idx}'
        if unit_id in self.units:
            return unit_id
        self.units[unit_id] = {
            'unit_id': unit_id,
            'page': page_num,
            'segment': segment_idx,
            'bbox': [int(v) for v in bbox] if bbox else None,
            'segment_file': segment_file,
            'state': 'detected',
            'history': [{'state': 'detected', 'ts': time.strftime('%Y-%m-%dT%H:%M:%S')}],
            'smiles': '',
            'compound_id': '',
            'visual_role': '',
            'corrections': [],
        }
        return unit_id

    def transition(self, unit_id, state, detail=None, persist=True, **fields):
        unit = self.units.get(unit_id)
        if unit is None:
            return
        if state not in KNOWN_STATES:
            raise ValueError(f'unknown ledger state: {state}')
        unit['state'] = state
        entry = {'state': state, 'ts': time.strftime('%Y-%m-%dT%H:%M:%S')}
        if detail:
            entry['detail'] = str(detail)[:300]
        unit['history'].append(entry)
        for key, value in fields.items():
            if key in {'smiles', 'compound_id', 'visual_role', 'filter_reason', 'strictness'}:
                unit[key] = str(value)[:400] if value is not None else ''
        if persist:
            self.save()

    def note_correction(self, unit_id, strategy, outcome, detail=None):
        unit = self.units.get(unit_id)
        if unit is None:
            return
        record = {
            'strategy': str(strategy),
            'outcome': str(outcome),
            'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        if detail:
            record['detail'] = str(detail)[:300]
        unit['corrections'].append(record)
        self.corrections.append({'unit_id': unit_id, **record})
        unit['state'] = 'correcting'
        self.save()

    def set_page(self, page_num, status, detail=None):
        entry = self.pages.setdefault(page_num, {'status': 'pending', 'history': []})
        entry['status'] = str(status)
        entry['history'].append(
            {'status': str(status), 'detail': str(detail)[:200] if detail else '',
             'ts': time.strftime('%Y-%m-%dT%H:%M:%S')})

    def finalize_from_rows(self, accepted_rows, filtered_rows):
        """Assign terminal states by matching rows back to units.

        Rows carry PAGE_NUM and SEGMENT_FILE; units were opened with
        (page, segment index, bbox, segment file). Matching is by page and
        segment file name; rows without a unit (e.g. assembled markush
        outputs) are recorded as extra accepted units.
        """
        index = {}
        for unit_id, unit in self.units.items():
            key = (unit.get('page'), os.path.basename(unit.get('segment_file') or ''))
            index[key] = unit_id
        for state, rows in (('accepted', accepted_rows), ('filtered', filtered_rows)):
            # terminal states are batched: transition() would persist per call,
            # which is O(N) full-file writes inside this loop.
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    page = int(row.get('PAGE_NUM'))
                except (TypeError, ValueError):
                    page = None
                seg = os.path.basename(str(row.get('SEGMENT_FILE') or ''))
                unit_id = index.get((page, seg))
                if unit_id is None:
                    unit_id = f'p{page}s{seg}'
                    self.units[unit_id] = {
                        'unit_id': unit_id, 'page': page, 'segment': seg,
                        'bbox': None, 'segment_file': seg, 'state': state,
                        'history': [{'state': state,
                                     'ts': time.strftime('%Y-%m-%dT%H:%M:%S')}],
                        'smiles': '', 'compound_id': '', 'visual_role': '',
                        'corrections': [],
                    }
                self.transition(
                    unit_id, state, persist=False,
                    detail=str(row.get('STRUCTURE_FILTER_REASON') or '')[:280],
                    compound_id=row.get('COMPOUND_ID'),
                    visual_role=row.get('VISUAL_ROLE'),
                    smiles=row.get('SMILES'),
                    filter_reason=row.get('STRUCTURE_FILTER_REASON'))

        self.save()

    # -- coverage -------------------------------------------------------
    def coverage(self):
        counts = {}
        for unit in self.units.values():
            counts[unit['state']] = counts.get(unit['state'], 0) + 1
        missed = [
            {'unit_id': u['unit_id'], 'page': u['page'], 'state': u['state']}
            for u in self.units.values() if u['state'] not in TERMINAL_STATES
        ]
        accepted = counts.get('accepted', 0)
        detected = len(self.units)
        return {
            'units_total': detected,
            'units_by_state': counts,
            'accepted': accepted,
            'filtered': counts.get('filtered', 0),
            'failed': counts.get('failed', 0),
            'non_terminal': missed,
            'complete': not self.failed_pages and not missed and detected > 0,
            'failed_pages': list(self.failed_pages),
            'pages': dict(self.pages),
            'corrections_total': len(self.corrections),
        }

    # -- persistence ------------------------------------------------------
    def save(self):
        payload = {
            'started_at': self.started_at,
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'goal': self.goal,
            'coverage': self.coverage(),
            'pages': self.pages,
            'units': list(self.units.values()),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.output_dir), suffix='.json.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, self.ledger_path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def load_ledger(output_dir):
    path = Path(output_dir) / 'extraction_ledger.json'
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)
