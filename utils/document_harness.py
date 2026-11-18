"""Document-level evidence and capability coordination."""
from __future__ import annotations

import importlib.util
import json
import os
import re

from utils.model_harness import ModelContractError, ModelEvidenceError

class DocumentHarness:
    """Document-level state shared by every capability. Sequences come from
    the VLM page read; figure/sequence entities, contradictions and gaps
    are filed in the per-document DocumentLedger."""

    def __init__(self, pdf_file: str, output_dir: str = '',
                 structure_pages: list[int] | None = None):
        self.pdf_file = pdf_file
        self.output_dir = output_dir
        self.structure_pages = [int(p) for p in (structure_pages or [])]
        self.page_entities: list[dict] = []
        self.sequence_entries: list[dict] = []
        self._links: list[dict] = []
        self._corrections: list[dict] = []
        self._seq_entity_ids: dict[tuple, str] = {}
        self._sequences_filed = False
        from services.document_ledger import DocumentLedger
        doc_id = os.path.splitext(os.path.basename(pdf_file or ''))[0] or 'document'
        self._ledger = DocumentLedger(doc_id)

    # ------------------------------------------------------- capability ---
    def ensure_sequences(self) -> list:
        """Sequences come from the VLM page read (ensure_page_entities)."""
        self.ensure_page_entities()
        all_seqs = []
        for records in self.by_id().values():
            all_seqs.extend(records)
        return all_seqs

    def by_id(self) -> dict[str, list]:
        """Sequence index built from the VLM page-read records. Each record
        is also filed as a SequenceEntity in the ledger (once)."""
        index: dict[str, list] = {}
        from utils.sequence_parser import SequenceRecord
        from utils.page_extractor import validate_one_letter_sequence
        for record in self.page_entities:
            page = record.get('page_num')
            for seq in record.get('sequences', []) or []:
                if not isinstance(seq, dict) or not isinstance(seq.get('seq_id'), str):
                    raise ModelContractError(f'Page {page} sequence requires a string seq_id')
                if type(seq.get('cyclic')) is not bool:
                    raise ModelContractError(f'Page {page} sequence cyclic must be a boolean')
                seq_id = seq['seq_id'].strip()
                sequence = seq.get('sequence_one_letter')
                validate_one_letter_sequence(sequence, f'Page {page} sequence_one_letter')
                if not sequence:
                    continue
                m = re.search(r'(\d{1,5})\s*$', seq_id) if seq_id else None
                key = m.group(1) if m else ''
                index.setdefault(key, []).append(SequenceRecord(
                    sequence=sequence, kind='aa', seq_id=key,
                    page_num=page, cyclic=seq['cyclic'],
                    confidence=0.9))
        if self._ledger is not None and not self._sequences_filed:
            self._file_sequences_in_ledger(index)
        return index

    def _file_sequences_in_ledger(self, index: dict[str, list]) -> None:
        """Register every page-read sequence record as a ledger
        SequenceEntity so figure gaps can be resolved against it."""
        from services.document_ledger import SequenceEntity
        for key in sorted(index):
            for i, record in enumerate(index[key]):
                entity_id = f"seq_p{record.page_num}_{key or 'x'}_{i}"
                self._ledger.add_sequence(SequenceEntity(
                    entity_id=entity_id, page=int(record.page_num or 0),
                    codes=list(record.sequence), seq_id=key,
                    source='vlm_page_read'))
                self._seq_entity_ids[
                    (record.page_num, key, record.sequence)] = entity_id
        self._sequences_filed = True

    def ensure_page_entities(self) -> list[dict]:
        if not self.structure_pages:
            return []
        if not self.page_entities:
            from utils.page_extractor import PageEntityExtractor
            audit_path = os.path.join(self.output_dir, 'model_calls.jsonl') if self.output_dir else None
            records = PageEntityExtractor(audit_path=audit_path).extract_pages(self.pdf_file, self.structure_pages)
        else:
            records = self.page_entities
        if not isinstance(records, list):
            raise ModelContractError('Page evidence must be a list')
        expected = set(self.structure_pages)
        seen = set()
        for record in records:
            if not isinstance(record, dict):
                raise ModelContractError('Page evidence record must be an object')
            page = record.get('page_num')
            if type(page) is not int or page not in expected or page in seen:
                raise ModelContractError(f'Unexpected or duplicate page evidence: {page!r}')
            seen.add(page)
            if record.get('ok') is not True:
                raise ModelEvidenceError(f'Page evidence failed on page {page}: {record.get("error", "unverified record")}')
            for key in ('compounds', 'sequences', 'sequence_entries', 'rgroup_definitions'):
                if not isinstance(record.get(key), list):
                    raise ModelContractError(f'Page {page} evidence requires {key} list')
            if not isinstance(record.get('page_summary'), str):
                raise ModelContractError(f'Page {page} evidence requires page_summary string')
        missing = expected - seen
        if missing:
            raise ModelEvidenceError(f'Missing page evidence: {sorted(missing)}')
        records.sort(key=lambda record: record['page_num'])
        self.page_entities = records
        return self.page_entities

    def _link_sequence_entries(self, rows: list[dict]) -> list[dict]:
        """Link sequence_entries from VLM table reads to structure rows via SEQ ID."""
        for row in rows:
            ref = str(row.get('SEQUENCE_REF') or '').strip()
            compound_id = str(row.get('COMPOUND_ID') or '').strip()
            import re as _re
            match = _re.match(r'^SEQ(\d+)$', compound_id, flags=_re.IGNORECASE)
            ref = ref or (match.group(1) if match else '')
            if not ref:
                continue
            # Find matching sequence_entries from page reads
            for entity in self.page_entities:
                for entry in entity.get('sequence_entries', []):
                    entry_id = str(entry.get('seq_id') or '').strip()
                    import re as _re2
                    m = _re2.search(r'(\d+)', entry_id)
                    if m and m.group(1) == ref:
                        row['SEQUENCE_TEXT'] = entry.get('sequence_text', '')
                        row['MOLECULAR_WEIGHT'] = entry.get('molecular_weight', '')
                        row['MOLECULAR_FORMULA'] = entry.get('molecular_formula', '')
                        row['EXACT_MASS'] = entry.get('exact_mass', '')
                        row['PURITY_PERCENT'] = entry.get('purity_percent', '')
                        row['OBSERVED_ION'] = entry.get('observed_ion', '')
                        break
        return rows

    def _label_index(self) -> dict[str, dict]:
        """Page+label -> compound entity, for row consolidation. Labels are
        normalized with the same SEQ-ID contract as the ID extractor so page
        entities join rows ('SEQ ID NO. 53' on both sides becomes 'seq53')."""
        index: dict[str, dict] = {}
        for record in self.page_entities:
            page = record.get('page_num')
            for compound in record.get('compounds', []) or []:
                label = str(compound.get('label_in_page') or '').strip()
                if not label:
                    continue
                seq_match = re.match(
                    r'^SEQ\.?\s*(?:ID\.?\s*)?NO\.?\s*[:.]?\s*([0-9]{1,5})\s*$',
                    label, flags=re.IGNORECASE)
                keys = {str(page), label.lower(), f"p{page}:{label.lower()}"}
                if seq_match:
                    keys.add(f"seq{seq_match.group(1)}")
                    keys.add(f"p{page}:seq{seq_match.group(1)}")
                for key in keys:
                    index.setdefault(key, compound)
        return index

    # ------------------------------------------------------------ ledger ---
    def _file_row_figure(self, row: dict, index: int, page) -> object | None:
        """File the decoded structure row as a ledger FigureEntity. The
        entity is registered up front; the reconciliation branches below
        mutate the same object (gaps, contradictions, resolution)."""
        if self._ledger is None:
            return None
        from services.document_ledger import EvidenceRecord, FigureEntity
        compound_id = str(row.get('COMPOUND_ID') or '').strip()
        smiles = str(row.get('SMILES') or '').strip()
        # row index keeps the id unique: the pipeline dedup can keep two
        # rows with the same compound_id but different decodes
        entity_id = f"fig_p{page}_{compound_id or 'row'}_{index}"
        figure = FigureEntity(
            entity_id=entity_id, page=int(page) if page else 0, bbox=[],
            smiles=smiles or None, compound_id=compound_id or None,
            compound_name=str(row.get('COMPOUND_NAME') or '').strip() or None)
        if smiles:
            figure.evidence.append(EvidenceRecord(
                skill='structure_decode', input_ref=f"page:{page}",
                output_digest=smiles[:80],
                confidence=0.9, detail='structures.csv decode'))
        self._ledger.add_figure(figure)
        return figure

    # --------------------------------------------------------  corrections ---
    def reconcile_structures(self, rows: list[dict]) -> list[dict]:
        """Cross-capability error correction over structure rows.

        Rows keep their image-decode truth. Only EMPTY decodes may be filled
        from the sequence domain, and only when the row is sequence-labeled
        and the peptide→SMILES construction parses. Every action is recorded
        in the corrections trail and mirrored in the DocumentLedger.
        """
        from utils.sequence_parser import sequence_to_smiles
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem
        RDLogger.DisableLog("rdApp.*")

        seq_ids = self.by_id()
        labels = self._label_index()
        rows = self._link_sequence_entries(rows)
        for row_index, row in enumerate(rows):
            # Page-entity consolidation runs first; names fill independently
            # of the sequence-record branch.
            if not str(row.get('COMPOUND_NAME') or '').strip():
                page = row.get('PAGE_NUM')
                for probe in (str(row.get('COMPOUND_ID') or '').strip(),
                              f"p{page}:{str(row.get('COMPOUND_ID') or '').strip().lower()}"):
                    entity = labels.get(probe)
                    if entity and str(entity.get('name') or '').strip():
                        row['COMPOUND_NAME'] = entity['name']
                        break
            ref = str(row.get('SEQUENCE_REF') or '').strip()
            compound_id = str(row.get('COMPOUND_ID') or '').strip()
            match = re.match(r'^SEQ(\d+)$', compound_id, flags=re.IGNORECASE)
            ref = ref or (match.group(1) if match else '')
            figure = self._file_row_figure(row, row_index, row.get('PAGE_NUM'))
            if not ref:
                if figure is not None:
                    figure.smiles_valid = bool(figure.smiles)
                continue
            candidates = seq_ids.get(ref, [])
            page = row.get('PAGE_NUM')
            variants = {(record.sequence, record.cyclic) for record in candidates}
            page_records = [r for r in candidates if not page or r.page_num == page] or candidates
            if len(variants) > 1:
                page_records = candidates
            link = {
                'compound_id': compound_id or f'SEQ{ref}',
                'sequence_ref': ref,
                'page_num': page,
                'sequences_found': [r.asdict() for r in page_records],
            }
            self._links.append(link)
            if figure is not None:
                figure.seq_id = ref
                from services.document_ledger import EvidenceRecord
                figure.evidence.append(EvidenceRecord(
                    skill='vlm_page_read', input_ref=f"page:{page}:seq:{ref}",
                    output_digest=page_records[0].sequence[:80] if len(variants) == 1 else '',
                    confidence=0.9 if len(variants) == 1 else 0.3,
                    detail='page-read sequence record'))
            if not page_records:
                self._corrections.append({
                    'compound_id': compound_id, 'sequence_ref': ref,
                    'action': 'no_sequence_record_on_page', 'page_num': page})
                if figure is not None:
                    from services.document_ledger import Gap
                    self._ledger.add_gap(Gap(
                        entity_id=figure.entity_id, field='sequence_ref',
                        reason_code='no_sequence_record_on_page',
                        attempted=['vlm_page_read']))
                continue
            if len(variants) > 1:
                self._corrections.append({
                    'compound_id': compound_id, 'sequence_ref': ref,
                    'action': 'conflicting_sequence_records', 'page_num': page})
                if figure is not None:
                    from services.document_ledger import Gap, Resolution
                    self._ledger.add_gap(Gap(
                        entity_id=figure.entity_id, field='sequence_ref',
                        reason_code='conflicting_sequence_records',
                        attempted=['vlm_page_read']))
                    figure.resolution = Resolution.REVIEW_QUEUED
                continue
            record = page_records[0]
            if figure is not None:
                figure.seq_residues = list(record.sequence)
            smiles = str(row.get('SMILES') or '').strip()
            derived = sequence_to_smiles(record.sequence, cyclic=record.cyclic)
            if not smiles:
                # A bare sequence cannot establish noncanonical residues,
                # stereochemistry, terminal modifications or ring-closure sites.
                # Keep its reconstruction as a candidate, never image truth.
                if derived:
                    row['SEQUENCE_DERIVED_CANDIDATE'] = derived
                    row['SEQUENCE_RECONSTRUCTION_STATUS'] = 'needs_typed_modification_and_closure_evidence'
                self._corrections.append({'compound_id': compound_id, 'sequence_ref': ref,
                    'action': 'sequence_candidate_requires_validation', 'sequence': record.sequence})
                if figure is not None:
                    from services.document_ledger import Gap, Resolution
                    self._ledger.add_gap(Gap(entity_id=figure.entity_id, field='smiles',
                        reason_code='typed_peptide_evidence_required', attempted=['structure_decode', 'base_sequence_candidate']))
                    figure.resolution = Resolution.REVIEW_QUEUED
                continue
            # Both present: cross-validate without overwriting the decode.
            pm, gm = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(derived) if derived else None
            if figure is not None:
                figure.smiles_valid = pm is not None
            if pm is not None and gm is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(pm, 2, 1024)
                fg = AllChem.GetMorganFingerprintAsBitVect(gm, 2, 1024)
                similarity = DataStructs.TanimotoSimilarity(fp, fg)
                link['tanimoto_vs_sequence'] = round(similarity, 3)
                same_graph = Chem.MolToSmiles(pm, isomericSmiles=True) == Chem.MolToSmiles(gm, isomericSmiles=True)
                link['exact_graph_agreement'] = same_graph
                if not same_graph:
                    row['SEQUENCE_STRUCTURE_MISMATCH'] = round(similarity, 3)
                    self._corrections.append({
                        'compound_id': compound_id, 'sequence_ref': ref,
                        'action': 'mismatch_flagged', 'tanimoto': round(similarity, 3)})
                    if figure is not None:
                        from services.document_ledger import (Contradiction,
                                                              Resolution)
                        self._ledger.add_contradiction(Contradiction(
                            kind='seq-mismatch', entity_id=figure.entity_id,
                            source_a='structure_decode',
                            source_b=f"sequence:{ref}",
                            detail_a=f"decode vs sequence tanimoto={round(similarity, 3)}",
                            detail_b=f"seq {record.sequence[:80]}"))
                        figure.resolution = Resolution.REVIEW_QUEUED
                        figure.reason_code = 'SEQUENCE_MISMATCH'
                else:
                    self._corrections.append({
                        'compound_id': compound_id, 'sequence_ref': ref,
                        'action': 'exact_base_graph_agreement', 'tanimoto': round(similarity, 3)})
                    if figure is not None:
                        from services.document_ledger import Resolution
                        figure.resolution = Resolution.RESOLVED
        return rows

    # ------------------------------------------------------------ goals ---
    def coverage_report(self, rows: list[dict]) -> dict:
        pages = {}
        for page in self.structure_pages:
            pages[page] = {'structures_total': 0, 'structures_with_smiles': 0,
                           'filtered': 0, 'sequences': 0}
        for row in rows:
            page = row.get('PAGE_NUM')
            bucket = pages.get(page)
            if bucket is None:
                continue
            if row.get('FILTERED_OUT'):
                bucket['filtered'] += 1
            else:
                bucket['structures_total'] += 1
                if str(row.get('SMILES') or '').strip():
                    bucket['structures_with_smiles'] += 1
        page_seqs = self.by_id()
        for ref, records in page_seqs.items():
            for record in records:
                bucket = pages.get(record.page_num)
                if bucket is not None:
                    bucket['sequences'] += 1
        return {
            'pdf': os.path.abspath(self.pdf_file),
            'pages': {str(k): v for k, v in sorted(pages.items())},
            'total_rows': len(rows),
            'rows_with_smiles': sum(
                1 for r in rows if str(r.get('SMILES') or '').strip() and not r.get('FILTERED_OUT')),
            'sequence_records': sum(len(v) for v in page_seqs.values()),
            'sequence_table_entries': sum(len(e.get('sequence_entries', [])) for e in self.page_entities),
            'links': self._links,
            'corrections': self._corrections,
        }

    # ------------------------------------------------------- environment ---
    def service_report(self) -> dict:
        """Environment registry: which backends this document can reach."""
        report = {}
        try:
            import constants
            import requests
            url = getattr(constants, 'LLM_OPENAI_COMPATIBLE_MODEL_URL', '')
            if url:
                resp = requests.get(url.rstrip('/') + '/models', timeout=5,
                                    headers={'Authorization': 'Bearer ' + str(
                                        getattr(constants, 'LLM_OPENAI_COMPATIBLE_MODEL_KEY', ''))})
                models = [m.get('id') for m in resp.json().get('data', [])][:5]
                report['llm_endpoint'] = {'url': url, 'models': models, 'ok': resp.ok}
        except Exception as exc:
            report['llm_endpoint'] = {'ok': False,
                                      'error': f"{type(exc).__name__}: {str(exc)[:80]}"}
        # The MarkushGlyph probe lives in experiments/markush_glyph until the
        # next-generation markush line wires it back into the pipeline.
        if importlib.util.find_spec('utils.markush_glyph_service') is not None:
            try:
                from utils.markush_glyph_service import markush_glyph_service
                report['markush_glyph'] = {'enabled': markush_glyph_service().enabled}
            except Exception as exc:
                report['markush_glyph'] = {'enabled': False, 'error': str(exc)[:80]}
        else:
            report['markush_glyph'] = {'enabled': False}
        try:
            from utils.molsight_service import molsight_service
            report['molsight'] = {'enabled': molsight_service().enabled}
        except Exception as exc:
            report['molsight'] = {'enabled': False, 'error': str(exc)[:80]}
        return report

    # ------------------------------------------------------------ output ---
    def write_outputs(self, rows: list[dict]) -> dict:
        """Persist sequence records + coverage report next to the CSVs."""
        if not self.output_dir:
            return {}
        os.makedirs(self.output_dir, exist_ok=True)
        paths = {}
        page_seqs = self.by_id()
        if page_seqs:
            seq_path = os.path.join(self.output_dir, 'sequences.csv')
            with open(seq_path, 'w', encoding='utf-8') as fh:
                fh.write('SEQ_ID,PAGE_NUM,KIND,CYCLIC,CONFIDENCE,SEQUENCE\n')
                for ref, records in sorted(page_seqs.items()):
                    for record in records:
                        fh.write(f"{record.seq_id},{record.page_num},{record.kind},"
                                 f"{record.cyclic},{record.confidence},{record.sequence}\n")
            paths['sequences'] = seq_path
        report = self.coverage_report(rows)
        report['services'] = self.service_report()
        report['page_entities'] = self.page_entities
        report_path = os.path.join(self.output_dir, 'coverage_report.json')
        with open(report_path, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
        paths['coverage'] = report_path
        ledger_path = os.path.join(self.output_dir, 'document_ledger.json')
        self._ledger.save(ledger_path)
        paths['ledger'] = ledger_path
        return paths
