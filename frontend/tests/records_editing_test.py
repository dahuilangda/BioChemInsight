"""Persistence, rollback and export semantics, isolated from inference and real jobs."""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class RecordsEditingTest(unittest.TestCase):
    def setUp(self):
        old = sys.stderr
        try: from frontend.backend import main
        finally: sys.stderr = old
        self.main = main
        self.folder = tempfile.TemporaryDirectory(); self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name); self.path = self.root / 'structures.csv'
        self.path.write_text('COMPOUND_ID,SMILES\nsubject,CCO\n')
        self.task = main.Task(id='editing-test', type='structure_extraction', status='completed', result_path=str(self.path), data=[{'COMPOUND_ID': 'subject', 'SMILES': 'CCO'}], params={})
        task = self.task
        class Manager:
            def get(self, key): return task if key == task.id else None
            def update(self, key, **fields):
                for name,value in fields.items(): setattr(task,name,value)
                return task
        self.manager = patch.object(main, 'task_manager', Manager()); self.manager.start(); self.addCleanup(self.manager.stop)

    def save(self, rows, revision='0'):
        return asyncio.run(self.main.update_task_structures(self.task.id, self.main.UpdateStructuresRequest(records=rows, base_revision=revision)))

    def test_row_identity_survives_reload_identifier_edits_and_reordering(self):
        raw = [{'COMPOUND_ID': 'same', 'SMILES': 'CCO'}, {'COMPOUND_ID': 'same', 'SMILES': 'CCO'}]
        first = self.main._normalize_records(raw, self.root, with_record_ids=True)
        repeated = self.main._normalize_records(raw, self.root, with_record_ids=True)
        self.assertEqual([row['RECORD_ID'] for row in first], [row['RECORD_ID'] for row in repeated])
        self.assertNotEqual(first[0]['RECORD_ID'], first[1]['RECORD_ID'])
        first[0]['COMPOUND_ID'] = 'corrected'
        moved = self.main._normalize_records(list(reversed(first)), self.root, with_record_ids=True)
        self.assertEqual(moved[1]['RECORD_ID'], first[0]['RECORD_ID'])

    def test_save_revision_conflict_and_restore(self):
        result = self.save([{'COMPOUND_ID': 'subject', 'SMILES': 'CCO', 'Endpoint (nM)': '1 ± .2'}])
        self.assertTrue(result.task.params['records_revision'])
        self.assertEqual(len(self.main.structure_revision_history(self.task.id)['revisions']), 1)
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as failure: self.save([])
        self.assertEqual(failure.exception.status_code, 409)
        restored = asyncio.run(self.main.undo_last_structure_edit(self.task.id, {'base_revision': result.task.params['records_revision']}))
        self.assertEqual(restored.records[0]['COMPOUND_ID'], 'subject')
        self.assertNotIn('Endpoint (nM)', restored.records[0])

    def test_corrected_values_and_removed_columns_survive_export(self):
        import pandas as pd
        rows = [{'COMPOUND_ID': 'subject', 'SMILES': 'CCO', 'USER_MEASUREMENT_EDITS': '{"Endpoint (nM)":""}', 'USER_REMOVED_MEASUREMENTS': '["Removed"]'}]
        path = self.main.merge_structure_activity_records(pd.DataFrame(rows), [{'COMPOUND_ID': 'subject', 'Endpoint (nM)': '100', 'Removed': '200'}], self.root)
        result = pd.read_csv(path, keep_default_na=False)
        self.assertEqual(result.iloc[0]['Endpoint (nM)'], '')
        self.assertNotIn('Removed', result.columns)

    def test_ambiguous_id_retains_unlinked_values(self):
        import pandas as pd
        rows = [{'COMPOUND_ID': 'subject', 'SMILES': 'CCO'}, {'COMPOUND_ID': 'subject', 'SMILES': 'CCN'}]
        path = self.main.merge_structure_activity_records(pd.DataFrame(rows), [{'COMPOUND_ID': 'subject', 'Endpoint': '100'}], self.root)
        result = pd.read_csv(path, keep_default_na=False)
        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[-1]['MEASUREMENT_LINK_STATUS'], 'ambiguous_identity')
        self.assertEqual(result.iloc[0]['Endpoint'], '')

    def test_scientific_column_preserves_endpoint_unit_and_provenance(self):
        records, metadata = self.main._records_from_assay_results({'generic label': {'subject': {'endpoint': 'EC50', 'target': 'Receptor', 'unit': 'uM', 'source_kind': 'predicted', 'value': '< 2 ± .1'}}})
        self.assertEqual(records[0]['Receptor · EC50 (µM) · predicted'], '< 2 ± .1')
        self.assertEqual(records[0]['Receptor · EC50 (µM) · predicted__source_kind'], 'predicted')


if __name__ == '__main__': unittest.main()
