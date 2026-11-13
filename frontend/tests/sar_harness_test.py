"""SAR control policy is tested with unrelated synthetic records and no model calls."""
import json
import tempfile
import unittest
from pathlib import Path
from utils.sar_harness import assess_sar, plan_recovery, save_sar_report
from utils.measurement_evidence import parse_measurement_table


class SARHarnessTest(unittest.TestCase):
    def report(self, rows, **options):
        return assess_sar(rows, result_pages=[], **options)

    def test_external_lookup_requires_failed_local_parse(self):
        row = {'COMPOUND_ID': 'subject', 'SMILES': '', 'COMPOUND_NAME': 'source-supported name'}
        report = self.report([row]); actions = plan_recovery(report)
        self.assertEqual(actions[0]['operation'], 'opsin_resolve')
        gap = actions[0]['gap_id']
        report['attempts'] = [{'gap_id': gap, 'operation': 'opsin_resolve', 'status': 'unresolved'}]
        self.assertEqual(plan_recovery(report)[0]['operation'], 'pubchem_lookup')

    def test_geometry_is_acquired_before_decoding_without_an_image(self):
        report = self.report([{'SMILES': 'CCO', 'PAGE_NUM': 2}])
        self.assertEqual(plan_recovery(report)[0]['operation'], 'locate_boxes')

    def test_an_empty_smiles_is_not_a_valid_empty_molecule(self):
        self.assertFalse(self.report([{'COMPOUND_ID': 'subject', 'SMILES': ''}])['units'][0]['graph_valid'])

    def test_conflicting_graphs_block_simple_identifier_join(self):
        report = self.report([{'COMPOUND_ID': 'subject', 'SMILES': 'CCO'}, {'COMPOUND_ID': 'subject', 'SMILES': 'CCN'}])
        self.assertEqual(sum(gap['field'] == 'identity_conflict' for gap in report['gaps']), 2)

    def test_user_structure_is_not_overwritten_by_automatic_recovery(self):
        report = self.report([{'COMPOUND_ID': 'subject', 'SMILES': '', 'STRUCTURE_REVIEW_SOURCE': 'user'}])
        self.assertEqual(plan_recovery(report), [])

    def test_completed_inventory_does_not_prove_image_correctness(self):
        report = self.report([{'COMPOUND_ID': 'subject', 'SMILES': 'CCO', 'PAGE_NUM': 2}],
            page_entities=[{'page_num': 2, 'compounds': [{'label_in_page': 'subject'}], 'ok': True}])
        self.assertTrue(report['coverage']['complete'])
        self.assertFalse(report['coverage']['correctness_confirmed'])

    def test_document_result_discovery_is_a_separate_goal(self):
        report = assess_sar([{'COMPOUND_ID': 'subject', 'SMILES': 'CCO'}])
        self.assertEqual(plan_recovery(report)[0]['operation'], 'discover_measurements')

    def test_same_local_crop_is_not_resampled_after_no_resolution(self):
        report = self.report([{'COMPOUND_ID': '', 'SMILES': 'CCO', 'SEGMENT_FILE': '/source/crop.png'}])
        action = plan_recovery(report)[0]
        report['attempts'] = [{**action, 'status': 'needs_review'}]
        self.assertEqual(plan_recovery(report)[0]['operation'], 'human_review')

    def test_missing_page_subjects_are_not_hidden_by_terminal_units(self):
        report = self.report([{'COMPOUND_ID': 'subject', 'SMILES': 'CCO', 'PAGE_NUM': 2}],
            page_entities=[{'page_num': 2, 'compounds': [{'label_in_page': 'subject'}, {'label_in_page': 'other'}]}])
        self.assertFalse(report['coverage']['complete'])
        self.assertEqual(plan_recovery(report)[0]['operation'], 'locate_boxes')

    def test_peptide_text_survives_as_typed_evidence(self):
        row = {'COMPOUND_ID': 'peptide', 'SMILES': 'NCC(=O)NCC(=O)NCC(=O)O', 'SEQUENCE_TEXT': 'source residue notation'}
        report = self.report([row])
        self.assertEqual(report['units'][0]['kind'], 'peptide')
        self.assertEqual(plan_recovery(report)[0]['operation'], 'peptide_graph_validate')
        self.assertEqual(row['SEQUENCE_TEXT'], 'source residue notation')

    def test_template_label_does_not_claim_a_complete_compounds_measurement(self):
        from utils.sar_harness import _append_observations
        rows=[{'COMPOUND_ID':'Q8','SMILES':'','STRUCTURE_TYPE':'markush'}, {'COMPOUND_ID':'Q8','SMILES':'CCO','LOCAL_EVIDENCE_STATUS':'verified'}]
        _append_observations(rows,{'records':[{'compound_id':'Q8','page':3,'table_index':0,'observations':[{'subject_id':'Q8','page':3,'table_index':0,'column':'Endpoint','value':'4'}]}]})
        self.assertNotIn('BIOACTIVITY_OBSERVATIONS',rows[0])
        self.assertIn('BIOACTIVITY_OBSERVATIONS',rows[1])

    def test_unverified_structure_identity_keeps_result_as_a_separate_record(self):
        from utils.sar_harness import _append_observations
        rows=[{'COMPOUND_ID':'Q8','SMILES':'CCO','MEASUREMENT_LINK_STATUS':'identifier_unverified'}]
        _append_observations(rows,{'records':[{'compound_id':'Q8','page':3,'table_index':0,'observations':[{'subject_id':'Q8','page':3,'table_index':0,'column':'Endpoint','value':'4'}]}]})
        self.assertEqual(len(rows),2)
        self.assertNotIn('BIOACTIVITY_OBSERVATIONS',rows[0])
        self.assertEqual(rows[1]['SMILES'],'')

    def test_result_table_crop_is_not_sent_to_structure_recognition(self):
        report = self.report([{'COMPOUND_ID': 'subject', 'SMILES': '', 'RESULT_ONLY': 'true', 'SEGMENT_FILE': '/source/table.png'}])
        self.assertFalse(any(a['operation'] == 'recognize_smiles' for a in plan_recovery(report)))

    def test_planner_reserves_data_coverage_instead_of_only_repeating_geometry(self):
        from utils.sar_harness import reserve_result_action
        selected = [{'operation': 'locate_boxes', 'page': 2}]
        eligible = [*selected, {'operation': 'read_result_table', 'page': 3}, {'operation': 'read_result_table', 'page': 42}]
        report = {'measurement_discovery': {'decisions': {'3': {'assay_names': ['one']}, '42': {'assay_names': ['one', 'two', 'three']}}}}
        actions = reserve_result_action(selected, eligible, report, 2)
        self.assertEqual(actions[0]['page'], 42)
        self.assertEqual(len(actions), 2)

    def test_disagreeing_reads_of_one_source_cell_are_preserved_and_flagged(self):
        from utils.sar_harness import _append_observations
        rows = [{'COMPOUND_ID': 'subject', 'SMILES': 'CCO'}]
        def extraction(value):
            return {'records': [{'compound_id': 'subject', 'page': 3, 'table_index': 0, 'observations': [{'subject_id': 'subject', 'column': 'Endpoint', 'page': 3, 'table_index': 0, 'value': value}]}]}
        _append_observations(rows, extraction('2'))
        _append_observations(rows, extraction('3'))
        self.assertEqual(len(json.loads(rows[0]['BIOACTIVITY_OBSERVATIONS'])), 2)
        self.assertTrue(any(g['field'] == 'measurement_conflict' for g in self.report(rows)['gaps']))

    def test_result_row_mentions_do_not_require_molecular_boxes(self):
        report = self.report([], page_entities=[{'page_num': 8, 'compounds': [{'label_in_page': 'subject', 'evidence_kind': 'result_row'}]}])
        self.assertFalse(any(a['operation'] == 'locate_boxes' for a in plan_recovery(report)))

    def test_page_recovery_is_not_repeated_for_each_missing_subject(self):
        report = self.report([], page_entities=[{'page_num': 4, 'compounds': [{'label_in_page': 'one'}, {'label_in_page': 'two'}]}])
        actions = plan_recovery(report)
        self.assertEqual(sum(a['operation'] == 'locate_boxes' for a in actions), 1)
        report['attempts'] = [{**actions[0], 'status': 'no_new_regions'}]
        self.assertFalse(any(a['operation'] == 'locate_boxes' for a in plan_recovery(report)))

    def test_planner_outage_falls_back_to_executable_data_recovery(self):
        import tempfile
        from unittest.mock import patch
        from utils.sar_harness import recover_sar_rows
        rows = [{'COMPOUND_ID': 'subject', 'SMILES': 'CCO', 'PAGE_NUM': 1}]
        with tempfile.TemporaryDirectory() as output, patch('utils.sar_harness.select_recovery_actions', side_effect=RuntimeError('planner offline')) as planner, patch('utils.measurement_evidence.read_measurement_page', return_value={'records': [], 'tables': [], 'gaps': []}) as reader:
            result, report = recover_sar_rows(rows, '/unused.pdf', output, result_pages=[7], max_actions=2)
        self.assertEqual(planner.call_count, 1)
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(report['recovery_actions_used'], 1)
        self.assertTrue(report['planning_failures'])

    def test_total_provider_outage_is_explicit_and_keeps_checkpoint(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from utils.sar_harness import recover_sar_rows
        rows = [{'COMPOUND_ID': 'subject', 'SMILES': 'CCO', 'PAGE_NUM': 1}]
        with tempfile.TemporaryDirectory() as output, patch('utils.sar_harness.select_recovery_actions', side_effect=RuntimeError('planner offline')), patch('utils.measurement_evidence.read_measurement_page', side_effect=RuntimeError('vision offline')):
            result, report = recover_sar_rows(rows, '/unused.pdf', output, result_pages=[7], max_actions=2)
            self.assertTrue((Path(output) / 'recovery_checkpoint.json').is_file())
        self.assertEqual(report['execution_status'], 'blocked_provider')
        self.assertEqual(result[0]['SMILES'], 'CCO')

    def test_new_task_refreshes_budget_while_checkpoint_resume_retains_usage(self):
        from utils.sar_harness import new_recovery_session
        parent = {'recovery_actions_used': 8, 'attempts': [{'operation': 'read_result_table', 'status': 'recovered'}]}
        child = new_recovery_session(parent)
        self.assertEqual(child['recovery_actions_used'], 0)
        self.assertEqual(child['attempts'], parent['attempts'])
        self.assertEqual(parent['recovery_actions_used'], 8)
        child['attempts'].append({'operation': 'human_review'})
        self.assertEqual(len(parent['attempts']), 1)

    def test_assessment_preserves_persisted_budget_usage(self):
        report = assess_sar([], previous={'recovery_actions_used': 5})
        self.assertEqual(report['recovery_actions_used'], 5)

    def test_dense_result_table_must_account_for_every_cell(self):
        table = {'tables': [{'columns': [{'endpoint': 'Endpoint', 'unit': 'nM', 'target': '', 'source_kind': 'experimental', 'header_evidence': 'Endpoint (nM)'}],
            'rows': [{'compound_id': 'subject', 'row_evidence': 'subject', 'values': []}]}], 'gaps': []}
        with self.assertRaises(ValueError): parse_measurement_table(json.dumps(table))
        table['tables'][0]['rows'][0]['values'] = ['< 4 ± 1']
        self.assertEqual(parse_measurement_table(json.dumps(table))['tables'][0]['columns'][0]['column'], 'Endpoint (nM)')


if __name__ == '__main__': unittest.main()
