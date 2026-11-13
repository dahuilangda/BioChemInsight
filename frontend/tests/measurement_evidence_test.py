"""Contract checks use unrelated synthetic subjects, independent of the demo PDF."""
import json
import unittest
from unittest.mock import patch
from utils.measurement_evidence import parse_local_evidence, parse_page_evidence, apply_local_evidence, measurement_column, user_measurement_edits


class MeasurementEvidenceTest(unittest.TestCase):
    def payload(self):
        return {'compound_id': 'Q7', 'annotation_text': 'endpoint result', 'confidence': 'high', 'gap': '',
            'observations': [{'subject_id': 'Q7', 'endpoint': 'EC₅₀', 'target': 'Receptor Z', 'unit': 'uM', 'value': '< 0.8 ± 0.2', 'source_kind': 'experimental', 'evidence': 'Q7 endpoint result'}]}

    def test_ownership_must_match_the_local_subject(self):
        payload = self.payload(); payload['observations'][0]['subject_id'] = 'Q9'
        result = parse_local_evidence(json.dumps(payload))
        self.assertEqual(result['observations'], [])
        self.assertEqual(len(result['rejected_observations']), 1)

    def test_columns_preserve_endpoint_units_and_provenance(self):
        result = parse_local_evidence(json.dumps(self.payload()))
        observation = result['observations'][0]
        self.assertEqual(observation['column'], 'Receptor Z · EC50 (µM)')
        self.assertEqual(observation['value'], '< 0.8 ± 0.2')
        predicted = dict(observation, source_kind='predicted')
        self.assertNotEqual(measurement_column(predicted), observation['column'])

    def test_uncertainty_header_does_not_create_a_second_assay(self):
        self.assertEqual(measurement_column({'endpoint': 'EC50 ± SD', 'target': 'Receptor Z', 'unit': 'nM'}), measurement_column({'endpoint': 'EC50', 'target': 'Receptor Z', 'unit': 'nM'}))

    def test_comparison_prose_cannot_assign_values_by_mention_order(self):
        from utils.measurement_evidence import quarantine_ambiguous_prose_ownership, observation_values
        rows = [{'COMPOUND_ID': 'Q7', 'BIOACTIVITY_OBSERVATIONS': json.dumps([{'subject_id': 'Q7', 'column': 'EC50 (nM)', 'value': '12', 'evidence': 'Q7 compared with Q8 gave 12 and 18, respectively.'}, {'subject_id': 'Q7', 'column': 'EC50 (nM)', 'value': '18', 'evidence': 'Q7', 'page': 4}])}, {'COMPOUND_ID': 'Q8'}]
        self.assertEqual(quarantine_ambiguous_prose_ownership(rows), 1)
        self.assertEqual(observation_values(rows[0])['EC50 (nM)'], '18')
        self.assertEqual(len(json.loads(rows[0]['BIOACTIVITY_OBSERVATIONS'])), 2)

    def test_target_is_not_repeated_in_a_complete_source_header(self):
        self.assertEqual(measurement_column({'endpoint': 'Receptor Z binding', 'target': 'Receptor Z', 'unit': 'nM'}), 'Receptor Z binding (nM)')

    def test_correction_commits_identifier_and_value_together(self):
        row = {'COMPOUND_ID': 'wrong', 'ANNOTATION_TEXT': 'unrelated result'}
        apply_local_evidence(row, parse_local_evidence(json.dumps(self.payload())))
        self.assertEqual(row['COMPOUND_ID'], 'Q7')
        self.assertEqual(row['PREVIOUS_COMPOUND_ID'], 'wrong')
        self.assertEqual(row['Receptor Z · EC50 (µM)'], '< 0.8 ± 0.2')

    def test_ambiguous_read_does_not_publish_a_measurement(self):
        payload = self.payload(); payload['gap'] = 'Two subjects share a caption'
        row = {'COMPOUND_ID': 'Q7', 'ANNOTATION_TEXT': 'old unverified text'}
        apply_local_evidence(row, parse_local_evidence(json.dumps(payload)))
        self.assertEqual(row['LOCAL_EVIDENCE_STATUS'], 'needs_review')
        self.assertEqual(row['BIOACTIVITY_OBSERVATIONS'], '[]')
        self.assertEqual(row['ANNOTATION_TEXT'], '')

    def test_predicted_page_results_are_not_filtered_out(self):
        payload = {'has_result_records': True, 'confidence': 'high', 'evidence': 'compound rows and disposition result column',
            'endpoints': [{'endpoint': 'Intrinsic clearance', 'target': '', 'unit': 'mL/min', 'source_kind': 'predicted'}]}
        result = parse_page_evidence(json.dumps(payload))
        self.assertIn('predicted', result['endpoints'][0]['column'])
        payload['endpoints'] = []
        with self.assertRaises(ValueError): parse_page_evidence(json.dumps(payload))

    def test_user_edits_preserve_intentional_blanks(self):
        self.assertEqual(user_measurement_edits({'USER_MEASUREMENT_EDITS': '{"EC50 (nM)":""}'}), {'EC50 (nM)': ''})

    def test_repeat_source_measurements_do_not_overwrite_values(self):
        from utils.measurement_evidence import observation_values
        observations = [{'subject_id': 'Q7', 'column': 'Clearance (mL/min)', 'value': value, 'page': page, 'table_index': 0} for value, page in [('2.1', 3), ('4.2', 5)]]
        values = observation_values({'COMPOUND_ID': 'Q7', 'BIOACTIVITY_OBSERVATIONS': json.dumps(observations)})
        self.assertEqual(values['Clearance (mL/min)'], '2.1 [p. 3, table 1]; 4.2 [p. 5, table 1]')
        self.assertEqual(observation_values({'COMPOUND_ID': 'Q9', 'BIOACTIVITY_OBSERVATIONS': json.dumps(observations)}), {})

    def test_empty_source_identifier_cannot_verify_a_previous_assignment(self):
        payload=self.payload();payload['compound_id']='';payload['observations']=[]
        row={'COMPOUND_ID':'old-assignment'}
        apply_local_evidence(row,parse_local_evidence(json.dumps(payload)))
        self.assertEqual(row['LOCAL_EVIDENCE_STATUS'],'needs_review')

    def test_invalid_provenance_is_rejected(self):
        payload = self.payload(); payload['observations'][0]['source_kind'] = 'guess'
        with self.assertRaises(ValueError): parse_local_evidence(json.dumps(payload))


if __name__ == '__main__': unittest.main()
