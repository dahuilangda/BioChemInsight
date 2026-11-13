"""The chemical assembler is exercised independently of any document or model."""
import asyncio
import unittest
from rdkit import Chem
from skills.markush_assemble.skill import assemble


class MarkushAssemblyTest(unittest.TestCase):
    def test_only_declared_fragments_form_members(self):
        result = asyncio.run(assemble({'template_smiles': 'c1cc([*:1])ccc1', 'fragments': {'1': ['[*:1]C', '[*:1]O']}}))
        self.assertTrue(result['ok'])
        self.assertEqual(len(result['members']), 2)
        for member in result['members']:
            self.assertNotIn('*', member['smiles'])
            self.assertIsNotNone(Chem.MolFromSmiles(member['smiles']))

    def test_missing_attachment_is_a_gap_not_an_invented_molecule(self):
        result = asyncio.run(assemble({'template_smiles': 'C([*:1])[*:2]', 'fragments': {'1': ['[*:1]C']}}))
        self.assertFalse(result['ok'])
        self.assertTrue(result['gaps'])

    def test_existing_stereochemistry_is_preserved(self):
        result = asyncio.run(assemble({'template_smiles': 'N[C@@H](C)C(=O)N[*:1]', 'fragments': {'1': ['[*:1]CC']}}))
        self.assertTrue(result['ok'])
        self.assertIn('@', result['members'][0]['smiles'])

    def test_formula_and_row_position_cannot_claim_an_activity(self):
        result = asyncio.run(assemble({'template_smiles': 'c1cc([*:1])ccc1', 'fragments': {'1': ['[*:1]C']},
            'activity_rows': [{'compound_id': 'member_1', 'EC50': '8'}, {'compound_id': 'C7H8', 'EC50': '9'}]}))
        self.assertTrue(result['ok'])
        self.assertTrue(all(not row['matched'] for row in result['activity_join']))

    def test_multi_atom_fragment_requires_an_explicit_attachment(self):
        result = asyncio.run(assemble({'template_smiles': 'c1cc([*:1])ccc1', 'fragments': {'1': ['CCO']}}))
        self.assertFalse(result['ok'])

    def test_independent_name_conflict_vetoes_visually_accepted_assembly(self):
        from unittest.mock import patch
        from utils.text_fragment_assembly import crosscheck_assembly_names
        rows = [{'compound_id': 'Q8', 'assembly_status': 'assembled', 'assembled_smiles': 'CCCO', 'assembled_visual_review': {'consistent': True}}]
        with patch('utils.series_member_synthesis.harvest_full_names_from_text', return_value={'Q8': 'source name'}), patch('py2opsin.py2opsin', return_value='CCO'):
            crosscheck_assembly_names(rows, [])
        self.assertEqual(rows[0]['assembly_status'], 'blocked')
        self.assertEqual(rows[0]['independent_name_review']['status'], 'conflict')

    def test_reused_identifier_with_distinct_names_is_ambiguous(self):
        from unittest.mock import patch
        from utils.series_member_synthesis import harvest_full_names_from_text
        with patch('utils.series_member_synthesis._opsin_resolves', return_value=True):
            names = harvest_full_names_from_text(['Q8'], [{'markdown': 'different name one (Q8).'}, {'markdown': 'different name two (Q8).'}])
        self.assertEqual(names, {})

    def test_name_heading_normalizes_typography_without_spectroscopy_prose(self):
        from unittest.mock import patch
        from utils.series_member_synthesis import harvest_full_names_from_text
        with patch('utils.series_member_synthesis._opsin_resolves', side_effect=lambda text: text == 'source-name'):
            names = harvest_full_names_from_text(['Q8'], [{'markdown': 'Previous spectroscopy prose.\nsource‐name (Q8). Characterization'}])
        self.assertEqual(names, {'Q8': 'source-name'})

    def test_declared_formula_mismatch_is_rejected(self):
        result = asyncio.run(assemble({'template_smiles': 'c1cc([*:1])ccc1', 'fragments': {'1': ['[*:1]C']}, 'declared_formula': 'C8H10'}))
        self.assertFalse(result['ok'])


if __name__ == '__main__': unittest.main()
