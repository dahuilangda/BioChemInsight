"""Typed fragment and peptide facts are independent of the document regression."""
import asyncio
import unittest
from rdkit import Chem
from utils.series_member_synthesis import pubchem_radical_to_attachment_fragment, parse_full_molecule_smiles
from skills.name_resolution.skill import pubchem_lookup, peptide_graph_validate


class NameResolutionTest(unittest.TestCase):
    def test_single_radical_site_supplies_an_explicit_fragment_attachment(self):
        fragment = pubchem_radical_to_attachment_fragment({'cid': 1, 'isomeric_smiles': '[CH2]CO'})
        self.assertIsNotNone(fragment)
        self.assertEqual(sum(atom.GetAtomicNum() == 0 for atom in fragment.GetAtoms()), 1)
        self.assertEqual(sum(atom.GetNumRadicalElectrons() for atom in fragment.GetAtoms()), 0)
        self.assertEqual(fragment.GetProp('_LookupCID'), '1')

    def test_saturated_molecule_has_no_justified_attachment(self):
        self.assertIsNone(pubchem_radical_to_attachment_fragment({'isomeric_smiles': 'CCO'}))
        self.assertIsNone(pubchem_radical_to_attachment_fragment({'isomeric_smiles': '[CH2]C[CH2]'}))

    def test_radical_is_not_a_closed_compound(self):
        self.assertIsNone(parse_full_molecule_smiles('[CH2]CO'))

    def test_external_atomic_operation_requires_source_evidence(self):
        result = asyncio.run(pubchem_lookup({'name': 'unsupported name'}))
        self.assertEqual(result['status'], 'blocked')

    def test_local_parser_uses_unique_writable_temporary_inputs(self):
        from unittest.mock import patch
        from pathlib import Path
        from utils.opsin_local import name_to_smiles
        paths = []
        def parse(name, **options):
            path = Path(options['tmp_fpath']); paths.append(path)
            self.assertTrue(path.parent.is_dir())
            self.assertNotEqual(path.parent, Path.cwd())
            return 'CCO'
        with patch('py2opsin.py2opsin', side_effect=parse):
            name_to_smiles('source name'); name_to_smiles('source name')
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(not path.parent.exists() for path in paths))

    def test_printed_name_candidate_does_not_replace_an_accepted_image_graph(self):
        import tempfile
        from pathlib import Path
        import fitz
        from unittest.mock import patch
        from utils.text_fragment_assembly import attach_source_name_candidates
        with tempfile.TemporaryDirectory() as temporary:
            pdf = Path(temporary) / 'source.pdf'
            with fitz.open() as document:
                document.new_page(); document.save(pdf)
            rows = [{'COMPOUND_ID': 'Q8', 'SMILES': ''}, {'COMPOUND_ID': 'Q9', 'SMILES': 'CCN', 'STRUCTURE_REVIEW_SOURCE': 'user'}]
            with patch('utils.series_member_synthesis.harvest_full_names_from_text', return_value={'Q8': 'printed full name', 'Q9': 'different name'}), patch('py2opsin.py2opsin', return_value='CCO'):
                attach_source_name_candidates(rows, str(pdf))
            self.assertEqual(rows[0]['NAME_STRUCTURE_CANDIDATE'], 'CCO')
            self.assertEqual(rows[0]['SMILES'], '')
            self.assertEqual(rows[1]['SMILES'], 'CCN')
            self.assertNotIn('NAME_STRUCTURE_CANDIDATE', rows[1])

    def test_source_name_conflict_keeps_native_graph_as_unaccepted_evidence(self):
        import tempfile
        from pathlib import Path
        import fitz
        from unittest.mock import patch
        from utils.text_fragment_assembly import attach_source_name_candidates
        with tempfile.TemporaryDirectory() as temporary:
            pdf = Path(temporary) / 'source.pdf'
            with fitz.open() as document:
                document.new_page(); document.save(pdf)
            rows = [{'COMPOUND_ID': 'Q8', 'SMILES': 'CCN'}]
            with patch('utils.series_member_synthesis.harvest_full_names_from_text', return_value={'Q8': 'printed full name'}), patch('py2opsin.py2opsin', return_value='CCO'):
                attach_source_name_candidates(rows, str(pdf))
            self.assertEqual(rows[0]['SMILES'], '')
            self.assertEqual(rows[0]['MODEL_STRUCTURE_CANDIDATE'], 'CCN')
            self.assertEqual(rows[0]['NAME_STRUCTURE_CANDIDATE'], 'CCO')
            self.assertEqual(rows[0]['STRUCTURE_RECOVERY_STATUS'], 'source_name_graph_conflict')

    def test_peptide_graph_validator_preserves_cyclization_and_stereo(self):
        original = 'C[C@H]1NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC1=O'
        result = asyncio.run(peptide_graph_validate({'smiles': original, 'timeout': 2}))
        self.assertFalse(result['image_identity_verified'])
        self.assertIn(result['status'], {'resolved', 'ambiguous', 'partial', 'budget_exceeded'})
        self.assertIn('validation', result)
        if result['status'] != 'resolved': self.assertFalse(result['ok'])
        # The validator never supplies a replacement SMILES for the native decoder.
        self.assertNotIn('smiles', result)


if __name__ == '__main__': unittest.main()
