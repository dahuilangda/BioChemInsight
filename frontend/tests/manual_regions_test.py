"""Manual selections preserve PDF coordinates and never save failed inference."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fitz


@unittest.skipUnless(importlib.util.find_spec('redis'), 'backend runtime required')
class ManualRegionsTest(unittest.TestCase):
    def setUp(self):
        from frontend.backend import main
        self.main = main
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.pdf = Path(self.directory.name) / 'source.pdf'
        with fitz.open() as document:
            document.new_page(width=600, height=800)
            document.save(self.pdf)
        self.document = SimpleNamespace(stored_path=self.pdf, total_pages=1)
        coordinate_mock = patch.object(main, 'recover_source_coordinates', return_value=None)
        self.coordinate_mock = coordinate_mock.start(); self.addCleanup(coordinate_mock.stop)

    def invoke(self, **fields):
        payload = self.main.ManualRegionRequest(page=1, x=.1, y=.25, w=.3, h=.2, **fields)
        with patch.object(self.main, '_ensure_pdf_or_404', return_value=self.document):
            return self.main.recognize_pdf_region('region-test', payload)

    def test_preserves_xyxy_evidence(self):
        with patch('utils.api_harness.APIHarness.image', return_value={'ok': True, 'smiles': 'CCO'}):
            record = self.invoke()
        box = json.loads(Path(record['BOX_COORDS_FILE']).read_text())
        self.assertEqual(box['box'], [60, 200, 240, 360])
        self.assertEqual(box['page_width'], 600)
        self.assertEqual(record['source_pages'], '1')
        self.assertTrue(Path(record['SEGMENT_FILE']).is_file())

    def test_manual_parse_retains_source_geometry(self):
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        mol=Chem.MolFromSmiles('CCO');rdDepictor.Compute2DCoords(mol,canonOrient=False)
        block=Chem.MolToMolBlock(mol)
        self.coordinate_mock.return_value=block
        with patch('utils.api_harness.APIHarness.image', return_value={'ok':True,'smiles':'CCO'}): record=self.invoke()
        self.assertEqual(record['MOLBLOCK'],block)
        self.assertEqual(record['STRUCTURE_POSE_SOURCE'],'source_image_coordinate_graph_mapping')

    def test_failed_recognition_does_not_create_evidence(self):
        from fastapi import HTTPException
        with patch('utils.api_harness.APIHarness.image', side_effect=RuntimeError('offline')):
            with self.assertRaises(HTTPException) as failure:
                self.invoke()
        self.assertEqual(failure.exception.status_code, 503)
        self.assertFalse((self.pdf.parent / 'manual_regions').exists())

    def test_first_preview_is_a_cached_webp(self):
        from PIL import Image
        import io
        with patch.object(self.main, '_ensure_pdf_or_404', return_value=self.document):
            response = self.main.get_pdf_preview('region-test', 1)
            again = self.main.get_pdf_preview('region-test', 1)
        self.assertEqual(response.media_type, 'image/webp')
        self.assertEqual(response.body, again.body)
        self.assertIn('max-age', response.headers['cache-control'])
        with Image.open(io.BytesIO(response.body)) as image:
            self.assertLessEqual(image.width, 1100)

    def test_recovery_detection_is_atomic_and_reuses_existing_geometry(self):
        from utils.sar_regions import locate_structure_regions
        detected = {'structures': [{'bbox': [150, 500, 600, 900], 'score': .5}]}
        with patch('utils.api_harness.APIHarness.image', return_value=detected) as operation:
            rows = locate_structure_regions(str(self.pdf), 1, self.root_path_for_regions(), [])
            self.assertEqual(operation.call_args.args[0], 'detect')
            self.assertEqual(rows[0]['SMILES'], '')
            self.assertEqual(rows[0]['COMPOUND_ID'], '')
            evidence = json.loads(Path(rows[0]['BOX_COORDS_FILE']).read_text())
            self.assertEqual(evidence['box'], [60, 200, 240, 360])
            again = locate_structure_regions(str(self.pdf), 1, self.root_path_for_regions(), rows)
        self.assertEqual(again, [])

    def root_path_for_regions(self):
        return str(self.pdf.parent / 'detected_regions')

    def test_table_region_preserves_rows_and_geometry_without_inventing_structure(self):
        result = {'tables': [{'columns': [{'column': 'Permeability (cm/s)', 'endpoint': 'Permeability', 'unit': 'cm/s', 'source_kind': 'experimental'}], 'rows': [{'compound_id': 'T8', 'values': ['1.4e-6'], 'row_evidence': 'visible row'}]}], 'gaps': []}
        payload = self.main.ManualRegionRequest(page=1, x=.1, y=.25, w=.3, h=.2)
        with patch.object(self.main, '_ensure_pdf_or_404', return_value=self.document), patch('utils.llm_utils.run_vision_json_task', return_value=result):
            response = self.main.parse_manual_measurement_region('region-test', payload)
        row = response['records'][0]
        self.assertEqual(row['SMILES'], '')
        self.assertEqual(row['MEASUREMENT_LINK_STATUS'], 'needs_structure_identity')
        self.assertEqual(json.loads(row['BIOACTIVITY_OBSERVATIONS'])[0]['value'], '1.4e-6')
        self.assertEqual(json.loads(Path(row['BOX_COORDS_FILE']).read_text())['box'], [60, 200, 240, 360])

    def test_table_failure_keeps_no_published_record(self):
        from fastapi import HTTPException
        payload = self.main.ManualRegionRequest(page=1, x=.1, y=.25, w=.3, h=.2)
        with patch.object(self.main, '_ensure_pdf_or_404', return_value=self.document), patch('utils.llm_utils.run_vision_json_task', side_effect=RuntimeError('offline')):
            with self.assertRaises(HTTPException) as failure:
                self.main.parse_manual_measurement_region('region-test', payload)
        self.assertEqual(failure.exception.status_code, 503)
        self.assertFalse((self.pdf.parent / 'manual_regions').exists())

    def test_invalid_regions_do_not_call_inference(self):
        from fastapi import HTTPException
        for fields in ({'x': -.1}, {'w': 0}, {'x': .9, 'w': .2}, {'page': 2}, {'h': float('nan')}):
            values = dict(page=1, x=.1, y=.2, w=.3, h=.2)
            values.update(fields)
            with patch.object(self.main, '_ensure_pdf_or_404', return_value=self.document), patch('utils.api_harness.APIHarness.image') as inference:
                with self.assertRaises(HTTPException):
                    self.main.recognize_pdf_region('region-test', self.main.ManualRegionRequest(**values))
                inference.assert_not_called()


if __name__ == '__main__':
    unittest.main()
