import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils import llm_utils


class StructureBorderReviewTests(unittest.TestCase):
    def classify_with_responses(self, responses, *, border_contact=None, strictness="strict"):
        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            image_path = Path(image_file.name)
            border_contact = border_contact or {
                "suspicious": True,
                "sides": ["left", "right"],
                "ratios": {"left": 0.1, "right": 0.1},
            }
            with mock.patch.object(llm_utils, "analyze_border_contact", return_value=border_contact):
                with mock.patch.object(llm_utils, "call_visual_model", side_effect=list(responses)) as call_mock:
                    result = llm_utils.classify_structure_candidate(str(image_path), strictness=strictness)
            return result, call_mock.call_count

    def test_strict_tight_crop_passes_when_visual_review_confirms_complete(self):
        result, call_count = self.classify_with_responses(
            [
                '{"structure_type":"complete_compound","is_complete_compound":true,"reason":"single exact molecule fully visible"}',
                '{"crop_status":"not_cropped","is_cropped":false,"reason":"tight crop but all chemistry is visible"}',
                '{"structure_type":"complete_compound","is_complete_compound":true,"reason":"whole molecule visible in tight crop"}',
            ]
        )

        self.assertEqual(call_count, 3)
        self.assertEqual(result["structure_type"], "complete_compound")
        self.assertTrue(result["is_complete_compound"])

    def test_strict_crop_check_fragment_still_blocks_candidate(self):
        result, call_count = self.classify_with_responses(
            [
                '{"structure_type":"complete_compound","is_complete_compound":true,"reason":"initial pass"}',
                '{"crop_status":"fragment","is_cropped":true,"reason":"bond exits image boundary"}',
            ]
        )

        self.assertEqual(call_count, 2)
        self.assertEqual(result["structure_type"], "fragment")
        self.assertFalse(result["is_complete_compound"])

    def test_strict_border_review_uncertain_still_blocks_candidate(self):
        result, call_count = self.classify_with_responses(
            [
                '{"structure_type":"complete_compound","is_complete_compound":true,"reason":"initial pass"}',
                '{"crop_status":"uncertain","is_cropped":false,"reason":"cannot confirm right edge"}',
                '{"structure_type":"uncertain","is_complete_compound":false,"reason":"right edge is ambiguous"}',
            ]
        )

        self.assertEqual(call_count, 3)
        self.assertEqual(result["structure_type"], "uncertain")
        self.assertFalse(result["is_complete_compound"])

    def test_strict_unconfirmed_border_contact_is_not_allowed_through(self):
        result, call_count = self.classify_with_responses(
            [
                '{"structure_type":"complete_compound","is_complete_compound":true,"reason":"initial pass"}',
                'not json',
                'not json',
            ]
        )

        self.assertEqual(call_count, 3)
        self.assertEqual(result["structure_type"], "uncertain")
        self.assertFalse(result["is_complete_compound"])
        self.assertIn("visual completeness was not confirmed", result["reason"])

    def test_markush_is_not_rescued_by_border_review_logic(self):
        result, call_count = self.classify_with_responses(
            [
                '{"structure_type":"markush","is_complete_compound":false,"reason":"R-group placeholder present"}',
            ]
        )

        self.assertEqual(call_count, 1)
        self.assertEqual(result["structure_type"], "markush")
        self.assertFalse(result["is_complete_compound"])

    def test_permissive_mode_preserves_existing_no_review_behavior(self):
        result, call_count = self.classify_with_responses(
            [
                '{"structure_type":"complete_compound","is_complete_compound":true,"reason":"single exact molecule fully visible"}',
            ],
            strictness="permissive",
        )

        self.assertEqual(call_count, 1)
        self.assertEqual(result["structure_type"], "complete_compound")
        self.assertTrue(result["is_complete_compound"])


if __name__ == "__main__":
    unittest.main()
