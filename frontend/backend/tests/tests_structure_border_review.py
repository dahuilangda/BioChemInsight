import tempfile
import unittest
from pathlib import Path
from unittest import mock

from activity_parser import _downgrade_low_confidence_visual_replacements, _normalize_multi_assay_payload
from utils import llm_utils
from utils.llm_utils import (
    _extract_assay_value_text,
    extract_confident_compound_id,
    normalize_multi_assay_dict_payload,
    normalize_single_assay_dict_payload,
)
from utils.skill_prompt_loader import load_skill_json, render_skill_prompt_with_examples, render_skill_reference


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


class SkillConfidencePathTests(unittest.TestCase):
    def test_extract_assay_value_text_supports_rich_object(self):
        self.assertEqual(
            _extract_assay_value_text(
                {"value": "8.30", "confidence": "high", "reason": "same row"}
            ),
            "8.30",
        )

    def test_normalize_single_assay_dict_payload_accepts_legacy_and_rich_values(self):
        payload = {
            "Example 31": {"value": "8.30", "confidence": "high", "reason": "row"},
            "Example 32": "9.10",
            "Example 33": {"confidence": "low", "reason": "missing value"},
        }
        self.assertEqual(
            normalize_single_assay_dict_payload(payload),
            {"Example 31": "8.30", "Example 32": "9.10"},
        )

    def test_normalize_multi_assay_dict_payload_preserves_requested_assays(self):
        payload = {
            "Assay A": {"Example 31": {"value": "8.30", "confidence": "high", "reason": "row"}},
            "Assay B": {"Example 7": "54"},
        }
        self.assertEqual(
            normalize_multi_assay_dict_payload(payload, ["Assay A", "Assay B", "Assay C"]),
            {
                "Assay A": {"Example 31": "8.30"},
                "Assay B": {"Example 7": "54"},
                "Assay C": {},
            },
        )

    def test_activity_parser_normalize_multi_assay_payload_accepts_rich_values(self):
        payload = {
            "Assay A": {
                "Example 31": {"value": 0, "confidence": "high", "reason": "row"},
                "Example 32": {"VALUE": "ND", "confidence": "medium", "reason": "row"},
                "Example 33": "12.5",
            }
        }
        self.assertEqual(
            _normalize_multi_assay_payload(payload, ["Assay A"]),
            {"Assay A": {"Example 31": "0", "Example 32": "ND", "Example 33": "12.5"}},
        )

    def test_low_confidence_visual_replace_is_downgraded(self):
        visual_report = {
            "corrections": [
                {
                    "assay_name": "Assay A",
                    "compound_id": "Example 31",
                    "current_value": "8.30",
                    "visual_value": "2",
                    "action": "replace",
                    "confidence": "low",
                    "evidence": "cell is blurry",
                },
                {
                    "assay_name": "Assay A",
                    "compound_id": "Example 32",
                    "current_value": "9.10",
                    "visual_value": "9.20",
                    "action": "replace",
                    "confidence": "high",
                    "evidence": "clear cell",
                },
            ]
        }
        result = _downgrade_low_confidence_visual_replacements(visual_report)
        self.assertEqual(result["corrections"][0]["action"], "uncertain")
        self.assertEqual(result["corrections"][1]["action"], "replace")

    def test_low_confidence_compound_id_is_rejected(self):
        self.assertIsNone(
            extract_confident_compound_id(
                {"COMPOUND_ID": "Example 31", "CONFIDENCE": "low", "REASON": "ambiguous"},
                fallback=None,
                none_value=None,
            )
        )
        self.assertEqual(
            extract_confident_compound_id(
                {"COMPOUND_ID": "Example 31", "CONFIDENCE": "high", "REASON": "explicit"},
                fallback=None,
                none_value=None,
            ),
            "Example 31",
        )

    def test_missing_confidence_defaults_to_medium_for_backwards_compatibility(self):
        self.assertEqual(
            extract_confident_compound_id({"COMPOUND_ID": "Example 31"}, fallback=None, none_value=None),
            "Example 31",
        )


class SkillPromptContractTests(unittest.TestCase):
    def test_assay_extraction_prompts_require_rich_value_objects(self):
        single_prompt = render_skill_prompt_with_examples(
            "biocheminsight-text-models",
            "references/content_to_dict_prompt.md",
            "references/examples/content_to_dict_examples.md",
            {
                "ASSAY_NAME": "Calcitonin Receptor Assay 1 EC50",
                "MARKDOWN_TEXT": "| Example | Calcitonin Receptor Assay 1 EC50 |\n| Example 31 | 8.30 |",
                "COMPOUND_ID_LIST_BLOCK": '"Example 31"',
            },
        )
        multi_prompt = render_skill_prompt_with_examples(
            "biocheminsight-text-models",
            "references/content_to_multi_assay_dict_prompt.md",
            "references/examples/content_to_multi_assay_dict_examples.md",
            {
                "ASSAY_NAMES_JSON": '["Calcitonin Receptor Assay 1 EC50"]',
                "MARKDOWN_TEXT": "| Example | Calcitonin Receptor Assay 1 EC50 |\n| Example 31 | 8.30 |",
                "COMPOUND_ID_LIST_BLOCK": '"Example 31"',
            },
        )

        for prompt in (single_prompt, multi_prompt):
            self.assertIn('"value"', prompt)
            self.assertIn('"confidence"', prompt)
            self.assertIn('"reason"', prompt)
            self.assertIn("Example 31", prompt)
            self.assertIn("不得只输出", prompt)

    def test_alias_and_description_prompts_require_confidence_and_reason(self):
        alias_prompt = render_skill_prompt_with_examples(
            "biocheminsight-text-models",
            "references/resolve_compound_id_alias_prompt.md",
            "references/examples/resolve_compound_id_alias_examples.md",
            {
                "RAW_ID": "1",
                "OPTIONAL_CONTEXT": "",
                "ALLOWLIST": '["Example 1", "Compound 1"]',
            },
        )
        description_prompt = render_skill_prompt_with_examples(
            "biocheminsight-text-models",
            "references/get_compound_id_from_description_prompt.md",
            "references/examples/get_compound_id_from_description_examples.md",
            {"DESCRIPTION": "Answer: Compound 2"},
        )

        for prompt in (alias_prompt, description_prompt):
            self.assertIn("CONFIDENCE", prompt)
            self.assertIn("REASON", prompt)

    def test_visual_review_prompts_have_confidence_guards(self):
        identify_prompt = render_skill_reference(
            "biocheminsight-text-models",
            "references/identify_assay_visual_review_requests_prompt.md",
            {"OCR_CONTEXT": "ctx", "ASSAY_DICTS_JSON": "{}", "PARSED_TABLES_JSON": "[]"},
        )
        review_prompt = render_skill_reference(
            "biocheminsight-vision-models",
            "references/review_assay_values_prompt.md",
            {"ASSAY_DICTS_JSON": "{}", "REVIEW_PAYLOAD_JSON": "{}"},
        )
        reconcile_prompt = render_skill_reference(
            "biocheminsight-text-models",
            "references/reconcile_assay_values_with_visual_report_prompt.md",
            {"OCR_CONTEXT": "ctx", "ASSAY_DICTS_JSON": "{}", "VISUAL_REPORT_JSON": "{}"},
        )

        self.assertIn("confidence", identify_prompt)
        self.assertIn("confidence", review_prompt)
        self.assertIn('confidence="high"', reconcile_prompt)

    def test_structure_classification_prompts_include_confidence(self):
        classify_prompt = render_skill_prompt_with_examples(
            "biocheminsight-vision-models",
            "references/classify_structure_candidate_prompt.md",
            "references/examples/classify_structure_candidate_examples.md",
        )
        crop_prompt = render_skill_prompt_with_examples(
            "biocheminsight-vision-models",
            "references/classify_structure_crop_check_prompt.md",
            "references/examples/classify_structure_crop_check_examples.md",
            {"BORDER_SIDES": "left"},
        )
        border_prompt = render_skill_prompt_with_examples(
            "biocheminsight-vision-models",
            "references/classify_structure_border_review_prompt.md",
            "references/examples/classify_structure_border_review_examples.md",
            {"BORDER_SIDES": "left"},
        )

        for prompt in (classify_prompt, crop_prompt, border_prompt):
            self.assertIn('"confidence"', prompt)
            self.assertIn("high|medium|low", prompt)

    def test_text_schema_documents_rich_assay_objects(self):
        schema = load_skill_json("biocheminsight-text-models", "references/output_schemas.json")
        self.assertEqual(
            schema["content_to_dict"]["additionalProperties"]["required_keys"],
            ["value", "confidence", "reason"],
        )
        self.assertEqual(
            schema["content_to_multi_assay_dict"]["additionalProperties"]["additionalProperties"]["required_keys"],
            ["value", "confidence", "reason"],
        )


if __name__ == "__main__":
    unittest.main()
