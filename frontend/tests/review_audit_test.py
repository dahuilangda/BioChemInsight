"""Audit and selective recovery must not silently discard newer unrelated edits."""

import copy
import tempfile
import json
import unittest
from pathlib import Path
from utils.review_audit import changes, restore_records, read_events


class ReviewAuditTest(unittest.TestCase):
    def setUp(self):
        self.before = [
            {"RECORD_ID": "a", "COMPOUND_ID": "same", "IC50 (µM)": "1"},
            {"RECORD_ID": "b", "COMPOUND_ID": "same", "SMILES": "CCO"},
        ]
        self.after = copy.deepcopy(self.before)
        self.after[0]["IC50 (µM)"] = "9"
        self.event = {"records": self.before, "after_records": self.after}

    def test_duplicate_names_use_stable_ids_and_fields(self):
        diff = changes(self.before, self.after)
        self.assertEqual(len(diff), 1)
        self.assertEqual(diff[0]["record_id"], "a")
        self.assertEqual(
            diff[0]["fields"], [{"field": "IC50 (µM)", "before": "1", "after": "9"}]
        )

    def test_field_restore_preserves_other_edits_and_measurement_override(self):
        current = copy.deepcopy(self.after)
        current[0]["COMPOUND_ID"] = "new"
        current[0]["USER_MEASUREMENT_EDITS"] = '{"IC50 (µM)":"9"}'
        result = restore_records(
            current, self.event, scope="field", record_id="a", field="IC50 (µM)"
        )
        self.assertEqual(result[0]["COMPOUND_ID"], "new")
        self.assertEqual(
            json.loads(result[0]["USER_MEASUREMENT_EDITS"])["IC50 (µM)"], "1"
        )
        self.assertEqual(current[0]["IC50 (µM)"], "9")

    def test_deleted_record_recovery_and_version_after(self):
        event = {"records": self.before, "after_records": self.before[1:]}
        result = restore_records(
            [dict(self.before[1], COMPOUND_ID="changed")],
            event,
            scope="record",
            record_id="a",
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]["COMPOUND_ID"], "changed")
        self.assertEqual(restore_records([], self.event, side="after"), self.after)

    def test_rejects_partial_structure_restore(self):
        with self.assertRaises(ValueError):
            restore_records(
                self.after, self.event, scope="field", record_id="b", field="SMILES"
            )

    def test_legacy_events_infer_after_and_do_not_invent_author(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "1.json").write_text(
                json.dumps({"revision": "one", "records": self.before})
            )
            Path(directory, "2.json").write_text(
                json.dumps({"revision": "two", "records": self.after})
            )
            events = read_events(directory, [])
        self.assertEqual(events[0]["after_records"], self.after)
        self.assertEqual(events[0]["actor"]["kind"], "unknown")
        self.assertEqual(events[1]["changes"][0]["operation"], "delete")

    def test_row_order_is_not_a_content_edit(self):
        self.assertEqual(changes(self.before, list(reversed(self.before))), [])

    def test_csv_padding_and_scalar_serialization_are_not_edits(self):
        before = [
            {
                "RECORD_ID": "a",
                "COMPOUND_ID": "0012",
                "PAGE_NUM": 2,
                "Endpoint": "1",
                "EMPTY": None,
            }
        ]
        after = [
            {
                "RECORD_ID": "a",
                "COMPOUND_ID": "0012",
                "PAGE_NUM": "2",
                "Endpoint": "1",
                "EMPTY": "",
                "PADDED": "",
            }
        ]
        self.assertEqual(changes(before, after), [])
        after[0]["COMPOUND_ID"] = "12"
        self.assertEqual(
            [field["field"] for field in changes(before, after)[0]["fields"]],
            ["COMPOUND_ID"],
        )

    def test_json_format_and_empty_containers_are_not_edits(self):
        before = [
            {
                "RECORD_ID": "a",
                "USER_MEASUREMENT_EDITS": '{"A":"1","B":"2"}',
                "USER_LOCKED_FIELDS": '["A","B"]',
                "USER_REMOVED_MEASUREMENTS": "[]",
            }
        ]
        after = [
            {
                "RECORD_ID": "a",
                "USER_MEASUREMENT_EDITS": '{ "B": "2", "A": "1" }',
                "USER_LOCKED_FIELDS": '["B","A","A"]',
                "USER_REMOVED_MEASUREMENTS": None,
            }
        ]
        self.assertEqual(changes(before, after), [])
        self.assertEqual(
            changes(
                [{"RECORD_ID": "a", "data": {}}], [{"RECORD_ID": "a", "data": "{}"}]
            ),
            [],
        )

    def test_only_the_changed_record_and_field_are_reported(self):
        after = copy.deepcopy(self.before)
        for row in after:
            row["PADDED"] = ""
            row["MEASUREMENT_REVIEWED_AT"] = "2026-11-16T10:00:00Z"
        after[1]["SMILES"] = "CCN"
        change = changes(self.before, after)
        self.assertEqual(len(change), 1)
        self.assertEqual(change[0]["record_id"], "b")
        self.assertEqual(
            change[0]["fields"], [{"field": "SMILES", "before": "CCO", "after": "CCN"}]
        )

    def test_add_delete_update_are_distinct_and_blank_is_a_real_change(self):
        after = [
            dict(self.before[0], **{"IC50 (µM)": ""}),
            {"RECORD_ID": "c", "COMPOUND_ID": "new"},
        ]
        diff = changes(self.before, after)
        self.assertEqual(
            {c["record_id"]: c["operation"] for c in diff},
            {"a": "update", "b": "delete", "c": "add"},
        )
        self.assertEqual(diff[0]["fields"][0]["after"], "")

    def test_molfile_header_changes_do_not_hide_coordinate_changes(self):
        before = [
            {"RECORD_ID": "a", "MOLBLOCK": "title\nprogram\ncomment\n0 1 2\nM  END\n"}
        ]
        after = [
            {
                "RECORD_ID": "a",
                "MOLBLOCK": "new title\nnew program\ncomment\n0 1 2\nM  END\n",
            }
        ]
        self.assertEqual(changes(before, after), [])
        after[0]["MOLBLOCK"] = after[0]["MOLBLOCK"].replace("0 1 2", "0 1 3")
        self.assertEqual(changes(before, after)[0]["fields"][0]["field"], "MOLBLOCK")

    def test_flattening_existing_measurements_is_not_a_user_edit(self):
        observations = json.dumps(
            [{"subject_id": "subject", "column": "Endpoint (nM)", "value": "7.5"}]
        )
        before = [
            {
                "RECORD_ID": "a",
                "COMPOUND_ID": "subject",
                "BIOACTIVITY_OBSERVATIONS": observations,
            }
        ]
        self.assertEqual(changes(before, [{**before[0], "Endpoint (nM)": "7.5"}]), [])
