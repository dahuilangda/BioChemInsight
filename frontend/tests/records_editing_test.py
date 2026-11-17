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
        try:
            from frontend.backend import main
        finally:
            sys.stderr = old
        self.main = main
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.path = self.root / "structures.csv"
        self.path.write_text("COMPOUND_ID,SMILES\nsubject,CCO\n")
        self.task = main.Task(
            id="editing-test",
            type="structure_extraction",
            status="completed",
            result_path=str(self.path),
            data=[{"COMPOUND_ID": "subject", "SMILES": "CCO"}],
            params={},
        )
        task = self.task

        class Manager:
            def get(self, key):
                return task if key == task.id else None

            def update(self, key, **fields):
                for name, value in fields.items():
                    setattr(task, name, value)
                return task

        self.manager = patch.object(main, "task_manager", Manager())
        self.manager.start()
        self.addCleanup(self.manager.stop)

    def save(self, rows, revision="0"):
        return asyncio.run(
            self.main.update_task_structures(
                self.task.id,
                self.main.UpdateStructuresRequest(records=rows, base_revision=revision),
            )
        )

    def test_row_identity_survives_reload_identifier_edits_and_reordering(self):
        raw = [
            {"COMPOUND_ID": "same", "SMILES": "CCO"},
            {"COMPOUND_ID": "same", "SMILES": "CCO"},
        ]
        first = self.main._normalize_records(raw, self.root, with_record_ids=True)
        repeated = self.main._normalize_records(raw, self.root, with_record_ids=True)
        self.assertEqual(
            [row["RECORD_ID"] for row in first], [row["RECORD_ID"] for row in repeated]
        )
        self.assertNotEqual(first[0]["RECORD_ID"], first[1]["RECORD_ID"])
        first[0]["COMPOUND_ID"] = "corrected"
        moved = self.main._normalize_records(
            list(reversed(first)), self.root, with_record_ids=True
        )
        self.assertEqual(moved[1]["RECORD_ID"], first[0]["RECORD_ID"])

    def test_save_revision_conflict_and_restore(self):
        result = self.save(
            [{"COMPOUND_ID": "subject", "SMILES": "CCO", "Endpoint (nM)": "1 ± .2"}]
        )
        self.assertTrue(result.task.params["records_revision"])
        self.assertEqual(
            len(self.main.structure_revision_history(self.task.id)["revisions"]), 1
        )
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as failure:
            self.save([])
        self.assertEqual(failure.exception.status_code, 409)
        restored = asyncio.run(
            self.main.undo_last_structure_edit(
                self.task.id, {"base_revision": result.task.params["records_revision"]}
            )
        )
        self.assertEqual(restored.records[0]["COMPOUND_ID"], "subject")
        self.assertNotIn("Endpoint (nM)", restored.records[0])

    def test_audit_restore_is_new_event_and_conflicts_do_not_write(self):
        first = self.save(
            [{"COMPOUND_ID": "subject", "SMILES": "CCO", "Endpoint (nM)": "1"}]
        )
        revision = first.task.params["records_revision"]
        rows = [
            dict(
                first.records[0],
                **{
                    "Endpoint (nM)": "9",
                    "USER_MEASUREMENT_EDITS": '{"Endpoint (nM)":"9"}',
                },
            )
        ]
        second = self.save(rows, revision)
        event = self.main.structure_revision_history(self.task.id)["revisions"][0]
        self.assertEqual(event["actor"]["kind"], "unknown")
        self.assertEqual(
            event["changes"][0]["record_id"], first.records[0]["RECORD_ID"]
        )
        restored = asyncio.run(
            self.main.restore_structure_revision(
                self.task.id,
                {
                    "revision": event["revision"],
                    "base_revision": second.task.params["records_revision"],
                    "scope": "field",
                    "record_id": first.records[0]["RECORD_ID"],
                    "field": "Endpoint (nM)",
                    "actor_session": "test-browser-session",
                },
            )
        )
        self.assertEqual(restored.records[0]["Endpoint (nM)"], "1")
        history = self.main.structure_revision_history(self.task.id)["revisions"]
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["restore_of"], event["revision"])
        self.assertEqual(history[0]["actor"]["kind"], "browser_session")
        from fastapi import HTTPException

        with self.assertRaises(HTTPException):
            asyncio.run(
                self.main.restore_structure_revision(
                    self.task.id,
                    {"revision": event["revision"], "base_revision": revision},
                )
            )
        self.assertEqual(
            len(self.main.structure_revision_history(self.task.id)["revisions"]), 3
        )

    def test_failed_persistence_rolls_back_export_and_does_not_publish_event(self):
        original = self.path.read_bytes()
        with patch.object(
            self.main.task_manager, "update", side_effect=RuntimeError("offline")
        ):
            with self.assertRaises(RuntimeError):
                self.save([{"COMPOUND_ID": "changed", "SMILES": "CCN"}])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(
            self.main.structure_revision_history(self.task.id)["revisions"], []
        )

    def test_duplicate_record_id_is_rejected(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as error:
            self.save(
                [
                    {"RECORD_ID": "same", "SMILES": "CCO"},
                    {"RECORD_ID": "same", "SMILES": "CCN"},
                ]
            )
        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(
            self.main.structure_revision_history(self.task.id)["revisions"], []
        )

    def test_unchanged_save_does_not_write_or_advance_revision(self):
        original = self.path.read_bytes()
        with patch.object(
            self.main.task_manager, "update", wraps=self.main.task_manager.update
        ) as update:
            result = self.save(
                [{"COMPOUND_ID": "subject", "SMILES": "CCO", "EMPTY": ""}]
            )
        update.assert_not_called()
        self.assertEqual(result.task.params.get("records_revision", "0"), "0")
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(
            self.main.structure_revision_history(self.task.id)["revisions"], []
        )

    def test_repeat_save_returns_the_same_revision(self):
        first = self.save(
            [{"COMPOUND_ID": "subject", "SMILES": "CCO", "Endpoint (nM)": "1"}]
        )
        revision = first.task.params["records_revision"]
        second = self.save(
            [{**first.records[0], "EMPTY": "", "USER_REMOVED_MEASUREMENTS": "[]"}],
            revision,
        )
        self.assertEqual(second.task.params["records_revision"], revision)
        self.assertEqual(
            len(self.main.structure_revision_history(self.task.id)["revisions"]), 1
        )

    def test_history_and_undo_skip_legacy_padding_events(self):
        first = self.save(
            [{"COMPOUND_ID": "subject", "SMILES": "CCO", "Endpoint (nM)": "1"}]
        )
        folder = self.root / "record_revisions"
        legacy = folder / "99991231-padding.json"
        legacy.write_text(
            json.dumps(
                {
                    "revision": "padding-only",
                    "records": first.records,
                    "after_records": [{**first.records[0], "EMPTY": ""}],
                    "changes": [
                        {
                            "record_id": first.records[0]["RECORD_ID"],
                            "operation": "update",
                            "fields": [{"field": "EMPTY", "before": None, "after": ""}],
                        }
                    ],
                }
            )
        )
        self.task.params["records_revision"] = "padding-only"
        history = self.main.structure_revision_history(self.task.id)["revisions"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["revision"], first.task.params["records_revision"])
        restored = asyncio.run(
            self.main.undo_last_structure_edit(
                self.task.id, {"base_revision": "padding-only"}
            )
        )
        self.assertNotIn("Endpoint (nM)", restored.records[0])
        self.assertTrue(legacy.exists())

    def test_saving_existing_observations_does_not_create_a_revision(self):
        import pandas as pd

        rows = [
            {
                "COMPOUND_ID": "subject",
                "SMILES": "CCO",
                "BIOACTIVITY_OBSERVATIONS": json.dumps(
                    [
                        {
                            "subject_id": "subject",
                            "column": "Endpoint (nM)",
                            "value": "7.5",
                        }
                    ]
                ),
            }
        ]
        pd.DataFrame(rows).to_csv(self.path, index=False)
        original = self.path.read_bytes()
        result = self.save(rows)
        self.assertEqual(result.task.params.get("records_revision", "0"), "0")
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(
            self.main.structure_revision_history(self.task.id)["revisions"], []
        )

    def test_corrected_values_and_removed_columns_survive_export(self):
        import pandas as pd

        rows = [
            {
                "COMPOUND_ID": "subject",
                "SMILES": "CCO",
                "USER_MEASUREMENT_EDITS": '{"Endpoint (nM)":""}',
                "USER_REMOVED_MEASUREMENTS": '["Removed"]',
            }
        ]
        path = self.main.merge_structure_activity_records(
            pd.DataFrame(rows),
            [{"COMPOUND_ID": "subject", "Endpoint (nM)": "100", "Removed": "200"}],
            self.root,
        )
        result = pd.read_csv(path, keep_default_na=False)
        self.assertEqual(result.iloc[0]["Endpoint (nM)"], "")
        self.assertNotIn("Removed", result.columns)

    def test_ambiguous_id_retains_unlinked_values(self):
        import pandas as pd

        rows = [
            {"COMPOUND_ID": "subject", "SMILES": "CCO"},
            {"COMPOUND_ID": "subject", "SMILES": "CCN"},
        ]
        path = self.main.merge_structure_activity_records(
            pd.DataFrame(rows),
            [{"COMPOUND_ID": "subject", "Endpoint": "100"}],
            self.root,
        )
        result = pd.read_csv(path, keep_default_na=False)
        self.assertEqual(len(result), 3)
        self.assertEqual(
            result.iloc[-1]["MEASUREMENT_LINK_STATUS"], "ambiguous_identity"
        )
        self.assertEqual(result.iloc[0]["Endpoint"], "")

    def test_scientific_column_preserves_endpoint_unit_and_provenance(self):
        records, metadata = self.main._records_from_assay_results(
            {
                "generic label": {
                    "subject": {
                        "endpoint": "EC50",
                        "target": "Receptor",
                        "unit": "uM",
                        "source_kind": "predicted",
                        "value": "< 2 ± .1",
                    }
                }
            }
        )
        self.assertEqual(records[0]["Receptor · EC50 (µM) · predicted"], "< 2 ± .1")
        self.assertEqual(
            records[0]["Receptor · EC50 (µM) · predicted__source_kind"], "predicted"
        )


if __name__ == "__main__":
    unittest.main()
