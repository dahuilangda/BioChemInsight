import fcntl
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from frontend.backend.pdf_search import (
    SearchBusy,
    _ocr_items,
    _rect,
    page_text,
    pages_text,
)


class PdfSearchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "document" / "source.pdf"
        self.path.parent.mkdir()

    def tearDown(self):
        self.directory.cleanup()

    def document(self, scanned=False, rotation=0):
        with fitz.open() as document:
            page = document.new_page(width=400, height=600)
            page.insert_text((60, 100), "Alpha beta gamma", fontsize=16)
            if scanned:
                image = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                page = document.new_page(width=400, height=600)
                page.insert_image(page.rect, stream=image.tobytes("png"))
                document.delete_page(0)
                page = document[0]
            page.set_rotation(rotation)
            document.save(self.path)

    def test_text_and_rotation(self):
        self.document(rotation=90)
        with patch("frontend.backend.pdf_search.request_pdf_to_markdown") as request:
            result = page_text(self.path, 1, False, "", "auto")
            self.assertFalse(result["needs_ocr"])
            self.assertEqual(
                [item["text"] for item in result["items"]], ["Alpha", "beta", "gamma"]
            )
            for item in result["items"]:
                rect = item["rect"]
                self.assertTrue(0 <= rect["x"] < rect["x"] + rect["w"] <= 1)
                self.assertTrue(0 <= rect["y"] < rect["y"] + rect["h"] <= 1)
            request.assert_not_called()

    def test_scan_cache_and_page_upload(self):
        self.document(scanned=True)
        payload = {
            "raw_predictions": [
                {
                    "width": 800,
                    "height": 1200,
                    "parsing_res_list": [
                        {
                            "block_content": "Alpha beta",
                            "block_bbox": [120, 160, 300, 210],
                        },
                    ],
                }
            ]
        }
        self.assertTrue(page_text(self.path, 1, False, "service", "auto")["needs_ocr"])

        def recognize(path, *args, **kwargs):
            with fitz.open(path) as document:
                self.assertEqual(len(document), 1)
            return payload

        with patch(
            "frontend.backend.pdf_search.request_pdf_to_markdown", side_effect=recognize
        ) as request:
            first = page_text(self.path, 1, True, "service", "auto")
            second = page_text(self.path, 1, False, "service", "auto")
            self.assertEqual(first, second)
            self.assertEqual(first["source"], "ocr")
            self.assertEqual(request.call_count, 1)
            self.assertEqual(first["items"][0]["rect"]["x"], 0.15)
        self.assertEqual(len(list(self.path.parent.rglob("*.json"))), 1)

    def test_failed_ocr_is_not_cached(self):
        self.document(scanned=True)
        with (
            patch(
                "frontend.backend.pdf_search.request_pdf_to_markdown",
                side_effect=TimeoutError,
            ),
            self.assertRaises(TimeoutError),
        ):
            page_text(self.path, 1, True, "service", "auto")
        self.assertEqual(list(self.path.parent.rglob("*.json")), [])

    def test_missing_coordinates_are_not_accepted(self):
        with self.assertRaises(ValueError):
            _ocr_items({"page_markdowns": ["Recognized text"]})
        with self.assertRaises(ValueError):
            _ocr_items(
                {
                    "page_markdowns": ["Recognized text"],
                    "raw_predictions": [{"width": 10, "height": 10}],
                }
            )

    def test_cache_recovers_from_invalid_json(self):
        self.document()
        first = page_text(self.path, 1, False, "", "auto")
        cache = next(self.path.parent.rglob("*.json"))
        cache.write_text("{")
        self.assertEqual(first, page_text(self.path, 1, False, "", "auto"))
        cache.write_text('{"items": [], "source": [], "needs_ocr": false}')
        self.assertEqual(first, page_text(self.path, 1, False, "", "auto"))

    def test_batch_keeps_successful_pages(self):
        self.document()
        with self.assertLogs("frontend.backend.pdf_search", level="ERROR"):
            result = pages_text(self.path, [1, 2], "", "auto")
        self.assertEqual([page["page"] for page in result["pages"]], [1])
        self.assertEqual(result["failed"], [2])

    def test_ocr_lock_limits_concurrency(self):
        self.document(scanned=True)
        with (self.path.parent.parent / "search-ocr.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch(
                "frontend.backend.pdf_search.request_pdf_to_markdown"
            ) as request:
                with self.assertRaises(SearchBusy):
                    page_text(self.path, 1, True, "service", "auto")
                request.assert_not_called()

    def test_nonfinite_coordinates(self):
        with self.assertRaises(ValueError):
            _rect([0, 0, float("nan"), 20], 100, 100)

    def test_invalid_page(self):
        self.document()
        with self.assertRaises(IndexError):
            page_text(self.path, 2, False, "", "auto")


if __name__ == "__main__":
    unittest.main()
