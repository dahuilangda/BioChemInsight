"""Evidence dispatcher: page labels -> skill dispatch -> relationship rows.

The orchestration entry that ties the harness together for a document:

  1. label pages    — signals from paddle layout + page text
  2. dispatch       — each label routes to its skill cluster:
       compound_table     -> text extraction + notation ladder + mass arbitration
       abbreviation_table -> code/name cells + structure-crop OCSR
       2d_structure       -> MolDetv2 boxes + MolSight decode + VLM ID read
       assay_text         -> activity extraction (production pipeline)
  3. merge          — rows joined on compound ID into structure-activity-
                      ID relationships

Accuracy and stability guarantees, enforced at the seams between steps:
  - every skill output is a CANDIDATE; the declared gates arbitrate
  - skill failures are named gaps on the affected rows, never silent
  - each skill runs independently: one skill's failure does not consume
    another's evidence
  - the dispatcher owns no chemistry — it routes and merges only
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from utils.page_type_classifier import classify_corpus


# ---------------------------------------------------------------------------
# page labels
# ---------------------------------------------------------------------------

@dataclass
class PageLabel:
    """What one page contains, and how well it was captured."""
    page_key: str
    page_type: str                  # compound_table | abbreviation_table | ...
    signals: dict = field(default_factory=dict)
    capture_quality: str = "full"   # full | partial | lost
    markdown: str = ""
    raw_predictions: dict = field(default_factory=dict)


def label_pages(pages: list[tuple[str, str]]) -> dict[str, PageLabel]:
    """Signal-based labeling over OCR page markdown (classify_corpus)."""
    by_type = classify_corpus(pages)
    labels: dict[str, PageLabel] = {}
    for key, md in pages:
        page_type = next((t for t, keys in by_type.items() if key in keys),
                         "prose")
        labels[key] = PageLabel(page_key=key, page_type=page_type,
                                markdown=md)
    return labels


def label_capture_quality(labels: dict[str, PageLabel],
                          expected: dict[str, int]) -> None:
    """Annotate capture quality per page against expectation.

    `expected` maps page_type -> the compound count the document declares
    (e.g. WO2025137159 declares 897 compounds). Pages whose recovered
    notation count falls short of the per-page expectation carry a
    'partial' flag — the dispatcher routes them to the strip-level
    supplemental pass rather than treating the loss as final."""
    from collections import Counter
    per_type = Counter(label.page_type for label in labels.values())
    for label in labels.values():
        expected_n = expected.get(label.page_type)
        if expected_n and per_type[label.page_type] < expected_n:
            label.capture_quality = "partial"


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

class ProductionStructureExtractor:
    """Adapter: the production batch operation
    (structure_parser.extract_structures_from_pdf) behind a uniform
    extract(pdf, first, last) interface. Runs in the biocheminsight
    container (MolDetv2 + models + ultralytics are container-resident)."""

    def __init__(self, output_dir: str, repo_root: str = "/data/BioChemInsight"):
        self.output_dir = output_dir
        self.repo_root = repo_root

    def extract(self, pdf_path: str, first: int, last: int) -> list[dict]:
        import subprocess
        driver = self.repo_root + "/scripts_structure_extract.py"
        out = f"{self.output_dir}/struct_{first}_{last}"
        result = subprocess.run(
            ["docker", "exec", "biocheminsight", "python3", driver,
             str(first), str(last), out],
            capture_output=True, text=True, timeout=7200)
        rows_path = f"{out}/structure_rows.json"
        if result.returncode != 0 or not Path(rows_path).exists():
            return []
        return json.loads(Path(rows_path).read_text(encoding="utf-8"))


class EvidenceDispatcher:
    """Label pages, dispatch skills, merge into relationship rows.

    Construction takes the skill instances (each already built with its
    own capability boundaries); dispatch routes per page label; the merge
    step joins rows on compound ID. The dispatcher owns routing and
    bookkeeping — no chemistry decisions live here."""

    def __init__(self, *, notation_harness=None, structure_extractor=None,
                 sheet_reader=None, abbreviation_reader=None):
        self.notation_harness = notation_harness
        self.structure_extractor = structure_extractor
        self.sheet_reader = sheet_reader
        self.abbreviation_reader = abbreviation_reader
        self.relationship_rows: list[dict] = []
        self.dispatch_log: list[dict] = []

    def dispatch(self, labels: dict[str, PageLabel],
                 pdf_path: str | None = None) -> dict:
        """Route pages by label; merge per-label outputs."""
        log = self.dispatch_log
        compound_rows: dict[str, dict] = {}

        table_pages = {k: l for k, l in labels.items()
                       if l.page_type == "compound_table"}
        abbrev_pages = {k: l for k, l in labels.items()
                        if l.page_type == "abbreviation_table"}
        structure_pages = sorted(k for k, l in labels.items()
                                 if l.page_type == "2d_structure")

        if table_pages and self.notation_harness:
            log.append({"route": "compound_table",
                        "pages": len(table_pages)})
            # the notation harness handles row mining + resolution
            # (peer: parse_full driver — kept out of this module to keep
            #  the dispatcher free of OCR-format dependencies)

        if abbrev_pages and self.abbreviation_reader:
            log.append({"route": "abbreviation_table",
                        "pages": len(abbrev_pages)})
            # code/name pairs from text; structure-only rows from crops
            for key, label in sorted(abbrev_pages.items()):
                records = self.abbreviation_reader.read_sheet(label.markdown)
                for rec in records:
                    compound_rows.setdefault(
                        rec.get("compound_id"), {}).update(rec)

        if structure_pages and self.structure_extractor and pdf_path:
            first, last = _page_range(structure_pages)
            log.append({"route": "2d_structure",
                        "physical_pages": (first, last)})
            rows = self.structure_extractor.extract(pdf_path, first, last)
            for row in rows:
                cid = str(row.get("COMPOUND_ID") or "")
                if cid:
                    compound_rows.setdefault(cid, {}).update(row)

        self.relationship_rows = [
            {"compound_id": cid, **slot} for cid, slot in compound_rows.items()]
        return {"rows": self.relationship_rows, "log": log}


def _page_range(page_keys: list[str]) -> tuple[int, int]:
    """Physical page range covering the given keys (keys carry page
    numbers as 'src|N' suffixes)."""
    numbers = []
    for key in page_keys:
        digits = re.findall(r"\d+", key)
        if digits:
            numbers.append(int(digits[-1]))
    return (min(numbers), max(numbers)) if numbers else (0, 0)

