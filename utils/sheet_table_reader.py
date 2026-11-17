"""Sheet-level reading through the vision model (serial, checkpointed).

LEGACY PATH for dense chemical sheets: on WO2025137159 EXAMPLE sheets
every per-compound transcription call timed out (90s budget, both
retries), which the vision contract already refuses. Dense sheets
now route through VLBlockReader (utils/vl_block_reader.py,
PaddleOCR-VL-1.6 block crops). This module remains for short
targeted queries that fit inside the vision timeout.

Self-hosted model servers are a shared, finite resource: this reader is
strictly serial (one request at a time), patient (probe-checked start, per-call
retry-then-skip), and resumable (a JSONL checkpoint holds every completed
sheet; reruns skip completed work). No caller may raise the concurrency —
parallel campaigns against the shared server wedge it for everyone.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from utils.llm_utils import call_visual_model

INVENTORY_PROMPT = """This patent sheet shows a compound table. List ONLY \
the compound ID numbers printed in the table's ID column (any format, e.g. \
14 or C14), in order. Output JSON only: {"compounds": ["14", "15"], \
"count": 2}. If no compound table is visible, output \
{"compounds": [], "count": 0}."""

COMPOUND_PROMPT = """This patent sheet shows a compound table. Transcribe \
compound number {cid} completely. It has three stacked rows: (1) attributes \
— Monomer/Dimer marker with MW in Daltons, cyclization positions/bond/\
linker, dimerization info, stability assays, IC50 grades; (2) the peptide \
line notation ending with an annotation like '(SEQ ID NO:18)'; (3) the \
printed SMILES. Output ONE JSON object:
{{"compound_id": "{cid}", "form": "<M or D>", "mw": <number or null>,
 "cyclization_positions": "<verbatim or empty>",
 "cyclization_bond": "<verbatim or empty>",
 "cyclization_linker": "<verbatim or empty>",
 "dimerization": "<verbatim or empty>",
 "stability": "<verbatim or empty>", "ic50": "<verbatim or empty>",
 "notation": "<the line notation exactly as printed, with its \
(SEQ ID NO:N) annotation>",
 "smiles": "<printed SMILES verbatim, or empty if unreadable>",
 "note": "<empty or what was unreadable>"}}
Transcribe verbatim; do not fix apparent OCR issues; unreadable fields are \
empty strings. Output the JSON object only."""

_PROBE_PROMPT = "How many table rows are on this page? One integer."
_JSON_RE = re.compile(r"[\[{].*[\]}]", re.S)


def _extract_json(reply: str):
    m = _JSON_RE.search(reply or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class SheetTableReader:
    """Serial per-sheet reading with a crash-safe checkpoint."""

    def __init__(self, checkpoint_path: str, retry_limit: int = 2,
                 probe_max_wait_s: int = 21600):
        self.checkpoint_path = Path(checkpoint_path)
        self.retry_limit = retry_limit
        self.probe_max_wait_s = probe_max_wait_s
        self.done: dict[str, list[dict]] = {}
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        for line in self.checkpoint_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.done[str(rec.get("sheet_id"))] = rec.get("rows", [])

    def _append_checkpoint(self, sheet_id: str, rows: list[dict]) -> None:
        with self.checkpoint_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"sheet_id": sheet_id, "rows": rows},
                                ensure_ascii=False) + "\n")

    def probe_ready(self, probe_png: str) -> bool:
        """Block until the vision server answers one tiny call, or the
        wait budget runs out. The campaign starts only on a live server."""
        deadline = time.time() + self.probe_max_wait_s
        while time.time() < deadline:
            try:
                reply = call_visual_model(probe_png, _PROBE_PROMPT)
                if reply and reply.strip():
                    return True
            except Exception:
                pass
            print(f"  vision server busy; retrying probe in 120s "
                  f"({int(deadline - time.time())}s left)", flush=True)
            time.sleep(120)
        return False

    def _call(self, png: str, prompt: str):
        last = None
        for attempt in range(self.retry_limit):
            try:
                return call_visual_model(png, prompt)
            except Exception as exc:
                last = f"{type(exc).__name__}: {str(exc)[:100]}"
                time.sleep(20 * (attempt + 1))     # polite backoff, serial
        raise RuntimeError(last or "vision call failed")

    def read_sheet(self, png: str, sheet_id: str) -> tuple[list[dict],
                                                            list[str], bool]:
        """Inventory + per-compound targeted transcription. Returns
        (records, gaps, retryable): a compound that fails twice is named
        and skipped; an unreadable inventory or a sheet whose compounds
        all gapped is flagged retryable so callers do not checkpoint it
        as done — a degraded-server empty read is not evidence of an
        empty sheet."""
        gaps: list[str] = []
        inv = _extract_json(self._call(png, INVENTORY_PROMPT)) or {}
        cids = [str(c) for c in inv.get("compounds", []) if str(c).strip()]
        if not cids:
            return [], [f"{sheet_id}: inventory unreadable"], True
        records: list[dict] = []
        for cid in cids:
            try:
                reply = self._call(png, COMPOUND_PROMPT.format(cid=cid))
            except RuntimeError as exc:
                gaps.append(f"{sheet_id}/C{cid}: {exc}")
                continue
            obj = _extract_json(reply)
            if not obj or not obj.get("notation"):
                gaps.append(f"{sheet_id}/C{cid}: transcription empty or "
                            "without notation")
                continue
            obj["compound_id"] = cid
            records.append(obj)
        retryable = bool(gaps) and not records
        return records, gaps, retryable


def run_serial(pages: list[int], png_of, checkpoint_path: str,
               probe_png: str, out_path: str | None = None) -> dict:
    """Serial campaign over `pages`. Resumable: completed sheets are
    skipped from the checkpoint."""
    reader = SheetTableReader(checkpoint_path)
    pending = [p for p in pages if str(p) not in reader.done]
    print(f"sheets: {len(pages)} total, {len(pending)} pending, "
          f"{len(reader.done)} done", flush=True)
    if pending and not reader.probe_ready(probe_png):
        print("vision server did not recover in budget; aborting run",
              flush=True)
        return {"started": False}
    for page in pending:
        png = png_of(page)
        t0 = time.time()
        records, gaps, retryable = reader.read_sheet(png, str(page))
        if retryable:
            print(f"sheet {page}: retryable ({len(gaps)} gaps); not "
                  "checkpointed", flush=True)
            continue
        reader._append_checkpoint(str(page), records)
        for gap in gaps:
            print(f"  gap: {gap}", flush=True)
        print(f"sheet {page}: {len(records)} compounds "
              f"({time.time() - t0:.0f}s)", flush=True)
    if out_path:
        Path(out_path).write_text(json.dumps(
            {"sheets": reader.done}, ensure_ascii=False, indent=1),
            encoding="utf-8")
    return {"sheets": len(reader.done)}
