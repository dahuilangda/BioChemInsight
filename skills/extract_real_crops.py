"""Extract verified structure crops from real patent PDFs for SFT pairing.

Pipeline: render (PyMuPDF) -> MolDetV2 box detection (harness :8011) ->
crops -> label-band OCR (PaddleOCR-VL :8010, raw mode) -> MolSight decode ->
sequence-pairing verification against the gold table -> manifest.

Verification gate — sequence pairing is the gate:
  verified : the crop's SEQ ID label pairs with a gold row that has a gold
             SMILES (verify_method="sequence"); `smiles` carries the gold
             SMILES as the training target.
  rejected : image-quality exclusions (truncated/fragment/blurry boxes),
             kept with a named reason.
  unpaired : everything else — no SEQ label, or the labeled compound has
             no gold SMILES yet (kept for future decode retests).
  The bare decode is advisory only: its mass delta against the declared
  TABLE 1 value is recorded as `formula_matched` evidence and never vetoes
  a pair (MolSight is not yet tuned to these dense macrocycles).

When the first decode disagrees with the declared mass, a padded variant is
decoded before giving up: tight detector boxes clip substituent labels, and
the declared mass arbitrates between padding variants of the same drawing.

Every failure is named (render/detect/ocr/decode/identity) in the manifest
"errors" list; nothing fails silently.

Resume contract: per-patent rows.jsonl stores one line per crop (plus
zero-box page markers) tagged with the (dpi, conf) it was produced under;
re-runs skip stored pages and decode only rows that still lack SMILES.

Usage:
    python skills/extract_real_crops.py \
        --patent /data/wo_patents/US20240287138A1.pdf \
        --patent /data/wo_patents/WO2024163691A1.pdf \
        --out /data/BioChemInsight/data/real_crops
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MASS_TOLERANCE_DA = 0.8   # formula_matched evidence threshold (advisory)
BAND_POLL_TIMEOUT_S = 150.0
OCR_SUBMIT_TIMEOUT_S = 30
DETECT_TIMEOUT_S = 120
CROP_PAD_PX = 10           # whitespace margin saved around the detected box
DECODE_RETRY_PAD_PX = 16   # extra margin for the padded decode variant
RIGHT_BAND_WIDTH_PX = 440  # narrow side bands truncate the layout block and
                           # lose trailing digits ("SEQ ID NO. 51" -> "SEQ ID NO")
LEFT_BAND_WIDTH_PX = 440   # GPC3 table rows print the SEQ ID / compound
                           # number in the cell left of the structure

SEQ_PATTERN = re.compile(r"SEQ\s*ID\s*NO[\s.:#]*([0-9]{1,4})", re.IGNORECASE)
FIG_PATTERN = re.compile(r"\bFIG[\s.]*([0-9]{1,3})", re.IGNORECASE)
COMPOUND_PATTERN = re.compile(
    r"\b(?:Compound|Example|Formula)\s+([A-Z0-9IVX-]{1,8})", re.IGNORECASE)
PARTIAL_SEQ_HINT = re.compile(r"SEQ|S\.?E\.?Q|NO[\s.]*[0-9]", re.IGNORECASE)


# --------------------------------------------------------------------------
# named error ledger
# --------------------------------------------------------------------------

class ErrorLedger:
    """Collects per-stage failures so a batch over 600+ pages never dies on
    one bad page; the manifest carries the full list for triage."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, stage: str, error: str, **context) -> None:
        self.items.append({"stage": stage, "error": error[:300], **context})


# --------------------------------------------------------------------------
# stages: render / detect / band OCR
# --------------------------------------------------------------------------

def render_page(document, page_index: int, dpi: int):
    """PDF page -> PIL RGB image; raises ValueError with a named cause."""
    from PIL import Image
    try:
        pixmap = document[page_index].get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height),
                               pixmap.samples)
    except Exception as exc:  # noqa: BLE001 - surfaced into the ledger
        raise ValueError(f"pymupdf_render_failed: {exc}") from exc


def png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def detect_structure_boxes(image_bytes: bytes, detect_url: str,
                           confidence: float) -> list[dict]:
    """MolDetV2 box detection through the harness service."""
    response = requests.post(
        f"{detect_url}/v1/structures/detect",
        params={"conf": confidence},
        files={"file": ("page.png", image_bytes, "image/png")},
        timeout=DETECT_TIMEOUT_S)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise ValueError(f"detect_service_error: {payload['error']}")
    return payload.get("structures", [])


def ink_fraction(image) -> float:
    histogram = image.convert("L").histogram()
    dark_pixels = sum(histogram[:200])
    return dark_pixels / max(1, image.size[0] * image.size[1])


def label_band_regions(page_image, bbox: list[int]) -> list[tuple[str, "Image.Image"]]:
    """Label text bands around one detected box: the caption band below and
    the side bands left/right of the drawing. The US FIG family prints the
    SEQ ID to the right; the WO table families print the compound sequence
    to the right (KRAS) and the row label to the left (GPC3). Ink-free
    bands are dropped without spending OCR jobs; a bare side label is
    ~0.1% ink, so the threshold sits below that."""
    page_width, page_height = page_image.size
    x0, y0, x1, y1 = bbox
    box_height = max(1, y1 - y0)
    candidates = [
        ("below", (max(0, x0 - CROP_PAD_PX), y1,
                   min(page_width, x1 + CROP_PAD_PX),
                   min(page_height, y1 + max(50, int(0.18 * box_height))))),
        ("right", (min(page_width, x1 + CROP_PAD_PX),
                   max(0, y0 - int(0.25 * box_height)),
                   min(page_width, x1 + CROP_PAD_PX + RIGHT_BAND_WIDTH_PX),
                   min(page_height, y1 + int(0.25 * box_height)))),
        ("left", (max(0, x0 - CROP_PAD_PX - LEFT_BAND_WIDTH_PX),
                  max(0, y0 - int(0.25 * box_height)),
                  max(0, x0 - CROP_PAD_PX),
                  min(page_height, y1 + int(0.25 * box_height)))),
    ]
    bands = []
    for position, (bx0, by0, bx1, by1) in candidates:
        if bx1 - bx0 < 8 or by1 - by0 < 8:
            continue
        band = page_image.crop((bx0, by0, bx1, by1))
        if ink_fraction(band) < 0.0005:
            continue
        bands.append((position, band))
    return bands


def ocr_image(image, ocr_url: str) -> str:
    """PaddleOCR-VL text for one band. Uses raw predictions because the
    markdown assembler drops header/footnote blocks — exactly where SEQ ID
    side labels land. A white frame gives the layout detector margin;
    without it tight bands lose the trailing digits."""
    from PIL import Image as PILImage
    framed = PILImage.new("RGB", (image.width + 64, image.height + 64), "white")
    framed.paste(image, (32, 32))
    response = requests.post(
        f"{ocr_url}/v1/image-to-markdown",
        files={"file": ("band.png", png_bytes(framed), "image/png")},
        data={"return_raw": "true"},
        timeout=OCR_SUBMIT_TIMEOUT_S)
    response.raise_for_status()
    job_id = response.json()["job_id"]
    deadline = time.time() + BAND_POLL_TIMEOUT_S
    status = None
    while time.time() < deadline:
        status = requests.get(f"{ocr_url}/v1/jobs/{job_id}",
                              timeout=15).json().get("status")
        if status in ("completed", "failed"):
            break
        time.sleep(2.0)
    if status != "completed":
        raise ValueError(f"ocr_job_{status or 'timeout'}")
    result = requests.get(f"{ocr_url}/v1/jobs/{job_id}/result",
                          timeout=30).json()
    block_texts = []
    for prediction in result.get("raw_predictions") or []:
        for block in prediction.get("parsing_res_list") or []:
            text = (block.get("block_content") or "").strip()
            if text:
                block_texts.append(text)
    if not block_texts:  # fallback: assembled markdown (non-header blocks)
        block_texts = [markdown for markdown in
                       (result.get("page_markdowns") or []) if markdown.strip()]
    return "\n".join(block_texts)


def extract_id_candidates(text: str) -> dict:
    """SEQ ID / FIG / compound label candidates from one band's text."""
    return {
        "seq_ids": sorted({int(match) for match in SEQ_PATTERN.findall(text)}),
        "figs": sorted({int(match) for match in FIG_PATTERN.findall(text)}),
        "compounds": sorted({match.upper() for match in
                             COMPOUND_PATTERN.findall(text)}),
        "text": text[:400],
    }


def run_label_ocr(band_tasks: list[dict], ocr_url: str, threads: int,
                  ledger: ErrorLedger) -> None:
    """OCR the queued bands with a small thread pool.

    Retry policy: patent drawings print the SEQ label right beside the
    structure, but scans and layout variance truncate it ("SEQ ID NO." with
    the digits lost). When a band hints at SEQ text without yielding a
    number, a doubled-width side band at 2x scale is tried once — the wider
    context lets the layout detector keep the whole label as one block.
    """
    def ocr_band_task(task: dict) -> None:
        try:
            candidates = extract_id_candidates(ocr_image(task["image"], ocr_url))
            if not candidates["seq_ids"] and task["position"] == "right" \
                    and PARTIAL_SEQ_HINT.search(candidates["text"] or ""):
                # the widened band gives the layout detector room to keep the
                # whole label as one block; upscaling is unnecessary and makes
                # the job 10x slower on text-heavy pages
                widened = task["widen"]()
                if widened is not None:
                    candidates = extract_id_candidates(ocr_image(widened, ocr_url))
                    if candidates["seq_ids"]:
                        candidates["recovered_by_widened_retry"] = True
            task["ids"] = candidates
        except Exception as exc:  # noqa: BLE001 - named per band
            ledger.add("ocr", str(exc), patent=task["patent"],
                       page=task["page"], box=task["box_index"],
                       band=task["position"])
            task["ids"] = None
        finally:
            task["image"] = None  # release band memory promptly

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(ocr_band_task, band_tasks))


def merge_label_ids(crop_row: dict, tasks: list[dict]) -> None:
    """Fold band OCR results into one crop row (below-band SEQ hits win;
    the right band carries the SEQ side label in this family)."""
    seq_id = None
    figs: list[int] = []
    compounds: list[str] = []
    band_texts: dict[str, str] = {}
    for task in tasks:
        candidates = task.get("ids")
        if not candidates:
            continue
        band_texts[task["position"]] = candidates["text"]
        figs.extend(candidates["figs"])
        compounds.extend(candidates["compounds"])
        if seq_id is None and candidates["seq_ids"]:
            seq_id = candidates["seq_ids"][0]
    crop_row["seq_id"] = seq_id
    crop_row["fig"] = sorted(set(figs))[:4]
    crop_row["labels"] = sorted(set(compounds))[:4]
    crop_row["label_text"] = band_texts


# --------------------------------------------------------------------------
# table-evidence label matching (WO table families)
# --------------------------------------------------------------------------
# The US FIG sheets label every drawing "SEQ ID NO. n" beside the structure,
# which SEQ_PATTERN recovers directly. The WO table families instead embed
# structures in compound-table rows: KRAS prints the cyclo(...) sequence and
# the masses in the cells right of the drawing; GPC3 prints the row's SEQ ID
# to the left and the analytical m/z to the right. The evidence index lets
# band OCR text resolve the row without a SEQ-pattern hit: exact/fuzzy
# sequence-string match and exact mass-number match, both recorded as
# labeled methods (never silently folded into "SEQ ID NO").

_FLOAT_TOKEN = re.compile(r"\d{2,4}\.\d{1,4}")
_CYCLO_FINDER = re.compile(r"cyclo\s*\(")
TOKEN_ALIASES = {"cl": "C", "c1": "C", "c2": "C", "dc": "C", "dc1": "C",
                 "dcl": "C", "nai2": "Nal2", "nal2": "Nal2",
                 "nai27n": "Nal27N", "nal27n": "Nal27N",
                 "f4c00": "F4COO", "f4aa0": "F4aao", "f4aa": "F4aao"}


class TableEvidenceIndex:
    """seq_id lookup for band-OCR text: normalized sequence strings and
    printed mass values of the compound table."""

    def __init__(self, table_rows: dict):
        self.by_sequence: list[tuple[str, str]] = []   # (norm, seq_id)
        self.by_mass: dict[float, str] = {}            # value -> seq_id
        for sid, row in table_rows.items():
            seq = row.get("sequence") or row.get("compound_sequence")
            if seq:
                self.by_sequence.append((self._norm(seq), str(sid)))
            for key in ("observed_mass", "mz_measured", "exact_mass",
                        "measured_mass"):
                value = row.get(key)
                if isinstance(value, (int, float)) and value > 50:
                    self.by_mass.setdefault(round(float(value), 4), str(sid))

    @staticmethod
    def _norm(seq: str) -> str:
        s = re.sub(r"\s+", "", seq)
        s = re.sub(r"^cyclo\(|\)$", "", s).strip("-")
        return s

    @staticmethod
    def _tokens(seq: str) -> list[str]:
        out = []
        for tok in (t for t in seq.split("-") if t):
            fold = TOKEN_ALIASES.get(tok.lower())
            out.append(fold if fold else tok)
        return out

    @classmethod
    def _similar(cls, a: str, b: str) -> bool:
        ta, tb = cls._tokens(a), cls._tokens(b)
        if len(ta) != len(tb):
            return False
        for x, y in zip(ta, tb):
            if x == y:
                continue
            if len(x) == len(y) and sum(1 for p, q in zip(x, y) if p != q) <= 1:
                continue
            if abs(len(x) - len(y)) == 1 and (x.startswith(y) or y.startswith(x)
                                              or x.endswith(y) or y.endswith(x)):
                continue
            return False
        return True

    def match_text(self, text: str) -> tuple[str | None, str | None]:
        """(seq_id, method) for one band's text; None when nothing matches."""
        if not text:
            return None, None
        flat = re.sub(r"\s+", "", text)
        # 1. sequence evidence: every cyclo(...) span in the band
        for m in _CYCLO_FINDER.finditer(flat):
            depth = 0
            for i in range(m.end() - 1, len(flat)):
                if flat[i] == "(":
                    depth += 1
                elif flat[i] == ")":
                    depth -= 1
                    if depth == 0:
                        cand = self._norm(flat[m.start():i + 1])
                        break
            else:
                cand = self._norm(flat[m.start():])
            for norm, sid in self.by_sequence:
                if norm == cand or self._similar(norm, cand):
                    return sid, "band_sequence_match"
        # 2. bare sequence evidence without the cyclo wrapper
        stripped = flat.strip("c").strip()
        if "-" in stripped:
            for norm, sid in self.by_sequence:
                if norm == stripped or self._similar(norm, stripped):
                    return sid, "band_sequence_match"
        # 3. mass evidence: printed mass values (observed m/z / exact mass)
        hits: dict[str, str] = {}
        for token in _FLOAT_TOKEN.findall(text):
            value = float(token)
            for key in (value, round(value, 1), round(value, 2),
                        round(value, 3), round(value, 4)):
                sid = self.by_mass.get(round(float(key), 4))
                if sid is not None:
                    hits[sid] = f"band_mass_match:{token}"
        if len(hits) == 1:
            sid, method = next(iter(hits.items()))
            return sid, method
        return None, None


def attach_table_evidence(rows: list[dict], index: TableEvidenceIndex | None,
                          ) -> None:
    """Second-pass label association for crops whose bands yielded no
    SEQ-pattern id; records the evidence method per crop."""
    if index is None:
        return
    for row in rows:
        if row.get("seq_id") is not None:
            continue
        texts = row.get("label_text") or {}
        ordered = [texts.get(pos) for pos in ("right", "left", "below")]
        for text in ordered:
            sid, method = index.match_text(text or "")
            if sid is not None:
                row["seq_id"] = int(sid) if str(sid).lstrip("-").isdigit() \
                    else sid
                row["label_method"] = method
                break


# --------------------------------------------------------------------------
# MolSight decode + verification gate
# --------------------------------------------------------------------------

class StructureDecoder:
    """MolSight decode through the persistent host-side worker (the harness
    container has no GPU; the worker venv on the host does)."""

    def __init__(self, checkpoint: str, gpu: str):
        os.environ["MOLSIGHT_CHECKPOINT"] = checkpoint
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
        os.environ.setdefault("MOLSIGHT_DEVICE", "cuda:0")
        from utils.molsight_service import MolsightService
        self._worker = MolsightService()

    def decode_paths(self, image_paths: list[str]) -> list[dict]:
        return self._worker.predict_many(image_paths)


def smiles_metrics(smiles: str):
    """(mol, exact_mass, formula, canonical) or (None, ...) when invalid."""
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcMolFormula
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, None, None, None
    return (molecule, CalcExactMolWt(molecule), CalcMolFormula(molecule),
            Chem.MolToSmiles(molecule))


def tanimoto_similarity(smiles_a: str, smiles_b: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    mol_a, mol_b = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None
    fingerprint_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
    fingerprint_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fingerprint_a, fingerprint_b)


def collect_image_quality_flags(page_image, bbox: list[int]) -> list[str]:
    """Cheap visual-quality proxies for the gate: boxes clipped at the page
    edge truncate the drawing; extreme aspect ratios flag fragments or
    merged multi-structure rows."""
    page_width, page_height = page_image.size
    x0, y0, x1, y1 = bbox
    flags = []
    if x0 <= 2 or y0 <= 2 or x1 >= page_width - 2 or y1 >= page_height - 2:
        flags.append("box_touches_page_edge")
    box_w, box_h = max(1, x1 - x0), max(1, y1 - y0)
    if box_w / box_h > 8 or box_h / box_w > 8:
        flags.append("extreme_aspect_ratio")
    if box_w < 40 or box_h < 40:
        flags.append("degenerate_box")
    return flags


def verify_pair(crop_row: dict, table_rows: dict, gold_smiles: dict | None) -> None:
    """Fill verified / verify_method / reason on one crop row, in place.

    The sequence check is the gate: a crop whose SEQ ID label pairs with a
    gold row is a real patent drawing of a known molecule, so the pair is
    verified by that identity alone. The bare decode stays advisory —
    MolSight is not yet tuned to these dense macrocycles, so its mass
    delta is recorded as `formula_matched` evidence and never vetoes.
    `smiles` carries the gold SMILES (the training target); the decoder
    output moves to `decoded_smiles`."""
    seq_id = crop_row.get("seq_id")
    declared = table_rows.get(str(seq_id)) if seq_id is not None else None
    if declared is not None and crop_row.get("exact_mass") is None:
        crop_row["exact_mass"] = declared.get("exact_mass")
        crop_row.setdefault("sequence", declared.get("sequence"))

    # image quality is part of the gate: a truncated or fragment crop must
    # not become a training pair even when its label pairs cleanly
    if crop_row.get("image_flags"):
        crop_row["verified"] = False
        crop_row["verify_method"] = "image_quality"
        crop_row["reason"] = "image_flags:" + ",".join(crop_row["image_flags"])
        return

    gold = gold_smiles.get(str(seq_id)) if (gold_smiles and seq_id is not None) \
        else None

    # advisory decode evidence, recorded but never a veto
    decoded_smiles = crop_row.get("smiles") or ""
    declared_mass = declared.get("exact_mass") if declared else None
    if decoded_smiles:
        molecule, exact_mass, formula, canonical = smiles_metrics(decoded_smiles)
        crop_row["decoded_smiles"] = decoded_smiles
        crop_row["decoded_formula"] = formula
        crop_row["decoded_mass"] = round(exact_mass, 3) if exact_mass else None
        crop_row["smiles_canonical"] = canonical
        crop_row["decode_ok"] = molecule is not None
        if molecule is not None and isinstance(declared_mass, (int, float)):
            mass_delta = round(exact_mass - declared_mass, 3)
            crop_row["mass_delta"] = mass_delta
            crop_row["formula_matched"] = abs(mass_delta) <= MASS_TOLERANCE_DA
        if molecule is not None and gold:
            similarity = tanimoto_similarity(canonical, gold.get('smiles', '') if isinstance(gold, dict) else gold)
            crop_row["tanimoto_vs_gold"] = (round(similarity, 4)
                                            if similarity is not None else None)

    if gold:
        from skills.patent_gold_review import validate_reference
        reviewed, why = validate_reference(gold, crop_row.get('crop'), seq_id)
        if reviewed:
            crop_row["verified"] = True
            crop_row["verify_method"] = "figure_review"
            crop_row["reason"] = "original_figure_and_structure_reviewed"
            crop_row["smiles"] = gold['smiles']
            crop_row["gold_review"] = gold['review']
        else:
            crop_row["verified"] = False
            crop_row["verify_method"] = "gold_review"
            crop_row["reason"] = why
            crop_row["smiles"] = None
        return

    crop_row["verified"] = "unpaired"
    crop_row["verify_method"] = "none"
    crop_row["smiles"] = None
    crop_row["reason"] = ("no_seq_label" if declared is None
                          else "no_gold_smiles_for_seq")


def mark_duplicate_crops(rows: list[dict]) -> None:
    """Point near-duplicate crops at their best representative (same molecule
    drawn more than once, e.g. US and WO publications of one family).
    Duplicate rows keep their own verification state; the summary counts
    both raw and unique verified pairs."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("seq_id") is not None:
            group_key = f"seq:{row['seq_id']}"
        elif row.get("smiles_canonical"):
            group_key = f"smiles:{row['smiles_canonical']}"
        else:
            continue
        groups.setdefault(group_key, []).append(row)
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda row: (
            -(row.get("detect_score") or 0.0),
            -((row["bbox"][2] - row["bbox"][0]) * (row["bbox"][3] - row["bbox"][1]))))
        for duplicate in group[1:]:
            duplicate["duplicate_of"] = group[0]["crop"]


# --------------------------------------------------------------------------
# per-patent extraction
# --------------------------------------------------------------------------

def crop_filename(page_number: int, box_index: int) -> str:
    return f"page_{page_number:03d}_box_{box_index}.png"


def save_padded_crop(page_image, bbox: list[int], path: Path,
                     pad: int = CROP_PAD_PX) -> None:
    page_width, page_height = page_image.size
    x0, y0, x1, y1 = bbox
    box = (max(0, x0 - pad), max(0, y0 - pad),
           min(page_width, x1 + pad), min(page_height, y1 + pad))
    page_image.crop(box).save(path, "PNG")


def render_params(args) -> dict:
    return {"dpi": args.dpi, "conf": args.conf}


def load_resume_state(state_path: Path, params: dict):
    """(visited pages, stored rows) recorded for these render params."""
    visited_pages: set[int] = set()
    stored_rows: list[dict] = []
    if not state_path.exists():
        return visited_pages, stored_rows
    for line in state_path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("_params") != params:
            continue
        if "_visited_page" in record:
            visited_pages.add(record["_visited_page"])
        elif record.get("page") is not None:
            visited_pages.add(record["page"])
            stored_rows.append(record)
    return visited_pages, stored_rows


def write_resume_state(state_path: Path, visited_pages: set[int],
                       rows: list[dict], params: dict) -> None:
    """Full rewrite of the resume state (post-OCR / post-decode rows)."""
    row_pages = {row["page"] for row in rows}
    marker_lines = [json.dumps({"_visited_page": page, "_params": params}) + "\n"
                    for page in sorted(visited_pages - row_pages)]
    row_lines = [json.dumps(row, ensure_ascii=False) + "\n" for row in rows]
    state_path.write_text("".join(marker_lines + row_lines))


def extract_structure_crops(pdf_path: Path, args, ledger: ErrorLedger) -> list[dict]:
    """Phase A: render every page, detect boxes, save crops, OCR label bands."""
    import pymupdf

    patent = pdf_path.stem
    patent_dir = Path(args.out) / patent
    patent_dir.mkdir(parents=True, exist_ok=True)
    state_path = patent_dir / "rows.jsonl"
    params = render_params(args)
    visited_pages, rows = load_resume_state(state_path, params)
    if visited_pages:
        print(f"[{patent}] resume: {len(visited_pages)} pages in state, "
              f"{len(rows)} rows loaded", flush=True)

    try:
        document = pymupdf.open(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        ledger.add("open", f"pymupdf_open_failed: {exc}", patent=patent)
        return rows

    page_filter = (args.pages or {}).get(patent)
    start_time = time.time()
    pages_processed = 0
    for page_index in range(document.page_count):
        page_number = page_index + 1
        if page_number in visited_pages:
            continue
        if page_filter and not any(lo <= page_number <= hi
                                   for lo, hi in page_filter):
            visited_pages.add(page_number)  # excluded pages count as visited
            continue
        try:
            page_image = render_page(document, page_index, args.dpi)
        except ValueError as exc:
            ledger.add("render", str(exc), patent=patent, page=page_number)
            visited_pages.add(page_number)
            continue
        try:
            detections = detect_structure_boxes(
                png_bytes(page_image), args.detect_url, args.conf)
        except Exception as exc:  # noqa: BLE001
            ledger.add("detect", str(exc), patent=patent, page=page_number)
            visited_pages.add(page_number)
            continue

        page_rows: list[dict] = []
        band_tasks: list[dict] = []
        for box_index, detection in enumerate(detections):
            bbox = [int(value) for value in detection["bbox"]]
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < args.min_area:
                continue
            filename = crop_filename(page_number, box_index)
            try:
                save_padded_crop(page_image, bbox, patent_dir / filename)
            except Exception as exc:  # noqa: BLE001
                ledger.add("crop_save", str(exc), patent=patent,
                           page=page_number, box=box_index)
                continue
            crop_row = {
                "crop": f"{patent}/{filename}",
                "patent": patent,
                "page": page_number,
                "bbox": bbox,
                "seq_id": None,
                "smiles": None,
                "exact_mass": None,
                "decoded_formula": None,
                "decoded_mass": None,
                "verified": "pending",
                "verify_method": None,
                "reason": None,
                "image_flags": collect_image_quality_flags(page_image, bbox),
                "detect_score": round(float(detection.get("score") or 0.0), 4),
                "style_meta": {
                    "patent": patent,
                    "page": page_number,
                    "dpi": args.dpi,
                    "bbox_w": bbox[2] - bbox[0],
                    "bbox_h": bbox[3] - bbox[1],
                    "page_px": list(page_image.size),
                },
                "_params": params,
            }
            page_rows.append(crop_row)
            if not args.skip_ocr:
                position_bands = dict(label_band_regions(page_image, bbox))
                for position, band_image in position_bands.items():
                    # bind loop values as defaults: the retry closure runs
                    # after the loop has advanced to the next box
                    def widen_band(position=position,
                                   page_image=page_image,
                                   bbox=bbox,
                                   page_size=page_image.size):
                        if position != "right":
                            return None
                        page_width, page_height = page_size
                        x0, y0, x1, y1 = bbox
                        box_height = max(1, y1 - y0)
                        widened = page_image.crop((
                            min(page_width, x1 + CROP_PAD_PX),
                            max(0, y0 - int(0.25 * box_height)),
                            min(page_width,
                                x1 + CROP_PAD_PX + 2 * RIGHT_BAND_WIDTH_PX),
                            min(page_height, y1 + int(0.25 * box_height))))
                        return None if widened.getbbox() is None else widened
                    band_tasks.append({
                        "row": crop_row, "position": position,
                        "image": band_image, "widen": widen_band,
                        "patent": patent, "page": page_number,
                        "box_index": box_index})

        if band_tasks:
            run_label_ocr(band_tasks, args.ocr_url, args.ocr_threads, ledger)
        for crop_row in page_rows:
            merge_label_ids(crop_row,
                            [task for task in band_tasks if task["row"] is crop_row])

        rows.extend(page_rows)
        visited_pages.add(page_number)
        pages_processed += 1
        if pages_processed % 20 == 0:
            # incremental persist: a crash mid-patent must not lose hours of
            # band OCR work
            write_resume_state(state_path, visited_pages, rows, params)
            print(f"[{patent}] {pages_processed} pages, {len(rows)} crops "
                  f"({time.time() - start_time:.0f}s)", flush=True)

    document.close()
    write_resume_state(state_path, visited_pages, rows, params)
    print(f"[{patent}] phase A done: {len(rows)} crops "
          f"({time.time() - start_time:.0f}s, errors={len(ledger.items)})",
          flush=True)
    return rows


# --------------------------------------------------------------------------
# decode + finalize
# --------------------------------------------------------------------------

def decode_detected_crops(rows: list[dict], args, ledger: ErrorLedger,
                          pdf_paths: list[Path]) -> None:
    """MolSight decode for every crop that lacks a SMILES.

    Declared-mass arbitration between padding variants: tight detector boxes
    clip substituent labels and the decoder then misreads single residues.
    When the first read disagrees with the declared mass, one padded variant
    is decoded; the read closest to the authoritative declared mass wins and
    the selection is recorded rather than hidden."""
    decode_queue = [row for row in rows if not row.get("smiles")]
    if args.min_decode_score:
        decode_queue = [row for row in decode_queue
                        if row["detect_score"] >= args.min_decode_score]
    if args.max_decode:
        decode_queue = decode_queue[: args.max_decode]
    if not decode_queue:
        print("[decode] nothing to decode", flush=True)
        return
    try:
        decoder = StructureDecoder(args.checkpoint, args.gpu)
    except Exception as exc:  # noqa: BLE001
        ledger.add("decode", f"worker_start_failed: {exc}",
                   count=len(decode_queue))
        for row in decode_queue:
            row["decode_error"] = "worker_start_failed"
        return

    import pymupdf
    documents = {path.stem: pymupdf.open(str(path)) for path in pdf_paths}

    def padded_variant_path(row: dict, pad: int) -> str | None:
        """Re-render the page and save the padded variant as a hidden file."""
        x0, y0, x1, y1 = row["bbox"]
        try:
            page_image = render_page(documents[row["patent"]],
                                     row["page"] - 1, args.dpi)
        except ValueError as exc:
            ledger.add("render", str(exc), patent=row["patent"],
                       page=row["page"])
            return None
        page_width, page_height = page_image.size
        box = (max(0, x0 - pad), max(0, y0 - pad),
               min(page_width, x1 + pad), min(page_height, y1 + pad))
        variant_path = Path(args.out) / row["patent"] / \
            f".padded_{Path(row['crop']).name}"
        page_image.crop(box).save(variant_path, "PNG")
        return str(variant_path)

    print(f"[decode] {len(decode_queue)} crops "
          f"(checkpoint={Path(args.checkpoint).name}, gpu={args.gpu})",
          flush=True)
    start_time = time.time()
    image_paths = [str(Path(args.out) / row["crop"]) for row in decode_queue]
    results = decoder.decode_paths(image_paths)

    # rows whose first read is missing/invalid, or disagrees with the
    # declared mass when one is associated
    mass_disputes: list[dict] = []
    for row, result in zip(decode_queue, results):
        smiles = result.get("smiles") or ""
        if result.get("ok") and smiles and smiles_metrics(smiles)[0] is not None:
            row["smiles"] = smiles
            row.pop("decode_error", None)
        else:
            row["decode_error"] = result.get("error") or "smiles_invalid"
        declared_mass = row.get("exact_mass")
        if not isinstance(declared_mass, (int, float)):
            continue
        _, decoded_mass, _, _ = (smiles_metrics(row["smiles"])
                                 if row.get("smiles") else (None, None, None, None))
        if decoded_mass is None or \
                abs(decoded_mass - declared_mass) > MASS_TOLERANCE_DA:
            mass_disputes.append(row)

    variant_rows: list[dict] = []
    variant_paths: list[str] = []
    for row in mass_disputes:
        variant_path = padded_variant_path(row, CROP_PAD_PX + DECODE_RETRY_PAD_PX)
        if variant_path:
            variant_rows.append(row)
            variant_paths.append(variant_path)
    if variant_paths:
        print(f"[decode] mass arbitration on {len(variant_paths)} crops "
              f"with +{DECODE_RETRY_PAD_PX}px variants", flush=True)
        for row, result in zip(variant_rows, decoder.decode_paths(variant_paths)):
            smiles = result.get("smiles") or ""
            if not (result.get("ok") and smiles and smiles_metrics(smiles)[0]
                    is not None):
                continue
            _, variant_mass, _, _ = smiles_metrics(smiles)
            declared_mass = row["exact_mass"]
            if row.get("smiles"):
                _, first_mass, _, _ = smiles_metrics(row["smiles"])
                first_delta = abs(first_mass - declared_mass)
            else:
                first_delta = None
            variant_delta = abs(variant_mass - declared_mass)
            improves = (first_delta is None and variant_delta <= MASS_TOLERANCE_DA) \
                or (first_delta is not None and variant_delta < first_delta)
            if improves:
                row["smiles"] = smiles
                row.pop("decode_error", None)
                row["padded_variant_selected"] = True

    for row in variant_rows:
        variant_path = Path(args.out) / row["patent"] / \
            f".padded_{Path(row['crop']).name}"
        variant_path.unlink(missing_ok=True)
    print(f"[decode] done in {time.time() - start_time:.0f}s", flush=True)
    for document in documents.values():
        document.close()


def load_gold_smiles(path: str | None):
    """Load references with review evidence; legacy strings stay untrusted.

    verify_pair checks the SMILES digest and exact crop binding before
    promoting a pair. Returning legacy entries permits a named rejection.
    """
    if not path or not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text())
    items = data.get("rows") if isinstance(data.get("rows"), (list, dict)) else data
    keyed = {}
    if isinstance(items, dict):
        for key, value in items.items():
            if isinstance(value, (dict, str)):
                keyed[str(key)] = value
    elif isinstance(items, list):
        keyed = {str(item['seq_id']):item for item in items
                 if isinstance(item, dict) and item.get('seq_id') is not None}
    return keyed or None


def parse_page_filter(spec: str, stems: list[str]) -> dict[str, list[tuple]]:
    """'US20240287138A1:1-12+260-290' -> {stem: [(1,12),(260,290)]}"""
    parsed: dict[str, list[tuple]] = {}
    if not spec:
        return parsed
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        stem, _, ranges = part.partition(":")
        if stem not in stems or not ranges:
            raise SystemExit(f"--pages: unknown patent '{stem}' "
                             f"(known: {stems})")
        spans = []
        for piece in ranges.split("+"):
            if "-" in piece:
                lo, hi = piece.split("-")
                spans.append((int(lo), int(hi)))
            else:
                spans.append((int(piece), int(piece)))
        parsed[stem] = spans
    return parsed


def summarize(rows: list[dict], ledger: ErrorLedger) -> dict:
    per_patent: dict[str, dict] = {}
    for row in rows:
        slot = per_patent.setdefault(row["patent"], {
            "crops": 0, "labeled": 0, "verified": 0, "rejected": 0,
            "unpaired": 0, "duplicates": 0, "image_flagged": 0})
        slot["crops"] += 1
        if row.get("seq_id") is not None:
            slot["labeled"] += 1
        if row.get("image_flags"):
            slot["image_flagged"] += 1
        verdict = row.get("verified")
        if verdict is True:
            slot["verified"] += 1
        elif verdict is False:
            slot["rejected"] += 1
        elif verdict == "unpaired":
            slot["unpaired"] += 1
        if row.get("duplicate_of"):
            slot["duplicates"] += 1
    return {
        "total_crops": len(rows),
        "verified_true": sum(1 for row in rows if row.get("verified") is True),
        "verified_unique": len({row["crop"] for row in rows
                                if row.get("verified") is True
                                and not row.get("duplicate_of")}),
        "verified_false": sum(1 for row in rows if row.get("verified") is False),
        "unpaired": sum(1 for row in rows if row.get("verified") == "unpaired"),
        "duplicates": sum(1 for row in rows if row.get("duplicate_of")),
        "per_patent": per_patent,
        "error_count": len(ledger.items),
        "errors_by_stage": _tally(ledger.items, "stage"),
    }


def _tally(items: list[dict], key: str) -> dict:
    tallies: dict[str, int] = {}
    for item in items:
        tallies[str(item.get(key))] = tallies.get(str(item.get(key)), 0) + 1
    return tallies


def main() -> int:
    argument_parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0])
    argument_parser.add_argument("--patent", action="append", default=[],
                                 help="patent PDF path (repeatable)")
    argument_parser.add_argument("--out",
                                 default=str(REPO_ROOT / "data/real_crops"))
    argument_parser.add_argument("--table", default="/tmp/table1_rows.json",
                                 help="TABLE 1 rows json (seq_id -> exact_mass)")
    argument_parser.add_argument("--gold", default="/tmp/gold_align_result.json",
                                 help="optional gold SMILES json (seq_id -> smiles)")
    argument_parser.add_argument("--dpi", type=int, default=150)
    argument_parser.add_argument("--conf", type=float, default=0.25,
                                 help="MolDetV2 confidence (pipeline default 0.25)")
    argument_parser.add_argument("--min-area", type=int, default=2500,
                                 help="min box area in px^2 to keep a crop")
    argument_parser.add_argument("--detect-url", default="http://localhost:8011")
    argument_parser.add_argument("--ocr-url", default="http://localhost:8010")
    argument_parser.add_argument("--ocr-threads", type=int, default=3)
    argument_parser.add_argument("--gpu", default="1",
                                 help="CUDA device for the decode worker")
    argument_parser.add_argument("--checkpoint", default=(
        "/data/molsight_repo/runs/markush_v13_rft/ckpt_model/epoch_0.pth"))
    argument_parser.add_argument("--pages", default="",
                                 help="optional page filter, e.g. "
                                      "'US20240287138A1:1-12+260-290,WO2024163691A1:290-328'")
    argument_parser.add_argument("--min-decode-score", type=float, default=0.0)
    argument_parser.add_argument("--max-decode", type=int, default=0,
                                 help="cap decode count (0 = all)")
    argument_parser.add_argument("--skip-ocr", action="store_true")
    argument_parser.add_argument("--skip-decode", action="store_true")
    argument_parser.add_argument("--skip-band-evidence", action="store_true",
                                 help="disable table-evidence label matching")
    args = argument_parser.parse_args()

    try:
        import pymupdf  # noqa: F401
        import rdkit  # noqa: F401
    except ImportError as exc:
        print(f"FATAL missing dependency: {exc}", flush=True)
        return 2

    pdf_paths = [Path(path) for path in args.patent]
    missing = [str(path) for path in pdf_paths if not path.exists()]
    if missing:
        print(f"FATAL patents not found: {missing}", flush=True)
        return 2
    args.pages = parse_page_filter(args.pages, [path.stem for path in pdf_paths])

    table_path = Path(args.table)
    table_rows = json.loads(table_path.read_text()) if table_path.exists() else {}
    if not table_rows:
        print(f"NOTE table rows not found at {args.table}; "
              f"only gold Tanimoto can verify", flush=True)
    evidence_index = TableEvidenceIndex(table_rows) \
        if (table_rows and not args.skip_band_evidence) else None
    gold_smiles = load_gold_smiles(args.gold)
    if gold_smiles is None:
        print("NOTE gold SMILES not available; sequence check skipped "
              "(mass gate only)", flush=True)

    ledger = ErrorLedger()
    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for pdf_path in pdf_paths:
        phase_start = time.time()
        patent_rows = extract_structure_crops(pdf_path, args, ledger)
        print(f"[{pdf_path.stem}] {len(patent_rows)} crops in "
              f"{time.time() - phase_start:.0f}s", flush=True)
        all_rows.extend(patent_rows)

    # association data must precede decode: the declared exact mass
    # arbitrates between padding variants of the same drawing
    attach_table_evidence(all_rows, evidence_index)
    for row in all_rows:
        declared = (table_rows.get(str(row["seq_id"]))
                    if row.get("seq_id") is not None else None)
        if declared:
            row["exact_mass"] = declared.get("exact_mass")
            row["sequence"] = declared.get("sequence")

    def persist_patent_state() -> None:
        # decode results must survive a crash between phases; zero-box page
        # markers are preserved so resume never re-OCRs a finished page
        for pdf_path in pdf_paths:
            patent = pdf_path.stem
            params = render_params(args)
            state_path = Path(args.out, patent, "rows.jsonl")
            stored_visited, _ = load_resume_state(state_path, params)
            patent_rows = [row for row in all_rows if row["patent"] == patent]
            visited = stored_visited | {row["page"] for row in patent_rows}
            write_resume_state(state_path, visited, patent_rows, params)

    persist_patent_state()
    if not args.skip_decode:
        decode_detected_crops(all_rows, args, ledger, pdf_paths)
        persist_patent_state()

    for row in all_rows:
        verify_pair(row, table_rows, gold_smiles)
    mark_duplicate_crops(all_rows)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "dpi": args.dpi, "conf": args.conf,
            "patents": [str(path) for path in pdf_paths],
            "mass_tolerance_da": MASS_TOLERANCE_DA,
            "checkpoint": args.checkpoint,
            "gold_smiles_used": gold_smiles is not None,
            "detector": "MolDetV2 via harness /v1/structures/detect",
            "decoder": "MolSight worker (markush_v13_rft)",
        },
        "summary": summarize(all_rows, ledger),
        "rows": all_rows,
        "errors": ledger.items,
    }
    manifest_path = output_root / "real_crops_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    for patent in {row["patent"] for row in all_rows}:
        patent_rows = [row for row in all_rows if row["patent"] == patent]
        (output_root / patent / f"{patent}_manifest.json").write_text(
            json.dumps({"summary": summarize(patent_rows, ledger),
                        "rows": patent_rows},
                       ensure_ascii=False, indent=1))
    print(json.dumps(manifest["summary"], indent=1), flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
