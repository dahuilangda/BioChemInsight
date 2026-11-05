"""Standalone structure-recognition API: the harness unit operations.

Runs inside the molsight_venv (torch + ultralytics + RDKit). Each pipeline
capability is exposed independently so external clients — and the document
harness itself — can call them as discrete steps:

    POST /v1/structures/detect       page image   -> boxes (+ optional crops)
    POST /v1/structures/recognize    crop image   -> SMILES (MolSight)
    POST /v1/structures/parse        page image   -> boxes + SMILES per box
    POST /v1/structures/identify-id  page + bbox  -> compound/sequence ID label
    POST /v1/structures/route        crop+smiles  -> markush vs complete routing
    POST /v1/ocr                     page image   -> markdown text (PaddleOCR-VL)
    POST /v1/sequences/extract       OCR text     -> sequence records
    POST /v1/sequences/to-smiles     sequence     -> peptide SMILES
    GET  /healthz

Launch:
    CUDA_VISIBLE_DEVICES=1 MOLSIGHT_GPU=1 \
        /data/molsight_venv/bin/uvicorn structure_api:app \
        --host 0.0.0.0 --port 8011
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from fastapi import Body, FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

app = FastAPI(
    title="BioChemInsight Structure API",
    description="Unit operations of the structure pipeline: box detection, "
                "MolSight recognition, compound-ID identification, markush "
                "routing, OCR and peptide-sequence handling.",
    version="1.1",
)

OCR_SERVICE_URL = os.environ.get("OCR_SERVICE_URL", "http://localhost:8010")
_OCR_TIMEOUT_SECONDS = 180.0

# The MolSight worker subprocess inherits CUDA_VISIBLE_DEVICES from this
# process; MOLSIGHT_GPU selects it inside the service wrapper.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("MOLSIGHT_GPU", "0"))

_DETECT_LOCK = threading.Lock()
_DETECTOR_READY = False


def _detect_boxes(page_bgr: np.ndarray, conf: float) -> list:
    from utils.moldetv2 import moldetv2_detect

    global _DETECTOR_READY
    with _DETECT_LOCK:
        boxes = moldetv2_detect(page_bgr, conf=conf)
        _DETECTOR_READY = True
    # Merge boxes with heavy overlap (same structure detected twice); keep
    # boxes that merely touch — giant multi-part figures must not be merged
    # away, only true duplicates collapsed.
    boxes.sort(key=lambda b: b[4], reverse=True)
    kept: list[list[float]] = []
    for box in boxes:
        x0, y0, x1, y1, score = box
        duplicate = False
        for k in kept:
            ix0, iy0 = max(x0, k[0]), max(y0, k[1])
            ix1, iy1 = min(x1, k[2]), min(y1, k[3])
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            smaller = min((x1 - x0) * (y1 - y0), (k[2] - k[0]) * (k[3] - k[1]))
            if inter / smaller > 0.7:
                duplicate = True
                break
        if not duplicate:
            kept.append([x0, y0, x1, y1, score])
    return kept


def _decode_upload(contents: bytes) -> np.ndarray:
    arr = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("request body is not a decodable image")
    return image


def _crop_payload(page_bgr: np.ndarray, bbox: list[int], include: bool) -> dict:
    x0, y0, x1, y1 = bbox
    crop = page_bgr[y0:y1, x0:x1]
    item = {"bbox": [int(v) for v in bbox]}
    if include and crop.size:
        ok, buf = cv2.imencode(".png", crop)
        if ok:
            item["crop_png_base64"] = base64.b64encode(buf.tobytes()).decode()
    return item


def _recognize_file(image_path: str) -> dict:
    from utils.molsight_service import molsight_service

    result = molsight_service().predict(image_path)
    return {
        "smiles": result.get("smiles") if result.get("ok") else "",
        "ok": bool(result.get("ok")),
        "error": result.get("error"),
        "raw": result.get("raw"),
    }


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "detector": "moldetv2",
        "recognizer": "molsight",
        "checkpoint": os.environ.get(
            "MOLSIGHT_CHECKPOINT",
            "/data/molsight_repo/runs/markush_ft_v12_3ep_clean/ckpt_model/epoch_0.pth"),
    }


@app.post("/v1/structures/detect")
async def detect_structures(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=0.95),
    include_crops: bool = Query(False, alias="include_crops"),
):
    """Detect structure bounding boxes on a page image."""
    contents = await file.read()
    try:
        page = _decode_upload(contents)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    boxes = _detect_boxes(page, conf)
    return {
        "count": len(boxes),
        "structures": [_crop_payload(page, b[:4], include_crops) | {"score": round(b[4], 4)}
                       for b in boxes],
    }


@app.post("/v1/structures/recognize")
async def recognize_structure(file: UploadFile = File(...)):
    """Recognize a single cropped structure image with MolSight."""
    contents = await file.read()
    try:
        image = _decode_upload(contents)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, image)
        try:
            result = _recognize_file(tmp.name)
        finally:
            os.unlink(tmp.name)
    return result


@app.post("/v1/structures/parse")
async def parse_page(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=0.95),
    include_crops: bool = Query(False, alias="include_crops"),
):
    """Full page -> detected boxes -> MolSight SMILES per box."""
    contents = await file.read()
    try:
        page = _decode_upload(contents)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    boxes = _detect_boxes(page, conf)
    structures = []
    tmp_paths = []
    try:
        for b in boxes:
            x0, y0, x1, y1 = [int(v) for v in b[:4]]
            crop = page[y0:y1, x0:x1]
            item = {"bbox": [x0, y0, x1, y1], "score": round(float(b[4]), 4)}
            if include_crops and crop.size:
                ok, buf = cv2.imencode(".png", crop)
                if ok:
                    item["crop_png_base64"] = base64.b64encode(buf.tobytes()).decode()
            if crop.size:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                cv2.imwrite(tmp.name, crop)
                tmp_paths.append(tmp.name)
                item["recognition"] = _recognize_file(tmp.name)
            else:
                item["recognition"] = {"ok": False, "error": "empty crop"}
            structures.append(item)
    finally:
        for path in tmp_paths:
            os.unlink(path)
    return {"count": len(structures), "structures": structures}


# --- compound-ID identification -------------------------------------------

# Label forms seen under patent figures: "SEQ ID NO: 53", "SEQ ID NO.53,",
# "Compound 14", "Compound I-23", "Example 5", "Formula (I)".
_ID_PATTERNS = [
    (re.compile(r"SEQ\s*ID(?:\s*NO)?\.?\s*[:.]?\s*(\d{1,5})", re.IGNORECASE), "seq_id"),
    (re.compile(r"COMPOUND(?:\s*(?:NO|NUMBER))?\.?\s*[:.]?\s*"
                r"([0-9]{1,4}[A-Za-z]?|[IVXLCDM]+(?:-[0-9]{1,3})?)",
                re.IGNORECASE), "compound_id"),
    (re.compile(r"EXAMPLE(?:\s*(?:NO|NUMBER))?\.?\s*[:.]?\s*(\d{1,4}[A-Za-z]?)",
                re.IGNORECASE), "example_id"),
    (re.compile(r"FORMULA\s*\(?\s*([IVXLCDM]{1,4})\s*\)?", re.IGNORECASE), "formula_id"),
]


def _extract_id_candidates(text: str) -> list[dict]:
    found = []
    seen = set()
    for pattern, kind in _ID_PATTERNS:
        for match in pattern.finditer(text or ""):
            key = (kind, match.group(1).lower())
            if key in seen:
                continue
            seen.add(key)
            found.append({"kind": kind, "id": match.group(1),
                          "span": match.group(0).strip()})
    return found


def _label_bands(page: np.ndarray, bbox: list[int]) -> list[dict]:
    """Figure-label regions: the band under the box (patent convention) and
    a slim band above, both padded to give the OCR engine some margin."""
    h, w = page.shape[:2]
    x0, y0, x1, y1 = bbox
    box_h = max(1, y1 - y0)
    under_px = int(min(max(0.22 * box_h, 50), 300))
    above_px = int(min(max(0.08 * box_h, 24), 120))
    pad_x = int(0.05 * max(1, x1 - x0))
    bands = [
        {"position": "below", "bbox": [max(0, x0 - pad_x), y1,
                                       min(w, x1 + pad_x), min(h, y1 + under_px)]},
        {"position": "above", "bbox": [max(0, x0 - pad_x), max(0, y0 - above_px),
                                       min(w, x1 + pad_x), y0]},
    ]
    for band in bands:
        bx0, by0, bx1, by1 = band["bbox"]
        if bx1 <= bx0 or by1 <= by0:
            band["image"] = None
            continue
        crop = page[by0:by1, bx0:bx1]
        # White frame so glyphs at the crop edge still OCR cleanly.
        framed = cv2.copyMakeBorder(crop, 16, 16, 16, 16,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
        band["image"] = framed
    return bands


def _ocr_image(image: np.ndarray, timeout: float = _OCR_TIMEOUT_SECONDS) -> str:
    """Send one image through the PaddleOCR-VL service; return its markdown."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("band encode failed")
    resp = requests.post(f"{OCR_SERVICE_URL}/v1/image-to-markdown",
                         files={"file": ("band.png", buf.tobytes(), "image/png")},
                         timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = requests.get(f"{OCR_SERVICE_URL}/v1/jobs/{job_id}", timeout=15).json()
        state = status.get("status")
        if state == "completed":
            result = requests.get(
                f"{OCR_SERVICE_URL}/v1/jobs/{job_id}/result", timeout=15).json()
            return "\n".join(result.get("page_markdowns") or [])
        if state == "failed":
            raise RuntimeError(f"OCR job failed: {status.get('error')}")
        time.sleep(1.5)
    raise TimeoutError(f"OCR job {job_id} did not finish in {timeout:.0f}s")


@app.post("/v1/structures/identify-id")
async def identify_compound_id(
    file: UploadFile = File(...),
    bbox: str | None = Form(default=None),
    conf: float = Query(0.25, ge=0.05, le=0.95),
):
    """Compound/sequence ID printed near a structure figure.

    Give the page plus one bbox (JSON list [x0,y0,x1,y1]); without bbox the
    page is auto-detected and every box is labelled. Returns the OCR text of
    the label bands and structured ID candidates."""
    contents = await file.read()
    try:
        page = _decode_upload(contents)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if bbox:
        try:
            boxes = [[float(v) for v in json.loads(bbox)]]
            if len(boxes[0]) != 4:
                raise ValueError
        except (ValueError, json.JSONDecodeError):
            return JSONResponse(status_code=400,
                                content={"error": "bbox must be a JSON [x0,y0,x1,y1]"})
    else:
        boxes = [b[:4] for b in _detect_boxes(page, conf)]
    results = []
    for box in boxes:
        bbox_int = [int(v) for v in box]
        entry: dict = {"bbox": bbox_int, "ids": [], "bands": []}
        for band in _label_bands(page, bbox_int):
            if band["image"] is None:
                continue
            try:
                text = _ocr_image(band["image"])
            except Exception as exc:
                entry["bands"].append({"position": band["position"],
                                       "error": str(exc)[:200]})
                continue
            entry["bands"].append({"position": band["position"], "text": text.strip()})
            entry["ids"].extend(_extract_id_candidates(text))
        results.append(entry)
    return {"count": len(results), "structures": results}


# --- routing ----------------------------------------------------------------

@app.post("/v1/structures/route")
async def route_structure(
    file: UploadFile = File(...),
    smiles: str = Form(...),
    structure_type: str = Form(default=""),
):
    """Markush-vs-complete routing decision for one structure.

    Combines the declared type, chemistry facts (dummy atoms on the decoded
    SMILES) and, when the image is readable, the VLM arbiter's judgment on
    whether the decode broke the macrocycle."""
    contents = await file.read()
    tmp_path = None
    try:
        image = _decode_upload(contents)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            tmp_path = tmp.name
        from utils.structure_router import judge

        decision = judge(structure_type, smiles, tmp_path)
        return {"kind": decision.kind, "backend": decision.backend,
                "reason": decision.reason, "signals": decision.signals}
    except ImportError as exc:
        return JSONResponse(status_code=503,
                            content={"error": f"routing deps unavailable: {exc}"})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)[:300]})
    finally:
        if tmp_path:
            os.unlink(tmp_path)


# --- OCR proxy ---------------------------------------------------------------

@app.post("/v1/ocr")
async def ocr_page(file: UploadFile = File(...)):
    """Page image -> markdown text via the PaddleOCR-VL service."""
    contents = await file.read()
    try:
        page = _decode_upload(contents)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    try:
        text = _ocr_image(page)
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": str(exc)[:300]})
    return {"text": text}


# --- peptide sequences --------------------------------------------------------

@app.post("/v1/sequences/extract")
async def extract_sequences(
    text: str = Form(...),
    page_num: int = Form(default=0),
    to_smiles: bool = Query(True, alias="to_smiles"),
):
    """OCR text -> sequence records (one-/three-letter residues), optionally
    with head-to-tail cyclic SMILES for each record."""
    from utils.sequence_parser import extract_sequences_from_text, sequence_to_smiles

    records = extract_sequences_from_text(text, page_num=page_num)
    out = []
    for record in records:
        item = record.asdict()
        if to_smiles:
            item["smiles"] = sequence_to_smiles(record.sequence, cyclic=record.cyclic)
        out.append(item)
    return {"count": len(out), "sequences": out}


@app.post("/v1/sequences/to-smiles")
async def sequence_smiles(
    sequence: str = Form(...),
    cyclic: bool = Form(default=True),
):
    """One-letter peptide sequence -> canonical SMILES."""
    from utils.sequence_parser import sequence_to_smiles

    smiles = sequence_to_smiles(sequence, cyclic=cyclic)
    return {"sequence": sequence, "cyclic": cyclic, "smiles": smiles}


@app.post("/v1/sequences/build-cyclopeptide")
async def build_cyclopeptide(
    residues: str = Form(..., description="JSON list of residue codes, e.g. "
                                         '["Nle","4Pal","PipOH","W","NMeG"]'),
    n_methyl_positions: str = Form(default="[]",
                                   description="JSON list of 0-based backbone positions"),
):
    """Modified-residue cyclopeptide -> canonical SMILES.

    Deterministic assembly from the patent cyclopeptide vocabulary (Nle,
    4Pal, Trp7Cl/OMe, bMeTrp, aMeS, Hyp, PipOH, ACPC, NMeG, Oxa, ...)."""
    from utils.sequence_parser import build_cyclopeptide_smiles

    try:
        codes = json.loads(residues)
        methyl = json.loads(n_methyl_positions)
        if not isinstance(codes, list) or not isinstance(methyl, list):
            raise ValueError
    except (ValueError, json.JSONDecodeError):
        return JSONResponse(status_code=400,
                            content={"error": "residues/n_methyl_positions must be JSON lists"})
    smiles = build_cyclopeptide_smiles(codes, n_methyl_positions=methyl)
    return {"residues": codes, "n_methyl_positions": methyl,
            "smiles": smiles, "ok": bool(smiles)}


# --- bidirectional sequence-SMILES validation ---------------------------------

@app.post("/v1/peptides/smiles-to-sequence")
async def smiles_to_sequence_endpoint(
    file: UploadFile = File(...),
):
    """Decode a cyclic-peptide SMILES into its ordered residue list.

    Uses amide-N segmentation + per-unit canonical substructure matching
    against the vocabulary (incl. non-standard: 4Pal, Trp7Cl, aMeS, ACPC,
    Nle, PipOH, Hyp, NMeG, 7azaTrp, Sbridge, Pseudoproline, ...)."""
    contents = await file.read()
    smi = contents.decode('utf-8').strip()
    from utils.residue_typer import smiles_to_sequence
    codes, ok, detail = smiles_to_sequence(smi)
    return {"residues": codes, "ok": ok, "detail": detail,
            "count": len(codes)}


@app.post("/v1/peptides/cross-validate")
async def cross_validate_bidirectional(
    file: UploadFile = File(...),
    sequence_text: str = Form(..., description="JSON list of anchor residue codes"),
):
    """Bidirectional validation: decoded SMILES vs sequence-anchor residues.

    Returns per-residue typed comparison after rotation alignment, naming
    which residues differ and whether the backbone atom count matches."""
    contents = await file.read()
    smi = contents.decode('utf-8').strip()
    try:
        anchor = json.loads(sequence_text)
        if not isinstance(anchor, list):
            raise ValueError
    except (ValueError, json.JSONDecodeError):
        return JSONResponse(status_code=400,
                            content={"error": "sequence_text must be a JSON array of residue codes"})
    from utils.residue_typer import compare_with_anchor
    result = compare_with_anchor(smi, anchor)
    result["anchor_count"] = len(anchor)
    return result



# --- skill registry: atomic operations, level-by-level exposure --------------


@app.get("/v1/skills")
async def skills_listing():
    from skills.registry import listing, load_skills
    load_skills()
    return {"skills": listing()}


@app.post("/v1/skills/{name}/invoke")
def skill_invoke(name: str, payload: dict = Body(default={})):
    """One atomic operation per call. The harness composes these; nothing
    inside a skill knows about orchestration."""
    from skills.registry import get, load_skills
    load_skills()
    skill = get(name)
    if skill is None:
        return JSONResponse(status_code=404,
                            content={"error": f"unknown skill '{name}'"})
    try:
        # sync endpoint: FastAPI runs it in a worker thread, so model calls
        # inside the skill cannot stall the event loop
        return asyncio.run(skill.handler(payload))
    except Exception as exc:
        return JSONResponse(status_code=500,
                            content={"error": f"{type(exc).__name__}: {exc}"[:300]})



# --- model-native activity join ----------------------------------------------

_ACTIVITY_JOIN_PROMPT = """You are joining a patent bioactivity table to the \
compound structures found on the same patent page.

The table text below was OCR'd from the page. The candidate compound IDs \
were read from the structure-figure labels on that page.

TASK: for every table row, decide which compound ID it belongs to and emit \
that row's activity values. Copy identifiers EXACTLY as they appear in the \
table text. A row whose identifier is not among the candidate IDs must \
still be emitted, with "in_page_structures": false. Do not invent values; \
omit a field rather than guessing it. Columns may include IC50, EC50, Ki, \
percent inhibition, target/assay name, cell line.

Respond with ONLY this JSON object — the output must start with '{':
{"links": [{"compound_id": "<exact id from table>", "in_page_structures": \
true|false, "evidence": "<the table row text this came from>", "values": \
{"<column name>": "<value as written>"}}]}

CANDIDATE COMPOUND IDS (from structure labels on this page):
{ids_json}

TABLE TEXT (OCR):
{table_text}
"""


def _llm_json(prompt: str, timeout: float = 180.0) -> dict:
    """One structured-question call to the visual LLM; strict JSON out."""
    import constants
    from openai import OpenAI

    client = OpenAI(base_url=constants.LLM_OPENAI_COMPATIBLE_MODEL_URL,
                    api_key=constants.LLM_OPENAI_COMPATIBLE_MODEL_KEY)
    resp = client.chat.completions.create(
        model=constants.TEXT_MODEL_NAME if hasattr(constants, "TEXT_MODEL_NAME")
        else constants.VISUAL_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=4000, timeout=timeout)
    text = resp.choices[0].message.content or ''
    start = text.find('{')
    depth = 0
    for i in range(start, len(text)) if start >= 0 else []:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    break
    print(f"harness: unparseable LLM response: {text[:200]!r}")
    return {"links": [], "error": "unparseable LLM response", "raw": text[:300]}


@app.post("/v1/harness/link-activity")
async def link_activity(
    file: UploadFile = File(...),
    table_bbox: str | None = Form(default=None),
    conf: float = Query(0.25, ge=0.05, le=0.95),
):
    """Compound-ID -> activity linkage for one patent page.

    The join runs where the evidence lives: the page's activity table text
    and the page's structure-label IDs enter one model call together, and
    the model emits the (compound -> values) mapping with evidence spans.
    No alias maps, no order assumptions."""
    contents = await file.read()
    try:
        page = _decode_upload(contents)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    region = page
    if table_bbox:
        try:
            x0, y0, x1, y1 = [int(v) for v in json.loads(table_bbox)]
            region = page[max(0, y0):y1, max(0, x0):x1]
        except (ValueError, json.JSONDecodeError):
            return JSONResponse(status_code=400,
                                content={"error": "table_bbox must be JSON [x0,y0,x1,y1]"})
    try:
        table_text = _ocr_image(region)
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": f"OCR failed: {exc}"[:300]})

    boxes = [b[:4] for b in _detect_boxes(page, conf)]
    structures = []
    for box in boxes:
        entry = {"bbox": [int(v) for v in box], "ids": []}
        for band in _label_bands(page, [int(v) for v in box]):
            if band["image"] is None:
                continue
            try:
                entry["ids"].extend(_extract_id_candidates(_ocr_image(band["image"])))
            except Exception:
                continue
        structures.append(entry)
    candidate_ids = sorted({found["span"] for s in structures for found in s["ids"]})

    payload = _llm_json(_ACTIVITY_JOIN_PROMPT
                        .replace("{ids_json}", json.dumps(candidate_ids, ensure_ascii=False))
                        .replace("{table_text}", table_text[:12000]))
    return {"links": payload.get("links", []),
            "llm_error": payload.get("error"),
            "candidate_ids": candidate_ids,
            "structures": structures,
            "table_text": table_text}


# --- page-level harness orchestration ---------------------------------------

_PAGE_LABELS_PROMPT = """IMAGE 1 is one page of a patent. It contains chemical \
structure figures, each with a printed label nearby (below or beside it): a \
compound number, a "SEQ ID NO: N" reference, and sometimes a compound name.

The structure figures were detected at these pixel bounding boxes \
[x0, y0, x1, y1] on this page:
{boxes_json}

The page's OCR text follows. It may contain the labels, compound names, \
sequence tables, or activity tables.

TASK: for EACH detected box, report the label information printed for that \
specific figure (match by spatial position on the page). Do not invent \
labels; use null for missing fields.

Respond with ONLY this JSON object — the output must start with '{':
{"figures": [{"bbox": [x0, y0, x1, y1], "compound_id": "<id or null>", \
"name": "<compound name or null>", "seq_id": "<SEQ number or null>", \
"label_text": "<the exact printed label>"}]}

BOXES:
{boxes_json}

PAGE OCR TEXT:
{page_text}
"""


@app.post("/v1/harness/parse-page")
async def harness_parse_page(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=0.95),
    include_crops: bool = Query(False, alias="include_crops"),
    max_structures: int = Query(12, ge=1, le=60),
):
    """One patent page in -> the complete structured record out.

    Runs every harness capability in document context: box detection,
    MolSight recognition per figure, VLM spatial label assignment (IDs,
    names, SEQ references — one model call for the whole page, not one OCR
    per label band), routing with the arbiter, sequence extraction from the
    page text, and the activity join when the page carries a table."""
    contents = await file.read()
    try:
        page = _decode_upload(contents)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    # 1. detection
    boxes = [b[:4] for b in _detect_boxes(page, conf)][:max_structures]

    # 2. page OCR text (one call for labels + sequences + activity tables)
    try:
        page_text = _ocr_image(page)
    except Exception as exc:
        page_text = ""
        ocr_error = str(exc)[:200]
    else:
        ocr_error = None

    # 3. MolSight recognition per figure (with crop kept for the caller)
    figures = []
    tmp_paths = []
    try:
        for box in boxes:
            x0, y0, x1, y1 = [int(v) for v in box]
            crop = page[y0:y1, x0:x1]
            item: dict = {"bbox": [x0, y0, x1, y1]}
            if include_crops and crop.size:
                ok, buf = cv2.imencode(".png", crop)
                if ok:
                    item["crop_png_base64"] = base64.b64encode(buf.tobytes()).decode()
            if crop.size:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                cv2.imwrite(tmp.name, crop)
                tmp_paths.append(tmp.name)
                recognition = _recognize_file(tmp.name)
                item["smiles"] = recognition.get("smiles", "")
                item["recognize_ok"] = recognition.get("ok")
                item["smiles_valid"] = bool(
                    recognition.get("smiles") and _rdkit_ok(recognition["smiles"]))
                item["route"] = _route_crop(tmp.name, recognition.get("smiles", ""))
            else:
                item["smiles"] = ""
                item["recognize_ok"] = False
            figures.append(item)
    finally:
        for path in tmp_paths:
            os.unlink(path)

    # 4. one VLM call: spatial label assignment for every box at once
    labels_payload = _llm_json(_PAGE_LABELS_PROMPT
                               .replace("{boxes_json}", json.dumps(boxes))
                               .replace("{page_text}", page_text[:14000]))
    by_bbox = {}
    for fig in labels_payload.get("figures", []):
        key = json.dumps(fig.get("bbox"))
        by_bbox[key] = fig
    for item in figures:
        fig = by_bbox.get(json.dumps(item["bbox"]), {})
        item["compound_id"] = fig.get("compound_id")
        item["name"] = fig.get("name")
        item["seq_id"] = fig.get("seq_id")
        item["label_text"] = fig.get("label_text")

    # 5. sequences from the page text (cross-capability structure source)
    sequences = []
    if page_text:
        from utils.sequence_parser import extract_sequences_from_text, sequence_to_smiles
        for record in extract_sequences_from_text(page_text):
            entry = record.asdict()
            entry["smiles"] = sequence_to_smiles(record.sequence, cyclic=record.cyclic)
            sequences.append(entry)

    # 6. activity join when the page carries a table
    activity_links = []
    if page_text and any(marker in page_text.lower() for marker in
                         ("ic50", "ec50", "ki ", "inhibition", "activity")):
        candidate_ids = sorted({str(f.get("compound_id")) for f in figures
                                if f.get("compound_id")})
        payload = _llm_json(_ACTIVITY_JOIN_PROMPT
                            .replace("{ids_json}", json.dumps(candidate_ids, ensure_ascii=False))
                            .replace("{table_text}", page_text[:12000]))
        activity_links = payload.get("links", [])

    return {
        "figure_count": len(figures),
        "figures": figures,
        "sequences": sequences,
        "activity_links": activity_links,
        "page_text": page_text,
        "ocr_error": ocr_error,
    }


def _rdkit_ok(smiles: str) -> bool:
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


# --- multi-signal structure verification ------------------------------------

def _ink_ring_signature(crop_path: str) -> dict:
    """Deterministic macrocycle indicator from the crop's ink geometry.

    A drawn giant ring has a large hollow interior inside a big convex
    outline — a feature no open-chain drawing imitates and no VLM needs to
    read. convex_deficit = 1 - (ink bbox fill of the convex hull); a ring
    with a hollow centre leaves a large deficit."""
    image = cv2.imread(crop_path)
    if image is None:
        return {"macrocycle_like": None}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = (gray < 200).astype(np.uint8)
    ys, xs = np.where(ink)
    if len(ys) < 50:
        return {"macrocycle_like": None}
    points = np.column_stack([xs, ys]).astype(np.int32)
    hull = cv2.convexHull(points)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return {"macrocycle_like": None}
    interior = cv2.pointPolygonTest(hull, (float(xs.mean()), float(ys.mean())), True)
    # white share inside the convex hull = hollow-centre evidence
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    hull_pixels = int(mask.sum())
    white_inside = int(((mask == 1) & (ink == 0)).sum())
    hollow_ratio = white_inside / max(1, hull_pixels)
    macrocycle_like = hollow_ratio >= 0.55 and hull_area > 40000
    return {"macrocycle_like": bool(macrocycle_like),
            "hollow_ratio": round(float(hollow_ratio), 3)}


def _verify_structure_impl(crop_path: str | None, smiles: str,
                           declared_formula: str | None,
                           declared_mw: float | None,
                           sequence_text: str | None) -> dict:
    """Independent-signal verification of a decoded structure.

    Every signal answers a different question and none trusts another:
      RDKit      — is the decode chemically parseable at all
      formula    — does the atom inventory match the declared formula
      MW         — does the molecular weight match the declared value
      sequence   — does a deterministic rebuild from the sequence text
                   agree with the decode (Tanimoto)
      ink geometry — does the crop drawing show a hollow-centre giant ring
                     that the decode must preserve (deterministic CV)
      arbiter    — does the VLM see the same macrocycle in crop and render
    The verdict is verified only when every available signal agrees."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import DataStructs, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")

    signals: dict = {}
    contradictions: list[str] = []

    mol = Chem.MolFromSmiles(smiles or "")
    signals["rdkit_valid"] = mol is not None
    if mol is None:
        return {"verdict": "rejected", "signals": signals,
                "contradictions": ["decode does not parse"],
                "confidence": "high"}

    computed_formula = rdMolDescriptors.CalcMolFormula(mol)
    computed_mw = rdMolDescriptors._CalcMolWt(mol)
    macro_rings = sorted({len(r) for r in mol.GetRingInfo().AtomRings()
                          if len(r) >= 12}, reverse=True)
    signals["rings"] = {"macro_ring_sizes": macro_rings}

    if crop_path and os.path.exists(crop_path):
        geom = _ink_ring_signature(crop_path)
        signals["ink_geometry"] = geom
        if geom.get("macrocycle_like") and not macro_rings:
            contradictions.append(
                "decode opened the macrocycle (crop ink shows a hollow-centre "
                "giant ring, decode has none)")

    if declared_formula:
        match = computed_formula == str(declared_formula).strip()
        signals["formula"] = {"computed": computed_formula,
                              "declared": declared_formula, "match": match}
        if not match:
            contradictions.append(
                f"formula mismatch: decode {computed_formula} vs declared {declared_formula}")
    if declared_mw:
        match = abs(computed_mw - float(declared_mw)) <= 1.0
        signals["mw"] = {"computed": round(computed_mw, 2),
                         "declared": float(declared_mw), "match": match}
        if not match:
            contradictions.append(
                f"MW mismatch: decode {computed_mw:.2f} vs declared {declared_mw}")

    if sequence_text:
        from utils.sequence_parser import build_cyclopeptide_smiles, \
            extract_sequences_from_text, sequence_to_smiles
        built = build_cyclopeptide_smiles(sequence_text)
        if not built:
            records = extract_sequences_from_text(sequence_text)
            for record in records:
                built = sequence_to_smiles(record.sequence, cyclic=record.cyclic)
                if built:
                    break
        if built:
            bm = Chem.MolFromSmiles(built)
            fa = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
            fb = rdMolDescriptors.GetMorganFingerprintAsBitVect(bm, 2)
            sim = DataStructs.TanimotoSimilarity(fa, fb)
            agree = sim >= 0.85
            signals["sequence_agreement"] = {
                "tanimoto": round(sim, 3), "built_formula":
                    rdMolDescriptors.CalcMolFormula(bm), "agree": agree}
            if not agree:
                contradictions.append(
                    f"sequence rebuild disagrees with decode (Tanimoto {sim:.2f})")
        else:
            signals["sequence_agreement"] = {"built": None,
                                             "note": "grammar could not build"}

    if crop_path and os.path.exists(crop_path) and smiles:
        try:
            from utils.model_arbiter import model_arbiter

            verdict = model_arbiter().macrocycle_check(crop_path, smiles)
            broke = verdict.get("candidate_broke_ring")
            signals["arbiter"] = verdict
            if broke is True:
                contradictions.append("arbiter: decode broke the macrocycle")
        except Exception as exc:
            signals["arbiter"] = {"verdict": "unavailable", "error": str(exc)[:150]}

    # a signal only counts when it reached an actual answer; inconclusive
    # witnesses (grammar could not build, arbiter unavailable, no ink
    # decision) must not make a decode look verified
    answered = (
        (signals.get("formula", {}).get("match") is not None)
        + (signals.get("mw", {}).get("match") is not None)
        + (signals.get("sequence_agreement", {}).get("agree") is not None)
        + (signals.get("arbiter", {}).get("candidate_broke_ring") is not None)
        + (signals.get("ink_geometry", {}).get("macrocycle_like") is not None)
    )
    if contradictions:
        return {"verdict": "contradiction", "signals": signals,
                "contradictions": contradictions, "confidence": "high"}
    if not answered:
        return {"verdict": "unverified", "signals": signals,
                "contradictions": [], "confidence": "low"}
    return {"verdict": "verified", "signals": signals,
            "contradictions": [], "confidence": "high"}


@app.post("/v1/harness/verify-structure")
async def verify_structure(
    file: UploadFile = File(...),
    smiles: str = Form(...),
    formula: str | None = Form(default=None),
    mw: float | None = Form(default=None),
    sequence_text: str | None = Form(default=None),
):
    """Multi-signal verification: does this decode deserve trust?

    RDKit validity, declared-formula match, declared-MW match, sequence-text
    rebuild agreement, and VLM ring comparison are independent witnesses;
    the verdict names every contradiction instead of picking a winner."""
    contents = await file.read()
    crop_path = None
    try:
        image = _decode_upload(contents)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            crop_path = tmp.name
        return _verify_structure_impl(crop_path, smiles, formula, mw, sequence_text)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    finally:
        if crop_path:
            os.unlink(crop_path)


# --- goal-directed figure resolution ----------------------------------------

def _verify_rank(verify: dict) -> int:
    """Trustworthiness rank: verified > unverified > contradiction > rejected."""
    return {"verified": 3, "unverified": 2, "contradiction": 1,
            "rejected": 0}.get(verify.get("verdict"), 0)


@app.post("/v1/harness/resolve-figure")
async def resolve_figure(
    file: UploadFile = File(...),
    sequence_text: str | None = Form(default=None),
    formula: str | None = Form(default=None),
    mw: float | None = Form(default=None),
):
    """Goal: the most trustworthy SMILES for this figure, with an evidence
    trail — or an explicit capability-boundary report.

    Strategy ladder: MolSight decode -> multi-signal verify; if the decode
    does not verify and a sequence anchor exists, rebuild deterministically
    from the sequence grammar and verify that; select the better-verified
    candidate. Contradictions are named, never silently resolved."""
    contents = await file.read()
    crop_path = None
    attempts: list[dict] = []
    try:
        image = _decode_upload(contents)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            crop_path = tmp.name

        # attempt 1: specialist decode — ensemble over image preparations,
        # deduplicated by canonical form, each candidate verified
        candidates: dict[str, dict] = {}
        for variant in _decode_variants(crop_path):
            smi = variant["smiles"]
            try:
                from rdkit import Chem, RDLogger
                RDLogger.DisableLog("rdApp.*")
                mol = Chem.MolFromSmiles(smi)
                key = Chem.MolToSmiles(mol) if mol else f"invalid:{len(smi)}"
            except Exception:
                key = f"invalid:{len(smi)}"
            if key not in candidates:
                candidates[key] = {"smiles": smi,
                                   "preparations": [variant["preparation"]],
                                   "valid": mol is not None}
            else:
                candidates[key]["preparations"].append(variant["preparation"])
        ranked = sorted(candidates.values(),
                        key=lambda c: (not c["valid"], -len(c["preparations"])))
        for cand in ranked[:4]:
            verify = _verify_structure_impl(crop_path, cand["smiles"],
                                            formula, mw, sequence_text)
            attempts.append({"method": f"molsight[{'+'.join(cand['preparations'])}]",
                             "smiles": cand["smiles"], "verify": verify})

        # attempt 2: deterministic rebuild from the sequence anchor. The
        # rebuild IS derived from the sequence text, so the self-agreement
        # signal is excluded from its verification — only RDKit validity,
        # declared anchors, ink geometry and the arbiter judge it.
        if sequence_text:
            from utils.sequence_parser import build_cyclopeptide_smiles, \
                parse_cyclopeptide_notation
            codes = parse_cyclopeptide_notation(sequence_text)
            built = build_cyclopeptide_smiles(codes) if codes else None
            if built:
                verify = _verify_structure_impl(crop_path, built,
                                                formula, mw, None)
                attempts.append({"method": "sequence_grammar", "smiles": built,
                                 "verify": verify})

        if not attempts:
            return {"goal": "trustworthy structure", "attempts": [],
                    "selected": None,
                    "capability_boundary": {"reason": "decode produced no SMILES and no sequence anchor was given"}}

        # ranking: verification verdict first; among equals, a candidate
        # that CAME from the sequence anchor (the patent's own notation)
        # outranks a decoded guess
        def _rank(a):
            base = _verify_rank(a["verify"])
            anchor_bonus = 1 if a["method"].startswith("sequence_grammar") else 0
            return (base, anchor_bonus)
        best = max(attempts, key=_rank)
        boundary = None
        if _verify_rank(best["verify"]) < 3:
            reasons = []
            for a in attempts:
                reasons.extend(a["verify"].get("contradictions") or
                               [f"{a['method']}: {a['verify'].get('verdict')}"])
            boundary = {"reason": "; ".join(reasons)[:500]}
        return {"goal": "trustworthy structure", "attempts": attempts,
                "selected": {"method": best["method"], "smiles": best["smiles"],
                             "verdict": best["verify"].get("verdict"),
                             "confidence": best["verify"].get("confidence")},
                "capability_boundary": boundary}
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    finally:
        if crop_path:
            os.unlink(crop_path)


def _route_crop(crop_path: str, smiles: str) -> dict:
    if not smiles:
        return {}
    try:
        from utils.structure_router import judge

        decision = judge("complete", smiles, crop_path)
        return {"kind": decision.kind, "backend": decision.backend,
                "reason": decision.reason, "signals": decision.signals}
    except Exception as exc:
        return {"error": str(exc)[:200]}


def _decode_variants(crop_path: str) -> list[dict]:
    """Decode one crop under several image preparations.

    Real patent figures differ from the training renders in aspect ratio,
    whitespace and resolution; no single preparation is canonical. Each
    variant is a cheap specialist decode; verification decides later."""
    image = cv2.imread(crop_path)
    if image is None:
        return []
    h, w = image.shape[:2]
    variants: list[tuple[str, np.ndarray]] = [("original", image)]

    # square pad (avoid aspect distortion in the 512 resize)
    side = max(h, w)
    square = np.full((side, side, 3), 255, dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    square[y0:y0 + h, x0:x0 + w] = image
    variants.append(("square_pad", square))

    # 2x upscale (sharper bonds after the model's 512 downsample)
    variants.append(("upscale2x", cv2.resize(image, (w * 2, h * 2),
                                             interpolation=cv2.INTER_CUBIC)))

    # white margin trim (figures with big borders resize the molecule smaller)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray < 235)
    if len(ys) > 0:
        ty0, ty1, tx0, tx1 = ys.min(), ys.max(), xs.min(), xs.max()
        if (ty1 - ty0) < h * 0.9 or (tx1 - tx0) < w * 0.9:
            pad = 12
            trimmed = image[max(0, ty0 - pad):min(h, ty1 + pad),
                            max(0, tx0 - pad):min(w, tx1 + pad)]
            if trimmed.size:
                variants.append(("trimmed", trimmed))

    out: list[dict] = []
    seen_names = set()
    for name, prep in variants:
        if prep.size == 0 or name in seen_names:
            continue
        seen_names.add(name)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, prep)
            try:
                result = _recognize_file(tmp.name)
            finally:
                os.unlink(tmp.name)
        if result.get("ok") and result.get("smiles"):
            out.append({"preparation": name, "smiles": result["smiles"]})
    return out
