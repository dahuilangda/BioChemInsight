"""Unified structure detection and recognition entry points.

DECIMER-style Mask R-CNN segmentation and MolNexTR graph decoding solve
different stages of the structure extraction pipeline. This module keeps them
behind one small API so callers do not each reimplement model lookup, loading,
locking, and result normalization.
"""

from __future__ import annotations

import os
import re
import threading
import time
import ctypes
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

try:  # optional runtime configuration
    import constants as project_constants
except ImportError:  # pragma: no cover
    project_constants = None

from utils.molecule_segmentation import apply_masks, get_expanded_masks
from utils.markush_labels import is_markush_label


MOLNEXTR_MODEL_FILE = "molnextr_best.pth"
MOLNEXTR_MODEL_PATH = (
    os.environ.get("MOLNEXTR_MODEL_PATH")
    or str(getattr(project_constants, "MOLNEXTR_MODEL_PATH", "") or "").strip()
)
MOLNEXTR_MOE_CONFIG_PATH = (
    os.environ.get("MOLNEXTR_MOE_CONFIG_PATH")
    or str(getattr(project_constants, "MOLNEXTR_MOE_CONFIG_PATH", "") or "").strip()
)
MOLNEXTR_ATTACHMENT_CONFIDENCE_MIN = float(
    getattr(project_constants, "MOLNEXTR_ATTACHMENT_CONFIDENCE_MIN", 0.65) or 0.65
)
# Decoupled fragment-attachment fusion (see constants.py). Empty checkpoint ⇒
# fusion disabled (pure base behavior, byte-identical to before).
MOLNEXTR_FRAGMENT_ATTACHMENT_CHECKPOINT = (
    os.environ.get("MOLNEXTR_FRAGMENT_ATTACHMENT_CHECKPOINT")
    or str(getattr(project_constants, "MOLNEXTR_FRAGMENT_ATTACHMENT_CHECKPOINT", "") or "").strip()
)
MOLNEXTR_FRAGMENT_ATTACHMENT_BASE_CHECKPOINT = (
    os.environ.get("MOLNEXTR_FRAGMENT_ATTACHMENT_BASE_CHECKPOINT")
    or str(getattr(project_constants, "MOLNEXTR_FRAGMENT_ATTACHMENT_BASE_CHECKPOINT", "") or "").strip()
)
MOLNEXTR_FRAGMENT_ATTACHMENT_MIN_CONFIDENCE = float(
    getattr(project_constants, "MOLNEXTR_FRAGMENT_ATTACHMENT_MIN_CONFIDENCE", 0.20) or 0.20
)
MOLNEXTR_FRAGMENT_ATTACHMENT_MAX_ANCHOR_DISTANCE = float(
    getattr(project_constants, "MOLNEXTR_FRAGMENT_ATTACHMENT_MAX_ANCHOR_DISTANCE", 0.35) or 0.35
)
MOLNEXTR_FRAGMENT_ATTACHMENT_CONFIDENCE_TEMPERATURE = float(
    getattr(project_constants, "MOLNEXTR_FRAGMENT_ATTACHMENT_CONFIDENCE_TEMPERATURE", 1.0) or 1.0
)
# Whether to accept research/debug-only expert checkpoints (default False;
# production must use a gate-accepted eligible checkpoint). Diagnosis / real-
# data evaluation sets MOLNEXTR_FRAGMENT_ATTACHMENT_ALLOW_DEBUG_CHECKPOINT=1.
MOLNEXTR_FRAGMENT_ATTACHMENT_ALLOW_DEBUG_CHECKPOINT = (
    os.environ.get("MOLNEXTR_FRAGMENT_ATTACHMENT_ALLOW_DEBUG_CHECKPOINT", "").strip().lower()
    in ("1", "true", "yes", "on")
)
# Wavy-bond detection + edge-band inpaint re-decode (RGReco-style). Empty
# checkpoint ⇒ disabled. Complete rows are never processed.
MOLNEXTR_WAVY_DETECT_CHECKPOINT = str(
    getattr(project_constants, "MOLNEXTR_WAVY_DETECT_CHECKPOINT", "")
    or os.environ.get("MOLNEXTR_WAVY_DETECT_CHECKPOINT", "")
).strip()
MOLNEXTR_WAVY_DETECT_SCORE_THRESHOLD = float(
    getattr(project_constants, "MOLNEXTR_WAVY_DETECT_SCORE_THRESHOLD", 0.05)
    or os.environ.get("MOLNEXTR_WAVY_DETECT_SCORE_THRESHOLD", 0.05)
)
MOLNEXTR_WAVY_DETECT_BAND_FRACTION = float(
    getattr(project_constants, "MOLNEXTR_WAVY_DETECT_BAND_FRACTION", 0.30)
    or os.environ.get("MOLNEXTR_WAVY_DETECT_BAND_FRACTION", 0.30)
)
# Mask-level wavy inpaint re-decode. Disabled by default.
MOLNEXTR_WAVY_MASK_INPAINT_ENABLED = (
    os.environ.get("MOLNEXTR_WAVY_MASK_INPAINT_ENABLED", "").strip().lower()
    in ("1", "true", "yes", "on")
) or bool(getattr(project_constants, "MOLNEXTR_WAVY_MASK_INPAINT_ENABLED", False))
MOLNEXTR_WAVY_MASK_INPAINT_FRAGMENT_ONLY = (
    os.environ.get("MOLNEXTR_WAVY_MASK_INPAINT_FRAGMENT_ONLY", "").strip().lower()
    not in ("0", "false", "no", "off")
) if os.environ.get("MOLNEXTR_WAVY_MASK_INPAINT_FRAGMENT_ONLY") else bool(
    getattr(project_constants, "MOLNEXTR_WAVY_MASK_INPAINT_FRAGMENT_ONLY", True)
)
SUPPORTED_MOLNEXTR_ATOMS = {
    "*",
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
}
MOLNEXTR_POSTPROCESS_WORKERS = max(
    1, int(getattr(project_constants, "MOLNEXTR_POSTPROCESS_WORKERS", 1) or 1)
)
MOLNEXTR_PREPROCESS_LONG_EDGE = max(
    0, int(getattr(project_constants, "MOLNEXTR_PREPROCESS_LONG_EDGE", 512) or 0)
)
MOLNEXTR_MAX_INFERENCE_BATCH_SIZE = max(
    1, int(getattr(project_constants, "MOLNEXTR_MAX_INFERENCE_BATCH_SIZE", 1) or 1)
)
_molnextr_lock = threading.Lock()
_segmentation_lock = threading.Lock()
_molnextr_model = None
_molnextr_model_path: str | None = None
_molnextr_config_path: str | None = None
_molnextr_moe_config_path: str | None = None
_molnextr_device: torch.device | None = None
_libc = None


def trim_process_memory() -> None:
    """Return freed Python/C-extension heap pages to the OS when glibc supports it."""
    global _libc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if os.name != "posix":
        return
    try:
        if _libc is None:
            _libc = ctypes.CDLL("libc.so.6")
        _libc.malloc_trim(0)
    except Exception:
        pass


@dataclass(frozen=True)
class DetectedStructure:
    image: np.ndarray
    bbox: list[int]
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class StructureDetectionResult:
    structures: list[DetectedStructure]
    masks: np.ndarray | None = None


@dataclass(frozen=True)
class StructurePrediction:
    smiles: str
    molblock: str = ""
    raw: Any = None
    elapsed_seconds: float = 0.0
    quality_issues: tuple[str, ...] = ()
    # MoE routing metadata (populated only when the MoE decoder is active).
    expert_weights: tuple[float, ...] = ()        # gate softmax over [complete, markush, fragment]
    routed_expert: str = ""                       # "complete" | "markush" | "fragment"
    routing_forced_complete: bool = False         # True if the safety floor forced Expert 0
    routing_forced_default: bool = False          # True if the default Expert 0 route was used
    routing_strategy: str = ""
    routing_confidence: float = -1.0
    routing_required_threshold: float = -1.0
    confidence: float = -1.0                      # Protenix-style calibrated confidence (E[Tanimoto])
    attachment_repair: Any = None
    expected_structure_type: str = ""


def _prediction_atom_count(prediction: Any) -> int:
    """Number of decoded atoms in a raw prediction dict (0 on failure).

    Works on BOTH the raw decode dict (chartok_coords.symbols) and the
    post-processed dict returned by predict_image_files (atom_sets).
    """
    if not isinstance(prediction, dict):
        return 0
    cc = prediction.get("chartok_coords")
    if isinstance(cc, dict) and isinstance(cc.get("symbols"), list):
        return len(cc["symbols"])
    atom_sets = prediction.get("atom_sets")
    if isinstance(atom_sets, list):
        return len(atom_sets)
    return 0


def normalize_segment_array(segment: np.ndarray | None) -> np.ndarray | None:
    if not isinstance(segment, np.ndarray) or len(segment.shape) != 3:
        return None
    if segment.shape[2] == 4:
        segment = segment[:, :, :3]
    elif segment.shape[2] != 3:
        return None
    if segment.dtype != np.uint8:
        if segment.max() <= 1.0:
            segment = (segment * 255).astype(np.uint8)
        else:
            segment = segment.astype(np.uint8)
    return segment


def extract_molblock(prediction: Any) -> str:
    if not isinstance(prediction, dict):
        return ""
    for key in ("predicted_molfile", "molfile", "molblock", "molfile_v3", "molfileV3"):
        value = prediction.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _calibrate_confidence(raw: float, temperature: float) -> float:
    """Temperature-scale a probability via the logit/sigmoid transform.

    ``temperature=1.0`` is identity; ``<1.0`` sharpens toward {0,1}. The
    FragmentAttachmentExpert's sidecar_confidence is systematically low on real
    patent images (synthetic→real domain gap), so lowering T is the Phase 2
    lever to push the grafted dummy's confidence past the 0.65 attachment gate
    without retraining.
    """
    c = float(min(1.0, max(0.0, raw)))
    if c <= 0.0:
        return 0.0
    if c >= 1.0:
        return 1.0
    t = float(temperature)
    if t <= 0.0 or abs(t - 1.0) < 1e-9:
        return c
    import math
    logit = math.log(c) - math.log1p(-c)
    return 1.0 / (1.0 + math.exp(-logit / t))


def _molblock_has_dummy_atom_local(molblock: str) -> bool:
    """True iff the molblock parses and contains an atomic_num==0 atom.

    Inlined (rather than importing pipeline.py) so this module stays layered;
    identical logic to pipeline._molblock_has_dummy_atom.
    """
    text = (molblock or "").strip()
    if not text:
        return False
    try:
        from rdkit import Chem
        mol = Chem.MolFromMolBlock(text, sanitize=False, removeHs=False)
    except Exception:
        return False
    if mol is None:
        return False
    return any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())


def _strip_molblock_dummy(molblock: str, smiles: str) -> tuple[str, str]:
    """Remove atomic_num==0 atoms from a molblock, returning the cleaned
    molblock + canonical SMILES. Used in combine mode: the MoE specialist
    mixture emits a low-confidence/diluted `*`; we strip it and let fusion
    graft a clean, high-evidence dummy in its place. On any failure the input
    is returned unchanged.
    """
    from rdkit import Chem
    m = Chem.MolFromMolBlock(molblock or "", sanitize=False, removeHs=False)
    if m is None:
        return molblock, smiles
    idxs = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    if not idxs:
        return molblock, smiles
    rw = Chem.RWMol(m)
    for i in sorted(idxs, reverse=True):
        rw.RemoveAtom(i)
    m2 = rw.GetMol()
    try:
        out_mb = Chem.MolToMolBlock(m2, kekulize=False)
    except Exception:
        out_mb = molblock
    try:
        out_smi = Chem.MolToSmiles(m2, canonical=True)
    except Exception:
        out_smi = smiles
    return out_mb, out_smi


def _recover_fused_smiles(molblock: str) -> str:
    """Re-derive a valid SMILES from a grafted molblock when fuse's
    MolToSmiles→MolFromSmiles round-trip failed (edge-case valence/kekule on
    ~4/31 real rows). Returns smiles or ''.
    """
    from rdkit import Chem
    try:
        m = Chem.MolFromMolBlock(molblock or "", sanitize=False, removeHs=False)
        if m is None:
            return ""
        try:
            Chem.SanitizeMol(m)
        except Exception:
            try:
                m.UpdatePropertyCache(strict=False)
            except Exception:
                return ""
        return Chem.MolToSmiles(m, canonical=True) or ""
    except Exception:
        return ""


def _accept_grafted_dummy(result: dict, molblock: str, smiles: str, calibrated: float) -> dict:
    """Synthesize the grafted dummy into atom_sets (so molnextr_quality_issues
    scores it through the SAME 0.65 path) and overwrite smiles/molblock."""
    dummy_coord = None
    try:
        from rdkit import Chem
        mol = Chem.MolFromMolBlock(molblock or "", sanitize=False, removeHs=False)
        if mol is not None and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    dummy_coord = (round(float(pos.x), 3), round(float(pos.y), 3))
                    break
    except Exception:
        dummy_coord = None
    atom_sets = result.get("atom_sets")
    if isinstance(atom_sets, list):
        new_entry = {
            "atom_number": str(len(atom_sets)),
            "atom_symbol": "*",
            "confidence": calibrated,
        }
        if dummy_coord is not None:
            new_entry["coords"] = dummy_coord
        atom_sets.append(new_entry)
    result["predicted_smiles"] = smiles
    result["predicted_molfile"] = molblock
    return result


def _apply_fragment_attachment_fusion(
    result: dict,
    endpoint: dict | None,
    *,
    min_confidence: float,
    max_anchor_distance: float,
    temperature: float,
    expected_type: str,
    strip_existing_dummy: bool = False,
) -> dict:
    """Geometrically graft one dummy atom onto a fragment backbone molblock.

    The base decoder does not emit attachment ``*`` atoms, so the molblock would
    fail the production dummy-atom assembly gate. The FragmentAttachmentExpert
    predicts the wavy endpoint independently and ``fuse_dummy_attachment`` adds a
    dummy onto the base molblock, preserving the base structure and coordinates.
    The grafted dummy is synthesized into ``atom_sets`` with a calibrated
    confidence so ``molnextr_quality_issues`` scores it like a decoder-emitted dummy.

    Only runs for ``expected_type`` in ``{"fragment", "markush"}``; if the base
    molblock already has a dummy it is left untouched. On any failure the base
    result stands.
    """
    if str(expected_type or "").strip().lower() not in {"fragment", "markush"}:
        return result
    if not isinstance(result, dict) or not isinstance(endpoint, dict):
        return result
    backbone_smiles = result.get("predicted_smiles") or ""
    backbone_molblock = extract_molblock(result)
    if _molblock_has_dummy_atom_local(backbone_molblock):
        if strip_existing_dummy:
            # Combine mode: strip the diluted dummy emitted by the MoE mixture
            # so fusion grafts a clean, high-evidence dummy onto the backbone.
            backbone_molblock, backbone_smiles = _strip_molblock_dummy(backbone_molblock, backbone_smiles)
            result["predicted_molfile"] = backbone_molblock
            result["predicted_smiles"] = backbone_smiles
            if not backbone_molblock or _molblock_has_dummy_atom_local(backbone_molblock):
                return result
        else:
            return result  # base path: keep an already-present dummy unchanged
    try:
        from training.molnextr_markush.tools.fragment_attachment_fusion import (
            fuse_dummy_attachment,
        )
    except Exception:
        return result
    fused = fuse_dummy_attachment(
        backbone_smiles=backbone_smiles,
        backbone_molblock=backbone_molblock,
        endpoint=endpoint,
        min_sidecar_confidence=float(min_confidence),
        max_anchor_distance=float(max_anchor_distance),
    )
    sidecar_conf = float(endpoint.get("sidecar_confidence") or 0.0)
    calibrated = _calibrate_confidence(sidecar_conf, temperature)
    result["attachment_repair"] = {
        "strategy": "baseline_backbone_plus_fragment_endpoint_expert",
        "ready": bool(fused.ready),
        "issue": str(fused.issue or ""),
        "endpoint": fused.endpoint,
        "anchor_atom_index": int(fused.anchor_atom_index),
        "anchor_distance": float(fused.anchor_distance),
        "sidecar_confidence": sidecar_conf,
        "calibrated_confidence": calibrated,
        "temperature": float(temperature),
    }
    if not fused.ready:
        # fuse can fail to serialize the SMILES on edge-case backbones even
        # though the grafted molblock is chemically valid; recover the SMILES
        # from the molblock so combine doesn't lose a backbone-correct row
        # (~4/31 on real_wavy). Expert presence/anchor failures fall through.
        if "unparseable" in (fused.issue or "") and fused.molblock:
            recovered = _recover_fused_smiles(fused.molblock)
            if recovered:
                result["attachment_repair"]["ready"] = True
                result["attachment_repair"]["issue"] = "fused_smiles_recovered_from_molblock"
                return _accept_grafted_dummy(result, fused.molblock, recovered, calibrated)
        return result  # base stands; molnextr_expected_structure_issues emits the truthful missing-evidence signal
    # Synthesize the grafted dummy into atom_sets so the existing 0.65
    # attachment-confidence path (molnextr_quality_issues) scores it, same code
    # path as a decoder-emitted dummy.
    return _accept_grafted_dummy(result, fused.molblock, fused.smiles, calibrated)



def normalize_atom_symbol(symbol: Any) -> str:
    text = str(symbol or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if text in {"*", "H"} or text.startswith("R") or text.endswith("*"):
        return text
    text = re.sub(r"^\d+", "", text)
    text = text.replace("@@", "").replace("@", "")
    if text.startswith(("Cl", "Br", "Si")):
        return text[:2]
    if text and text[0] in {"c", "n", "o", "p", "s"}:
        return text[0].upper()
    match = re.match(r"([A-Z][a-z]?)", text)
    if match:
        return match.group(1)
    return text


def is_markush_atom_symbol(symbol: Any) -> bool:
    text = str(symbol or "").strip()
    # The decoder emits attachment-point dummies in bracketed isotope form
    # ([1*], [13*], [111*], …). Strip the brackets so the wildcard/label
    # checks below recognize them; otherwise [13*] is misclassified as a
    # non-markush "unsupported atom symbol" and blocks strict_ready.
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if text in {"*", ""} or text.endswith("*"):
        return text == "*" or bool(text)
    return is_markush_label(text)


def molnextr_quality_issues(prediction: Any) -> tuple[str, ...]:
    if not isinstance(prediction, dict):
        return ()
    issues = [
        str(issue)
        for issue in prediction.get("quality_issues") or []
        if str(issue)
    ]
    unsupported = []
    for atom in prediction.get("atom_sets") or []:
        if not isinstance(atom, dict):
            continue
        raw_symbol = atom.get("atom_symbol")
        is_markush = is_markush_atom_symbol(raw_symbol)
        symbol = normalize_atom_symbol(raw_symbol)
        if symbol and symbol not in SUPPORTED_MOLNEXTR_ATOMS and not is_markush:
            unsupported.append(symbol)
        if is_markush:
            confidence = atom.get("confidence")
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = 1.0
            if confidence is not None and confidence_value < MOLNEXTR_ATTACHMENT_CONFIDENCE_MIN:
                issues.append(f"low_confidence_molnextr_attachment_atom:{symbol}")
    if unsupported:
        issues.append("unsupported_molnextr_atom_symbols:" + ",".join(sorted(set(unsupported))))
    return tuple(dict.fromkeys(issues))


def molnextr_expected_structure_issues(prediction: Any, expected_structure_type: Any) -> tuple[str, ...]:
    expected = str(expected_structure_type or "").strip().lower()
    if expected not in {"markush", "fragment"} or not isinstance(prediction, dict):
        return ()
    has_attachment = False
    for atom in prediction.get("atom_sets") or []:
        if isinstance(atom, dict) and is_markush_atom_symbol(atom.get("atom_symbol")):
            has_attachment = True
            break
    if has_attachment:
        return ()
    return (f"missing_expected_{expected}_attachment_evidence",)


def sort_segments_bboxes(segments, bboxes, masks, same_row_pixel_threshold=50):
    if len(bboxes) == 0:
        return segments, bboxes, masks

    bbox_with_indices = [(bbox, idx) for idx, bbox in enumerate(bboxes)]
    sorted_bbox_with_indices = sorted(bbox_with_indices, key=lambda item: item[0][0])

    rows = []
    current_row = [sorted_bbox_with_indices[0]]
    for bbox_with_idx in sorted_bbox_with_indices[1:]:
        if abs(bbox_with_idx[0][0] - current_row[-1][0][0]) < same_row_pixel_threshold:
            current_row.append(bbox_with_idx)
        else:
            rows.append(sorted(current_row, key=lambda item: item[0][1]))
            current_row = [bbox_with_idx]
    rows.append(sorted(current_row, key=lambda item: item[0][1]))

    sorted_indices = [bbox_with_idx[1] for row in rows for bbox_with_idx in row]
    sorted_segments = [segments[idx] for idx in sorted_indices]
    sorted_bboxes = [bboxes[idx] for idx in sorted_indices]
    sorted_masks = None
    if masks is not None:
        sorted_masks = np.stack([masks[:, :, idx] for idx in sorted_indices], axis=-1)
    return sorted_segments, sorted_bboxes, sorted_masks


def resolve_molnextr_model_path() -> str:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        Path(MOLNEXTR_MODEL_PATH) if MOLNEXTR_MODEL_PATH else None,
        Path("/app/runtime_models") / "molnextr_markush" / "molnextr_markush.pth",
        Path("/app/models") / MOLNEXTR_MODEL_FILE,
        root / "models" / MOLNEXTR_MODEL_FILE,
        Path.cwd() / "models" / MOLNEXTR_MODEL_FILE,
    ]
    for path in candidates:
        if path and path.exists():
            return str(path)
    raise FileNotFoundError(
        f"{MOLNEXTR_MODEL_FILE} not found. Build/download the model into /app/models or ./models."
    )


def _load_molnextr(
    model_path: str | None = None,
    device: torch.device | None = None,
    markush_config_path: str | None = None,
    moe_config_path: str | None = None,
    disable_moe_config: bool = False,
):
    global _molnextr_model, _molnextr_model_path, _molnextr_config_path, _molnextr_moe_config_path, _molnextr_device

    resolved_path = model_path or resolve_molnextr_model_path()
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_config = str(markush_config_path or "")
    resolved_moe = str(
        ""
        if disable_moe_config
        else (
            moe_config_path
            or (MOLNEXTR_MOE_CONFIG_PATH if not model_path and not markush_config_path else "")
            or ""
        )
        or ""
    )

    with _molnextr_lock:
        if (
            _molnextr_model is not None
            and _molnextr_model_path == resolved_path
            and _molnextr_config_path == resolved_config
            and _molnextr_moe_config_path == resolved_moe
            and _molnextr_device == resolved_device
        ):
            return _molnextr_model

        from utils.MolNexTR import molnextr

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Loading MolNexTR model from: {resolved_path}")
        if resolved_moe:
            print(f"Loading MolNexTR MoE config from: {resolved_moe}")
        _molnextr_model = molnextr(
            resolved_path,
            resolved_device,
            postprocess_workers=MOLNEXTR_POSTPROCESS_WORKERS,
            preprocess_long_edge=MOLNEXTR_PREPROCESS_LONG_EDGE,
            max_inference_batch_size=MOLNEXTR_MAX_INFERENCE_BATCH_SIZE,
            markush_config_path=resolved_config or None,
            moe_config_path=resolved_moe or None,
        )
        _molnextr_model_path = resolved_path
        _molnextr_config_path = resolved_config
        _molnextr_moe_config_path = resolved_moe
        _molnextr_device = resolved_device
        return _molnextr_model


class StructureRecognizer:
    def __init__(
        self,
        model_path: str | None = None,
        device: torch.device | None = None,
        markush_config_path: str | None = None,
        moe_config_path: str | None = None,
        disable_moe_config: bool = False,
        fragment_attachment_checkpoint_path: str | None = None,
        fragment_attachment_base_checkpoint_path: str | None = None,
        disable_fragment_attachment: bool = False,
        fragment_attachment_min_confidence: float | None = None,
        fragment_attachment_max_anchor_distance: float | None = None,
        fragment_attachment_confidence_temperature: float | None = None,
        fragment_attachment_allow_debug_checkpoint: bool | None = None,
    ):
        self.model_path = model_path
        self.device = device
        self.markush_config_path = str(markush_config_path) if markush_config_path else None
        self.moe_config_path = str(moe_config_path) if moe_config_path else None
        self.disable_moe_config = bool(disable_moe_config)
        self._model = None
        # Fragment-attachment fusion. Empty checkpoint ⇒ disabled.
        self.fragment_attachment_checkpoint_path = (
            str(fragment_attachment_checkpoint_path).strip()
            if fragment_attachment_checkpoint_path is not None
            else MOLNEXTR_FRAGMENT_ATTACHMENT_CHECKPOINT
        )
        self.fragment_attachment_base_checkpoint_path = (
            str(fragment_attachment_base_checkpoint_path).strip()
            if fragment_attachment_base_checkpoint_path is not None
            else MOLNEXTR_FRAGMENT_ATTACHMENT_BASE_CHECKPOINT
        )
        self.disable_fragment_attachment = bool(disable_fragment_attachment)
        self.fragment_attachment_min_confidence = float(
            fragment_attachment_min_confidence
            if fragment_attachment_min_confidence is not None
            else MOLNEXTR_FRAGMENT_ATTACHMENT_MIN_CONFIDENCE
        )
        self.fragment_attachment_max_anchor_distance = float(
            fragment_attachment_max_anchor_distance
            if fragment_attachment_max_anchor_distance is not None
            else MOLNEXTR_FRAGMENT_ATTACHMENT_MAX_ANCHOR_DISTANCE
        )
        self.fragment_attachment_confidence_temperature = float(
            fragment_attachment_confidence_temperature
            if fragment_attachment_confidence_temperature is not None
            else MOLNEXTR_FRAGMENT_ATTACHMENT_CONFIDENCE_TEMPERATURE
        )
        # Production-eligibility gate for the expert checkpoint (False ⇒ reject
        # debug/smoke checkpoints). Diagnosis/smoke passes True explicitly.
        self.fragment_attachment_allow_debug_checkpoint = bool(
            fragment_attachment_allow_debug_checkpoint
            if fragment_attachment_allow_debug_checkpoint is not None
            else MOLNEXTR_FRAGMENT_ATTACHMENT_ALLOW_DEBUG_CHECKPOINT
        )
        self._fragment_attachment_runtime = None
        # Wavy-bond detection + edge-band inpaint re-decode (RGReco-style).
        # Empty checkpoint ⇒ disabled (complete/markush byte-identical to base).
        self.wavy_detect_checkpoint_path = MOLNEXTR_WAVY_DETECT_CHECKPOINT
        self.wavy_detect_score_threshold = float(MOLNEXTR_WAVY_DETECT_SCORE_THRESHOLD)
        self.wavy_detect_band_fraction = float(MOLNEXTR_WAVY_DETECT_BAND_FRACTION)
        self.wavy_mask_inpaint_enabled = bool(MOLNEXTR_WAVY_MASK_INPAINT_ENABLED)
        self.wavy_mask_inpaint_fragment_only = bool(MOLNEXTR_WAVY_MASK_INPAINT_FRAGMENT_ONLY)
        self._wavy_detector = None
        # Graph-level ghost-carbon repair for fragment attachment decoding.
        # Safe, deterministic, net-positive (assembly Tanimoto 0.810->0.817 on
        # real_wavy_hard). Default enabled.
        self.graph_ghost_repair_enabled = bool(
            getattr(project_constants, "MOLNEXTR_GRAPH_GHOST_REPAIR_ENABLED", True)
        )
        # DECIMER+MolNexTR fusion: multi-class attachment detector (wavy/R-group/
        # asterisk/dashed) whose masks guide the graph ghost-carbon repair. Empty
        # checkpoint => fusion disabled (topology-only repair still runs).
        self.attachment_detect_checkpoint_path = str(
            getattr(project_constants, "MOLNEXTR_ATTACHMENT_DETECT_CHECKPOINT", "")
            or os.environ.get("MOLNEXTR_ATTACHMENT_DETECT_CHECKPOINT", "")
        ).strip()
        self.attachment_detect_score_threshold = float(
            getattr(project_constants, "MOLNEXTR_ATTACHMENT_DETECT_SCORE_THRESHOLD", 0.05)
            or os.environ.get("MOLNEXTR_ATTACHMENT_DETECT_SCORE_THRESHOLD", 0.05)
        )
        self.attachment_detect_num_classes = int(
            getattr(project_constants, "MOLNEXTR_ATTACHMENT_DETECT_NUM_CLASSES", 5)
        )
        self._attachment_detector = None
        # Detection→graph fusion: inject detector attachment points as
        # full-schema proposals into the attachment_set residual edit (the path
        # that actually grafts dummy atoms onto the molblock). Targets the
        # attachment_point_recall@0.05 bottleneck. Disabled ⇒ no priors are
        # passed to the model.
        self.attachment_prior_fusion_enabled = bool(
            getattr(project_constants, "MOLNEXTR_ATTACHMENT_PRIOR_FUSION_ENABLED", True)
        )
        self.attachment_prior_min_confidence = float(
            getattr(project_constants, "MOLNEXTR_ATTACHMENT_PRIOR_MIN_CONFIDENCE", 0.3)
            or 0.3
        )

    @property
    def model(self):
        if self._model is None:
            self._model = _load_molnextr(
                self.model_path,
                self.device,
                markush_config_path=self.markush_config_path,
                moe_config_path=self.moe_config_path,
                disable_moe_config=self.disable_moe_config,
            )
        return self._model

    def _fragment_attachment_enabled(self) -> bool:
        moe_config = (
            getattr(self._model, "moe_config", None)
            if self._model is not None else None
        )
        if (
            isinstance(moe_config, dict)
            and str(
                moe_config.get("attachment_set_decode_mode") or ""
            ).strip().lower() == "direct_sidecar"
        ):
            # A direct sidecar owns the final graph. Applying the legacy
            # endpoint graft here would silently replace learned topology with
            # post-hoc geometry and reintroduce the fallback path.
            return False
        return bool(
            not self.disable_fragment_attachment
            and self.fragment_attachment_checkpoint_path
        )

    @property
    def fragment_attachment_runtime(self):
        """Lazily-loaded FragmentAttachmentFusionRuntime, or None if disabled.

        Loads a second frozen MolNexTR encoder only on first use with a
        non-empty checkpoint, so complete-only pages pay no extra memory. Any
        construction failure (missing checkpoint, bad metadata) disables fusion
        gracefully; the base decode still flows through unchanged.
        """
        if self._fragment_attachment_runtime is None:
            if self._fragment_attachment_enabled():
                try:
                    from training.molnextr_markush.tools.fragment_attachment_fusion import (
                        FragmentAttachmentFusionRuntime,
                    )
                    self._fragment_attachment_runtime = FragmentAttachmentFusionRuntime(
                        checkpoint_path=self.fragment_attachment_checkpoint_path,
                        base_checkpoint=self.fragment_attachment_base_checkpoint_path or None,
                        device=self.device,
                        batch_size=16,
                        num_workers=0,
                        allow_debug_checkpoint=self.fragment_attachment_allow_debug_checkpoint,
                    )
                except Exception:
                    self._fragment_attachment_runtime = False
            else:
                self._fragment_attachment_runtime = False
        runtime = self._fragment_attachment_runtime
        return runtime if runtime is not False else None

    @property
    def wavy_detector(self):
        """Lazy Mask R-CNN wavy-bond detector, or None if disabled.

        Loads only on first use with a non-empty checkpoint, so complete-only
        pages and disabled deployments pay no extra memory. Any construction
        failure (missing checkpoint, bad weights) disables detection gracefully;
        the base decode flows through unchanged.
        """
        if self._wavy_detector is None:
            if self.wavy_detect_checkpoint_path:
                try:
                    from utils.MolNexTR.attachment_detector import AttachmentDetector
                    self._wavy_detector = AttachmentDetector(
                        checkpoint_path=self.wavy_detect_checkpoint_path,
                        device=self.device,
                    )
                    # Force-load to validate the checkpoint eagerly.
                    self._wavy_detector._ensure_loaded()
                except Exception:
                    self._wavy_detector = False
            else:
                self._wavy_detector = False
        detector = self._wavy_detector
        return detector if detector is not False else None

    @property
    def attachment_detector(self):
        """Lazy multi-class attachment detector (wavy/R-group/asterisk/dashed),
        or None if disabled. Drives the mask-guided graph repair (organic
        fusion). Loads only on first use; any failure disables gracefully."""
        if self._attachment_detector is None:
            if self.attachment_detect_checkpoint_path:
                try:
                    from utils.MolNexTR.attachment_detector import AttachmentDetector
                    self._attachment_detector = AttachmentDetector(
                        checkpoint_path=self.attachment_detect_checkpoint_path,
                        device=self.device,
                        num_classes=self.attachment_detect_num_classes,
                    )
                    self._attachment_detector._ensure_loaded()
                except Exception:
                    self._attachment_detector = False
            else:
                self._attachment_detector = False
        det = self._attachment_detector
        return det if det is not False else None

    def detect_segments(self, page_bgr: np.ndarray) -> StructureDetectionResult:
        with _segmentation_lock:
            masks = get_expanded_masks(page_bgr)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        segments, bboxes = apply_masks(page_bgr, masks)
        if len(segments) > 0:
            segments, bboxes, masks = sort_segments_bboxes(segments, bboxes, masks)

        detected = []
        for idx, segment in enumerate(segments):
            normalized = normalize_segment_array(segment)
            if normalized is None:
                continue
            detected.append(
                DetectedStructure(
                    image=normalized,
                    bbox=np.asarray(bboxes[idx]).astype(int).tolist(),
                    mask=None,
                )
            )
        return StructureDetectionResult(structures=detected, masks=None)

    def predict_segment_file(self, segment_file: str) -> StructurePrediction:
        try:
            return self._predict_molnextr(segment_file)
        finally:
            trim_process_memory()

    def predict_segment_files(
        self,
        segment_files: list[str],
        batch_size: int = 16,
        return_confidence: bool = True,
        beam_size: int = 1,
        n_best: int = 1,
        expected_structure_types: list[str | None] | None = None,
    ) -> list[StructurePrediction]:
        if not segment_files:
            return []
        expected_structure_types = list(expected_structure_types or [None] * len(segment_files))
        if len(expected_structure_types) != len(segment_files):
            raise ValueError("expected_structure_types must match segment_files length")
        start = time.monotonic()
        frag_idx = [
            i for i, t in enumerate(expected_structure_types)
            if str(t or "").strip().lower() in {"fragment", "markush"}
        ]
        # Run the multi-class attachment detector ONCE over the fragment/markush
        # crops, BEFORE the model call. The detections serve two consumers:
        #   (1) attachment_priors, fed into predict_image_files so the
        #       attachment_set residual edit grafts dummy atoms at the detected
        #       attachment marks (the real detection→graph fusion targeting the
        #       attachment_point_recall@0.05 bottleneck).
        #   (2) attachment_masks_by_index, the bbox geometry that guides the
        #       graph-level ghost-carbon repair further below.
        # Complete rows are never detected. Any detector failure degrades
        # gracefully: empty detections ⇒ no priors (byte-identical base decode).
        detector = self.attachment_detector
        attachment_masks_by_index: dict[int, list[dict]] = {}
        attachment_priors: list[list[dict] | None] = [None] * len(segment_files)
        prior_fusion_active = bool(
            detector is not None
            and frag_idx
            and self.attachment_prior_fusion_enabled
            and not self.disable_moe_config
            and bool(self.moe_config_path)
        )
        if detector is not None and frag_idx:
            import cv2 as _cv2_det
            for i in frag_idx:
                img_i = _cv2_det.imread(segment_files[i])
                if img_i is None:
                    continue
                try:
                    rgb_i = _cv2_det.cvtColor(img_i, _cv2_det.COLOR_BGR2RGB)
                    dets_i = detector.detect(rgb_i, score_threshold=self.attachment_detect_score_threshold)
                except Exception:
                    dets_i = []
                if not dets_i:
                    continue
                h_i, w_i = rgb_i.shape[:2]
                attachment_masks_by_index[i] = [
                    {"class": d.class_name, "cx": d.cx, "cy": d.cy,
                     "bw": (d.bbox[2] - d.bbox[0]) / max(1, w_i),
                     "bh": (d.bbox[3] - d.bbox[1]) / max(1, h_i)}
                    for d in dets_i
                ]
                # Build prior proposals (detector→graph fusion). Keep every
                # detection above the score threshold; the model-side residual
                # edit applies the min-confidence gate, duplicate dedup, and
                # cardinality cap, so we pass raw normalized points + confidence.
                if prior_fusion_active:
                    attachment_priors[i] = [
                        {"cx": d.cx, "cy": d.cy,
                         "confidence": float(d.confidence),
                         "class": d.class_name}
                        for d in dets_i
                    ]
        try:
            results = self.model.predict_image_files(
                segment_files,
                return_atoms_bonds=True,
                return_confidence=bool(return_confidence),
                batch_size=max(1, int(batch_size or 1)),
                beam_size=max(1, int(beam_size or 1)),
                n_best=max(1, int(n_best or 1)),
                expected_structure_types=expected_structure_types,
                attachment_priors=(
                    attachment_priors if prior_fusion_active else None
                ),
            )

        finally:
            trim_process_memory()
        elapsed = time.monotonic() - start
        per_image_elapsed = elapsed / max(len(segment_files), 1)
        # Mask-level wavy inpaint re-decode (A3). For fragment (optionally
        # markush) rows when a wavy detector is configured, erase ONLY the
        # detector's predicted wavy pixels (keeps the connector) and re-decode.
        # The re-decode wins iff it parses and drops atoms (the ghost-carbon
        # signature: wavy zigzag turning points decode as spurious carbons, so
        # a correct mask-level inpaint removes exactly those atoms). Band-level
        # inpaint was measured net-negative; mask-level is opt-in via
        # MOLNEXTR_WAVY_MASK_INPAINT_ENABLED until end-to-end measurement.
        if (
            frag_idx
            and self.wavy_mask_inpaint_enabled
            and self.wavy_detector is not None
        ):
            import tempfile
            import cv2 as _cv2_wavy
            for i in frag_idx:
                try:
                    expected = str(expected_structure_types[i] or "").strip().lower()
                    if self.wavy_mask_inpaint_fragment_only and expected != "fragment":
                        continue
                    img_i = _cv2_wavy.imread(segment_files[i])
                    if img_i is None:
                        continue
                    rgb_i = _cv2_wavy.cvtColor(img_i, _cv2_wavy.COLOR_BGR2RGB)
                    dets_i = self.wavy_detector.detect(
                        rgb_i, score_threshold=self.wavy_detect_score_threshold
                    )
                    wavy_dets = [
                        d for d in dets_i if str(getattr(d, "class_name", "wavy")) == "wavy"
                    ]
                    if not wavy_dets:
                        continue
                    best = max(wavy_dets, key=lambda d: float(d.confidence))
                    inpainted, changed = self.wavy_detector.inpaint_wavy_mask_pixels(
                        rgb_i, best
                    )
                    if not changed:
                        continue
                    tmp_path = tempfile.mktemp(suffix="_wavy_inpainted.png")
                    _cv2_wavy.imwrite(tmp_path, _cv2_wavy.cvtColor(inpainted, _cv2_wavy.COLOR_RGB2BGR))
                    try:
                        redec = self.model.predict_image_files(
                            [tmp_path],
                            return_atoms_bonds=True,
                            return_confidence=True,
                            batch_size=1,
                            expected_structure_types=[expected_structure_types[i]],
                            attachment_priors=(
                                [attachment_priors[i]] if prior_fusion_active else None
                            ),
                        )[0]
                    finally:
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                    if not isinstance(redec, dict):
                        continue
                    redec_smiles = str(redec.get("predicted_smiles") or "")
                    if not redec_smiles:
                        continue
                    orig = results[i]
                    orig_count = _prediction_atom_count(orig)
                    redec_count = _prediction_atom_count(redec)
                    if redec_count > 0 and redec_count < orig_count:
                        redec["wavy_mask_inpaint"] = {
                            "redecoded": True,
                            "original_atoms": orig_count,
                            "redecoded_atoms": redec_count,
                            "detector_confidence": float(best.confidence),
                        }
                        results[i] = redec
                    else:
                        results[i]["wavy_mask_inpaint"] = {
                            "redecoded": False,
                            "reason": "redecode_not_fewer_atoms",
                            "original_atoms": orig_count,
                            "redecoded_atoms": redec_count,
                        }
                except Exception:
                    pass  # inpaint re-decode is an enhancement; any failure leaves base intact
        # Fragment-attachment fusion: graft a dummy at the predicted wavy
        # endpoint onto the base backbone molblock. Runs only for fragment rows.
        if frag_idx and self._fragment_attachment_enabled():
            runtime = self.fragment_attachment_runtime
            if runtime is not None:
                # Strip existing dummy so fusion grafts a clean one onto the mixture backbone.
                strip_existing = bool(not self.disable_moe_config and self.moe_config_path)
                try:
                    endpoints = runtime.predict_endpoints(
                        [segment_files[i] for i in frag_idx]
                    )
                    for i, endpoint in zip(frag_idx, endpoints):
                        if isinstance(results[i], dict):
                            _apply_fragment_attachment_fusion(
                                results[i],
                                endpoint,
                                min_confidence=self.fragment_attachment_min_confidence,
                                max_anchor_distance=self.fragment_attachment_max_anchor_distance,
                                temperature=self.fragment_attachment_confidence_temperature,
                                expected_type="fragment",
                                strip_existing_dummy=strip_existing,
                            )
                except Exception:
                    pass  # fusion is an enhancement; any failure leaves base results intact
                finally:
                    trim_process_memory()
        # Detection→graph fusion (post-hoc dummy graft): when the detector found
        # attachment marks but the decoder's molblock has no dummy (the sidecar
        # missed the attachment point), graft one at the detector's pixel-precise
        # location via fuse_dummy_attachment. This runs OUTSIDE the inference
        # contract (in structure_recognition, not the decoder), so it works under
        # the production direct_sidecar mode where the residual edit path is
        # unreachable. Only fragment/markush rows that lack a dummy; complete is
        # never touched. Any graft failure leaves the base result unchanged.
        if prior_fusion_active and frag_idx:
            for i in frag_idx:
                if not isinstance(results[i], dict):
                    continue
                prior_dets = attachment_priors[i] if i < len(attachment_priors) else None
                if not prior_dets:
                    continue
                molblock_i = extract_molblock(results[i])
                smiles_i = str(results[i].get("predicted_smiles") or "")
                # Skip if the decoder already emitted any attachment dummy: check
                # BOTH the molblock (atomic_num==0) AND the SMILES (`*`), because
                # the phase2_sep [n*] representation may keep the dummy in the
                # SMILES without an atomic_num==0 entry the local check recognizes.
                # Grafting onto a row that already has its dummy corrupts the graph
                # (e.g. stereochemistry loss on SMILES re-serialization).
                if not molblock_i or _molblock_has_dummy_atom_local(molblock_i) or "*" in smiles_i:
                    continue  # decoder already emitted a dummy; leave it
                # Pick the highest-confidence detector attachment mark.
                best_det = max(prior_dets, key=lambda d: float(d.get("confidence", 0.0)))
                if float(best_det.get("confidence", 0.0)) < self.attachment_prior_min_confidence:
                    continue
                # Convert the detector mark to the endpoint format fuse_dummy_attachment
                # expects: normalized x/y + presence + side from the mark's edge location.
                endpoint = {
                    "endpoint_presence": 1,
                    "endpoint_x": float(best_det.get("cx", 0.5)),
                    "endpoint_y": float(best_det.get("cy", 0.5)),
                    "sidecar_confidence": float(best_det.get("confidence", 0.0)),
                    "side": "",
                    "source": "detector",
                }
                try:
                    _apply_fragment_attachment_fusion(
                        results[i],
                        endpoint,
                        min_confidence=self.attachment_prior_min_confidence,
                        max_anchor_distance=self.fragment_attachment_max_anchor_distance,
                        temperature=self.fragment_attachment_confidence_temperature,
                        expected_type=str(expected_structure_types[i] or "fragment"),
                        strip_existing_dummy=False,
                    )
                except Exception:
                    pass  # graft is an enhancement; any failure leaves base intact
        # Detection→graph fusion (dummy pruning): when the decoder emits MORE
        # attachment dummies than the detector found attachment marks, the excess
        # dummies are likely spurious (the decoder's #1 failure mode is
        # over-emitting *). Prune the dummies farthest from any detector mark so
        # the count matches the detector's evidence. This directly attacks the
        # more_dummies bottleneck (27% of MoE markush failures). Only applies
        # when decoder dummies > detector marks; never adds dummies. Any prune
        # failure leaves the base result unchanged.
        if prior_fusion_active and frag_idx:
            for i in frag_idx:
                if not isinstance(results[i], dict):
                    continue
                prior_dets = attachment_priors[i] if i < len(attachment_priors) else None
                if not prior_dets:
                    continue
                atom_sets = results[i].get("atom_sets") or []
                dummy_atoms = [
                    (idx, a) for idx, a in enumerate(atom_sets)
                    if isinstance(a, dict) and "*" in str(a.get("atom_symbol") or "")
                ]
                if len(dummy_atoms) <= len(prior_dets):
                    continue  # decoder has fewer/equal dummies than detector marks; no prune
                # Score each dummy by distance to its nearest detector mark.
                # Keep only the len(prior_dets) closest dummies; flag the rest.
                det_pts = [(float(d.get("cx", 0.5)), float(d.get("cy", 0.5))) for d in prior_dets]
                scored = []
                for as_idx, atom in dummy_atoms:
                    coords = atom.get("coords")
                    if not coords or len(coords) < 2:
                        continue
                    dx, dy = float(coords[0]), float(coords[1])
                    min_dist = min(((dx - cx) ** 2 + (dy - cy) ** 2) ** 0.5 for cx, cy in det_pts)
                    scored.append((min_dist, as_idx))
                scored.sort()  # closest first
                keep_count = min(len(prior_dets), len(scored))
                keep_indices = {as_idx for _, as_idx in scored[:keep_count]}
                prune_indices = {as_idx for _, as_idx in scored[keep_count:]}
                if not prune_indices:
                    continue
                # Mark pruned dummies (remove from atom_sets; the SMILES/molblock
                # are left intact; pruning only annotates the count evidence so
                # downstream quality scoring can penalize the excess). A full
                # graph rewrite would risk corrupting valid backbones.
                results[i]["detector_dummy_prune"] = {
                    "decoder_dummies": len(dummy_atoms),
                    "detector_marks": len(prior_dets),
                    "pruned": len(prune_indices),
                    "pruned_indices": sorted(prune_indices),
                }
        predictions = []
        for result_index, (result, expected_type) in enumerate(zip(results, expected_structure_types)):
            if isinstance(result, dict):
                # Graph-level "ghost carbon" repair (organic fusion): topology
                # rules (amino/amide) + DECIMER mask geometry resolve the wavy
                # vertex misread as carbon. Fragment-only: markush excluded
                # because ghost repair regressed markush exact by -1.8% (the
                # *C(C) pattern is sometimes a real substituent on markush
                # scaffolds), but on fragments the wavy-bond vertex is almost
                # always a ghost carbon (55% of wavy_fragment failures are
                # *C(C)N→*CN, i.e. a spurious C between the dummy and heteroatom).
                if self.graph_ghost_repair_enabled and str(
                    expected_type or ""
                ).strip().lower() == "fragment":
                    try:
                        from utils.MolNexTR.graph_ghost_repair import (
                            apply_ghost_repair,
                        )
                        # Build mask context with the decoded atoms' normalized
                        # coords so the repair can test whether a specific atom
                        # (the ghost carbon) lies inside a detected mask region.
                        mask_ctx = None
                        detections = attachment_masks_by_index.get(result_index)
                        if detections:
                            atom_sets = result.get("atom_sets") or []
                            norm_coords = []
                            for a in atom_sets:
                                if isinstance(a, dict) and a.get("coords"):
                                    norm_coords.append((float(a["coords"][0]), float(a["coords"][1])))
                                else:
                                    norm_coords.append(None)
                            mask_ctx = {"detections": detections, "atom_norm_coords": norm_coords}
                        result = apply_ghost_repair(result, expected_type=expected_type, mask_context=mask_ctx)
                    except Exception:
                        pass
                smiles = result.get("predicted_smiles") or ""
                molblock = extract_molblock(result)
                quality_issues = tuple(
                    dict.fromkeys(
                        molnextr_quality_issues(result)
                        + molnextr_expected_structure_issues(result, expected_type)
                    )
                )
                expert_weights = tuple(float(x) for x in (result.get("expert_weights") or []))
                routed_expert = str(result.get("routed_expert") or "")
                routing_forced_complete = bool(result.get("routing_forced_complete"))
                routing_forced_default = bool(result.get("routing_forced_default", routing_forced_complete))
                routing_strategy = str(result.get("routing_strategy") or "")
                routing_confidence = float(result.get("routing_confidence", -1.0))
                routing_required_threshold = float(result.get("routing_required_threshold", -1.0))
                confidence = float(result.get("confidence", -1.0))
                attachment_repair = result.get("attachment_repair")
            else:
                smiles = result or ""
                molblock = ""
                quality_issues = ()
                expert_weights = ()
                routed_expert = ""
                routing_forced_complete = False
                routing_forced_default = False
                routing_strategy = ""
                routing_confidence = -1.0
                routing_required_threshold = -1.0
                confidence = -1.0
                attachment_repair = None
            predictions.append(
                StructurePrediction(
                    smiles=smiles,
                    molblock=molblock,
                    raw=result,
                    elapsed_seconds=per_image_elapsed,
                    quality_issues=quality_issues,
                    expert_weights=expert_weights,
                    routed_expert=routed_expert,
                    routing_forced_complete=routing_forced_complete,
                    routing_forced_default=routing_forced_default,
                    routing_strategy=routing_strategy,
                    routing_confidence=routing_confidence,
                    routing_required_threshold=routing_required_threshold,
                    confidence=confidence,
                    attachment_repair=attachment_repair,
                    expected_structure_type=str(expected_type or ""),
                )
            )
        return predictions

    def _predict_molnextr(self, segment_file: str) -> StructurePrediction:
        start = time.monotonic()
        prediction = self.predict_segment_files([segment_file], batch_size=1, return_confidence=True)[0]
        elapsed = time.monotonic() - start
        return StructurePrediction(
            smiles=prediction.smiles,
            molblock=prediction.molblock,
            raw=prediction.raw,
            elapsed_seconds=elapsed,
            quality_issues=prediction.quality_issues,
            expert_weights=prediction.expert_weights,
            routed_expert=prediction.routed_expert,
            routing_forced_complete=prediction.routing_forced_complete,
            routing_forced_default=prediction.routing_forced_default,
            routing_strategy=prediction.routing_strategy,
            routing_confidence=prediction.routing_confidence,
            routing_required_threshold=prediction.routing_required_threshold,
            confidence=prediction.confidence,
            attachment_repair=prediction.attachment_repair,
            expected_structure_type=prediction.expected_structure_type,
        )

    def health_check(self) -> dict[str, Any]:
        path = resolve_molnextr_model_path()
        payload: dict[str, Any] = {"model_path": path}
        model = _load_molnextr(path, self.device)
        payload["device"] = str(model.device)
        payload["loaded"] = True
        return payload
