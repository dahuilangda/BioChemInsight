"""Markush-domain post-decode attachment repair for the cascade path.

Operator audit (design §36) finding #3: production's markush exact lives in
P9 (post-hoc dummy grafting when the decoder misses an attachment) + P10
(excess-dummy pruning) + the attachment Mask R-CNN priors that feed them.
The cascade returns bare decode dicts with none of this. This module brings
P9/P10 to the cascade using the SAME attachment detector production uses.
"""
from __future__ import annotations

from typing import Any


def collect_attachment_priors(recognizer, crop_rgb):
    """Run the production attachment Mask R-CNN on one crop; normalized dets.

    Returns [] on any failure — grafts are enhancements, never blockers.
    """
    try:
        detector = recognizer.attachment_detector
        dets = detector.detect(crop_rgb, score_threshold=0.05)
        out = []
        for det in dets or []:
            try:
                out.append({
                    "class": str(getattr(det, "class_name", "") or ""),
                    "cx": float(det.cx),
                    "cy": float(det.cy),
                    "confidence": float(det.confidence),
                })
            except (TypeError, ValueError, AttributeError):
                continue
        return out
    except Exception:
        return []


def graft_missing_dummy(decoded: dict, priors: list[dict],
                         min_confidence: float = 0.3) -> dict:
    """P9: decoder emitted no dummy but detector found an attachment mark.

    Grafts a `*` onto the backbone via the existing fragment fusion helper,
    mirroring structure_recognition.py:1172-1212 semantics.
    """
    if not isinstance(decoded, dict) or not priors:
        return decoded
    atom_sets = decoded.get("atom_sets") or []
    has_dummy = any(
        isinstance(a, dict) and "*" in str(a.get("atom_symbol") or "")
        for a in atom_sets
    )
    if has_dummy:
        return decoded
    best = max(priors, key=lambda d: float(d.get("confidence", 0.0)))
    if float(best.get("confidence", 0.0)) < min_confidence:
        return decoded
    result = dict(decoded)
    result["atom_sets"] = list(atom_sets) + [{
        "atom_symbol": "*",
        "coords": [float(best.get("cx", 0.5)), float(best.get("cy", 0.5))],
        "confidence": float(best.get("confidence", 0.0)),
    }]
    edges = decoded.get("edges")
    if edges is not None:
        # attach the dummy to the nearest existing atom
        import numpy as _np

        coords = [
            (a.get("coords") or [0.5, 0.5]) if isinstance(a, dict) else [0.5, 0.5]
            for a in result["atom_sets"][:-1]
        ]
        if coords:
            arr = _np.array(coords, dtype=float)
            d = ((arr[:, 0] - best.get("cx", 0.5)) ** 2
                 + (arr[:, 1] - best.get("cy", 0.5)) ** 2) ** 0.5
            nearest = int(d.argmin())
            new_edges = _np.array(edges, dtype=int)
            if new_edges.ndim == 2:
                n = len(result["atom_sets"])
                grown = _np.zeros((n, n), dtype=new_edges.dtype)
                grown[: new_edges.shape[0], : new_edges.shape[1]] = new_edges
                grown[nearest, n - 1] = 1
                grown[n - 1, nearest] = 1
                result["edges"] = grown
    result["grafted_dummy"] = True
    return result


def prune_excess_dummies(decoded: dict, priors: list[dict]) -> dict:
    """P10: decoder emitted more dummies than detector marks — annotate.

    Mirrors structure_recognition.py:1215-1248 (annotation-only; a full graph
    rewrite would risk corrupting valid backbones).
    """
    if not isinstance(decoded, dict) or not priors:
        return decoded
    atom_sets = decoded.get("atom_sets") or []
    dummies = [
        (idx, a) for idx, a in enumerate(atom_sets)
        if isinstance(a, dict) and "*" in str(a.get("atom_symbol") or "")
    ]
    if len(dummies) <= len(priors):
        return decoded
    det_pts = [(float(d.get("cx", 0.5)), float(d.get("cy", 0.5))) for d in priors]
    scored = []
    for idx, atom in dummies:
        coords = atom.get("coords")
        if not coords or len(coords) < 2:
            continue
        dx, dy = float(coords[0]), float(coords[1])
        min_dist = min(((dx - cx) ** 2 + (dy - cy) ** 2) ** 0.5 for cx, cy in det_pts)
        scored.append((min_dist, idx))
    scored.sort()
    keep = min(len(priors), len(scored))
    prune = {idx for _, idx in scored[keep:]}
    if not prune:
        return decoded
    result = dict(decoded)
    result["detector_dummy_prune"] = {
        "decoder_dummies": len(dummies),
        "detector_marks": len(priors),
        "pruned": len(prune),
        "pruned_indices": sorted(prune),
    }
    return result


def apply_markush_attachment_repair(decoded: Any, recognizer, crop_rgb,
                                    min_confidence: float = 0.3) -> Any:
    """P9+P10 combined for one cascade prediction."""
    if not isinstance(decoded, dict):
        return decoded
    priors = collect_attachment_priors(recognizer, crop_rgb)
    if not priors:
        return decoded
    out = graft_missing_dummy(decoded, priors, min_confidence=min_confidence)
    out = prune_excess_dummies(out, priors)
    return out
