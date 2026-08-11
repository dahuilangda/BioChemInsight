from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SIDE_NAMES = ("left", "right", "top", "bottom")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
        return rows
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_json_dict(value: Any) -> dict[str, Any]:
    try:
        if value is None or str(value).strip() == "":
            return {}
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def row_image_path(row: dict[str, Any]) -> str:
    for key in ("file_path", "SEGMENT_FILE", "segment_file", "image", "IMAGE_FILE", "image_file"):
        value = str(row.get(key) or "").strip()
        if value:
            return value.replace("/app/", "")
    return ""


def expected_direction(row: dict[str, Any]) -> str:
    quality = parse_json_dict(row.get("render_quality"))
    direction = str(quality.get("attachment_direction") or "").strip()
    if direction:
        return direction
    reason = str(row.get("STRUCTURE_FILTER_REASON") or row.get("filter_reason") or "").lower()
    if "left" in reason:
        return "left"
    if "right" in reason:
        return "right"
    return ""


def direction_to_side(direction: str) -> str:
    text = str(direction or "").lower()
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    if text == "up":
        return "top"
    if text == "down":
        return "bottom"
    return ""


def expected_has_endpoint(row: dict[str, Any]) -> bool | None:
    quality = parse_json_dict(row.get("render_quality"))
    if quality.get("attachment_render_mode") == "wavy":
        return True
    target = str(row.get("SMILES") or row.get("target") or "")
    if "*" in target:
        return True
    structure_type = str(row.get("STRUCTURE_TYPE") or row.get("structure_type") or "").lower()
    if structure_type == "fragment":
        return True
    if structure_type in {"markush", "ordinary", "complete"}:
        return False
    if target:
        return False
    return None


def fragment_prior(row: dict[str, Any]) -> bool:
    structure_type = str(row.get("STRUCTURE_TYPE") or row.get("structure_type") or "").lower()
    if structure_type == "fragment":
        return True
    annotations = parse_json_dict(row.get("annotations"))
    document = annotations.get("document") if isinstance(annotations.get("document"), dict) else {}
    if document.get("visual_role") == "attachment_fragment":
        return True
    return "wavy" in str(row.get("STRUCTURE_FILTER_REASON") or row.get("filter_reason") or "").lower()


def foreground_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fixed = (gray < 235).astype(np.uint8) * 255
    mask = cv2.bitwise_or(otsu, fixed)
    kernel = np.ones((2, 2), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def side_band(mask: np.ndarray, bbox: tuple[int, int, int, int], side: str) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    band_w = max(6, int(round(width * 0.28)))
    band_h = max(6, int(round(height * 0.28)))
    if side == "left":
        return mask[y1:y2, x1 : min(x2, x1 + band_w)]
    if side == "right":
        return mask[y1:y2, max(x1, x2 - band_w) : x2]
    if side == "top":
        return mask[y1 : min(y2, y1 + band_h), x1:x2]
    return mask[max(y1, y2 - band_h) : y2, x1:x2]


def band_complexity(band: np.ndarray) -> dict[str, float]:
    if band.size == 0:
        return {"ink": 0.0, "edge_density": 0.0, "component_density": 0.0}
    ink = float((band > 0).mean())
    edges = cv2.Canny(band, 50, 150)
    edge_density = float((edges > 0).mean())
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((band > 0).astype(np.uint8), 8)
    useful = 0
    for index in range(1, num_labels):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= 3:
            useful += 1
    component_density = useful / max(1.0, math.sqrt(float(band.shape[0] * band.shape[1])))
    return {
        "ink": ink,
        "edge_density": edge_density,
        "component_density": component_density,
    }


def detect_image(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "image_readable": False,
            "visual_endpoint_score": 0.0,
            "predicted_side": "",
            "side_scores": {},
        }
    mask = foreground_mask(image)
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return {
            "image_readable": True,
            "visual_endpoint_score": 0.0,
            "predicted_side": "",
            "side_scores": {},
        }
    image_h, image_w = mask.shape[:2]
    x1, y1, x2, y2 = bbox
    side_scores = {}
    side_features = {}
    for side in SIDE_NAMES:
        band = side_band(mask, bbox, side)
        features = band_complexity(band)
        if side == "left":
            proximity = 1.0 - min(1.0, x1 / max(1.0, image_w * 0.18))
        elif side == "right":
            proximity = 1.0 - min(1.0, (image_w - x2) / max(1.0, image_w * 0.18))
        elif side == "top":
            proximity = 1.0 - min(1.0, y1 / max(1.0, image_h * 0.18))
        else:
            proximity = 1.0 - min(1.0, (image_h - y2) / max(1.0, image_h * 0.18))
        score = (
            0.58 * min(1.0, features["edge_density"] * 18.0)
            + 0.25 * min(1.0, features["ink"] * 7.0)
            + 0.12 * min(1.0, features["component_density"] * 5.0)
            + 0.05 * proximity
        )
        side_scores[side] = float(score)
        side_features[side] = {**features, "proximity": float(proximity)}
    predicted_side = max(side_scores, key=side_scores.get)
    visual_score = float(side_scores[predicted_side])
    return {
        "image_readable": True,
        "image_width": int(image_w),
        "image_height": int(image_h),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "visual_endpoint_score": visual_score,
        "predicted_side": predicted_side,
        "side_scores": side_scores,
        "side_features": side_features,
    }


def precision_recall(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("expected_has_endpoint") is not None]
    tp = sum(bool(row.get(key)) and row.get("expected_has_endpoint") is True for row in labeled)
    fp = sum(bool(row.get(key)) and row.get("expected_has_endpoint") is False for row in labeled)
    fn = sum(not bool(row.get(key)) and row.get("expected_has_endpoint") is True for row in labeled)
    tn = sum(not bool(row.get(key)) and row.get("expected_has_endpoint") is False for row in labeled)
    return {
        "labeled": len(labeled),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def side_accuracy(rows: list[dict[str, Any]], key: str = "predicted_side") -> dict[str, Any]:
    labeled = [
        row
        for row in rows
        if row.get("expected_side") and row.get("expected_has_endpoint") is True
    ]
    correct = sum(row.get(key) == row.get("expected_side") for row in labeled)
    counts = Counter(str(row.get(key) or "") for row in labeled)
    expected_counts = Counter(str(row.get("expected_side") or "") for row in labeled)
    return {
        "labeled": len(labeled),
        "correct": correct,
        "accuracy": correct / len(labeled) if labeled else 0.0,
        "predicted_counts": dict(sorted(counts.items())),
        "expected_counts": dict(sorted(expected_counts.items())),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prototype wavy/cut attachment endpoint detector for L1d evaluation."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visual-threshold", type=float, default=0.28)
    parser.add_argument(
        "--use-fragment-prior",
        action="store_true",
        help="Allow the BioChemInsight/synthetic fragment decision to trigger predicted_has_endpoint.",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source_rows = load_rows(Path(args.csv))
    if args.limit > 0:
        source_rows = source_rows[: args.limit]
    results = []
    for index, row in enumerate(source_rows):
        image_path = row_image_path(row)
        detection = detect_image(Path(image_path))
        visual_has_endpoint = detection["visual_endpoint_score"] >= float(args.visual_threshold)
        prior_endpoint = fragment_prior(row)
        predicted_has_endpoint = bool(visual_has_endpoint or (args.use_fragment_prior and prior_endpoint))
        direction = expected_direction(row)
        expected_side = direction_to_side(direction)
        expected_endpoint = expected_has_endpoint(row)
        results.append(
            {
                "row_index": index,
                "image": image_path,
                "source_id": str(row.get("source_id") or row.get("PAGE_NUM") or ""),
                "structure_type": str(row.get("STRUCTURE_TYPE") or row.get("structure_type") or ""),
                "expected_has_endpoint": expected_endpoint,
                "expected_direction": direction,
                "expected_side": expected_side,
                "fragment_prior_endpoint": prior_endpoint,
                "visual_has_endpoint": visual_has_endpoint,
                "predicted_has_endpoint": predicted_has_endpoint,
                **detection,
            }
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "endpoint_detector.results.jsonl", results)
    visual_metrics = precision_recall(results, "visual_has_endpoint")
    prior_metrics = precision_recall(results, "predicted_has_endpoint")
    report = {
        "csv": args.csv,
        "rows": len(results),
        "visual_threshold": float(args.visual_threshold),
        "use_fragment_prior": bool(args.use_fragment_prior),
        "visual_endpoint": visual_metrics,
        "predicted_endpoint": prior_metrics,
        "side": side_accuracy(results),
        "score_summary": {
            "min": min((row["visual_endpoint_score"] for row in results), default=0.0),
            "max": max((row["visual_endpoint_score"] for row in results), default=0.0),
            "mean": (
                sum(row["visual_endpoint_score"] for row in results) / len(results)
                if results
                else 0.0
            ),
        },
        "examples": {
            "low_score_positive": [
                row
                for row in sorted(results, key=lambda item: item["visual_endpoint_score"])
                if row.get("expected_has_endpoint") is True
            ][:8],
            "false_positive_visual": [
                row
                for row in results
                if row.get("visual_has_endpoint") and row.get("expected_has_endpoint") is False
            ][:8],
            "side_mismatch": [
                row
                for row in results
                if row.get("expected_side")
                and row.get("expected_has_endpoint") is True
                and row.get("predicted_side") != row.get("expected_side")
            ][:8],
        },
    }
    (output_dir / "endpoint_detector.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
