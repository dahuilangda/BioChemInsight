"""Source discovery helpers for MolNexTR MoE sidecar training."""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any
import json

PRODUCTION_DATASET_ID = "molnextr_moe_production_v1"
PRODUCTION_RELATIVE_ROOT = f"data/generated/pose_factory/{PRODUCTION_DATASET_ID}"
PRODUCTION_SHARD_RE = re.compile(r"s\d{3,}")

PRODUCTION_PATTERNS = {
    0: "ordinary/s*/ordinary_negative.csv",
    1: "markush/s*/accepted_candidate/markush_layout_positive.csv",
    2: "fragment/s*/attachment_fragment_positive.csv",
}

LABEL_NAMES = {
    0: "complete",
    1: "markush",
    2: "fragment",
}


def _shard_dir_for_csv(path: Path, *, label: int) -> Path:
    if int(label) == 1 and path.parent.name == "accepted_candidate":
        return path.parent.parent
    return path.parent


def is_production_shard_name(name: str) -> bool:
    """Accept the zero-padded shard ids emitted by the production generator.

    The dataset grew past ``s999``.  Requiring exactly three digits silently
    dropped every later shard even though those shards had the same manifests
    and passing contracts as the earlier data.
    """
    return PRODUCTION_SHARD_RE.fullmatch(str(name or "")) is not None


def _contracts_root_for_csv(path: Path, *, root: Path) -> Path:
    return root.parents[3] / "runs" / f"{PRODUCTION_DATASET_ID}_contracts"


def _gate_passes(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing_gate:{path.name}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, f"unreadable_gate:{path.name}"
    if not isinstance(payload, dict):
        return False, f"invalid_gate:{path.name}"
    passed = payload.get("passed")
    if passed is None:
        passed = payload.get("trainable")
    if passed is not True:
        return False, f"gate_not_passing:{path.name}"
    return True, "passing"


def _required_gate_paths(path: Path, *, label: int, root: Path) -> list[Path]:
    shard_dir = _shard_dir_for_csv(path, label=label)
    contracts = _contracts_root_for_csv(path, root=root)
    shard = shard_dir.name
    if int(label) == 0:
        return [
            contracts / "ordinary" / shard / "validation.json",
            contracts / "ordinary" / shard / "ordinary_molnextr_quality_contract.json",
        ]
    if int(label) == 1:
        return [
            contracts / "markush" / shard / "accepted_validation.json",
            contracts / "markush" / shard / "accepted_formal_nonlinear_warp_contract.json",
            contracts / "markush" / shard / "accepted_pose_alignment.json",
            contracts / "markush" / shard / "markush_substitution_anchor_contract.json",
        ]
    if int(label) == 2:
        return [
            contracts / "fragment" / shard / "attachment_visual_contract.json",
            contracts / "fragment" / shard / "validation.json",
            contracts / "fragment" / shard / "fragment_formal_nonlinear_warp_contract.json",
        ]
    return []


def _csv_stability(path: Path, *, label: int, root: Path, now: float, min_age_seconds: float) -> tuple[bool, str]:
    shard_dir = _shard_dir_for_csv(path, label=label)
    if not is_production_shard_name(shard_dir.name):
        return False, "non_production_shard_name"
    if any(token in part.lower() for part in path.parts for token in ("smoke", "debug", "probe", "tmp")):
        return False, "non_production_shard_name"
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False, "missing"
    except OSError as exc:
        return False, f"stat_failed:{exc.__class__.__name__}"
    if stat.st_size <= 0:
        return False, "empty"
    age = now - stat.st_mtime
    if min_age_seconds > 0 and age < min_age_seconds:
        return False, f"too_new:{age:.1f}s"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = handle.readline()
    except OSError as exc:
        return False, f"open_failed:{exc.__class__.__name__}"
    if "file_path" not in header or ("SMILES" not in header and "smiles" not in header):
        return False, "unexpected_header"
    manifest_path = path.parent / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False, "manifest_unreadable"
        if manifest.get("rejected") is True:
            return False, "manifest_rejected"
    # The per-shard provenance gate files live under runs/<dataset>_contracts/,
    # which is produced during generation and may be absent when rebuilding the
    # train_df from already-accepted shards (e.g. after a runs/ cleanup). The
    # accepted_candidate CSV + in-shard manifest above already prove the rows
    # were accepted by the generation gates, so the contracts check is
    # skippable for a rebuild. Set MOE_SKIP_PROVENANCE_GATE=1 to do so.
    if not os.environ.get("MOE_SKIP_PROVENANCE_GATE"):
        for gate_path in _required_gate_paths(path, label=label, root=root):
            ok, reason = _gate_passes(gate_path)
            if not ok:
                return False, reason
    return True, "stable"


def _rank_key(path: Path, *, label: int, seed: int) -> tuple[str, str]:
    text = f"{int(seed)}:{int(label)}:{path.as_posix()}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), path.as_posix()


def _split_csvs(
    csv_paths: list[Path],
    *,
    label: int,
    calibration_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    if not csv_paths:
        return [], []
    paths = sorted(csv_paths, key=lambda p: p.as_posix())
    fraction = min(max(float(calibration_fraction), 0.0), 0.8)
    if fraction <= 0.0 or len(paths) < 2:
        return paths, []
    calibration_count = max(1, int(round(len(paths) * fraction)))
    calibration_count = min(calibration_count, len(paths) - 1)
    ranked = sorted(paths, key=lambda p: _rank_key(p, label=label, seed=seed))
    calibration = set(ranked[:calibration_count])
    train = [p for p in paths if p not in calibration]
    cal = [p for p in paths if p in calibration]
    return train, cal


def discover_available_production_pose_factory(
    root: str | Path,
    *,
    min_age_seconds: float = 120.0,
    calibration_fraction: float = 0.2,
    seed: int = 2026062701,
) -> tuple[dict[int, list[str]], dict[int, list[str]], dict[str, Any]]:
    """Discover stable production shard CSVs and split by whole shard.

    Markush discovery intentionally selects only accepted-candidate shard CSVs.
    Raw Markush candidates, and any ordinary/fragment/Markush shard without
    the required production gates, are excluded from training.
    """
    root_path = Path(root).resolve()
    now = time.time()
    train_sources: dict[int, list[str]] = {}
    calibration_sources: dict[int, list[str]] = {}
    labels_report: dict[str, Any] = {}

    for label, pattern in sorted(PRODUCTION_PATTERNS.items()):
        candidates = sorted(root_path.glob(pattern), key=lambda p: p.as_posix())
        stable: list[Path] = []
        skipped: list[dict[str, str]] = []
        for path in candidates:
            ok, reason = _csv_stability(path, label=label, root=root_path, now=now, min_age_seconds=min_age_seconds)
            if ok:
                stable.append(path)
            else:
                skipped.append({"path": str(path), "reason": reason})
        train, calibration = _split_csvs(
            stable,
            label=label,
            calibration_fraction=calibration_fraction,
            seed=seed,
        )
        train_sources[label] = [str(p) for p in train]
        calibration_sources[label] = [str(p) for p in calibration]
        labels_report[LABEL_NAMES.get(label, str(label))] = {
            "label": int(label),
            "pattern": pattern,
            "candidate_csvs": int(len(candidates)),
            "stable_csvs": int(len(stable)),
            "train_csvs": int(len(train)),
            "calibration_csvs": int(len(calibration)),
            "train_csv_paths": [str(p) for p in train],
            "calibration_csv_paths": [str(p) for p in calibration],
            "skipped_csvs": skipped,
        }

    report = {
        "schema_version": "molnextr_moe_source_discovery_v1",
        "source_mode": "available_production_pose_factory",
        "dataset_id": PRODUCTION_DATASET_ID,
        "root": str(root_path),
        "min_age_seconds": float(min_age_seconds),
        "calibration_fraction": float(calibration_fraction),
        "seed": int(seed),
        "policy": {
            "uses_currently_available_stable_csvs": True,
            "requires_production_gate": True,
            "requires_production_dataset_id": PRODUCTION_DATASET_ID,
            "split_unit": "whole_shard_csv",
            "markush_raw_candidate_csvs_excluded": True,
            "markush_requires_accepted_candidate_gates": True,
            "ordinary_requires_molnextr_quality_contract": True,
            "fragment_requires_visual_schema_and_warp_gates": True,
        },
        "labels": labels_report,
    }
    return train_sources, calibration_sources, report
