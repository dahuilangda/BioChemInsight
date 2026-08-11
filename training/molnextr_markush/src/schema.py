from __future__ import annotations

import json
from typing import Any

from training.molnextr_markush.src.labels import label_difficulty


def infer_structure_type(smiles: str, label_instances: list[dict[str, Any]]) -> str:
    if label_instances:
        return "markush"
    if "*" in str(smiles or ""):
        return "fragment"
    return "complete_compound"


def build_annotations(
    *,
    smiles: str,
    label_instances: list[dict[str, Any]],
    structure_type: str | None = None,
    compound_id: str = "",
    visual_role: str = "",
    variable_positions: list[str] | None = None,
    fragment_refs: list[str] | None = None,
    relationships: list[dict[str, Any]] | None = None,
    assay_links: list[dict[str, Any]] | None = None,
) -> str:
    labels = sorted({str(item["text"]) for item in label_instances})
    families = sorted({str(item["family"]) for item in label_instances})
    payload = {
        "document": {
            "compound_id": compound_id,
            "visual_role": visual_role,
        },
        "structure": {
            "structure_type": structure_type or infer_structure_type(smiles, label_instances),
            "variable_positions": variable_positions or labels,
            "fragment_refs": fragment_refs or [],
        },
        "markush": {
            "instances": label_instances,
            "labels": labels,
            "label_count": len(labels),
            "families": families,
            "difficulty": label_difficulty(labels),
        },
        "relationships": relationships or [],
        "assay_links": assay_links or [],
    }
    return json.dumps(payload, sort_keys=True)


def parse_annotations(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        raise ValueError("missing annotations")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("annotations must be a JSON object")
    return payload


def markush_instances(annotations: dict[str, Any]) -> list[dict[str, Any]]:
    values = ((annotations.get("markush") or {}).get("instances") or [])
    if not isinstance(values, list):
        raise ValueError("annotations.markush.instances must be a JSON list")
    return [item for item in values if isinstance(item, dict)]
