from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


csv.field_size_limit(sys.maxsize)

POSE_FACTORY_SCHEMA_VERSION = "pose_factory_v1"
SVG_CENTER_FIT_METHOD = "svg_bond_axis_atom_center_hybrid_intersections_with_single_axis_affine_seed_projection"
MARKUSH_DOCUMENT_REALISM_SCHEMA_VERSION = "markush_document_realism_v1"
MARKUSH_DOCUMENT_REALISM_POLICY = "patent_literature_markushgenerator_cdk_svg_v1"
FORMAL_NONLINEAR_WARP_POLICY = "formal_synchronized_warped_svg_polyline_pose_preservation_v1"
FRAGMENT_FORMAL_NONLINEAR_WARP_POLICY = "fragment_synchronized_endpoint_connector_mark_document_warp_v1"
MARKUSH_ATOM_INDEX_ALIGNMENT_SCHEMA_VERSION = "cdk_rdkit_atom_index_alignment_v2"
FORMAL_SIM_DATASET_FAMILY = "molnextr_markush_fragment_router_formal_sim_v1"
FORMAL_SIM_LINEAGE_SCHEMA_VERSION = "formal_sim_dataset_lineage_v1"
FORMAL_SIM_POLICY_SCHEMA_VERSION = "formal_simulation_policy_v1"
SHORT_WAVY_CONNECTOR_POLICY = "patent_real_short_stub_v1"
MAX_PATENT_WAVY_STRAIGHT_CONNECTOR_PX_AT_384 = 48.0
MAX_PATENT_WAVY_VISIBLE_CONNECTOR_PX_AT_384 = 58.0
MAX_PATENT_CUT_CONNECTOR_PX_AT_384 = 56.0
MAX_PATENT_DUMMY_CONNECTOR_PX_AT_384 = 54.0
MAX_PATENT_QUERY_CONNECTOR_PX_AT_384 = 58.0
MARKUSH_REALISM_RENDER_PARAMETER_KEYS = {
    "seed",
    "stroke_ratio",
    "bond_separation",
    "symbol_margin_ratio",
    "font_name",
    "font_size",
    "render_atom_numbers",
    "render_carbon_symbols",
    "render_aromatic_display",
    "render_deuterium_symbol",
    "render_terminal_carbons",
}

SUPPORTED_STRUCTURE_TYPES = {
    "attachment_fragment",
    "complete_compound",
    "wavy_fragment",
    "cut_fragment",
    "markush_layout",
}


def formal_sim_dataset_lineage(
    *,
    branch: str,
    source_record_id: Any,
    parent_source_group: Any,
    generation_stage: str,
    source_dataset: str = "",
    source_file: str = "",
    shard_id: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": FORMAL_SIM_LINEAGE_SCHEMA_VERSION,
        "dataset_family": FORMAL_SIM_DATASET_FAMILY,
        "branch": str(branch),
        "source_dataset": str(source_dataset or ""),
        "source_file": str(source_file or ""),
        "source_record_id": str(source_record_id),
        "parent_source_group": str(parent_source_group),
        "generation_stage": str(generation_stage),
        "shard_id": str(shard_id or ""),
    }


def formal_simulation_policy(
    *,
    branch: str,
    allowed_operations: list[str],
    coordinate_mutation_policy: str,
    geometry_contract: str = "",
    formal_capable: bool = True,
    image_synchronized: bool = True,
    atom_coordinates_synchronized: bool = True,
    endpoint_coordinates_synchronized: bool | None = None,
    ocr_boxes_synchronized: bool | None = None,
    svg_or_connector_anchors_synchronized: bool | None = None,
) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "schema_version": FORMAL_SIM_POLICY_SCHEMA_VERSION,
        "dataset_family": FORMAL_SIM_DATASET_FAMILY,
        "branch": str(branch),
        "allowed_operations": [str(value) for value in allowed_operations],
        "coordinate_mutation_policy": str(coordinate_mutation_policy),
        "geometry_contract": str(geometry_contract or ""),
        "formal_capable": bool(formal_capable),
        "research_only": False,
        "debug_only": False,
        "image_synchronized": bool(image_synchronized),
        "atom_coordinates_synchronized": bool(atom_coordinates_synchronized),
    }
    if endpoint_coordinates_synchronized is not None:
        policy["endpoint_coordinates_synchronized"] = bool(endpoint_coordinates_synchronized)
    if ocr_boxes_synchronized is not None:
        policy["ocr_boxes_synchronized"] = bool(ocr_boxes_synchronized)
    if svg_or_connector_anchors_synchronized is not None:
        policy["svg_or_connector_anchors_synchronized"] = bool(svg_or_connector_anchors_synchronized)
    return policy


def is_svg_center_fit_method(method: str) -> bool:
    return SVG_CENTER_FIT_METHOD in str(method or "")


def pose_affine_diagnostics(pose_mapping: dict[str, Any], fit_diagnostics: dict[str, Any]) -> dict[str, Any]:
    if is_svg_center_fit_method(str(pose_mapping.get("fit_method") or "")):
        seed = pose_mapping.get("affine_seed_diagnostics")
        if isinstance(seed, dict) and isinstance(seed.get("affine"), dict):
            return seed["affine"]
        nested = fit_diagnostics.get("affine_seed_diagnostics")
        if isinstance(nested, dict) and isinstance(nested.get("affine"), dict):
            return nested["affine"]
        return {}
    return fit_diagnostics.get("affine") if isinstance(fit_diagnostics.get("affine"), dict) else {}


def validate_markush_atom_index_alignment_contract(render_quality: dict[str, Any]) -> list[str]:
    if str(render_quality.get("structure_type") or "") != "markush_layout":
        return []
    issues: list[str] = []
    alignment = render_quality.get("atom_index_alignment")
    if not isinstance(alignment, dict):
        return ["missing_atom_index_alignment"]
    if alignment.get("schema_version") != MARKUSH_ATOM_INDEX_ALIGNMENT_SCHEMA_VERSION:
        issues.append("atom_index_alignment_schema_mismatch")
    if alignment.get("applied") is not True:
        issues.append("atom_index_alignment_not_applied")
    if alignment.get("validated") is not True:
        issues.append("atom_index_alignment_not_validated")

    validation = alignment.get("validation") if isinstance(alignment.get("validation"), dict) else {}
    required_validation_flags = {
        "atomic_numbers_match_after_remap",
        "bond_edge_set_matches_after_remap",
        "dummy_pseudo_atom_count_matches",
        "dummy_label_or_isotope_constraints_checked",
    }
    for flag in sorted(required_validation_flags):
        if validation.get(flag) is not True:
            issues.append(f"atom_index_alignment_{flag}_failed")

    graph = render_quality.get("graph_consistency") if isinstance(render_quality.get("graph_consistency"), dict) else {}
    atom_count = int(graph.get("atom_count") or validation.get("atom_count") or 0)
    raw_mapping = alignment.get("cdk_to_rdkit_atom_index")
    mapping: dict[int, int] = {}
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        issues.append("atom_index_alignment_missing_mapping")
    else:
        for key, value in raw_mapping.items():
            try:
                mapping[int(key)] = int(value)
            except (TypeError, ValueError):
                issues.append("atom_index_alignment_mapping_index_invalid")
        if atom_count > 0:
            expected = set(range(atom_count))
            if set(mapping) != expected:
                issues.append("atom_index_alignment_cdk_indices_not_contiguous")
            if set(mapping.values()) != expected:
                issues.append("atom_index_alignment_rdkit_indices_not_contiguous")
        if len(set(mapping.values())) != len(mapping):
            issues.append("atom_index_alignment_not_one_to_one")

    if atom_count > 0 and int(validation.get("atom_count") or -1) != atom_count:
        issues.append("atom_index_alignment_atom_count_mismatch")

    return sorted(set(issues))

SUPPORTED_ENDPOINT_MARKS = {
    "",
    "wavy",
    "cut",
    "dummy_atom",
    "query_attachment",
}

REQUIRED_RENDER_QUALITY_KEYS = {
    "schema_version",
    "generator_version",
    "backend",
    "backend_references",
    "source_dataset",
    "source_record_id",
    "structure_type",
    "image_width",
    "image_height",
    "atom_coordinates",
    "bonds",
    "layout_seed",
    "render_style",
    "coord_policy",
    "quality_gates",
    "graph_consistency",
}

FORMAL_SIM_REQUIRED_LINEAGE_KEYS = {
    "schema_version",
    "dataset_family",
    "branch",
    "source_record_id",
    "parent_source_group",
    "generation_stage",
}

FORMAL_SIM_REQUIRED_POLICY_KEYS = {
    "schema_version",
    "dataset_family",
    "branch",
    "allowed_operations",
    "coordinate_mutation_policy",
    "formal_capable",
    "research_only",
    "debug_only",
}

POSE_ALIGNMENT_KEYS = {
    "image_to_graph_orientation_alignment",
    "coordinate_mutation_after_render",
    "orientation_policy",
    "synchronized_after_augmentation",
}

FRAGMENT_RENDER_QUALITY_KEYS = {
    "attachment_anchor",
    "attachment_anchor_index",
    "attachment_endpoint",
    "attachment_direction",
    "attachment_render_mode",
    "attachment_render_geometry",
    "attachment_anchor_depiction_mode",
    "attachment_anchor_label_is_visible_text",
    "endpoint_label_quality",
}

FRAGMENT_STRUCTURE_TYPES = {
    "attachment_fragment",
    "wavy_fragment",
    "cut_fragment",
}

SUPPORTED_FRAGMENT_GEOMETRIES = {
    "straight_connector_with_terminal_perpendicular_wavy_cut_mark",
    "custom_markush_attachment_perpendicular_wavy",
    "straight_connector_with_terminal_perpendicular_cut_bar",
    "rdkit_moldraw2d_terminal_perpendicular_cut_bar",
    "straight_connector_with_terminal_double_cut_bar",
    "straight_connector_with_terminal_stub_cut",
    "straight_connector_with_terminal_query_label",
    "straight_connector_with_terminal_dummy_atom",
}

REAL_ATOM_TOKENS = {
    "B",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "Si",
    "Se",
    "Te",
    "As",
}


@dataclass(frozen=True)
class ExternalBackendReference:
    name: str
    url: str
    role: str
    decision: str


@dataclass(frozen=True)
class PoseFactoryPlan:
    schema_version: str = POSE_FACTORY_SCHEMA_VERSION
    required_backends: tuple[ExternalBackendReference, ...] = (
        ExternalBackendReference(
            name="MolDepictor",
            url="https://github.com/DS4SD/MolDepictor",
            role="RDKit/CDK-style molecular depiction diversity, fonts, metadata-derived keypoints, and image noise primitives.",
            decision="Use as design reference; do not vendor wholesale into this clean training tree.",
        ),
        ExternalBackendReference(
            name="MarkushGenerator",
            url="https://github.com/DS4SD/MarkushGenerator",
            role="CXSMILES-aware Markush image generation, CDK depiction, OCR boxes, and image/text page composition.",
            decision="Use as design reference for future Markush layout factory and page-style crops.",
        ),
        ExternalBackendReference(
            name="MarkushGrapher",
            url="https://github.com/DS4SD/MarkushGrapher",
            role="MarkushGrapher-2 task data, augmentation, OCR/page cells, and Markush tokenizer conventions.",
            decision="Use as source-data and validation reference; keep raw data protected.",
        ),
        ExternalBackendReference(
            name="MolScribe",
            url="https://github.com/thomas0809/MolScribe",
            role="Image-to-graph OCSR coordinate/token training precedent and Indigo rendering reference.",
            decision="Use as comparison reference for ordinary molecule scale and coordinate-aware supervision.",
        ),
    )


@dataclass
class ValidationIssue:
    row_index: int
    severity: str
    code: str
    message: str
    source_id: str = ""
    file_path: str = ""


@dataclass
class ShardValidationReport:
    csv_path: str
    row_count: int
    valid_rows: int
    invalid_rows: int
    trainable: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)
    structure_type_counts: dict[str, int] = field(default_factory=dict)
    endpoint_side_counts: dict[str, int] = field(default_factory=dict)
    endpoint_mark_counts: dict[str, int] = field(default_factory=dict)
    fragment_geometry_counts: dict[str, int] = field(default_factory=dict)
    render_style_counts: dict[str, int] = field(default_factory=dict)
    issue_severity_counts: dict[str, int] = field(default_factory=dict)
    issue_code_counts: dict[str, int] = field(default_factory=dict)
    missing_required_field_counts: dict[str, int] = field(default_factory=dict)
    quality_gate_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    graph_consistency_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    molnextr_pose_counts: dict[str, int] = field(default_factory=dict)
    fragment_attachment_counts: dict[str, int] = field(default_factory=dict)
    markush_layout_counts: dict[str, int] = field(default_factory=dict)
    markush_realism_counts: dict[str, int] = field(default_factory=dict)
    image_quality_counts: dict[str, int] = field(default_factory=dict)
    numeric_summaries: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "csv_path": self.csv_path,
            "schema_version": POSE_FACTORY_SCHEMA_VERSION,
            "row_count": self.row_count,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "trainable": self.trainable,
            "source_counts": self.source_counts,
            "structure_type_counts": self.structure_type_counts,
            "endpoint_side_counts": self.endpoint_side_counts,
            "endpoint_mark_counts": self.endpoint_mark_counts,
            "fragment_geometry_counts": self.fragment_geometry_counts,
            "render_style_counts": self.render_style_counts,
            "issue_severity_counts": self.issue_severity_counts,
            "issue_code_counts": self.issue_code_counts,
            "missing_required_field_counts": self.missing_required_field_counts,
            "quality_gate_counts": self.quality_gate_counts,
            "graph_consistency_counts": self.graph_consistency_counts,
            "molnextr_pose_counts": self.molnextr_pose_counts,
            "fragment_attachment_counts": self.fragment_attachment_counts,
            "markush_layout_counts": self.markush_layout_counts,
            "markush_realism_counts": self.markush_realism_counts,
            "image_quality_counts": self.image_quality_counts,
            "numeric_summaries": self.numeric_summaries,
            "issues": [issue.__dict__ for issue in self.issues],
        }


def external_backend_plan() -> dict[str, Any]:
    plan = PoseFactoryPlan()
    return {
        "schema_version": plan.schema_version,
        "required_backends": [reference.__dict__ for reference in plan.required_backends],
        "implementation_policy": {
            "no_local_renderer_revival": True,
            "base_molnextr_checkpoint_immutable": True,
            "complete_rows_bypass_sidecar": True,
            "generated_rows_must_be_pose_aware": True,
        },
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def validate_formal_sim_metadata(render_quality: dict[str, Any]) -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []
    lineage = render_quality.get("dataset_lineage") if isinstance(render_quality.get("dataset_lineage"), dict) else {}
    policy = render_quality.get("simulation_policy") if isinstance(render_quality.get("simulation_policy"), dict) else {}
    if not lineage and not policy:
        return issues
    if not lineage:
        issues.append(("formal_sim_lineage_missing", "formal-sim rows require dataset_lineage"))
    if not policy:
        issues.append(("formal_sim_policy_missing", "formal-sim rows require simulation_policy"))
    missing_lineage = sorted(FORMAL_SIM_REQUIRED_LINEAGE_KEYS - set(lineage))
    missing_policy = sorted(FORMAL_SIM_REQUIRED_POLICY_KEYS - set(policy))
    if missing_lineage:
        issues.append(("formal_sim_lineage_keys_missing", ",".join(missing_lineage)))
    if missing_policy:
        issues.append(("formal_sim_policy_keys_missing", ",".join(missing_policy)))
    if lineage and lineage.get("schema_version") != FORMAL_SIM_LINEAGE_SCHEMA_VERSION:
        issues.append(("formal_sim_lineage_schema_mismatch", "dataset_lineage schema_version is not accepted"))
    if policy and policy.get("schema_version") != FORMAL_SIM_POLICY_SCHEMA_VERSION:
        issues.append(("formal_sim_policy_schema_mismatch", "simulation_policy schema_version is not accepted"))
    if lineage and lineage.get("dataset_family") != FORMAL_SIM_DATASET_FAMILY:
        issues.append(("formal_sim_lineage_family_mismatch", "dataset_lineage dataset_family is not accepted"))
    if policy and policy.get("dataset_family") != FORMAL_SIM_DATASET_FAMILY:
        issues.append(("formal_sim_policy_family_mismatch", "simulation_policy dataset_family is not accepted"))
    if policy and policy.get("research_only") is not False:
        issues.append(("formal_sim_policy_research_only", "formal-sim simulation_policy must have research_only=false"))
    if policy and policy.get("debug_only") is not False:
        issues.append(("formal_sim_policy_debug_only", "formal-sim simulation_policy must have debug_only=false"))
    if policy and policy.get("formal_capable") is not True:
        issues.append(("formal_sim_policy_not_formal_capable", "formal-sim simulation_policy must be formal_capable=true"))
    if policy and not isinstance(policy.get("allowed_operations"), list):
        issues.append(("formal_sim_policy_operations_invalid", "simulation_policy allowed_operations must be a list"))
    if lineage and not str(lineage.get("parent_source_group") or "").strip():
        issues.append(("formal_sim_parent_source_group_missing", "dataset_lineage parent_source_group is required"))
    return issues
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def as_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalized_point(value: Any) -> tuple[float | None, float | None]:
    if isinstance(value, dict):
        return as_float(value.get("x")), as_float(value.get("y"))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return as_float(value[0]), as_float(value[1])
    return None, None


def point_in_unit_square(x: float | None, y: float | None) -> bool:
    return x is not None and y is not None and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def bbox_area(cell: Any) -> float:
    if not isinstance(cell, dict):
        return 0.0
    bbox = cell.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return 0.0
    values = [as_float(item) for item in bbox]
    if any(value is None for value in values):
        return 0.0
    x1, y1, x2, y2 = values
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def real_atom_count(atom_coordinates: Any) -> int:
    if not isinstance(atom_coordinates, list):
        return 0
    return sum(1 for item in atom_coordinates if isinstance(item, dict) and str(item.get("token") or "") in REAL_ATOM_TOKENS)


def literal_star_markush_issues(render_quality: dict[str, Any], atom_coordinates: Any, cells: Any) -> list[str]:
    issues: list[str] = []
    graph = render_quality.get("graph_consistency") if isinstance(render_quality.get("graph_consistency"), dict) else {}
    visible_star_indices = graph.get("visible_star_token_indices")
    if isinstance(visible_star_indices, list) and visible_star_indices:
        issues.append("markush_visible_literal_star_atom_tokens")
    if any(
        isinstance(atom, dict) and str(atom.get("token") or "").strip() == "*"
        for atom in (atom_coordinates if isinstance(atom_coordinates, list) else [])
    ):
        issues.append("markush_atom_coordinates_contain_literal_star_token")
    if any(
        isinstance(cell, dict) and str(cell.get("text") or "").strip() == "*"
        for cell in (cells if isinstance(cells, list) else [])
    ):
        issues.append("markush_ocr_cells_contain_literal_star_text")
    return issues


def bool_bucket(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "missing"
    return "other"


def formal_nonlinear_pose_contract_passed(render_quality: dict[str, Any]) -> bool:
    pose_alignment = render_quality.get("pose_alignment") if isinstance(render_quality.get("pose_alignment"), dict) else {}
    quality_gates = render_quality.get("quality_gates") if isinstance(render_quality.get("quality_gates"), dict) else {}
    warp = (
        render_quality.get("nonlinear_document_warp")
        if isinstance(render_quality.get("nonlinear_document_warp"), dict)
        else {}
    )
    preservation = (
        render_quality.get("nonlinear_pose_preservation")
        if isinstance(render_quality.get("nonlinear_pose_preservation"), dict)
        else {}
    )
    geometry = render_quality.get("svg_bond_geometry") if isinstance(render_quality.get("svg_bond_geometry"), dict) else {}
    return (
        pose_alignment.get("image_to_graph_orientation_alignment") is True
        and pose_alignment.get("coordinate_mutation_after_render") is True
        and pose_alignment.get("synchronized_after_augmentation") is True
        and pose_alignment.get("formal_nonlinear_document_warp") is True
        and pose_alignment.get("nonlinear_pose_preservation_policy") == FORMAL_NONLINEAR_WARP_POLICY
        and warp.get("enabled") is True
        and warp.get("research_only") is not True
        and warp.get("debug_only") is not True
        and warp.get("formal_training_allowed") is True
        and warp.get("policy") == FORMAL_NONLINEAR_WARP_POLICY
        and warp.get("image_synchronized") is True
        and warp.get("atom_coordinates_synchronized") is True
        and warp.get("ocr_boxes_synchronized") is True
        and warp.get("svg_bond_axis_line_anchors_synchronized") is True
        and int(warp.get("atom_outside_count") or 0) == 0
        and int(warp.get("invalid_bbox_count") or 0) == 0
        and preservation.get("schema_version") == "markush_nonlinear_pose_preservation_v1"
        and preservation.get("policy") == FORMAL_NONLINEAR_WARP_POLICY
        and preservation.get("passed") is True
        and geometry.get("schema_version") == "markush_svg_bond_geometry_v1"
        and geometry.get("nonlinear_warp_synchronized") is True
        and quality_gates.get("formal_nonlinear_document_warp_allowed") is True
        and quality_gates.get("formal_nonlinear_pose_preservation_passed") is True
        and quality_gates.get("svg_bond_axis_line_anchors_synchronized") is True
    )


def fragment_formal_nonlinear_contract_passed(render_quality: dict[str, Any]) -> bool:
    pose_alignment = render_quality.get("pose_alignment") if isinstance(render_quality.get("pose_alignment"), dict) else {}
    quality_gates = render_quality.get("quality_gates") if isinstance(render_quality.get("quality_gates"), dict) else {}
    warp = (
        render_quality.get("fragment_nonlinear_document_warp")
        if isinstance(render_quality.get("fragment_nonlinear_document_warp"), dict)
        else {}
    )
    preservation = (
        render_quality.get("fragment_nonlinear_pose_preservation")
        if isinstance(render_quality.get("fragment_nonlinear_pose_preservation"), dict)
        else {}
    )
    return (
        render_quality.get("structure_type") in FRAGMENT_STRUCTURE_TYPES
        and pose_alignment.get("image_to_graph_orientation_alignment") is True
        and pose_alignment.get("coordinate_mutation_after_render") is True
        and pose_alignment.get("synchronized_after_augmentation") is True
        and pose_alignment.get("formal_nonlinear_document_warp") is True
        and pose_alignment.get("nonlinear_pose_preservation_policy") == FRAGMENT_FORMAL_NONLINEAR_WARP_POLICY
        and warp.get("schema_version") == "fragment_nonlinear_document_warp_v1"
        and warp.get("enabled") is True
        and warp.get("policy") == FRAGMENT_FORMAL_NONLINEAR_WARP_POLICY
        and warp.get("research_only") is not True
        and warp.get("debug_only") is not True
        and warp.get("formal_training_allowed") is True
        and warp.get("image_synchronized") is True
        and warp.get("atom_coordinates_synchronized") is True
        and warp.get("endpoint_coordinates_synchronized") is True
        and warp.get("connector_anchors_synchronized") is True
        and warp.get("terminal_mark_geometry_synchronized") is True
        and int(warp.get("atom_outside_count") or 0) == 0
        and warp.get("blank") is not True
        and warp.get("dense") is not True
        and preservation.get("schema_version") == "fragment_formal_nonlinear_document_warp_contract_v1"
        and preservation.get("policy") == FRAGMENT_FORMAL_NONLINEAR_WARP_POLICY
        and preservation.get("passed") is True
        and preservation.get("endpoint_matches_dummy_atom") is True
        and preservation.get("anchor_coord_matches_anchor_atom") is True
        and preservation.get("anchor_label_matches_atom_token") is True
        and preservation.get("anchor_dummy_bond_present") is True
        and preservation.get("single_dummy_atom") is True
        and preservation.get("connector_samples_valid") is True
        and preservation.get("terminal_mark_samples_valid") is True
        and quality_gates.get("fragment_formal_nonlinear_document_warp_allowed") is True
        and quality_gates.get("fragment_formal_nonlinear_contract_passed") is True
        and quality_gates.get("fragment_connector_mark_anchors_synchronized") is True
    )


def ordinary_document_context_pose_passed(render_quality: dict[str, Any]) -> bool:
    pose_alignment = render_quality.get("pose_alignment") if isinstance(render_quality.get("pose_alignment"), dict) else {}
    quality_gates = render_quality.get("quality_gates") if isinstance(render_quality.get("quality_gates"), dict) else {}
    domain = (
        render_quality.get("ordinary_document_domain_policy")
        if isinstance(render_quality.get("ordinary_document_domain_policy"), dict)
        else {}
    )
    operations = domain.get("operations") if isinstance(domain.get("operations"), list) else []
    operation_set = {str(item) for item in operations}
    required_operations = {
        "moldepictor_source_ink_only_composite",
        "alpha_antialiased_transparent_white_structure_composite",
        "diverse_patent_paper_before_structure_composite",
    }
    return (
        render_quality.get("structure_type") == "complete_compound"
        and pose_alignment.get("image_to_graph_orientation_alignment") is True
        and pose_alignment.get("coordinate_mutation_after_render") is True
        and pose_alignment.get("synchronized_after_augmentation") is True
        and domain.get("schema_version") == "ordinary_document_domain_policy_v2"
        and domain.get("enabled") is True
        and domain.get("geometry_mutation_allowed") is False
        and domain.get("graph_topology_mutation_allowed") is False
        and domain.get("atom_coordinates_synchronized") is True
        and domain.get("image_to_graph_orientation_alignment_preserved") is True
        and domain.get("attachment_like_rows_allowed") is False
        and domain.get("complete_path") == "direct_original_molnextr"
        and domain.get("molnextr_input_quality_passed") is True
        and str(domain.get("paper_profile") or "") in {"clean_white", "white_scan", "gray_scan", "aged_scan"}
        and required_operations.issubset(operation_set)
        and quality_gates.get("ordinary_molnextr_input_quality_passed") is True
        and quality_gates.get("ordinary_document_context_present") is True
    )


def summarize_numbers(values: list[float]) -> dict[str, float | int]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return {"count": 0}
    return {
        "count": len(finite_values),
        "min": min(finite_values),
        "max": max(finite_values),
        "mean": sum(finite_values) / len(finite_values),
    }


def row_image_path(row: dict[str, Any], base_dir: Path) -> Path:
    raw = str(row.get("file_path") or row.get("image_path") or "").strip()
    if not raw:
        return Path("")
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path)


def image_basic_stats(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on runtime image stack
        return {"readable": False, "error": f"image_dependencies_unavailable:{exc}"}

    try:
        with Image.open(path) as image:
            image = image.convert("L")
            width, height = image.size
            array = np.asarray(image)
    except Exception as exc:
        return {"readable": False, "error": f"image_unreadable:{exc}"}

    if width <= 0 or height <= 0:
        return {"readable": False, "error": "invalid_image_size"}
    dark_ratio = float((array < 245).mean())
    return {
        "readable": True,
        "width": int(width),
        "height": int(height),
        "dark_pixel_ratio": dark_ratio,
        "blank": dark_ratio < 0.0005,
        "dense": dark_ratio > 0.85,
    }


def validate_pose_factory_row(
    row: dict[str, Any],
    *,
    row_index: int,
    base_dir: Path,
    check_images: bool,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    issues: list[ValidationIssue] = []
    source_id = str(row.get("source_id") or "")
    file_path_text = str(row.get("file_path") or row.get("image_path") or "")

    def add(severity: str, code: str, message: str) -> None:
        issues.append(
            ValidationIssue(
                row_index=row_index,
                severity=severity,
                code=code,
                message=message,
                source_id=source_id,
                file_path=file_path_text,
            )
        )

    render_quality = parse_json_object(row.get("render_quality"))
    if not render_quality:
        add("error", "missing_render_quality", "render_quality must be a JSON object")
        render_quality = {}

    missing_keys = sorted(REQUIRED_RENDER_QUALITY_KEYS - set(render_quality))
    if missing_keys:
        add("error", "render_quality_missing_keys", ",".join(missing_keys))
    for code, message in validate_formal_sim_metadata(render_quality):
        add("error", code, message)

    source_arrow = str(row.get("source_arrow") or "")
    if not source_arrow.startswith("pose_factory:"):
        add("error", "invalid_source_arrow", "source_arrow must start with pose_factory:")

    pose_alignment = render_quality.get("pose_alignment")
    if not isinstance(pose_alignment, dict):
        add("error", "missing_pose_alignment", "pose_alignment is required to preserve MolNexTR image-to-graph orientation alignment")
        pose_alignment = {}
    else:
        missing_pose_keys = sorted(POSE_ALIGNMENT_KEYS - set(pose_alignment))
        if missing_pose_keys:
            add("error", "pose_alignment_missing_keys", ",".join(missing_pose_keys))
        formal_nonlinear_pose = formal_nonlinear_pose_contract_passed(render_quality)
        fragment_formal_nonlinear_pose = fragment_formal_nonlinear_contract_passed(render_quality)
        ordinary_document_context_pose = ordinary_document_context_pose_passed(render_quality)
        if pose_alignment.get("image_to_graph_orientation_alignment") is not True:
            add("error", "pose_alignment_not_synchronized", "image-to-graph orientation alignment must be explicitly preserved")
        if (
            pose_alignment.get("coordinate_mutation_after_render") is not False
            and not formal_nonlinear_pose
            and not fragment_formal_nonlinear_pose
            and not ordinary_document_context_pose
        ):
            add(
                "error",
                "pose_alignment_mutated_after_render",
                "pose coordinates may mutate after render only with a passing formal nonlinear Markush/fragment contract or ordinary document-context pose contract",
            )
        if not str(pose_alignment.get("orientation_policy") or ""):
            add("error", "pose_alignment_missing_policy", "pose_alignment must record the orientation policy")

    structure_type = str(render_quality.get("structure_type") or row.get("structure_type") or "")
    if structure_type not in SUPPORTED_STRUCTURE_TYPES:
        add("error", "invalid_structure_type", f"unsupported structure_type={structure_type!r}")

    backend_references = render_quality.get("backend_references")
    if not isinstance(backend_references, list) or not backend_references:
        add("error", "missing_backend_references", "external backend references are required")

    atom_coordinates = render_quality.get("atom_coordinates")
    atom_coordinate_count = 0
    atom_coordinate_bad_count = 0
    atom_coordinate_by_index: dict[int, dict[str, Any]] = {}
    if not isinstance(atom_coordinates, list) or not atom_coordinates:
        add("error", "missing_atom_coordinates", "atom coordinates are required for MolNexTR pose supervision")
    else:
        atom_coordinate_count = len(atom_coordinates)
        for item in atom_coordinates:
            if isinstance(item, dict):
                try:
                    atom_coordinate_by_index[int(item.get("atom_index"))] = item
                except (TypeError, ValueError):
                    pass
            x, y = normalized_point(item)
            if not point_in_unit_square(x, y):
                atom_coordinate_bad_count += 1
        if atom_coordinate_bad_count:
            add("error", "atom_coordinate_out_of_range", f"{atom_coordinate_bad_count} atom coordinates are outside [0,1]")

    bonds = render_quality.get("bonds")
    bond_count = len(bonds) if isinstance(bonds, list) else 0
    if isinstance(atom_coordinates, list) and not isinstance(bonds, list):
        add("error", "missing_bonds", "bond list is required")

    graph_consistency = render_quality.get("graph_consistency")
    if not isinstance(graph_consistency, dict):
        add("error", "missing_graph_consistency", "graph_consistency is required")
        graph_consistency = {}
    else:
        if graph_consistency.get("row_smiles_canonical_matches_mol") is not True:
            add("error", "smiles_graph_mismatch", "row SMILES must canonicalize to the rendered molecule")
        atom_count = int(graph_consistency.get("atom_count") or -1)
        if isinstance(atom_coordinates, list) and atom_count != len(atom_coordinates):
            add("error", "atom_count_coordinate_mismatch", "graph atom count must match atom coordinate count")
        if isinstance(bonds, list) and int(graph_consistency.get("bond_count") or -1) != len(bonds):
            add("error", "bond_count_mismatch", "graph bond count must match bond record count")

    image_width = as_float(render_quality.get("image_width") or row.get("image_width"))
    image_height = as_float(render_quality.get("image_height") or row.get("image_height"))
    if not image_width or not image_height or image_width <= 0 or image_height <= 0:
        add("error", "invalid_image_dimensions", "image_width/image_height must be positive")

    endpoint_side = str(render_quality.get("attachment_direction") or row.get("endpoint_side") or "")
    endpoint_mark = str(render_quality.get("attachment_render_mode") or "")
    render_geometry = str(render_quality.get("attachment_render_geometry") or "")
    missing_fragment_keys: list[str] = []
    endpoint_x: float | None = None
    endpoint_y: float | None = None
    wavy_geometry: dict[str, Any] = {}
    mark_geometry: dict[str, Any] = {}
    fragment_pixel_rmse: float | None = None
    fragment_pixel_rmse_threshold: float | None = None
    fragment_pixel_abs_p95: float | None = None
    fragment_pixel_abs_p95_threshold: float | None = None
    fragment_pixel_abs_max: float | None = None
    fragment_pixel_abs_max_threshold: float | None = None
    fragment_pixel_fit_points = 0
    fragment_molnextr_wavy_length_384: float | None = None
    fragment_molnextr_mark_length_384: float | None = None
    fragment_molnextr_connector_length_384: float | None = None
    fragment_molnextr_visible_connector_length_384: float | None = None
    fragment_molnextr_straight_connector_length_384: float | None = None
    fragment_molnextr_min_atom_pair_384: float | None = None
    fragment_document_realism_policy = ""
    fragment_visual_quality_passed: bool | None = None
    fragment_document_realism_machine_audit_passed: bool | None = None
    fragment_pixel_geometry_passed: bool | None = None
    fragment_molnextr_input_quality_passed: bool | None = None
    fragment_attachment_bond_semantics_passed: bool | None = None
    fragment_visual_wavy_not_rdkit_stereo: bool | None = None
    markush_ocr_cell_count = 0
    markush_bad_box_count = 0
    markush_real_atom_count = 0
    markush_ocr_cell_area_sum = 0.0
    markush_ocr_cell_area_max = 0.0
    markush_pose_rmse: float | None = None
    markush_pose_threshold: float | None = None
    markush_pose_fit_points = 0
    markush_realism_policy = ""
    markush_realism_status = ""
    markush_realism_machine_audit_passed: bool | None = None
    markush_realism_manual_review_required: bool | None = None
    markush_realism_missing_render_parameter_count = 0
    markush_realism_font_size: float | None = None
    markush_realism_stroke_ratio: float | None = None
    markush_molnextr_min_atom_pair_384: float | None = None
    markush_molnextr_min_ocr_cell_side_384: float | None = None
    markush_molnextr_dark_pixel_ratio: float | None = None
    markush_molnextr_purewhite_pixel_ratio: float | None = None
    if structure_type in FRAGMENT_STRUCTURE_TYPES:
        missing_fragment_keys = sorted(FRAGMENT_RENDER_QUALITY_KEYS - set(render_quality))
        if missing_fragment_keys:
            add("error", "fragment_keys_missing", ",".join(missing_fragment_keys))
        endpoint_x, endpoint_y = normalized_point(render_quality.get("attachment_endpoint"))
        if not point_in_unit_square(endpoint_x, endpoint_y):
            add("error", "endpoint_out_of_range", "fragment endpoint must be normalized to [0,1]")
        if endpoint_side not in {"left", "right", "top", "bottom"}:
            add("error", "invalid_endpoint_side", f"endpoint side={endpoint_side!r}")
        if endpoint_mark not in SUPPORTED_ENDPOINT_MARKS:
            add("error", "invalid_endpoint_mark", f"endpoint mark={endpoint_mark!r}")
        if structure_type == "wavy_fragment" and endpoint_mark != "wavy":
            add("error", "wavy_fragment_requires_wavy_mark", "wavy_fragment rows must use attachment_render_mode=wavy")
        if render_geometry not in SUPPORTED_FRAGMENT_GEOMETRIES:
            add(
                "error",
                "unsupported_fragment_geometry",
                f"fragment attachment geometry is not supported: {render_geometry!r}",
            )
        if endpoint_mark == "wavy" and render_geometry != "custom_markush_attachment_perpendicular_wavy":
            add(
                "error",
                "wavy_fragment_requires_custom_markush_perpendicular_geometry",
                "wavy fragment rows must use the custom terminal perpendicular Markush attachment primitive, not along-bond or RDKit stereo/UNKNOWN wavy",
            )
        if endpoint_mark == "cut" and render_geometry not in {
            "straight_connector_with_terminal_perpendicular_cut_bar",
            "rdkit_moldraw2d_terminal_perpendicular_cut_bar",
            "straight_connector_with_terminal_double_cut_bar",
            "straight_connector_with_terminal_stub_cut",
        }:
            add("error", "cut_fragment_requires_cut_geometry", "cut rows must use a terminal cut-bar or stub geometry")
        if endpoint_mark == "query_attachment" and render_geometry != "straight_connector_with_terminal_query_label":
            add("error", "query_fragment_requires_query_geometry", "query attachment rows must use a terminal query-label geometry")
        if endpoint_mark == "dummy_atom" and render_geometry != "straight_connector_with_terminal_dummy_atom":
            add("error", "dummy_fragment_requires_dummy_geometry", "dummy attachment rows must use a terminal dummy-atom geometry")
        if not str(render_quality.get("attachment_anchor") or ""):
            add("error", "missing_attachment_anchor", "fragment rows require an anchor token")
        if graph_consistency.get("single_dummy_atom") is not True:
            add("error", "fragment_requires_single_dummy_atom", "fragment rows require exactly one dummy attachment atom")
        if graph_consistency.get("anchor_dummy_bond_present") is not True:
            add("error", "fragment_requires_anchor_dummy_bond", "anchor and dummy attachment atom must be bonded")
        endpoint_length_policy = render_quality.get("endpoint_length_policy")
        if not isinstance(endpoint_length_policy, dict):
            add("error", "missing_endpoint_length_policy", "fragment rows must declare endpoint length and coordinate mutation policy")
        else:
            fragment_formal_nonlinear_pose = fragment_formal_nonlinear_contract_passed(render_quality)
            if (
                endpoint_length_policy.get("coordinate_mutation_after_render") is not False
                and not fragment_formal_nonlinear_pose
            ):
                add(
                    "error",
                    "endpoint_coordinate_mutated_after_render",
                    "fragment endpoint coordinates may mutate after render only with a passing fragment formal nonlinear contract",
                )
            connector_length = as_float(endpoint_length_policy.get("connector_length_px"))
            minimum_connector_length = as_float(endpoint_length_policy.get("minimum_connector_length_px"))
            if minimum_connector_length is not None and minimum_connector_length > 0.0:
                if connector_length is None or connector_length + 1e-6 < minimum_connector_length:
                    add(
                        "error",
                        "endpoint_connector_shorter_than_declared_minimum",
                        "fragment connector length must satisfy the declared native-render minimum",
                    )
        dummy_index = graph_consistency.get("dummy_index")
        anchor_index = graph_consistency.get("anchor_index")
        try:
            dummy_coord = atom_coordinate_by_index[int(dummy_index)]
        except (KeyError, TypeError, ValueError):
            dummy_coord = None
        if not isinstance(dummy_coord, dict):
            add("error", "fragment_missing_dummy_atom_coordinate", "attachment dummy atom must have a coordinate row")
        else:
            dummy_x, dummy_y = normalized_point(dummy_coord)
            if (
                endpoint_x is None
                or endpoint_y is None
                or dummy_x is None
                or dummy_y is None
                or abs(endpoint_x - dummy_x) > 1e-6
                or abs(endpoint_y - dummy_y) > 1e-6
            ):
                add("error", "fragment_endpoint_dummy_coordinate_mismatch", "attachment_endpoint must match the dummy atom coordinate")
        try:
            anchor_coord_row = atom_coordinate_by_index[int(anchor_index)]
        except (KeyError, TypeError, ValueError):
            anchor_coord_row = None
        if not isinstance(anchor_coord_row, dict):
            add("error", "fragment_missing_anchor_atom_coordinate", "attachment anchor atom must have a coordinate row")
        elif str(render_quality.get("attachment_anchor") or "") != str(anchor_coord_row.get("token") or ""):
            add("error", "fragment_anchor_label_token_mismatch", "attachment_anchor must match the anchor atom token")
        if endpoint_mark == "wavy":
            wavy_geometry = parse_json_object(render_quality.get("wavy_geometry"))
            length_ratio = as_float(wavy_geometry.get("wavy_length_to_connector_ratio"))
            dot_abs = as_float(wavy_geometry.get("wavy_axis_dot_connector_abs"))
            cycles = as_float(wavy_geometry.get("wavy_cycles"))
            visible_connector_length = as_float(wavy_geometry.get("visible_connector_length_px"))
            straight_connector_length = as_float(wavy_geometry.get("straight_connector_length_px"))
            anchor_depiction_mode = str(render_quality.get("attachment_anchor_depiction_mode") or "")
            anchor_label_is_visible_text = render_quality.get("attachment_anchor_label_is_visible_text") is True
            if render_geometry == "custom_markush_attachment_perpendicular_wavy":
                if wavy_geometry.get("not_rdkit_stereo_wavy") is not True:
                    add("error", "custom_wavy_must_not_be_rdkit_stereo", "custom Markush attachment wavy must not use RDKit stereo/UNKNOWN bond semantics")
                if str(wavy_geometry.get("short_wavy_connector_policy") or "") != SHORT_WAVY_CONNECTOR_POLICY:
                    add(
                        "error",
                        "custom_wavy_short_connector_policy_missing",
                        "custom Markush wavy rows must use the production short patent connector policy",
                    )
                if length_ratio is None or not 0.35 <= length_ratio <= 2.80:
                    add(
                        "error",
                        "invalid_custom_markush_perpendicular_wavy_length_ratio",
                        "custom Markush wavy must be a local terminal mark scaled to the connector",
                    )
                if dot_abs is None or dot_abs > 0.15:
                    add(
                        "error",
                        "invalid_custom_markush_perpendicular_wavy_axis_alignment",
                        "custom Markush wavy must run approximately perpendicular to the attachment connector",
                    )
                if cycles is None or not 2.0 <= cycles <= 5.8:
                    add("error", "invalid_custom_markush_perpendicular_wavy_cycle_count", "custom Markush wavy must use a visible patent-style terminal squiggle")
                if visible_connector_length is None or visible_connector_length <= 0.0:
                    add("error", "custom_wavy_visible_connector_length_missing", "custom Markush wavy rows must record visible connector length")
                if straight_connector_length is None or straight_connector_length <= 0.0:
                    add("error", "custom_wavy_straight_connector_length_missing", "custom Markush wavy rows must record straight connector length")
                measured_line_width = as_float(wavy_geometry.get("measured_native_line_width_px"))
                draw_line_width = as_float(wavy_geometry.get("draw_line_width_px"))
                if measured_line_width is None or measured_line_width <= 0.0:
                    add("error", "custom_wavy_native_line_width_missing", "custom Markush wavy rows must record measured native bond width")
                if draw_line_width is None or draw_line_width <= 0.0:
                    add("error", "custom_wavy_draw_line_width_missing", "custom Markush wavy rows must record actual drawn custom line width")
                if (
                    measured_line_width is not None
                    and draw_line_width is not None
                    and draw_line_width > max(1.15, measured_line_width * 1.02)
                ):
                    add("error", "custom_wavy_draw_line_width_too_thick", "custom Markush wavy/connector stroke must not exceed native bond stroke width")
                if wavy_geometry.get("line_width_matches_native_bonds") is not True:
                    add("error", "custom_wavy_line_width_not_native_bond_width", "custom Markush wavy and connector must match the native molecule bond stroke width")
                if str(wavy_geometry.get("line_width_source") or "") not in {
                    "rdkit_native_bond_line_width",
                    "final_rendered_native_attachment_bond_width",
                }:
                    add("error", "custom_wavy_line_width_source_not_native", "custom Markush wavy line width must come from the RDKit native bond line width")
                if not str(wavy_geometry.get("depiction_profile") or ""):
                    add("error", "custom_wavy_missing_depiction_profile", "custom Markush wavy rows must record the ChemDraw/Marvin-like depiction profile")
                font_range = wavy_geometry.get("atom_font_size_range_px") if isinstance(wavy_geometry.get("atom_font_size_range_px"), list) else []
                if len(font_range) != 2:
                    add("error", "custom_wavy_missing_atom_font_profile", "custom Markush wavy rows must record atom font size range")
                draw_start = (
                    wavy_geometry.get("connector_draw_start")
                    if isinstance(wavy_geometry.get("connector_draw_start"), dict)
                    else {}
                )
                erase_start = (
                    wavy_geometry.get("native_connector_erase_start")
                    if isinstance(wavy_geometry.get("native_connector_erase_start"), dict)
                    else {}
                )
                if not draw_start or not erase_start:
                    add(
                        "error",
                        "custom_wavy_missing_connector_draw_start",
                        "custom wavy rows must record the redrawn connector start and native erase start",
                    )
                else:
                    draw_start_gap = math.hypot(
                        float(draw_start.get("x") or 0.0) - float(erase_start.get("x") or 0.0),
                        float(draw_start.get("y") or 0.0) - float(erase_start.get("y") or 0.0),
                    )
                    draw_width_for_gap = draw_line_width if draw_line_width is not None else 1.0
                    if draw_start_gap > max(0.95, float(draw_width_for_gap) * 0.65):
                        add(
                            "error",
                            "custom_wavy_connector_draw_start_not_erased_start",
                            "custom wavy redrawn connector must start where the native dummy bond was erased",
                        )
                start_policy = (
                    wavy_geometry.get("visible_connector_start_policy")
                    if isinstance(wavy_geometry.get("visible_connector_start_policy"), dict)
                    else {}
                )
                clearance = (
                    wavy_geometry.get("visible_anchor_label_clearance")
                    if isinstance(wavy_geometry.get("visible_anchor_label_clearance"), dict)
                    else {}
                )
                if start_policy.get("protects_visible_atom_label") is True:
                    if clearance.get("passed") is not True:
                        add(
                            "error",
                            "custom_wavy_visible_anchor_label_clearance_failed",
                            "custom wavy connector must start outside visible anchor atom labels",
                        )
                    measured_start = as_float(clearance.get("measured_visible_start_offset_px"))
                    minimum_start = as_float(clearance.get("minimum_visible_start_offset_px"))
                    if measured_start is None or minimum_start is None or measured_start + 1e-6 < minimum_start:
                        add(
                            "error",
                            "custom_wavy_connector_starts_inside_anchor_label",
                            "custom wavy connector starts too close to the visible anchor label",
                        )
                    clearance_straight = as_float(clearance.get("straight_connector_length_px"))
                    minimum_straight = as_float(clearance.get("minimum_straight_connector_length_px"))
                    if clearance_straight is None or minimum_straight is None or clearance_straight + 1e-6 < minimum_straight:
                        add(
                            "error",
                            "custom_wavy_connector_too_short_after_anchor_label",
                            "custom wavy connector must remain visibly connected after label clearance",
                        )
                if anchor_depiction_mode == "visible_carbon_attachment_label" or anchor_label_is_visible_text:
                    if start_policy.get("protects_visible_atom_label") is not True:
                        add(
                            "error",
                            "custom_wavy_visible_anchor_label_not_protected",
                            "custom wavy connector must protect explicit carbon and hetero atom labels",
                        )
                    if wavy_geometry.get("anchor_label_is_visible_text") is not True:
                        add(
                            "error",
                            "custom_wavy_visible_anchor_label_detection_missing",
                            "custom wavy geometry must record whether the anchor label is actually visible",
                        )
                if anchor_depiction_mode == "implicit_carbon_skeleton_endpoint" and anchor_label_is_visible_text:
                    add(
                        "error",
                        "implicit_carbon_declared_as_visible_text",
                        "implicit carbon skeleton endpoints must not be marked as visible atom labels",
                    )
                junction_style = str(wavy_geometry.get("wavy_connector_intersection_style") or "")
                if junction_style not in {"center_cross", "side_touch"}:
                    add(
                        "error",
                        "invalid_custom_markush_perpendicular_wavy_junction_style",
                        "custom terminal perpendicular wavy must declare whether the connector crosses the mark center or touches the side",
                    )
                if junction_style == "side_touch" and wavy_geometry.get("side_touch_generation_allowed") is not True:
                    add(
                        "error",
                        "custom_wavy_side_touch_not_allowed_in_production",
                        "production custom wavy fragments must use center-cross unless a side-touch dataset explicitly enables it",
                    )
                center_distance = as_float(wavy_geometry.get("connector_wavy_center_distance_px"))
                if junction_style == "center_cross":
                    if wavy_geometry.get("connector_crosses_wavy_center") is not True:
                        add("error", "custom_wavy_center_cross_not_declared", "center-cross terminal wavy must declare connector crossing through center")
                    if wavy_geometry.get("center_cross_curve_passes_connector") is not True:
                        add("error", "custom_wavy_center_cross_curve_not_declared", "center-cross terminal wavy curve must pass through the connector endpoint")
                    if wavy_geometry.get("center_cross_connector_is_solid_line") is not True:
                        add("error", "custom_wavy_center_cross_not_solid_connector", "center-cross terminal wavy connector must be a solid line through the wavy center")
                    if center_distance is None or center_distance > 2.5:
                        add("error", "custom_wavy_connector_not_through_center", "center-cross terminal wavy center must lie on connector endpoint")
                elif junction_style == "side_touch":
                    if wavy_geometry.get("connector_crosses_wavy_center") is True:
                        add("error", "custom_wavy_side_touch_declares_center_cross", "side-touch terminal wavy must not declare center crossing")
                    if wavy_geometry.get("connector_contact_is_terminal_wavy_endpoint") is not True:
                        add("error", "custom_wavy_side_touch_endpoint_contact_not_declared", "side-touch terminal wavy must declare endpoint contact")
                externality = parse_json_object(render_quality.get("terminal_wavy_externality"))
                if externality and externality.get("passed") is not True:
                    add("error", "terminal_wavy_externality_failed", "custom terminal wavy must remain outside the non-dummy molecule body")
            elif length_ratio is None or not 0.55 <= length_ratio <= 1.30:
                add(
                    "error",
                    "invalid_wavy_length_ratio",
                    "wavy cut length must be tied to connector length, not a fixed tiny mark or a long stereo-any bond",
                )
            if render_geometry != "custom_markush_attachment_perpendicular_wavy" and (dot_abs is None or dot_abs > 0.12):
                add(
                    "error",
                    "invalid_wavy_perpendicularity",
                    "wavy cut axis must be approximately perpendicular to the connector",
                )
            if render_geometry != "custom_markush_attachment_perpendicular_wavy" and (cycles is None or not 2.6 <= cycles <= 5.8):
                add("error", "invalid_wavy_cycle_count", "wavy cut should use a literature-like multi-cycle mark")
        elif endpoint_mark in {"cut", "query_attachment", "dummy_atom"}:
            mark_geometry = parse_json_object(render_quality.get("fragment_mark_geometry"))
            length_ratio = as_float(mark_geometry.get("mark_length_to_connector_ratio"))
            dot_abs = as_float(mark_geometry.get("mark_axis_dot_connector_abs"))
            if endpoint_mark == "cut" and render_geometry == "rdkit_moldraw2d_terminal_perpendicular_cut_bar":
                if length_ratio is None or not 0.45 <= length_ratio <= 2.80:
                    add("error", "invalid_cut_mark_length_ratio", "renderer-native cut mark length must remain visually local to the connector")
            elif endpoint_mark == "cut" and (length_ratio is None or not 0.35 <= length_ratio <= 1.15):
                add("error", "invalid_cut_mark_length_ratio", "cut mark length must be tied to connector length")
            if dot_abs is None or dot_abs > 0.16:
                add("error", "invalid_fragment_mark_perpendicularity", "terminal fragment mark axis must be approximately perpendicular to the connector")
        fragment_document_realism = (
            render_quality.get("document_realism") if isinstance(render_quality.get("document_realism"), dict) else {}
        )
        fragment_visual_quality = (
            render_quality.get("fragment_visual_quality") if isinstance(render_quality.get("fragment_visual_quality"), dict) else {}
        )
        fragment_pixel_geometry = (
            render_quality.get("fragment_pixel_geometry_detection")
            if isinstance(render_quality.get("fragment_pixel_geometry_detection"), dict)
            else {}
        )
        fragment_molnextr_quality = (
            render_quality.get("fragment_molnextr_input_quality")
            if isinstance(render_quality.get("fragment_molnextr_input_quality"), dict)
            else {}
        )
        fragment_document_realism_policy = str(fragment_document_realism.get("policy") or "")
        fragment_visual_quality_passed = fragment_visual_quality.get("passed") is True
        if not fragment_document_realism:
            add("error", "fragment_missing_document_realism", "fragment rows must include Markush-style document realism evidence")
        else:
            if fragment_document_realism.get("schema_version") != "fragment_document_realism_v1":
                add("error", "fragment_document_realism_schema_mismatch", "fragment document realism schema is not accepted")
            if fragment_document_realism_policy != "patent_literature_markush_fragment_document_crop_v1":
                add("error", "fragment_document_realism_policy_mismatch", "fragment document realism policy is not accepted")
            if fragment_document_realism.get("machine_audit_passed") is not True:
                add("error", "fragment_document_realism_machine_audit_failed", "fragment document realism machine audit must pass")
            if fragment_document_realism.get("manual_visual_review_required") is not True:
                add("error", "fragment_document_realism_manual_review_not_required", "fragment rows must keep manual visual review required")
            if fragment_document_realism.get("manual_visual_review_passed") is True:
                add("error", "fragment_document_realism_manual_review_overclaimed", "row-level fragment generator must not self-certify manual review")
            fragment_document_realism_machine_audit_passed = fragment_document_realism.get("machine_audit_passed") is True
            if endpoint_mark == "wavy" and fragment_document_realism.get("custom_markush_attachment_primitive") is not True:
                add("error", "fragment_document_realism_missing_custom_wavy_primitive", "fragment realism must record the custom Markush attachment primitive")
            if fragment_document_realism.get("not_rdkit_stereo_wavy") is not True and endpoint_mark == "wavy":
                add("error", "fragment_document_realism_allows_rdkit_stereo_wavy", "wavy fragment realism must forbid RDKit stereo/UNKNOWN wavy semantics")
            external_basis = fragment_document_realism.get("external_basis")
            if not isinstance(external_basis, list) or not external_basis:
                add("error", "fragment_document_realism_missing_external_basis", "fragment realism must cite Markush/MolNexTR basis")
        if not fragment_visual_quality:
            add("error", "fragment_missing_visual_quality", "fragment rows must include visual quality audit evidence")
        else:
            if fragment_visual_quality.get("schema_version") != "fragment_visual_quality_v1":
                add("error", "fragment_visual_quality_schema_mismatch", "fragment visual quality schema is not accepted")
            if fragment_visual_quality.get("passed") is not True:
                add("error", "fragment_visual_quality_failed", "fragment visual quality machine audit must pass")
            if fragment_visual_quality.get("machine_audit_passed") is not True:
                add("error", "fragment_visual_quality_machine_audit_failed", "fragment visual quality machine audit must pass")
            if fragment_visual_quality.get("manual_visual_review_required") is not True:
                add("error", "fragment_visual_quality_manual_review_not_required", "fragment visual quality must keep manual review required")
        if not fragment_pixel_geometry:
            add("error", "fragment_missing_pixel_geometry_detection", "fragment rows must include final-pixel attachment geometry RMSE evidence")
        else:
            fragment_pixel_geometry_passed = fragment_pixel_geometry.get("passed") is True
            fragment_pixel_rmse = as_float(fragment_pixel_geometry.get("line_constraint_rmse_px"))
            fragment_pixel_rmse_threshold = as_float(fragment_pixel_geometry.get("line_constraint_rmse_threshold_px"))
            fragment_pixel_abs_p95 = as_float(fragment_pixel_geometry.get("line_constraint_abs_p95_px"))
            fragment_pixel_abs_p95_threshold = as_float(fragment_pixel_geometry.get("line_constraint_abs_p95_threshold_px"))
            fragment_pixel_abs_max = as_float(fragment_pixel_geometry.get("line_constraint_abs_max_px"))
            fragment_pixel_abs_max_threshold = as_float(fragment_pixel_geometry.get("line_constraint_abs_max_threshold_px"))
            fragment_pixel_fit_points = int(fragment_pixel_geometry.get("line_constraint_fit_point_count") or 0)
            min_fit_points = int(fragment_pixel_geometry.get("line_constraint_min_fit_point_count") or 0)
            if fragment_pixel_geometry.get("schema_version") != "fragment_pixel_geometry_detection_v1":
                add("error", "fragment_pixel_geometry_schema_mismatch", "fragment pixel geometry schema is not accepted")
            if fragment_pixel_geometry.get("passed") is not True:
                add("error", "fragment_pixel_geometry_detection_failed", "fragment final-pixel attachment geometry detection must pass")
            if fragment_pixel_rmse is None or fragment_pixel_rmse_threshold is None or fragment_pixel_rmse > fragment_pixel_rmse_threshold:
                add("error", "fragment_pixel_geometry_rmse_failed", "fragment final-pixel connector/mark RMSE must pass")
            if fragment_pixel_abs_p95 is None or fragment_pixel_abs_p95_threshold is None or fragment_pixel_abs_p95 > fragment_pixel_abs_p95_threshold:
                add("error", "fragment_pixel_geometry_p95_failed", "fragment final-pixel connector/mark p95 residual must pass")
            if fragment_pixel_abs_max is None or fragment_pixel_abs_max_threshold is None or fragment_pixel_abs_max > fragment_pixel_abs_max_threshold:
                add("error", "fragment_pixel_geometry_max_failed", "fragment final-pixel connector/mark max residual must pass")
            if min_fit_points > 0 and fragment_pixel_fit_points < min_fit_points:
                add("error", "fragment_pixel_geometry_too_few_fit_points", "fragment pixel geometry detection has too few fit points")
        if not fragment_molnextr_quality:
            add("error", "fragment_missing_molnextr_input_quality", "fragment rows must include MolNexTR CropWhite+Resize input quality evidence")
        else:
            fragment_molnextr_input_quality_passed = fragment_molnextr_quality.get("passed") is True
            fragment_molnextr_wavy_length_384 = as_float(fragment_molnextr_quality.get("wavy_length_px_at_384"))
            fragment_molnextr_mark_length_384 = as_float(fragment_molnextr_quality.get("mark_length_px_at_384"))
            fragment_molnextr_connector_length_384 = as_float(fragment_molnextr_quality.get("connector_length_px_at_384"))
            fragment_molnextr_visible_connector_length_384 = as_float(fragment_molnextr_quality.get("visible_connector_length_px_at_384"))
            fragment_molnextr_straight_connector_length_384 = as_float(fragment_molnextr_quality.get("straight_connector_length_px_at_384"))
            fragment_molnextr_min_atom_pair_384 = as_float(fragment_molnextr_quality.get("min_atom_pair_distance_px_at_384"))
            quality_thresholds = (
                fragment_molnextr_quality.get("thresholds")
                if isinstance(fragment_molnextr_quality.get("thresholds"), dict)
                else {}
            )
            max_visible_connector_384 = as_float(quality_thresholds.get("max_visible_connector_length_px_at_384"))
            max_straight_connector_384 = as_float(quality_thresholds.get("max_straight_connector_length_px_at_384"))
            max_connector_384 = as_float(quality_thresholds.get("max_connector_length_px_at_384"))
            if fragment_molnextr_quality.get("schema_version") != "fragment_molnextr_input_quality_v1":
                add("error", "fragment_molnextr_input_quality_schema_mismatch", "fragment MolNexTR input quality schema is not accepted")
            if fragment_molnextr_quality.get("passed") is not True:
                add("error", "fragment_molnextr_input_quality_failed", "fragment must remain readable after MolNexTR CropWhite+Resize preprocessing")
            if endpoint_mark == "wavy" and render_geometry == "custom_markush_attachment_perpendicular_wavy":
                if max_visible_connector_384 is None or max_visible_connector_384 > MAX_PATENT_WAVY_VISIBLE_CONNECTOR_PX_AT_384:
                    add(
                        "error",
                        "fragment_molnextr_short_wavy_visible_threshold_missing_or_loose",
                        "custom wavy rows must declare the production short visible-connector threshold",
                    )
                if max_straight_connector_384 is None or max_straight_connector_384 > MAX_PATENT_WAVY_STRAIGHT_CONNECTOR_PX_AT_384:
                    add(
                        "error",
                        "fragment_molnextr_short_wavy_straight_threshold_missing_or_loose",
                        "custom wavy rows must declare the production short straight-connector threshold",
                    )
                if (
                    fragment_molnextr_visible_connector_length_384 is None
                    or fragment_molnextr_visible_connector_length_384 > MAX_PATENT_WAVY_VISIBLE_CONNECTOR_PX_AT_384
                ):
                    add(
                        "error",
                        "fragment_molnextr_visible_connector_too_long_for_patent_wavy",
                        "custom wavy visible connector is longer than the production patent-fragment contract",
                    )
                if (
                    fragment_molnextr_straight_connector_length_384 is None
                    or fragment_molnextr_straight_connector_length_384 > MAX_PATENT_WAVY_STRAIGHT_CONNECTOR_PX_AT_384
                ):
                    add(
                        "error",
                        "fragment_molnextr_straight_connector_too_long_for_patent_wavy",
                        "custom wavy straight connector is longer than the production patent-fragment contract",
                    )
            elif endpoint_mark in {"cut", "dummy_atom", "query_attachment"}:
                expected_max_connector = {
                    "cut": MAX_PATENT_CUT_CONNECTOR_PX_AT_384,
                    "dummy_atom": MAX_PATENT_DUMMY_CONNECTOR_PX_AT_384,
                    "query_attachment": MAX_PATENT_QUERY_CONNECTOR_PX_AT_384,
                }[endpoint_mark]
                if max_connector_384 is None or max_connector_384 > expected_max_connector:
                    add(
                        "error",
                        "fragment_molnextr_connector_threshold_missing_or_loose",
                        "non-wavy attachment fragments must declare a production short-connector threshold",
                    )
                if (
                    fragment_molnextr_connector_length_384 is None
                    or fragment_molnextr_connector_length_384 > expected_max_connector
                ):
                    add(
                        "error",
                        "fragment_molnextr_connector_too_long_for_patent_fragment",
                        "non-wavy attachment connector is longer than the production patent-fragment contract",
                    )
        attachment_bond_semantics = (
            render_quality.get("fragment_attachment_bond_semantics")
            if isinstance(render_quality.get("fragment_attachment_bond_semantics"), dict)
            else {}
        )
        if not attachment_bond_semantics:
            add("error", "fragment_missing_attachment_bond_semantics", "fragment rows must prove visual attachment marks do not alter RDKit bond semantics")
        else:
            fragment_attachment_bond_semantics_passed = attachment_bond_semantics.get("passed") is True
            fragment_visual_wavy_not_rdkit_stereo = attachment_bond_semantics.get("not_rdkit_stereo_wavy") is True
            if attachment_bond_semantics.get("schema_version") != "fragment_attachment_bond_semantics_v1":
                add("error", "fragment_attachment_bond_semantics_schema_mismatch", "fragment attachment bond semantics schema is not accepted")
            if attachment_bond_semantics.get("passed") is not True:
                add("error", "fragment_attachment_bond_semantics_failed", "fragment attachment dummy bond must stay SINGLE/NONE with no RDKit stereo wavy")
            if endpoint_mark == "wavy" and attachment_bond_semantics.get("not_rdkit_stereo_wavy") is not True:
                add("error", "fragment_wavy_bond_semantics_allows_rdkit_stereo", "wavy attachment must be visual-only and not RDKit stereo/UNKNOWN")
        quality_gates = render_quality.get("quality_gates") if isinstance(render_quality.get("quality_gates"), dict) else {}
        if quality_gates.get("fragment_visual_quality_passed") is not True:
            add("error", "fragment_visual_quality_gate_missing", "quality_gates must include fragment_visual_quality_passed=true")
        if quality_gates.get("fragment_document_realism_machine_audit_passed") is not True:
            add("error", "fragment_document_realism_quality_gate_missing", "quality_gates must include fragment_document_realism_machine_audit_passed=true")
        if quality_gates.get("fragment_pixel_geometry_detection_passed") is not True:
            add("error", "fragment_pixel_geometry_quality_gate_missing", "quality_gates must include fragment_pixel_geometry_detection_passed=true")
        if quality_gates.get("fragment_molnextr_input_quality_passed") is not True:
            add("error", "fragment_molnextr_input_quality_gate_missing", "quality_gates must include fragment_molnextr_input_quality_passed=true")
        if quality_gates.get("fragment_attachment_bond_semantics_passed") is not True:
            add("error", "fragment_attachment_bond_semantics_gate_missing", "quality_gates must include fragment_attachment_bond_semantics_passed=true")
        if endpoint_mark == "wavy" and quality_gates.get("fragment_visual_wavy_not_rdkit_stereo") is not True:
            add("error", "fragment_visual_wavy_not_rdkit_stereo_gate_missing", "quality_gates must prove custom wavy is not RDKit stereo")
    elif structure_type == "markush_layout":
        for issue in validate_markush_atom_index_alignment_contract(render_quality):
            add(
                "error",
                issue,
                "Markush rows require validated CDK/SVG/V3000 atom indices aligned to RDKit CXSMILES atom indices",
            )
        markush = render_quality.get("markush") if isinstance(render_quality.get("markush"), dict) else {}
        cells = markush.get("ocr_cells") if isinstance(markush.get("ocr_cells"), list) else render_quality.get("ocr_cells")
        if not isinstance(cells, list):
            cells = []
        for issue in literal_star_markush_issues(render_quality, atom_coordinates, cells):
            add(
                "error",
                issue,
                "Markush accepted rows must use real R/X/Y/Z variable labels and must not expose literal '*' atom or OCR tokens",
            )
        markush_ocr_cell_count = len(cells)
        if not cells:
            add("error", "missing_markush_ocr_cells", "markush_layout rows require OCR/layout cells")
        cell_areas = [bbox_area(cell) for cell in cells]
        markush_ocr_cell_area_sum = float(sum(cell_areas))
        markush_ocr_cell_area_max = float(max(cell_areas) if cell_areas else 0.0)
        markush_real_atom_count = real_atom_count(atom_coordinates)
        if markush_real_atom_count < 3:
            add("error", "markush_label_only_or_no_real_backbone", "markush_layout rows require at least 3 real chemical atoms")
        if markush_ocr_cell_area_sum > 0.12:
            add("error", "markush_text_dominated_image", "Markush OCR/text boxes occupy too much image area")
        if markush_ocr_cell_area_max > 0.08:
            add("error", "markush_oversized_ocr_cell", "a single Markush OCR/text box is too large for trainable layout data")
        for cell in cells:
            bbox = cell.get("bbox") if isinstance(cell, dict) else None
            if not isinstance(bbox, list) or len(bbox) != 4:
                markush_bad_box_count += 1
                continue
            coords = [as_float(value) for value in bbox]
            if any(value is None for value in coords):
                markush_bad_box_count += 1
                continue
            x1, y1, x2, y2 = coords
            if not (0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0):
                markush_bad_box_count += 1
        if markush_bad_box_count:
            add("error", "markush_ocr_cell_bbox_out_of_range", f"{markush_bad_box_count} OCR/layout cell boxes are outside [0,1]")
        pose_mapping = render_quality.get("pose_mapping") if isinstance(render_quality.get("pose_mapping"), dict) else {}
        formal_nonlinear_pose = formal_nonlinear_pose_contract_passed(render_quality)
        nonlinear_pose = (
            render_quality.get("nonlinear_pose_preservation")
            if formal_nonlinear_pose and isinstance(render_quality.get("nonlinear_pose_preservation"), dict)
            else {}
        )
        markush_pose_rmse = as_float(pose_mapping.get("line_constraint_rmse_svg_units"))
        markush_pose_threshold = as_float(pose_mapping.get("line_constraint_rmse_threshold"))
        markush_pose_fit_points = int(pose_mapping.get("line_constraint_fit_equation_count") or 0)
        if formal_nonlinear_pose:
            markush_pose_rmse = as_float(nonlinear_pose.get("line_constraint_rmse_svg_units"))
            markush_pose_threshold = as_float(nonlinear_pose.get("line_constraint_rmse_threshold"))
            markush_pose_fit_points = int(nonlinear_pose.get("line_constraint_fit_equation_count") or 0)
            line_abs_p95 = as_float(nonlinear_pose.get("line_constraint_abs_p95_svg_units"))
            line_abs_p95_threshold = as_float(nonlinear_pose.get("line_constraint_abs_p95_threshold"))
            line_abs_max = as_float(nonlinear_pose.get("line_constraint_abs_max_svg_units"))
            line_abs_max_threshold = as_float(nonlinear_pose.get("line_constraint_abs_max_threshold"))
            anchor_rmse = as_float(nonlinear_pose.get("intersection_anchor_rmse_svg_units"))
            anchor_rmse_threshold = as_float(nonlinear_pose.get("intersection_anchor_rmse_threshold"))
            anchor_abs_max = as_float(nonlinear_pose.get("intersection_anchor_abs_max_svg_units"))
            anchor_abs_max_threshold = as_float(nonlinear_pose.get("intersection_anchor_abs_max_threshold"))
            if line_abs_p95 is None or line_abs_p95_threshold is None or line_abs_p95 > line_abs_p95_threshold:
                add("error", "markush_nonlinear_line_abs_p95_failed", "warped SVG-polyline line residual p95 must pass")
            if line_abs_max is None or line_abs_max_threshold is None or line_abs_max > line_abs_max_threshold:
                add("error", "markush_nonlinear_line_abs_max_failed", "warped SVG-polyline line residual max must pass")
            if anchor_rmse is None or anchor_rmse_threshold is None or anchor_rmse > anchor_rmse_threshold:
                add("error", "markush_nonlinear_anchor_rmse_failed", "warped SVG-polyline intersection anchor RMSE must pass")
            if anchor_abs_max is None or anchor_abs_max_threshold is None or anchor_abs_max > anchor_abs_max_threshold:
                add("error", "markush_nonlinear_anchor_abs_max_failed", "warped SVG-polyline intersection anchor max residual must pass")
        fit_diagnostics = pose_mapping.get("fit_diagnostics") if isinstance(pose_mapping.get("fit_diagnostics"), dict) else {}
        affine_diagnostics = pose_affine_diagnostics(pose_mapping, fit_diagnostics)
        if markush_pose_rmse is None or markush_pose_threshold is None or markush_pose_rmse > markush_pose_threshold:
            add("error", "markush_pose_mapping_rmse_failed", "markush_layout rows require a passing CDK/SVG bond-line pose mapping RMSE")
        if markush_pose_fit_points < 6:
            add("error", "markush_pose_mapping_too_few_points", "markush_layout pose mapping requires at least 6 bond-line equations")
        if not fit_diagnostics:
            add("error", "markush_pose_mapping_missing_fit_diagnostics", "Markush pose mapping must include line/intersection diagnostics")
        if not affine_diagnostics:
            add("error", "markush_pose_mapping_missing_affine_diagnostics", "Markush pose mapping must include affine determinant and scale diagnostics")
        else:
            determinant = as_float(affine_diagnostics.get("determinant")) if isinstance(affine_diagnostics, dict) else None
            scale_ratio = as_float(affine_diagnostics.get("scale_ratio")) if isinstance(affine_diagnostics, dict) else None
            intersection_anchor_count = int(fit_diagnostics.get("intersection_anchor_count") or 0)
            if determinant is None or determinant >= 0.0:
                add("error", "markush_pose_mapping_invalid_affine_orientation", "Markush affine determinant must reflect SVG y-axis orientation")
            if scale_ratio is None or scale_ratio > 1.25:
                add("error", "markush_pose_mapping_affine_anisotropy", "Markush affine scale ratio is too anisotropic for synchronized CDK depiction")
            if intersection_anchor_count < 2:
                add("error", "markush_pose_mapping_too_few_intersection_anchors", "Markush pose mapping needs multibond atom anchors to constrain endpoint positions")
        pose_alignment_rmse = as_float(pose_alignment.get("pose_mapping_rmse_svg_units"))
        pose_alignment_threshold = as_float(pose_alignment.get("pose_mapping_rmse_threshold"))
        pose_alignment_fit_points = int(pose_alignment.get("pose_mapping_fit_point_count") or 0)
        pose_alignment_metric = str(pose_alignment.get("pose_mapping_metric") or "")
        if formal_nonlinear_pose:
            pose_alignment_rmse = as_float(nonlinear_pose.get("line_constraint_rmse_svg_units"))
            pose_alignment_threshold = as_float(nonlinear_pose.get("line_constraint_rmse_threshold"))
            pose_alignment_fit_points = int(nonlinear_pose.get("line_constraint_fit_equation_count") or 0)
            pose_alignment_metric = "nonlinear_warped_svg_polyline_line_constraint_rmse_svg_units"
        elif pose_alignment_metric != "line_constraint_rmse_svg_units":
            add("error", "markush_pose_alignment_metric_not_line_constraint", "Markush pose_alignment must use the bond-line constraint RMSE metric")
        if pose_alignment_rmse is None or pose_alignment_threshold is None:
            add("error", "markush_pose_alignment_missing_rmse", "Markush pose_alignment must record CDK/SVG RMSE evidence")
        elif abs(pose_alignment_rmse - float(markush_pose_rmse or 0.0)) > 1e-6 or abs(pose_alignment_threshold - float(markush_pose_threshold or 0.0)) > 1e-6:
            expected_source = "nonlinear_pose_preservation" if formal_nonlinear_pose else "pose_mapping"
            add("error", "markush_pose_alignment_rmse_mismatch", f"pose_alignment RMSE evidence must match {expected_source}")
        if pose_alignment_fit_points != markush_pose_fit_points:
            expected_source = "nonlinear_pose_preservation" if formal_nonlinear_pose else "pose_mapping"
            add("error", "markush_pose_alignment_fit_point_mismatch", f"pose_alignment fit-point evidence must match {expected_source}")
        document_realism = (
            render_quality.get("document_realism") if isinstance(render_quality.get("document_realism"), dict) else {}
        )
        if not document_realism:
            markush_realism_missing_render_parameter_count = len(MARKUSH_REALISM_RENDER_PARAMETER_KEYS)
            add(
                "error",
                "markush_missing_document_realism",
                "markush_layout rows must preserve patent/literature realism provenance and machine audit evidence",
            )
        else:
            markush_realism_policy = str(document_realism.get("policy") or "")
            markush_realism_status = str(document_realism.get("status") or "")
            markush_realism_machine_audit_passed = document_realism.get("machine_audit_passed") is True
            markush_realism_manual_review_required = document_realism.get("manual_visual_review_required") is True
            if document_realism.get("schema_version") != MARKUSH_DOCUMENT_REALISM_SCHEMA_VERSION:
                add("error", "markush_document_realism_schema_mismatch", "Markush document_realism schema is not accepted")
            if markush_realism_policy != MARKUSH_DOCUMENT_REALISM_POLICY:
                add("error", "markush_document_realism_policy_mismatch", "Markush document_realism policy is not accepted")
            if document_realism.get("generation_method") != "existing_markushgenerator_cdk_svg_depictor":
                add("error", "markush_document_realism_generation_method_mismatch", "Markush rows must use the existing MarkushGenerator/CDK SVG pipeline")
            if document_realism.get("no_new_renderer_or_image_method") is not True:
                add("error", "markush_document_realism_new_renderer_not_blocked", "document_realism must forbid unreviewed renderer/image-method replacement")
            if document_realism.get("machine_audit_passed") is not True:
                add("error", "markush_document_realism_machine_audit_failed", "patent/literature realism machine audit must pass")
            if document_realism.get("manual_visual_review_required") is not True:
                add("error", "markush_document_realism_manual_review_not_required", "machine realism audit must not replace manual visual review")
            if document_realism.get("manual_visual_review_passed") is True:
                add("error", "markush_document_realism_manual_review_overclaimed", "row-level generator output must not self-certify manual visual acceptance")
            if str(document_realism.get("source_visual_domain") or "") != "patent_literature_markush_document_crop":
                add("error", "markush_document_realism_domain_mismatch", "source visual domain must target patent/literature Markush document crops")
            external_basis = document_realism.get("external_basis")
            if not isinstance(external_basis, list) or not external_basis:
                add("error", "markush_document_realism_missing_external_basis", "document_realism must cite external implementation/paper basis")
            missing_params = document_realism.get("render_parameter_keys_missing")
            if isinstance(missing_params, list):
                markush_realism_missing_render_parameter_count = len(missing_params)
            else:
                markush_realism_missing_render_parameter_count = len(MARKUSH_REALISM_RENDER_PARAMETER_KEYS)
                add("error", "markush_document_realism_missing_parameter_audit", "document_realism must list missing render parameters")
            render_params = document_realism.get("render_parameters") if isinstance(document_realism.get("render_parameters"), dict) else {}
            markush_realism_font_size = as_float(render_params.get("font_size"))
            markush_realism_stroke_ratio = as_float(render_params.get("stroke_ratio"))
            if markush_realism_missing_render_parameter_count:
                add("error", "markush_document_realism_missing_render_parameters", "CDK render metadata must be complete")
            if markush_realism_font_size is None or markush_realism_font_size <= 0.0:
                add("error", "markush_document_realism_invalid_font_size", "document_realism must preserve a positive CDK font size")
            if markush_realism_stroke_ratio is None or markush_realism_stroke_ratio <= 0.0:
                add("error", "markush_document_realism_invalid_stroke_ratio", "document_realism must preserve a positive CDK stroke ratio")
            quality_gates = render_quality.get("quality_gates") if isinstance(render_quality.get("quality_gates"), dict) else {}
            if quality_gates.get("patent_literature_realism_machine_audit_passed") is not True:
                add("error", "markush_document_realism_quality_gate_missing", "quality_gates must include patent_literature_realism_machine_audit_passed=true")
            markush_molnextr_quality = (
                render_quality.get("markush_molnextr_input_quality")
                if isinstance(render_quality.get("markush_molnextr_input_quality"), dict)
                else {}
            )
            if not markush_molnextr_quality:
                add(
                    "error",
                    "markush_missing_molnextr_input_quality",
                    "Markush rows must include MolNexTR CropWhite+Resize input quality evidence",
                )
            else:
                if markush_molnextr_quality.get("schema_version") != "markush_molnextr_input_quality_v1":
                    add("error", "markush_molnextr_input_quality_schema_mismatch", "Markush MolNexTR input quality schema is not accepted")
                if markush_molnextr_quality.get("passed") is not True:
                    add("error", "markush_molnextr_input_quality_failed", "Markush must remain readable after MolNexTR CropWhite+Resize preprocessing")
                markush_molnextr_min_atom_pair_384 = as_float(
                    markush_molnextr_quality.get("min_atom_pair_distance_px_at_384")
                )
                markush_molnextr_min_ocr_cell_side_384 = as_float(
                    markush_molnextr_quality.get("min_ocr_cell_side_px_at_384")
                )
                markush_molnextr_dark_pixel_ratio = as_float(markush_molnextr_quality.get("dark_pixel_ratio"))
                markush_molnextr_purewhite_pixel_ratio = as_float(markush_molnextr_quality.get("purewhite_pixel_ratio"))
                thresholds = (
                    markush_molnextr_quality.get("thresholds")
                    if isinstance(markush_molnextr_quality.get("thresholds"), dict)
                    else {}
                )
                min_atom_threshold = as_float(thresholds.get("min_atom_pair_distance_px_at_384"))
                min_ocr_threshold = as_float(thresholds.get("min_ocr_cell_side_px_at_384"))
                max_ocr_area_sum = as_float(thresholds.get("max_ocr_cell_area_sum"))
                if min_atom_threshold is None or min_atom_threshold < 4.0:
                    add("error", "markush_molnextr_atom_distance_threshold_missing_or_loose", "Markush input quality must declare the atom-distance threshold")
                if min_ocr_threshold is None or min_ocr_threshold < 3.5:
                    add("error", "markush_molnextr_ocr_side_threshold_missing_or_loose", "Markush input quality must declare the OCR label size threshold")
                if max_ocr_area_sum is None or max_ocr_area_sum > 0.12:
                    add("error", "markush_molnextr_ocr_area_threshold_missing_or_loose", "Markush input quality must declare the OCR area threshold")
                if markush_molnextr_min_atom_pair_384 is not None and markush_molnextr_min_atom_pair_384 < 4.0:
                    add("error", "markush_molnextr_atom_coordinates_too_close_after_resize", "Markush atom centers are too close after MolNexTR resize")
                if markush_molnextr_min_ocr_cell_side_384 is None or markush_molnextr_min_ocr_cell_side_384 < 3.5:
                    add("error", "markush_molnextr_ocr_label_too_small_after_resize", "Markush OCR label is too small after MolNexTR resize")
                if int(markush_molnextr_quality.get("atom_outside_cropwhite_pad_count") or 0) != 0:
                    add("error", "markush_molnextr_atom_outside_cropwhite_pad", "Markush atom coordinates must survive CropWhite+pad")
                if quality_gates.get("markush_molnextr_input_quality_passed") is not True:
                    add("error", "markush_molnextr_input_quality_gate_missing", "quality_gates must include markush_molnextr_input_quality_passed=true")

    image_stats: dict[str, Any] = {}
    if not file_path_text:
        add("error", "missing_file_path", "file_path is required")
    elif check_images:
        image_path = row_image_path(row, base_dir)
        if not image_path.exists():
            add("error", "missing_image_file", str(image_path))
        else:
            image_stats = image_basic_stats(image_path)
            if not image_stats.get("readable"):
                add("error", "unreadable_image", str(image_stats.get("error") or ""))
            if image_stats.get("blank"):
                add("error", "blank_image", "image has too little ink")
            if image_stats.get("dense"):
                add("error", "dense_image", "image has too many dark pixels")

    return issues, {
        "source": source_arrow.split(":", 1)[1] if ":" in source_arrow else source_arrow,
        "structure_type": structure_type,
        "endpoint_side": endpoint_side,
        "endpoint_mark": endpoint_mark,
        "fragment_geometry": render_geometry,
        "render_style": str(render_quality.get("render_style") or ""),
        "image_stats": image_stats,
        "missing_required_keys": missing_keys,
        "missing_fragment_keys": missing_fragment_keys,
        "quality_gates": render_quality.get("quality_gates") if isinstance(render_quality.get("quality_gates"), dict) else {},
        "pose_alignment": pose_alignment,
        "graph_consistency": graph_consistency,
        "atom_coordinate_count": atom_coordinate_count,
        "atom_coordinate_bad_count": atom_coordinate_bad_count,
        "bond_count": bond_count,
        "endpoint_in_unit_square": point_in_unit_square(endpoint_x, endpoint_y)
        if structure_type in FRAGMENT_STRUCTURE_TYPES
        else None,
        "anchor_present": bool(str(render_quality.get("attachment_anchor") or "")),
        "single_dummy_atom": graph_consistency.get("single_dummy_atom"),
        "anchor_dummy_bond_present": graph_consistency.get("anchor_dummy_bond_present"),
        "wavy_length_ratio": as_float(wavy_geometry.get("wavy_length_to_connector_ratio")),
        "wavy_axis_dot_connector_abs": as_float(wavy_geometry.get("wavy_axis_dot_connector_abs")),
        "wavy_cycles": as_float(wavy_geometry.get("wavy_cycles")),
        "short_wavy_connector_policy": str(wavy_geometry.get("short_wavy_connector_policy") or ""),
        "fragment_mark_length_ratio": as_float(mark_geometry.get("mark_length_to_connector_ratio")),
        "fragment_mark_axis_dot_connector_abs": as_float(mark_geometry.get("mark_axis_dot_connector_abs")),
        "fragment_pixel_rmse": fragment_pixel_rmse,
        "fragment_pixel_rmse_threshold": fragment_pixel_rmse_threshold,
        "fragment_pixel_abs_p95": fragment_pixel_abs_p95,
        "fragment_pixel_abs_p95_threshold": fragment_pixel_abs_p95_threshold,
        "fragment_pixel_abs_max": fragment_pixel_abs_max,
        "fragment_pixel_abs_max_threshold": fragment_pixel_abs_max_threshold,
        "fragment_pixel_fit_points": fragment_pixel_fit_points,
        "fragment_molnextr_wavy_length_384": fragment_molnextr_wavy_length_384,
        "fragment_molnextr_mark_length_384": fragment_molnextr_mark_length_384,
        "fragment_molnextr_connector_length_384": fragment_molnextr_connector_length_384,
        "fragment_molnextr_visible_connector_length_384": fragment_molnextr_visible_connector_length_384,
        "fragment_molnextr_straight_connector_length_384": fragment_molnextr_straight_connector_length_384,
        "fragment_molnextr_min_atom_pair_384": fragment_molnextr_min_atom_pair_384,
        "fragment_document_realism_policy": fragment_document_realism_policy,
        "fragment_document_realism_machine_audit_passed": fragment_document_realism_machine_audit_passed,
        "fragment_visual_quality_passed": fragment_visual_quality_passed,
        "fragment_pixel_geometry_passed": fragment_pixel_geometry_passed,
        "fragment_molnextr_input_quality_passed": fragment_molnextr_input_quality_passed,
        "fragment_attachment_bond_semantics_passed": fragment_attachment_bond_semantics_passed,
        "fragment_visual_wavy_not_rdkit_stereo": fragment_visual_wavy_not_rdkit_stereo,
        "markush_ocr_cell_count": markush_ocr_cell_count,
        "markush_bad_box_count": markush_bad_box_count,
        "markush_real_atom_count": markush_real_atom_count,
        "markush_ocr_cell_area_sum": markush_ocr_cell_area_sum,
        "markush_ocr_cell_area_max": markush_ocr_cell_area_max,
        "markush_pose_rmse": markush_pose_rmse,
        "markush_pose_threshold": markush_pose_threshold,
        "markush_pose_fit_points": markush_pose_fit_points,
        "markush_realism_policy": markush_realism_policy,
        "markush_realism_status": markush_realism_status,
        "markush_realism_machine_audit_passed": markush_realism_machine_audit_passed,
        "markush_realism_manual_review_required": markush_realism_manual_review_required,
        "markush_realism_missing_render_parameter_count": markush_realism_missing_render_parameter_count,
        "markush_realism_font_size": markush_realism_font_size,
        "markush_realism_stroke_ratio": markush_realism_stroke_ratio,
        "markush_molnextr_min_atom_pair_384": markush_molnextr_min_atom_pair_384,
        "markush_molnextr_min_ocr_cell_side_384": markush_molnextr_min_ocr_cell_side_384,
        "markush_molnextr_dark_pixel_ratio": markush_molnextr_dark_pixel_ratio,
        "markush_molnextr_purewhite_pixel_ratio": markush_molnextr_purewhite_pixel_ratio,
    }


def validate_pose_factory_shard(
    csv_path: Path,
    *,
    check_images: bool = True,
    max_issues: int = 200,
) -> ShardValidationReport:
    base_dir = csv_path.parent
    issues: list[ValidationIssue] = []
    source_counts: Counter[str] = Counter()
    structure_type_counts: Counter[str] = Counter()
    endpoint_side_counts: Counter[str] = Counter()
    endpoint_mark_counts: Counter[str] = Counter()
    fragment_geometry_counts: Counter[str] = Counter()
    render_style_counts: Counter[str] = Counter()
    issue_severity_counts: Counter[str] = Counter()
    issue_code_counts: Counter[str] = Counter()
    missing_required_field_counts: Counter[str] = Counter()
    quality_gate_counts: dict[str, Counter[str]] = {}
    graph_consistency_counts: dict[str, Counter[str]] = {}
    molnextr_pose_counts: Counter[str] = Counter()
    fragment_attachment_counts: Counter[str] = Counter()
    markush_layout_counts: Counter[str] = Counter()
    markush_realism_counts: Counter[str] = Counter()
    image_quality_counts: Counter[str] = Counter()
    numeric_values: dict[str, list[float]] = {
        "atom_coordinate_count": [],
        "bond_count": [],
        "image_dark_pixel_ratio": [],
        "wavy_length_ratio": [],
        "wavy_axis_dot_connector_abs": [],
        "wavy_cycles": [],
        "fragment_mark_length_ratio": [],
        "fragment_mark_axis_dot_connector_abs": [],
        "fragment_pixel_rmse": [],
        "fragment_pixel_rmse_threshold": [],
        "fragment_pixel_abs_p95": [],
        "fragment_pixel_abs_p95_threshold": [],
        "fragment_pixel_abs_max": [],
        "fragment_pixel_abs_max_threshold": [],
        "fragment_pixel_fit_points": [],
        "fragment_molnextr_wavy_length_384": [],
        "fragment_molnextr_mark_length_384": [],
        "fragment_molnextr_connector_length_384": [],
        "fragment_molnextr_visible_connector_length_384": [],
        "fragment_molnextr_straight_connector_length_384": [],
        "fragment_molnextr_min_atom_pair_384": [],
        "markush_ocr_cell_count": [],
        "markush_bad_box_count": [],
        "markush_real_atom_count": [],
        "markush_ocr_cell_area_sum": [],
        "markush_ocr_cell_area_max": [],
        "markush_pose_rmse": [],
        "markush_pose_threshold": [],
        "markush_pose_fit_points": [],
        "markush_realism_missing_render_parameter_count": [],
        "markush_realism_font_size": [],
        "markush_realism_stroke_ratio": [],
        "markush_molnextr_min_atom_pair_384": [],
        "markush_molnextr_min_ocr_cell_side_384": [],
        "markush_molnextr_dark_pixel_ratio": [],
        "markush_molnextr_purewhite_pixel_ratio": [],
    }
    invalid_rows = 0
    row_count = 0

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=2):
            row_count += 1
            row_issues, summary = validate_pose_factory_row(
                row,
                row_index=row_index,
                base_dir=base_dir,
                check_images=check_images,
            )
            if row_issues:
                invalid_rows += 1
                issue_severity_counts.update(issue.severity for issue in row_issues)
                issue_code_counts.update(issue.code for issue in row_issues)
                if len(issues) < max_issues:
                    issues.extend(row_issues[: max_issues - len(issues)])
            source_counts.update([summary["source"] or "missing"])
            structure_type = summary["structure_type"] or "missing"
            structure_type_counts.update([structure_type])
            endpoint_side_counts.update([summary["endpoint_side"] or "missing"])
            endpoint_mark_counts.update([summary["endpoint_mark"] or "missing"])
            fragment_geometry_counts.update([summary["fragment_geometry"] or "missing"])
            render_style_counts.update([summary["render_style"] or "missing"])
            missing_required_field_counts.update(summary["missing_required_keys"])
            missing_required_field_counts.update(summary["missing_fragment_keys"])

            quality_gates = summary["quality_gates"] if isinstance(summary["quality_gates"], dict) else {}
            for key, value in quality_gates.items():
                quality_gate_counts.setdefault(str(key), Counter()).update([bool_bucket(value)])

            graph_consistency = summary["graph_consistency"] if isinstance(summary["graph_consistency"], dict) else {}
            for key in [
                "row_smiles_canonical_matches_mol",
                "single_dummy_atom",
                "anchor_dummy_bond_present",
            ]:
                graph_consistency_counts.setdefault(key, Counter()).update([bool_bucket(graph_consistency.get(key))])

            if summary["atom_coordinate_count"]:
                molnextr_pose_counts.update(["rows_with_atom_coordinates"])
            else:
                molnextr_pose_counts.update(["rows_missing_atom_coordinates"])
            if summary["atom_coordinate_bad_count"]:
                molnextr_pose_counts.update(["rows_with_bad_atom_coordinates"])
            else:
                molnextr_pose_counts.update(["rows_without_bad_atom_coordinates"])
            if summary["bond_count"]:
                molnextr_pose_counts.update(["rows_with_bonds"])
            else:
                molnextr_pose_counts.update(["rows_missing_bonds"])

            if structure_type in FRAGMENT_STRUCTURE_TYPES:
                fragment_attachment_counts.update(["fragment_rows"])
                fragment_attachment_counts.update(
                    [
                        "endpoint_in_unit_square"
                        if summary["endpoint_in_unit_square"] is True
                        else "endpoint_missing_or_out_of_range"
                    ]
                )
                fragment_attachment_counts.update(["anchor_present" if summary["anchor_present"] else "anchor_missing"])
                fragment_attachment_counts.update(
                    ["single_dummy_atom" if summary["single_dummy_atom"] is True else "single_dummy_atom_missing_or_false"]
                )
                fragment_attachment_counts.update(
                    [
                        "anchor_dummy_bond_present"
                        if summary["anchor_dummy_bond_present"] is True
                        else "anchor_dummy_bond_missing_or_false"
                    ]
                )
                fragment_attachment_counts.update(
                    [
                        "document_realism_machine_audit_passed"
                        if summary["fragment_document_realism_machine_audit_passed"] is True
                        else "document_realism_machine_audit_missing_or_failed"
                    ]
                )
                fragment_attachment_counts.update(
                    [
                        "visual_quality_passed"
                        if summary["fragment_visual_quality_passed"] is True
                        else "visual_quality_missing_or_failed"
                    ]
                )
                fragment_attachment_counts.update(
                    [
                        "pixel_geometry_passed"
                        if summary["fragment_pixel_geometry_passed"] is True
                        else "pixel_geometry_missing_or_failed"
                    ]
                )
                fragment_attachment_counts.update(
                    [
                        "molnextr_input_quality_passed"
                        if summary["fragment_molnextr_input_quality_passed"] is True
                        else "molnextr_input_quality_missing_or_failed"
                    ]
                )
                fragment_attachment_counts.update(
                    [
                        "bond_semantics_passed"
                        if summary["fragment_attachment_bond_semantics_passed"] is True
                        else "bond_semantics_missing_or_failed"
                    ]
                )
                if summary["endpoint_mark"] == "wavy":
                    fragment_attachment_counts.update(
                        [
                            "visual_wavy_not_rdkit_stereo"
                            if summary["fragment_visual_wavy_not_rdkit_stereo"] is True
                            else "visual_wavy_rdkit_stereo_not_blocked"
                        ]
                    )
                fragment_attachment_counts.update([f"document_realism_policy:{summary['fragment_document_realism_policy'] or 'missing'}"])

            if structure_type == "markush_layout":
                markush_layout_counts.update(["markush_rows"])
                markush_layout_counts.update(
                    [
                        "rows_with_ocr_cells"
                        if int(summary["markush_ocr_cell_count"] or 0) > 0
                        else "rows_missing_ocr_cells"
                    ]
                )
                markush_layout_counts.update(
                    [
                        "rows_with_valid_ocr_boxes"
                        if int(summary["markush_bad_box_count"] or 0) == 0
                        else "rows_with_bad_ocr_boxes"
                    ]
                )
                rmse = summary["markush_pose_rmse"]
                threshold = summary["markush_pose_threshold"]
                if rmse is not None and threshold is not None and rmse <= threshold:
                    markush_layout_counts.update(["rows_with_pose_mapping_rmse_pass"])
                else:
                    markush_layout_counts.update(["rows_with_pose_mapping_rmse_fail"])
                markush_realism_counts.update([f"policy:{summary['markush_realism_policy'] or 'missing'}"])
                markush_realism_counts.update([f"status:{summary['markush_realism_status'] or 'missing'}"])
                markush_realism_counts.update(
                    [
                        "machine_audit_passed"
                        if summary["markush_realism_machine_audit_passed"] is True
                        else "machine_audit_missing_or_failed"
                    ]
                )
                markush_realism_counts.update(
                    [
                        "manual_visual_review_required"
                        if summary["markush_realism_manual_review_required"] is True
                        else "manual_visual_review_requirement_missing"
                    ]
                )

            image_stats = summary["image_stats"] if isinstance(summary["image_stats"], dict) else {}
            if image_stats:
                image_quality_counts.update(["readable" if image_stats.get("readable") else "unreadable"])
                image_quality_counts.update(["blank" if image_stats.get("blank") else "not_blank"])
                image_quality_counts.update(["dense" if image_stats.get("dense") else "not_dense"])
                dark_ratio = as_float(image_stats.get("dark_pixel_ratio"))
                if dark_ratio is not None:
                    numeric_values["image_dark_pixel_ratio"].append(dark_ratio)

            numeric_keys = ["atom_coordinate_count", "bond_count"]
            if summary["endpoint_mark"] == "wavy":
                numeric_keys.extend(["wavy_length_ratio", "wavy_axis_dot_connector_abs", "wavy_cycles"])
            if structure_type in FRAGMENT_STRUCTURE_TYPES and summary["endpoint_mark"] in {
                "cut",
                "query_attachment",
                "dummy_atom",
            }:
                numeric_keys.extend(["fragment_mark_length_ratio", "fragment_mark_axis_dot_connector_abs"])
            if structure_type in FRAGMENT_STRUCTURE_TYPES:
                numeric_keys.extend(
                    [
                        "fragment_pixel_rmse",
                        "fragment_pixel_rmse_threshold",
                        "fragment_pixel_abs_p95",
                        "fragment_pixel_abs_p95_threshold",
                        "fragment_pixel_abs_max",
                        "fragment_pixel_abs_max_threshold",
                        "fragment_pixel_fit_points",
                        "fragment_molnextr_wavy_length_384",
                        "fragment_molnextr_mark_length_384",
                        "fragment_molnextr_connector_length_384",
                        "fragment_molnextr_visible_connector_length_384",
                        "fragment_molnextr_straight_connector_length_384",
                        "fragment_molnextr_min_atom_pair_384",
                    ]
                )
            if structure_type == "markush_layout":
                numeric_keys.extend(
                    [
                        "markush_ocr_cell_count",
                        "markush_bad_box_count",
                        "markush_real_atom_count",
                        "markush_ocr_cell_area_sum",
                        "markush_ocr_cell_area_max",
                        "markush_pose_rmse",
                        "markush_pose_threshold",
                        "markush_pose_fit_points",
                        "markush_realism_missing_render_parameter_count",
                        "markush_realism_font_size",
                        "markush_realism_stroke_ratio",
                        "markush_molnextr_min_atom_pair_384",
                        "markush_molnextr_min_ocr_cell_side_384",
                        "markush_molnextr_dark_pixel_ratio",
                        "markush_molnextr_purewhite_pixel_ratio",
                    ]
                )
            for key in numeric_keys:
                value = as_float(summary.get(key))
                if value is not None:
                    numeric_values[key].append(value)

    return ShardValidationReport(
        csv_path=str(csv_path),
        row_count=row_count,
        valid_rows=row_count - invalid_rows,
        invalid_rows=invalid_rows,
        trainable=row_count > 0 and invalid_rows == 0,
        issues=issues,
        source_counts=dict(sorted(source_counts.items())),
        structure_type_counts=dict(sorted(structure_type_counts.items())),
        endpoint_side_counts=dict(sorted(endpoint_side_counts.items())),
        endpoint_mark_counts=dict(sorted(endpoint_mark_counts.items())),
        fragment_geometry_counts=dict(sorted(fragment_geometry_counts.items())),
        render_style_counts=dict(sorted(render_style_counts.items())),
        issue_severity_counts=dict(sorted(issue_severity_counts.items())),
        issue_code_counts=dict(sorted(issue_code_counts.items())),
        missing_required_field_counts=dict(sorted(missing_required_field_counts.items())),
        quality_gate_counts={
            key: dict(sorted(counter.items())) for key, counter in sorted(quality_gate_counts.items())
        },
        graph_consistency_counts={
            key: dict(sorted(counter.items())) for key, counter in sorted(graph_consistency_counts.items())
        },
        molnextr_pose_counts=dict(sorted(molnextr_pose_counts.items())),
        fragment_attachment_counts=dict(sorted(fragment_attachment_counts.items())),
        markush_layout_counts=dict(sorted(markush_layout_counts.items())),
        markush_realism_counts=dict(sorted(markush_realism_counts.items())),
        image_quality_counts=dict(sorted(image_quality_counts.items())),
        numeric_summaries={
            key: summarize_numbers(values)
            for key, values in sorted(numeric_values.items())
            if values
        },
    )
