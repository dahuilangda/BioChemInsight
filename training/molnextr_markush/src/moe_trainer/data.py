"""Dataframe contract validation and frozen-partition loading.

Extracted from the original ``tools/train_moe.py`` monolith.
"""
from __future__ import annotations

from ._common import *  # noqa: F401,F403
from .runtime import rank0_print  # noqa: F401,E402

def validate_dataframe_contract(
    frame: pd.DataFrame,
    *,
    context: str,
    native_attachment_extension: bool = False,
    skip_fragment_linearization: bool = False,
) -> None:
    required = {
        "file_path",
        "SMILES",
        "edges",
        "structure_type_label",
        "data_contract_version",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")
    versions = set(frame["data_contract_version"].dropna().astype(str).unique().tolist())
    if versions != {MOE_DATA_CONTRACT_VERSION}:
        raise ValueError(
            f"{context} data contract mismatch: {sorted(versions)}; "
            f"expected only {MOE_DATA_CONTRACT_VERSION!r}. Rebuild the dataframe."
        )
    fragment_mask = frame["structure_type_label"].astype(int).eq(2)
    if bool(fragment_mask.any()):
        alignment_columns = {
            "atom_index_alignment_verified",
            "atom_index_alignment_method",
            "smiles_to_source_atom_indices",
            "source_dummy_atom_index",
            "smiles_dummy_atom_index",
            "atom_index_reordered",
            "decoder_smiles",
            "fragment_linearization_verified",
            "fragment_linearization_method",
            "fragment_backbone_smiles",
            "fragment_decoder_backbone_prefix",
            "decoder_to_smiles_atom_indices",
            "decoder_to_source_atom_indices",
            "chemical_smiles_dummy_atom_index",
            "decoder_dummy_atom_index",
            "fragment_dummy_is_final_token",
            "fragment_backbone_prefix_exact",
            "decoder_atom_index_reordered",
        }
        if native_attachment_extension:
            alignment_columns.add("node_coords")
        missing_alignment = sorted(alignment_columns - set(frame.columns))
        if missing_alignment:
            raise ValueError(
                f"{context} fragment rows lack atom-index alignment proof: "
                f"{missing_alignment}. Rebuild the dataframe."
            )
        fragment_rows = frame.loc[fragment_mask]
        if not fragment_rows["atom_index_alignment_verified"].fillna(False).astype(bool).all():
            raise ValueError(f"{context} contains unverified fragment atom-index alignment")
        methods = set(
            fragment_rows["atom_index_alignment_method"]
            .fillna("")
            .astype(str)
            .unique()
            .tolist()
        )
        if methods != {FRAGMENT_ATOM_INDEX_ALIGNMENT_METHOD}:
            raise ValueError(
                f"{context} fragment alignment methods are {sorted(methods)}; "
                f"expected {FRAGMENT_ATOM_INDEX_ALIGNMENT_METHOD!r}"
            )
        if fragment_rows["smiles_to_source_atom_indices"].fillna("").astype(str).eq("").any():
            raise ValueError(f"{context} contains an empty fragment atom-index mapping")
        if (
            fragment_rows["source_dummy_atom_index"].fillna(-1).astype(int).lt(0).any()
            or fragment_rows["smiles_dummy_atom_index"].fillna(-1).astype(int).lt(0).any()
        ):
            raise ValueError(f"{context} contains invalid fragment dummy-index alignment")
        linearization_methods = set(
            fragment_rows["fragment_linearization_method"]
            .fillna("")
            .astype(str)
            .unique()
            .tolist()
        )
        if linearization_methods != {FRAGMENT_DECODER_LINEARIZATION_METHOD}:
            raise ValueError(
                f"{context} fragment linearization methods are "
                f"{sorted(linearization_methods)}; expected "
                f"{FRAGMENT_DECODER_LINEARIZATION_METHOD!r}"
            )
        required_true = (
            "fragment_linearization_verified",
            "fragment_dummy_is_final_token",
            "fragment_backbone_prefix_exact",
        )
        for column in required_true:
            if not fragment_rows[column].fillna(False).astype(bool).all():
                raise ValueError(f"{context} contains false fragment contract flag {column}")

        row_issues = []
        for row_index, row in fragment_rows.iterrows():
            issues = []
            decoder_smiles = str(row["decoder_smiles"] or "")
            tokens = atomwise_tokenizer(decoder_smiles)
            atom_tokens = [
                token
                for token in tokens
                if token.isalpha() or token.startswith("[") or token == "*"
            ]
            atom_count = len(atom_tokens)
            try:
                chemical_to_source = [
                    int(value)
                    for value in ast.literal_eval(
                        str(row["smiles_to_source_atom_indices"])
                    )
                ]
                decoder_to_chemical = [
                    int(value)
                    for value in ast.literal_eval(
                        str(row["decoder_to_smiles_atom_indices"])
                    )
                ]
                decoder_to_source = [
                    int(value)
                    for value in ast.literal_eval(
                        str(row["decoder_to_source_atom_indices"])
                    )
                ]
                edges = ast.literal_eval(str(row["edges"]))
                node_coords = (
                    ast.literal_eval(str(row["node_coords"]))
                    if native_attachment_extension
                    else []
                )
            except (SyntaxError, TypeError, ValueError) as exc:
                issues.append(f"unparseable_payload:{exc}")
                chemical_to_source = []
                decoder_to_chemical = []
                decoder_to_source = []
                edges = []
                node_coords = []
            if not tokens or tokens[-1] != "*":
                issues.append("dummy_is_not_final_plain_star_token")
            if atom_tokens.count("*") != 1:
                issues.append(f"decoder_dummy_count:{atom_tokens.count('*')}")
            prefix = str(row["fragment_decoder_backbone_prefix"] or "")
            if decoder_smiles != f"{prefix}*":
                issues.append("decoder_target_is_not_backbone_prefix_plus_dummy")
            if not str(row["fragment_backbone_smiles"] or ""):
                issues.append("missing_canonical_backbone_smiles")
            expected_permutation = list(range(atom_count))
            for name, values in (
                ("chemical_to_source", chemical_to_source),
                ("decoder_to_chemical", decoder_to_chemical),
                ("decoder_to_source", decoder_to_source),
            ):
                if len(values) != atom_count or sorted(values) != expected_permutation:
                    issues.append(f"{name}_is_not_permutation")
            chemical_dummy = int(row["chemical_smiles_dummy_atom_index"])
            decoder_dummy = int(row["decoder_dummy_atom_index"])
            source_dummy = int(row["source_dummy_atom_index"])
            if chemical_dummy != int(row["smiles_dummy_atom_index"]):
                issues.append("chemical_dummy_index_disagrees_with_alignment")
            if decoder_dummy != atom_count - 1:
                issues.append("decoder_dummy_is_not_last_atom")
            if decoder_to_chemical and (
                not (0 <= decoder_dummy < len(decoder_to_chemical))
                or decoder_to_chemical[decoder_dummy] != chemical_dummy
            ):
                issues.append("decoder_to_chemical_dummy_mapping_mismatch")
            if decoder_to_source and (
                not (0 <= decoder_dummy < len(decoder_to_source))
                or decoder_to_source[decoder_dummy] != source_dummy
            ):
                issues.append("decoder_to_source_dummy_mapping_mismatch")
            if (
                decoder_to_chemical
                and chemical_to_source
                and decoder_to_source
                and decoder_to_source
                != [chemical_to_source[index] for index in decoder_to_chemical]
            ):
                issues.append("composed_source_mapping_mismatch")
            attachment_ok, attachment_reason = fragment_attachment_contract(
                decoder_smiles,
                edges,
            )
            if not attachment_ok:
                issues.append(f"attachment_contract:{attachment_reason}")
            dummy_bond_types = [
                int(bond_type)
                for begin, end, bond_type in edges
                if int(begin) == decoder_dummy or int(end) == decoder_dummy
            ]
            if dummy_bond_types != [1]:
                issues.append(
                    f"terminal_dummy_bond_types:{dummy_bond_types}"
                )
            attachment_neighbors = [
                int(end) if int(begin) == decoder_dummy else int(begin)
                for begin, end, bond_type in edges
                if (
                    int(bond_type) != 0
                    and (
                        int(begin) == decoder_dummy
                        or int(end) == decoder_dummy
                    )
                )
            ]
            if native_attachment_extension and len(attachment_neighbors) != 1:
                issues.append(
                    f"native_extension_anchor_count:{len(attachment_neighbors)}"
                )
            elif native_attachment_extension:
                native_anchor = int(attachment_neighbors[0])
                if not 0 <= native_anchor < atom_count - 1:
                    issues.append(
                        f"native_extension_anchor_out_of_range:{native_anchor}"
                    )
                extension_text = f"[{native_anchor}:*]"
                native_target_length = (
                    2
                    + len(prefix)
                    + 2 * max(0, atom_count - 1)
                    + 1
                    + len(extension_text)
                )
                if native_target_length > int(
                    FORMAT_INFO["chartok_coords"]["max_len"]
                ):
                    issues.append(
                        f"native_extension_target_too_long:{native_target_length}"
                    )
            if native_attachment_extension and len(node_coords) != atom_count:
                issues.append(
                    f"decoder_coordinate_count:{len(node_coords)}!={atom_count}"
                )
            if bool(row["decoder_atom_index_reordered"]) != (
                decoder_to_chemical != expected_permutation
            ):
                issues.append("decoder_reordered_flag_mismatch")
            if issues and len(row_issues) < 10:
                row_issues.append((int(row_index), issues))
        if row_issues and not skip_fragment_linearization:
            raise ValueError(
                f"{context} contains invalid fragment decoder linearizations: "
                f"{row_issues}"
            )
    if frame["file_path"].astype(str).duplicated().any():
        raise ValueError(f"{context} contains duplicate file_path rows")
    unique_paths = frame["file_path"].astype(str).unique().tolist()
    missing_paths = [path for path in unique_paths if not os.path.isfile(path)]
    if missing_paths:
        preview = "; ".join(missing_paths[:5])
        raise FileNotFoundError(
            f"{context} contains {len(missing_paths)} missing/unmounted image "
            f"paths; refusing blank-image fallback. First paths: {preview}"
        )


def load_frozen_source_partition(
    report_path: str | Path,
    *,
    production_source_root: str | Path,
) -> tuple[dict[int, list[str]], dict[int, list[str]], dict]:
    """Load and revalidate a previously gated whole-shard source partition."""

    path = Path(report_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen source report is missing: {path}")
    report_bytes = path.read_bytes()
    report = json.loads(report_bytes)
    if report.get("schema_version") != "molnextr_moe_source_discovery_v1":
        raise ValueError("frozen source report schema mismatch")
    if report.get("source_mode") != "available_production_pose_factory":
        raise ValueError("frozen source report has an unsupported source mode")
    root = Path(production_source_root).resolve()
    labels = report.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("frozen source report has no label partitions")

    train: dict[int, list[str]] = {}
    calibration: dict[int, list[str]] = {}
    all_train: set[str] = set()
    all_calibration: set[str] = set()
    expected = {"complete": 0, "markush": 1, "fragment": 2}
    for bucket, expected_label in expected.items():
        partition = labels.get(bucket)
        if not isinstance(partition, dict):
            raise ValueError(f"frozen source report lacks {bucket!r}")
        if int(partition.get("label", -1)) != expected_label:
            raise ValueError(f"frozen source report label mismatch for {bucket}")
        train_paths = [str(Path(value).resolve()) for value in partition.get("train_csv_paths") or []]
        calibration_paths = [
            str(Path(value).resolve())
            for value in partition.get("calibration_csv_paths") or []
        ]
        if not train_paths or not calibration_paths:
            raise ValueError(f"frozen source report has an empty {bucket} partition")
        for source_path in [*train_paths, *calibration_paths]:
            candidate = Path(source_path)
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"frozen source CSV escapes production root: {candidate}"
                ) from exc
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"frozen source CSV is missing: {candidate}"
                )
        train[expected_label] = train_paths
        calibration[expected_label] = calibration_paths
        all_train.update(train_paths)
        all_calibration.update(calibration_paths)
    overlap = all_train & all_calibration
    if overlap:
        raise ValueError(
            f"frozen source train/calibration partitions overlap by {len(overlap)} CSVs"
        )
    validated = dict(report)
    validated["frozen_partition_contract"] = {
        "schema_version": "molnextr_frozen_source_partition_v1",
        "source_report_path": str(path),
        "source_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "production_source_root": str(root),
        "all_referenced_csvs_exist": True,
        "all_referenced_csvs_under_production_root": True,
        "train_calibration_overlap_count": 0,
        "train_csv_count": len(all_train),
        "calibration_csv_count": len(all_calibration),
    }
    return train, calibration, validated


def validate_attachment_set_capacity(
    frame: pd.DataFrame,
    *,
    num_queries: int,
    max_count: int,
) -> dict[str, int]:
    """Reject datasets that cannot be represented by the set predictor."""

    num_queries = int(num_queries)
    max_count = int(max_count)
    if num_queries <= 0:
        raise ValueError("--attachment-set-num-queries must be positive")
    if max_count <= 0:
        raise ValueError("--attachment-set-max-count must be positive")
    if max_count > num_queries:
        raise ValueError(
            "attachment-set cardinality cannot exceed the number of set queries: "
            f"max_count={max_count} num_queries={num_queries}"
        )
    if "SMILES" not in frame.columns:
        raise ValueError("attachment-set capacity audit requires the SMILES column")

    counts = frame["SMILES"].fillna("").astype(str).str.count(r"\*").astype(int)
    max_required = int(counts.max()) if len(counts) else 0
    capacity = min(num_queries, max_count)
    if max_required > capacity:
        examples = frame.loc[counts > capacity, "SMILES"].astype(str).head(3).tolist()
        raise ValueError(
            "attachment-set capacity is smaller than the training targets; "
            "refusing silent cardinality truncation: "
            f"required={max_required} capacity={capacity} examples={examples}"
        )
    return {
        "max_required": max_required,
        "num_queries": num_queries,
        "max_count": max_count,
    }


def load_real_original_training_frame(args, rank: int) -> tuple[pd.DataFrame | None, dict | None]:
    if not bool(args.use_real_original_data):
        return None, None
    data_path = Path(args.real_original_train_df)
    report_path = Path(args.real_original_train_report)
    if not data_path.exists() or not report_path.exists():
        message = (
            "validated real-original training data is missing; run "
            "tools/build_real_markushgrapher_ocsr.py first: "
            f"data={data_path} report={report_path}"
        )
        if bool(args.require_real_original_data):
            raise FileNotFoundError(message)
        rank0_print(rank, f"  WARNING: {message}")
        return None, None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise ValueError(f"real-original report is not passing: {report_path}")
    if report.get("schema_version") != "real_markushgrapher_ocsr_v2":
        raise ValueError("real-original report is not the pose-verified v2 contract")
    policy = report.get("policy") if isinstance(report.get("policy"), dict) else {}
    if policy.get("original_image_pixels_preserved") is not True:
        raise ValueError("real-original report does not preserve source image pixels")
    if policy.get("coordinate_targets_are_source_derived") is not True:
        raise ValueError("real-original coordinates are not source-derived")
    if policy.get("training_requires_verified_pose") is not True:
        raise ValueError("real-original training rows are not pose-gated")
    if policy.get("data_contract_version") != MOE_DATA_CONTRACT_VERSION:
        raise ValueError("real-original report data contract version mismatch")
    frame = pd.read_parquet(data_path)
    validate_dataframe_contract(frame, context="real-original dataframe")
    if not set(frame["image_domain"].astype(str).unique()) <= {"real_original"}:
        raise ValueError("real-original dataframe contains a non-real image domain")
    if not frame["coordinate_targets_available"].astype(bool).all():
        raise ValueError("real-original training dataframe contains an unverified pose")
    if not frame["coordinate_pose_verified"].astype(bool).all():
        raise ValueError("real-original training dataframe failed the pose verification flag")
    if set(frame["node_coords_space"].astype(str).unique()) != {
        "normalized_image_cxsmiles_ocr_similarity"
    }:
        raise ValueError("real-original training dataframe has an invalid pose space")
    if not set(frame["structure_type_label"].astype(int).unique()) <= {1, 2}:
        raise ValueError("real-original dataframe contains complete/noise labels")
    rank0_print(
        rank,
        f"  real-original rows: {len(frame)} "
        f"labels={frame['structure_type_label'].value_counts().to_dict()}",
    )
    return frame, report


