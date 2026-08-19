from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.molnextr_markush.src.molnextr_dataset_toolkit.config import (
    FRAGMENT_TARGET_BUCKETS,
    MARKUSH_R_COUNT_BUCKETS,
    FormalSimDatasetConfig,
)
from training.molnextr_markush.src.molnextr_dataset_toolkit.contracts import CommandSpec, GateResult, ShardPaths
from training.molnextr_markush.src.molnextr_dataset_toolkit.retry import RetryDiagnostics, analyze_retry_diagnostics


@dataclass(frozen=True)
class CandidatePlanWindow:
    shard_index: int
    accepted_rows_per_bucket: int
    candidate_rows_per_bucket: int
    bucket_count: int = len(MARKUSH_R_COUNT_BUCKETS)

    @classmethod
    def from_targets(
        cls,
        shard_index: int,
        *,
        accepted_rows_per_bucket: int,
        candidate_multiplier: int = 2,
        candidate_rows_per_bucket: int | None = None,
    ) -> "CandidatePlanWindow":
        accepted = int(accepted_rows_per_bucket)
        candidates = int(candidate_rows_per_bucket or accepted * int(candidate_multiplier))
        if accepted <= 0:
            raise ValueError("accepted_rows_per_bucket must be positive")
        if candidates < accepted:
            raise ValueError("candidate_rows_per_bucket must be >= accepted_rows_per_bucket")
        return cls(
            shard_index=int(shard_index),
            accepted_rows_per_bucket=accepted,
            candidate_rows_per_bucket=candidates,
        )

    @property
    def row_count(self) -> int:
        return self.candidate_rows_per_bucket * self.bucket_count

    @property
    def plan_start(self) -> int:
        return self.shard_index * self.row_count

    @property
    def plan_end(self) -> int:
        return (self.shard_index + 1) * self.row_count

    def to_dict(self) -> dict[str, int]:
        return {
            "shard_index": self.shard_index,
            "accepted_rows_per_bucket": self.accepted_rows_per_bucket,
            "candidate_rows_per_bucket": self.candidate_rows_per_bucket,
            "bucket_count": self.bucket_count,
            "candidate_window_rows": self.row_count,
            "plan_start": self.plan_start,
            "plan_end": self.plan_end,
        }


def parse_bucket_window_overrides(value: str) -> dict[str, tuple[int, int]]:
    text = str(value or "").strip()
    if not text:
        return {}
    windows: dict[str, tuple[int, int]] = {}
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item or ":" not in item:
            raise ValueError(f"invalid bucket window override {item!r}; expected bucket=start:end")
        bucket, range_text = [part.strip() for part in item.split("=", 1)]
        if bucket not in MARKUSH_R_COUNT_BUCKETS:
            raise ValueError(f"invalid Markush R-count bucket {bucket!r}")
        start_text, end_text = [part.strip() for part in range_text.split(":", 1)]
        start = int(start_text)
        end = int(end_text)
        if start < 0 or end < start:
            raise ValueError(f"invalid bucket window override {item!r}; require 0 <= start <= end")
        windows[bucket] = (start, end)
    if windows and set(windows) != set(MARKUSH_R_COUNT_BUCKETS):
        missing = sorted(set(MARKUSH_R_COUNT_BUCKETS) - set(windows))
        raise ValueError(f"bucket window overrides must cover all Markush buckets; missing={missing}")
    return windows


@dataclass(frozen=True)
class ExplicitBucketPlanWindow:
    shard_index: int
    accepted_rows_per_bucket: int
    bucket_windows: dict[str, tuple[int, int]]

    @property
    def plan_start(self) -> int:
        return min(start for start, _end in self.bucket_windows.values())

    @property
    def plan_end(self) -> int:
        return max(end for _start, end in self.bucket_windows.values())

    @property
    def candidate_rows_per_bucket_arg(self) -> str:
        return ",".join(
            f"{bucket}={self.bucket_windows[bucket][1] - self.bucket_windows[bucket][0]}"
            for bucket in MARKUSH_R_COUNT_BUCKETS
        )

    @property
    def bucket_window_arg(self) -> str:
        return ",".join(
            f"{bucket}={self.bucket_windows[bucket][0]}:{self.bucket_windows[bucket][1]}"
            for bucket in MARKUSH_R_COUNT_BUCKETS
        )

    def to_dict(self) -> dict[str, Any]:
        nonempty_windows = [
            (start, end) for start, end in self.bucket_windows.values() if int(end) > int(start)
        ]
        return {
            "shard_index": int(self.shard_index),
            "accepted_rows_per_bucket": int(self.accepted_rows_per_bucket),
            "bucket_windows": {
                bucket: {
                    "start": int(self.bucket_windows[bucket][0]),
                    "end": int(self.bucket_windows[bucket][1]),
                    "candidate_rows": int(self.bucket_windows[bucket][1] - self.bucket_windows[bucket][0]),
                }
                for bucket in MARKUSH_R_COUNT_BUCKETS
            },
            "candidate_window_rows": int(
                sum(end - start for start, end in self.bucket_windows.values())
            ),
            "plan_start": int(self.plan_start),
            "plan_end": int(self.plan_end),
            "plan_start_end_semantics": "bucket_local_min_start_max_end_for_noncontiguous_explicit_windows",
            "nonempty_bucket_window_count": int(len(nonempty_windows)),
            "nonempty_bucket_local_min_start": int(min((start for start, _end in nonempty_windows), default=0)),
            "nonempty_bucket_local_max_end": int(max((end for _start, end in nonempty_windows), default=0)),
            "terminal_tail_window": True,
        }


class MolNexTRDatasetPipeline:
    def __init__(self, config: FormalSimDatasetConfig | None = None) -> None:
        self.config = config or FormalSimDatasetConfig()

    def markush_paths(self, shard_index: int) -> ShardPaths:
        shard = self.config.pose_root / "markush" / f"s{int(shard_index):03d}"
        accepted = shard / "accepted_candidate"
        return ShardPaths(
            branch="markush",
            shard_index=int(shard_index),
            shard_dir=shard,
            csv_path=shard / "markush_layout_positive.csv",
            contracts_dir=self.config.contract_root / "markush" / f"s{int(shard_index):03d}",
            accepted_dir=accepted,
            accepted_csv_path=accepted / "markush_layout_positive.csv",
        )

    def fragment_paths(self, shard_index: int) -> ShardPaths:
        shard = self.config.pose_root / "fragment" / f"s{int(shard_index):03d}"
        return ShardPaths(
            branch="fragment",
            shard_index=int(shard_index),
            shard_dir=shard,
            csv_path=shard / "attachment_fragment_positive.csv",
            contracts_dir=self.config.contract_root / "fragment" / f"s{int(shard_index):03d}",
        )

    def ordinary_paths(self, shard_index: int) -> ShardPaths:
        shard = self.config.pose_root / "ordinary" / f"s{int(shard_index):03d}"
        return ShardPaths(
            branch="ordinary",
            shard_index=int(shard_index),
            shard_dir=shard,
            csv_path=shard / "ordinary_negative.csv",
            contracts_dir=self.config.contract_root / "ordinary" / f"s{int(shard_index):03d}",
        )

    def markush_commands(
        self,
        shard_index: int,
        *,
        rows_per_bucket: int,
        seed: int,
        candidate_multiplier: int = 2,
        candidate_rows_per_bucket: int | None = None,
        renderer_seed_retries: int = 3,
        bucket_window_overrides: str = "",
        bucket_targets_override: str = "",
    ) -> list[CommandSpec]:
        paths = self.markush_paths(shard_index)
        explicit_windows = parse_bucket_window_overrides(bucket_window_overrides)
        if explicit_windows:
            window: CandidatePlanWindow | ExplicitBucketPlanWindow = ExplicitBucketPlanWindow(
                shard_index=int(shard_index),
                accepted_rows_per_bucket=int(rows_per_bucket),
                bucket_windows=explicit_windows,
            )
            candidate_bucket_targets_arg = window.candidate_rows_per_bucket_arg
            extra_window_args = ["--bucket-window-overrides", window.bucket_window_arg]
        else:
            window = CandidatePlanWindow.from_targets(
                shard_index,
                accepted_rows_per_bucket=rows_per_bucket,
                candidate_multiplier=candidate_multiplier,
                candidate_rows_per_bucket=candidate_rows_per_bucket,
            )
            candidate_bucket_targets_arg = str(window.candidate_rows_per_bucket)
            extra_window_args = []
        bucket_targets_arg = str(bucket_targets_override or "").strip() or str(window.accepted_rows_per_bucket)
        thresholds = self.config.markush_thresholds
        branch = "markush"
        idx = int(shard_index)
        raw_csv = paths.csv_path
        accepted_csv = paths.accepted_csv_path
        if accepted_csv is None:
            raise ValueError("markush accepted_csv_path is required")
        contracts = paths.contracts_dir
        return [
            CommandSpec(
                stage="generate_raw_markush_cdk_svg_formal_nonlinear",
                branch=branch,
                shard_index=idx,
                expected_output=raw_csv,
                argv=[
                    self.config.python_executable,
                    "training/molnextr_markush/tools/build_pose_factory_markush_shard.py",
                    "--raw-root",
                    self.config.rel(self.config.markush_raw_root),
                    "--subset-glob",
                    "markushgrapher2/**/*.parquet,markushgrapher-synthetic-training-source/**/*.arrow",
                    "--row-prefix",
                    f"formal_sim_markush_s{idx:03d}",
                    "--output-dir",
                    self.config.rel(paths.shard_dir),
                    "--dataset-name",
                    f"{self.config.pose_root.name}_markush_s{idx:03d}",
                    "--rows",
                    str(window.accepted_rows_per_bucket),
                    "--bucket-targets",
                    bucket_targets_arg,
                    "--candidate-bucket-targets",
                    candidate_bucket_targets_arg,
                    "--candidate-plan-csv",
                    self.config.rel(self.config.markush_candidate_plan_csv),
                    "--candidate-plan-json",
                    self.config.rel(self.config.markush_candidate_plan_json),
                    "--plan-start",
                    str(window.plan_start),
                    "--plan-end",
                    str(window.plan_end),
                    *extra_window_args,
                    "--seed",
                    str(int(seed)),
                    "--enable-formal-nonlinear-document-warp",
                    "--renderer-seed-retries",
                    str(int(renderer_seed_retries)),
                    "--min-atom-pair-distance",
                    str(thresholds.min_atom_pair_distance),
                    "--max-failure-details",
                    "1000",
                ],
            ),
            CommandSpec(
                stage="validate_raw_markush_schema",
                branch=branch,
                shard_index=idx,
                expected_output=contracts / "raw_validation.json",
                formal_gate=True,
                argv=[
                    self.config.python_executable,
                    "training/molnextr_markush/tools/validate_pose_factory_shard.py",
                    "--csv",
                    self.config.rel(raw_csv),
                    "--output",
                    self.config.rel(contracts / "raw_validation.json"),
                ],
            ),
            CommandSpec(
                stage="audit_raw_markush_formal_nonlinear_warp",
                branch=branch,
                shard_index=idx,
                expected_output=contracts / "raw_formal_nonlinear_warp_contract.json",
                formal_gate=True,
                argv=[
                    self.config.python_executable,
                    "training/molnextr_markush/tools/audit_markush_formal_nonlinear_warp_contract.py",
                    "--csv",
                    self.config.rel(raw_csv),
                    "--output",
                    self.config.rel(contracts / "raw_formal_nonlinear_warp_contract.json"),
                ],
            ),
            CommandSpec(
                stage="prepare_markush_accepted_candidate",
                branch=branch,
                shard_index=idx,
                expected_output=accepted_csv,
                formal_gate=True,
                argv=[
                    self.config.python_executable,
                    "training/molnextr_markush/tools/prepare_markush_cdk_accepted_candidate.py",
                    "--csv",
                    self.config.rel(raw_csv),
                    "--output-dir",
                    self.config.rel(paths.accepted_dir or paths.shard_dir / "accepted_candidate"),
                    *thresholds.argv(),
                    "--min-fit-points",
                    "6",
                ],
            ),
            CommandSpec(
                stage="validate_accepted_markush_schema",
                branch=branch,
                shard_index=idx,
                expected_output=contracts / "accepted_validation.json",
                formal_gate=True,
                argv=[
                    self.config.python_executable,
                    "training/molnextr_markush/tools/validate_pose_factory_shard.py",
                    "--csv",
                    self.config.rel(accepted_csv),
                    "--output",
                    self.config.rel(contracts / "accepted_validation.json"),
                ],
            ),
            CommandSpec(
                stage="audit_accepted_markush_formal_nonlinear_warp",
                branch=branch,
                shard_index=idx,
                expected_output=contracts / "accepted_formal_nonlinear_warp_contract.json",
                formal_gate=True,
                argv=[
                    self.config.python_executable,
                    "training/molnextr_markush/tools/audit_markush_formal_nonlinear_warp_contract.py",
                    "--csv",
                    self.config.rel(accepted_csv),
                    "--output",
                    self.config.rel(contracts / "accepted_formal_nonlinear_warp_contract.json"),
                ],
            ),
            CommandSpec(
                stage="audit_accepted_markush_pose_alignment",
                branch=branch,
                shard_index=idx,
                expected_output=contracts / "accepted_pose_alignment.json",
                formal_gate=True,
                argv=[
                    self.config.python_executable,
                    "training/molnextr_markush/tools/check_markush_pose_alignment.py",
                    "--csv",
                    self.config.rel(accepted_csv),
                    "--output",
                    self.config.rel(contracts / "accepted_pose_alignment.json"),
                    *thresholds.argv(),
                    "--min-fit-points",
                    "6",
                    "--min-rows",
                    "1",
                    "--min-variable-cell-fraction",
                    "0.60",
                    "--min-r-tag-rows",
                    "1",
                ],
            ),
            CommandSpec(
                stage="audit_markush_substitution_anchor_contract",
                branch=branch,
                shard_index=idx,
                expected_output=contracts / "markush_substitution_anchor_contract.json",
                formal_gate=True,
                argv=[
                    self.config.python_executable,
                    "training/molnextr_markush/tools/audit_markush_substitution_anchor_contract.py",
                    "--csv",
                    self.config.rel(accepted_csv),
                    "--output",
                    self.config.rel(contracts / "markush_substitution_anchor_contract.json"),
                ],
            ),
        ]

    def fragment_commands(
        self,
        shard_index: int,
        *,
        rows: int,
        seed: int,
        source_start_index: int | None = None,
        max_source_attempts_override: int | None = None,
    ) -> list[CommandSpec]:
        paths = self.fragment_paths(shard_index)
        target_bucket = FRAGMENT_TARGET_BUCKETS[int(shard_index) % len(FRAGMENT_TARGET_BUCKETS)]
        max_source_attempts = (
            int(max_source_attempts_override)
            if max_source_attempts_override is not None and int(max_source_attempts_override) > 0
            else max(int(rows) * (30 if target_bucket else 12), int(rows) + 80)
        )
        start_index = int(source_start_index) if source_start_index is not None else int(shard_index) * int(rows)
        # --target-mode wavy conflicts with --target-bucket, so pass wavy mode
        # + real-tight-crop-style instead of the bucket.
        target_args = ["--target-mode", "wavy", "--real-tight-crop-style", "--target-attempts-per-backbone", "12"]
        return [
            CommandSpec(
                stage="generate_fragment_formal_nonlinear",
                branch="fragment",
                shard_index=int(shard_index),
                expected_output=paths.csv_path,
                argv=[
                    self.config.python_executable,
                    "training/molnextr_markush/tools/build_pose_factory_fragment_shard.py",
                    "--output-dir",
                    self.config.rel(paths.shard_dir),
                    "--rows",
                    str(int(rows)),
                    "--seed",
                    str(int(seed)),
                    "--strict-orientation",
                    "--enable-formal-nonlinear-document-warp",
                    "--formal-nonlinear-warp-amplitude-px",
                    "1.2",
                    "--heldout-fragment-csv",
                    self.config.rel(self.config.fragment_heldout_csv),
                    "--input-smiles-csv",
                    self.config.rel(self.config.fragment_input_smiles_csv),
                    "--start-index",
                    str(start_index),
                    *target_args,
                    "--max-source-attempts",
                    str(max_source_attempts),
                ],
            ),
            CommandSpec(
                stage="audit_fragment_attachment_visual_contract",
                branch="fragment",
                shard_index=int(shard_index),
                expected_output=paths.contracts_dir / "attachment_visual_contract.json",
                formal_gate=True,
                argv=[
                    self.config.python_executable,
                    "training/molnextr_markush/tools/audit_attachment_visual_contract.py",
                    "--input-csv",
                    self.config.rel(paths.csv_path),
                    "--report",
                    self.config.rel(paths.contracts_dir / "attachment_visual_contract.json"),
                    "--output-clean-csv",
                    self.config.rel(paths.csv_path),
                    "--filter-rejected-rows",
                ],
            ),
            CommandSpec(
                stage="validate_fragment_schema",
                branch="fragment",
                shard_index=int(shard_index),
                expected_output=paths.contracts_dir / "validation.json",
                formal_gate=True,
                argv=[
                    self.config.python_executable,
                    "training/molnextr_markush/tools/validate_pose_factory_shard.py",
                    "--csv",
                    self.config.rel(paths.csv_path),
                    "--output",
                    self.config.rel(paths.contracts_dir / "validation.json"),
                ],
            ),
            CommandSpec(
                stage="audit_fragment_formal_nonlinear_warp",
                branch="fragment",
                shard_index=int(shard_index),
                expected_output=paths.contracts_dir / "fragment_formal_nonlinear_warp_contract.json",
                formal_gate=True,
                argv=[
                    self.config.python_executable,
                    "training/molnextr_markush/tools/audit_fragment_formal_nonlinear_warp_contract.py",
                    "--csv",
                    self.config.rel(paths.csv_path),
                    "--output",
                    self.config.rel(paths.contracts_dir / "fragment_formal_nonlinear_warp_contract.json"),
                ],
            ),
        ]

    def ordinary_commands(
        self,
        shard_index: int,
        *,
        rows: int,
        source_start_row: int | None = None,
        molgrapher_parquet: str | Path | None = None,
    ) -> list[CommandSpec]:
        paths = self.ordinary_paths(shard_index)
        start_row = int(source_start_row) if source_start_row is not None else int(shard_index) * int(rows)
        max_source_rows = max(int(rows) * 4, int(rows)) if int(rows) > 0 else 0
        parquet_path = self.config.resolve(molgrapher_parquet) if molgrapher_parquet else self.config.molgrapher_parquet
        return [
            CommandSpec(
                stage="convert_molgrapher_ordinary_negative",
                branch="ordinary",
                shard_index=int(shard_index),
                expected_output=paths.csv_path,
                argv=[
                    self.config.python_executable,
                    "training/molnextr_markush/tools/convert_molgrapher_pose_shard.py",
                    "--parquet",
                    self.config.rel(parquet_path),
                    "--output-dir",
                    self.config.rel(paths.shard_dir),
                    "--max-rows",
                    str(int(rows)),
                    "--start-row",
                    str(start_row),
                    "--max-source-rows",
                    str(max_source_rows),
                    "--heldout-csv",
                    self.config.rel(self.config.fragment_heldout_csv),
                ],
            ),
            CommandSpec(
                stage="validate_ordinary_schema",
                branch="ordinary",
                shard_index=int(shard_index),
                expected_output=paths.contracts_dir / "validation.json",
                formal_gate=True,
                argv=[
                    self.config.python_executable,
                    "training/molnextr_markush/tools/validate_pose_factory_shard.py",
                    "--csv",
                    self.config.rel(paths.csv_path),
                    "--output",
                    self.config.rel(paths.contracts_dir / "validation.json"),
                ],
            ),
            CommandSpec(
                stage="audit_ordinary_molnextr_quality",
                branch="ordinary",
                shard_index=int(shard_index),
                expected_output=paths.contracts_dir / "ordinary_molnextr_quality_contract.json",
                formal_gate=True,
                argv=[
                    self.config.python_executable,
                    "training/molnextr_markush/tools/audit_ordinary_molnextr_quality_contract.py",
                    "--csv",
                    self.config.rel(paths.csv_path),
                    "--output",
                    self.config.rel(paths.contracts_dir / "ordinary_molnextr_quality_contract.json"),
                    "--output-clean-csv",
                    self.config.rel(paths.csv_path),
                    "--filter-rejected-rows",
                ],
            ),
        ]

    def attachment_role_contract_command(
        self,
        *,
        fragment_shards: list[int],
        ordinary_shards: list[int],
        output_name: str = "attachment_role_contract.json",
    ) -> CommandSpec:
        if not fragment_shards:
            raise ValueError("fragment_shards must be non-empty")
        if not ordinary_shards:
            raise ValueError("ordinary_shards must be non-empty")
        output = self.config.contract_root / output_name
        argv = [
            self.config.python_executable,
            "training/molnextr_markush/tools/check_attachment_role_contract.py",
        ]
        for shard_index in fragment_shards:
            argv.extend(["--fragment-csv", self.config.rel(self.fragment_paths(shard_index).csv_path)])
        for shard_index in ordinary_shards:
            argv.extend(["--ordinary-csv", self.config.rel(self.ordinary_paths(shard_index).csv_path)])
        argv.extend(["--output", self.config.rel(output)])
        return CommandSpec(
            stage="audit_fragment_ordinary_attachment_role_contract",
            branch="aggregate",
            shard_index=-1,
            expected_output=output,
            formal_gate=True,
            argv=argv,
        )

    def markush_gate_results(self, shard_index: int) -> list[GateResult]:
        paths = self.markush_paths(shard_index)
        names = [
            "raw_validation.json",
            "raw_formal_nonlinear_warp_contract.json",
            "accepted_validation.json",
            "accepted_formal_nonlinear_warp_contract.json",
            "accepted_pose_alignment.json",
            "markush_substitution_anchor_contract.json",
        ]
        return [GateResult.from_json_file(paths.contracts_dir / name) for name in names]

    def markush_retry_diagnostics(self, shard_index: int) -> RetryDiagnostics:
        paths = self.markush_paths(shard_index)
        return analyze_retry_diagnostics(
            csv_path=paths.accepted_csv_path,
            manifest_path=paths.shard_dir / "manifest.json",
        )

    def dry_run_manifest(self, commands: list[CommandSpec]) -> dict[str, Any]:
        return {
            "schema_version": "molnextr_dataset_toolkit_dry_run_v1",
            "dataset_id": self.config.dataset_id,
            "dataset_family": self.config.dataset_family,
            "pose_root": self.config.rel(self.config.pose_root),
            "contract_root": self.config.rel(self.config.contract_root),
            "commands": [command.to_dict(root=self.config.root) for command in commands],
            "formal_training_start_allowed": False,
            "blockers": [
                "dry-run only",
                "aggregate merge, source-leak, source-disjoint split, coverage, visual review, router, training-scale, model-scale, runtime/DDP and readiness gates still required",
            ],
        }

    def write_json(self, payload: dict[str, Any], output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
