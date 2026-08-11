from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MARKUSH_R_COUNT_BUCKETS: tuple[str, ...] = ("1", "2", "3-4", "5-8", "9+")

FRAGMENT_TARGET_BUCKETS: tuple[str, ...] = (
    "",
    "fragment|left|C|wavy",
    "fragment|right|C|wavy",
    "fragment|top|C|wavy",
    "fragment|bottom|C|wavy",
    "fragment|left|C|query_attachment",
    "fragment|right|C|query_attachment",
    "fragment|top|C|query_attachment",
    "fragment|bottom|C|query_attachment",
    "fragment|left|C|dummy_atom",
    "fragment|right|C|dummy_atom",
    "fragment|top|C|dummy_atom",
    "fragment|bottom|C|dummy_atom",
    "fragment|left|C|cut",
    "fragment|right|C|cut",
    "fragment|top|C|cut",
    "fragment|bottom|C|cut",
    "fragment|left|N|cut",
    "fragment|right|N|cut",
    "fragment|top|N|cut",
    "fragment|bottom|N|cut",
    "fragment|left|N|wavy",
    "fragment|right|N|wavy",
    "fragment|top|N|wavy",
    "fragment|bottom|N|wavy",
    "fragment|left|N|query_attachment",
    "fragment|top|N|query_attachment",
    "fragment|right|N|query_attachment",
    "fragment|bottom|N|query_attachment",
    "fragment|left|N|dummy_atom",
    "fragment|right|N|dummy_atom",
    "fragment|top|N|dummy_atom",
    "fragment|bottom|N|dummy_atom",
    "fragment|left|O|cut",
    "fragment|right|O|wavy",
    "fragment|top|O|query_attachment",
    "fragment|bottom|O|dummy_atom",
    "fragment|left|S|cut",
    "fragment|right|S|wavy",
    "fragment|top|S|query_attachment",
    "fragment|bottom|S|dummy_atom",
    "fragment|left|P|cut",
    "fragment|right|P|wavy",
    "fragment|top|P|query_attachment",
    "fragment|bottom|P|dummy_atom",
    "fragment|left|NH|cut",
    "fragment|right|NH|wavy",
    "fragment|top|OH|query_attachment",
    "fragment|bottom|OH|dummy_atom",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class MarkushGateThresholds:
    max_rmse: float = 0.05
    max_line_abs_p95: float = 0.05
    max_line_abs_max: float = 0.10
    max_intersection_anchor_rmse: float = 0.05
    max_intersection_anchor_abs_max: float = 0.10
    max_affine_scale_ratio: float = 1.25
    min_intersection_anchors: int = 2
    min_atom_pair_distance: float = 0.005

    def argv(self) -> list[str]:
        return [
            "--max-rmse",
            str(self.max_rmse),
            "--max-line-abs-p95",
            str(self.max_line_abs_p95),
            "--max-line-abs-max",
            str(self.max_line_abs_max),
            "--max-intersection-anchor-rmse",
            str(self.max_intersection_anchor_rmse),
            "--max-intersection-anchor-abs-max",
            str(self.max_intersection_anchor_abs_max),
            "--max-affine-scale-ratio",
            str(self.max_affine_scale_ratio),
            "--min-intersection-anchors",
            str(self.min_intersection_anchors),
            "--min-atom-pair-distance",
            str(self.min_atom_pair_distance),
        ]


@dataclass(frozen=True)
class FormalSimDatasetConfig:
    dataset_id: str = "molnextr_moe_production_v1"
    dataset_family: str = "molnextr_markush_fragment_router_formal_sim_v1"
    root: Path = repo_root()
    python_executable: str = "/home/dahuilangda/miniconda3/envs/llm/bin/python"
    markush_raw_root: Path | None = None
    markush_candidate_plan_csv: Path | None = None
    markush_candidate_plan_json: Path | None = None
    molgrapher_parquet: Path | None = None
    fragment_input_smiles_csv: Path | None = None
    fragment_heldout_csv: Path | None = None
    literature_heldout_csv: Path | None = None
    original_heldout_csv: Path | None = None
    markush_thresholds: MarkushGateThresholds = MarkushGateThresholds()

    def __post_init__(self) -> None:
        root = self.root
        defaults = {
            "markush_raw_root": root / "training/molnextr_markush/data/raw",
            "markush_candidate_plan_csv": root
            / "training/molnextr_markush/data/generated/pose_factory/"
            "molnextr_moe_production_v1_markush_source_anchor_plan/candidate_plan.csv",
            "markush_candidate_plan_json": root
            / "training/molnextr_markush/data/generated/pose_factory/"
            "molnextr_moe_production_v1_markush_source_anchor_plan/candidate_plan.json",
            "molgrapher_parquet": root
            / "training/molnextr_markush/data/raw/molgrapher_synthetic_300k/data/train-00000-of-00022-11eee679e7d7b976.parquet",
            "fragment_input_smiles_csv": root
            / "training/molnextr_markush/data/generated/pose_factory/"
            "molnextr_moe_production_v1_fragment_source_backbone_seeds/seeds.csv",
            "fragment_heldout_csv": root / "training/molnextr_markush/data/rgreco_fragment_eval/eval.csv",
            "literature_heldout_csv": root / "training/molnextr_markush/data/literature_eval/eval.csv",
            "original_heldout_csv": root / "training/molnextr_markush/data/original_eval/eval.csv",
        }
        for name, value in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)

    @property
    def pose_root(self) -> Path:
        return self.root / f"training/molnextr_markush/data/generated/pose_factory/{self.dataset_id}"

    @property
    def contract_root(self) -> Path:
        return self.root / f"training/molnextr_markush/runs/{self.dataset_id}_contracts"

    @property
    def aggregate_root(self) -> Path:
        return self.pose_root / "aggregate"

    @property
    def split_root(self) -> Path:
        return self.pose_root / "splits/source_disjoint_candidate"

    def resolve(self, path: str | Path) -> Path:
        value = Path(path)
        return value if value.is_absolute() else self.root / value

    def rel(self, path: str | Path) -> str:
        value = self.resolve(path)
        try:
            return str(value.relative_to(self.root))
        except ValueError:
            return str(value)
