from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.pose_factory import external_backend_plan


def parquet_metadata(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        return {
            "path": str(path),
            "readable": False,
            "error": f"pyarrow_unavailable:{exc}",
        }

    try:
        parquet_file = pq.ParquetFile(path)
        return {
            "path": str(path),
            "readable": True,
            "rows": int(parquet_file.metadata.num_rows),
            "columns": list(parquet_file.schema.names),
        }
    except Exception as exc:
        return {
            "path": str(path),
            "readable": False,
            "error": f"parquet_unreadable:{exc}",
        }


def inspect_markushgrapher2(raw_root: Path) -> dict[str, Any]:
    dataset_root = raw_root / "markushgrapher2"
    parquet_files = sorted(dataset_root.glob("*/*.parquet"))
    subsets: dict[str, dict[str, Any]] = {}
    total_rows = 0
    readable_files = 0
    for path in parquet_files:
        subset = path.parent.name
        metadata = parquet_metadata(path)
        subset_record = subsets.setdefault(
            subset,
            {
                "files": 0,
                "readable_files": 0,
                "rows": 0,
                "columns": [],
                "paths": [],
                "errors": [],
            },
        )
        subset_record["files"] += 1
        subset_record["paths"].append(str(path))
        if metadata.get("readable"):
            readable_files += 1
            subset_record["readable_files"] += 1
            subset_record["rows"] += int(metadata.get("rows") or 0)
            total_rows += int(metadata.get("rows") or 0)
            if not subset_record["columns"]:
                subset_record["columns"] = list(metadata.get("columns") or [])
        else:
            subset_record["errors"].append(metadata.get("error"))

    readme_path = dataset_root / "README.md"
    readme_present = readme_path.exists()
    return {
        "dataset": "markushgrapher2",
        "raw_root": str(dataset_root),
        "readme_present": readme_present,
        "parquet_file_count": len(parquet_files),
        "readable_parquet_files": readable_files,
        "total_rows_read_from_parquet": total_rows,
        "subsets": dict(sorted(subsets.items())),
        "pose_factory_use": {
            "ordinary_molecule_pool": "uspto-mol-m-54k can seed ordinary negatives and complete-molecule style references, but complete rows still bypass sidecar.",
            "markush_layout_pool": "ip5-markush, m2s, uspto-markush, and uspto-mol-m-54k provide page_image, cells, annotation, cxsmiles, and cxsmiles_opt for future Markush layout expert.",
            "fragment_endpoint_pool": "MG2 alone is not a fragment endpoint label source; wavy endpoint coordinates must be generated or manually reviewed.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect protected source data for the pose-aware factory.")
    parser.add_argument("--raw-root", default="training/molnextr_markush/data/raw")
    parser.add_argument("--output", default="training/molnextr_markush/runs/sidecar_contract/pose_source_inventory.json")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "external_backend_plan": external_backend_plan(),
        "sources": {
            "markushgrapher2": inspect_markushgrapher2(Path(args.raw_root)),
        },
        "formal_training_status": "blocked_until_pose_factory_shards_pass_validation_and_visual_review",
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
