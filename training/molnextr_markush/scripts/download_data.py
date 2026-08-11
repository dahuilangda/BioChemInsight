from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


MG2_REPO = "docling-project/MarkushGrapher-2-Datasets"
MG1_REPO = "docling-project/MarkushGrapher-Datasets"
RGRECO_REPO = "yuanjier/RGReco"

MG2_FILES = [
    ".gitattributes",
    "README.md",
    "ip5-markush/test-00000-of-00001.parquet",
    "m2s/test-00000-of-00001.parquet",
    "uspto-markush/test-00000-of-00001.parquet",
    "uspto-mol-m-54k/test-00000-of-00001.parquet",
    "uspto-mol-m-54k/train-00000-of-00006.parquet",
    "uspto-mol-m-54k/train-00001-of-00006.parquet",
    "uspto-mol-m-54k/train-00002-of-00006.parquet",
    "uspto-mol-m-54k/train-00003-of-00006.parquet",
    "uspto-mol-m-54k/train-00004-of-00006.parquet",
    "uspto-mol-m-54k/train-00005-of-00006.parquet",
]

MG1_METADATA_FILES = [
    ".gitattributes",
    "README.md",
    "m2s/dataset_dict.json",
    "m2s/test/dataset_info.json",
    "m2s/test/state.json",
    "markushgrapher-synthetic-training/dataset_dict.json",
    "markushgrapher-synthetic-training/train/dataset_info.json",
    "markushgrapher-synthetic-training/train/state.json",
    "markushgrapher-synthetic-training/test/dataset_info.json",
    "markushgrapher-synthetic-training/test/state.json",
    "markushgrapher-synthetic/dataset_dict.json",
    "markushgrapher-synthetic/test/dataset_info.json",
    "markushgrapher-synthetic/test/state.json",
    "uspto-markush/dataset_dict.json",
    "uspto-markush/test/dataset_info.json",
    "uspto-markush/test/state.json",
]

MG1_SYNTHETIC_TRAIN_FILES = [
    f"markushgrapher-synthetic-training/train/data-{index:05d}-of-00050.arrow"
    for index in range(50)
]

MG1_SYNTHETIC_TEST_FILES = [
    f"markushgrapher-synthetic-training/test/data-{index:05d}-of-00006.arrow"
    for index in range(6)
]

MG1_EVAL_FILES = [
    "m2s/test/data-00000-of-00001.arrow",
    "markushgrapher-synthetic/test/data-00000-of-00001.arrow",
    "uspto-markush/test/data-00000-of-00001.arrow",
]


def dataset_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def run_curl(url: str, output: Path, *, proxy: str, retries: int, dry_run: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return
    command = ["curl", "-L", "--fail", "--retry", str(retries), "--retry-delay", "5"]
    if proxy:
        command.extend(["-x", proxy])
    command.extend(["-o", str(output), url])
    if dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, check=True)


def download_files(
    *,
    repo: str,
    files: list[str],
    target_root: Path,
    proxy: str,
    retries: int,
    dry_run: bool,
) -> list[str]:
    downloaded = []
    for relative in files:
        run_curl(
            dataset_url(repo, relative),
            target_root / relative,
            proxy=proxy,
            retries=retries,
            dry_run=dry_run,
        )
        downloaded.append(relative)
    return downloaded


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore protected raw Markush/RGReco source datasets.")
    parser.add_argument("--output-root", default="training/molnextr_markush/data/raw")
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or "")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-mg2", action="store_true")
    parser.add_argument("--skip-rgreco", action="store_true")
    parser.add_argument("--include-mg1", action="store_true", help="Download the ~28 GB original MarkushGrapher-1 raw set.")
    parser.add_argument("--mg1-train-shards", type=int, default=50)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    manifest: dict[str, object] = {
        "protected_raw_data": True,
        "policy": "Do not delete raw/source datasets during training workspace cleanup.",
        "sources": {},
    }

    if not args.skip_mg2:
        mg2_files = download_files(
            repo=MG2_REPO,
            files=MG2_FILES,
            target_root=output_root / "markushgrapher2",
            proxy=args.proxy,
            retries=args.retries,
            dry_run=args.dry_run,
        )
        manifest["sources"]["markushgrapher2"] = {"repo": MG2_REPO, "files": mg2_files}

    if not args.skip_rgreco:
        # The RGReco repository contains many small files. Use git-lfs when
        # available; otherwise leave this source for a separate snapshot pull.
        target = output_root / "rgreco"
        command = [
            "git",
            "clone",
            "https://huggingface.co/datasets/yuanjier/RGReco",
            str(target),
        ]
        if args.dry_run:
            print(" ".join(command))
        elif not target.exists():
            env = os.environ.copy()
            if args.proxy:
                env["HTTPS_PROXY"] = args.proxy
                env["HTTP_PROXY"] = args.proxy
            subprocess.run(command, check=True, env=env)
        manifest["sources"]["rgreco"] = {"repo": RGRECO_REPO, "method": "git clone"}

    if args.include_mg1:
        shard_count = max(0, min(50, int(args.mg1_train_shards)))
        mg1_files = [
            *MG1_METADATA_FILES,
            *MG1_EVAL_FILES,
            *MG1_SYNTHETIC_TEST_FILES,
            *MG1_SYNTHETIC_TRAIN_FILES[:shard_count],
        ]
        downloaded = download_files(
            repo=MG1_REPO,
            files=mg1_files,
            target_root=output_root / "markushgrapher-synthetic-training-source",
            proxy=args.proxy,
            retries=args.retries,
            dry_run=args.dry_run,
        )
        manifest["sources"]["markushgrapher1"] = {
            "repo": MG1_REPO,
            "files": downloaded,
            "train_shards_requested": shard_count,
        }

    write_manifest(output_root / "RAW_DATA_MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
