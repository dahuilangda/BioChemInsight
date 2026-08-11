from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


REPO = "docling-project/MarkushGrapher-Datasets"
METADATA_FILES = [
    ".gitattributes",
    "README.md",
    "markushgrapher-synthetic-training/dataset_dict.json",
    "markushgrapher-synthetic-training/train/dataset_info.json",
    "markushgrapher-synthetic-training/train/state.json",
    "markushgrapher-synthetic-training/test/dataset_info.json",
    "markushgrapher-synthetic-training/test/state.json",
]
TRAIN_FILES = [
    f"markushgrapher-synthetic-training/train/data-{index:05d}-of-00050.arrow"
    for index in range(50)
]
TEST_FILES = [
    f"markushgrapher-synthetic-training/test/data-{index:05d}-of-00006.arrow"
    for index in range(6)
]


def dataset_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"


def valid_existing(path: Path, *, min_bytes: int) -> bool:
    return path.exists() and path.stat().st_size >= int(min_bytes)


def download_one(
    relative: str,
    *,
    target_root: Path,
    proxy: str,
    retries: int,
    retry_delay: int,
    min_bytes: int,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if valid_existing(target, min_bytes=min_bytes):
        return {"file": relative, "status": "skipped_existing", "bytes": int(target.stat().st_size)}
    temp = target.with_suffix(target.suffix + ".part")
    command = [
        "curl",
        "--http1.1",
        "-L",
        "--fail",
        "--retry",
        str(retries),
        "--retry-all-errors",
        "--retry-delay",
        str(retry_delay),
        "--connect-timeout",
        "30",
        "--speed-time",
        "120",
        "--speed-limit",
        "1024",
    ]
    if proxy:
        command[1:1] = ["-x", proxy]
    if resume and temp.exists() and temp.stat().st_size > 0:
        command.extend(["-C", "-"])
    command.extend(["-o", str(temp), dataset_url(relative)])
    if dry_run:
        return {"file": relative, "status": "dry_run", "command": command}
    if temp.exists() and not resume:
        temp.unlink()
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return {
            "file": relative,
            "status": "failed",
            "returncode": int(proc.returncode),
            "stderr_tail": proc.stderr[-2000:],
        }
    if not valid_existing(temp, min_bytes=min_bytes):
        size = temp.stat().st_size if temp.exists() else 0
        return {"file": relative, "status": "failed_too_small", "bytes": int(size), "min_bytes": int(min_bytes)}
    temp.replace(target)
    return {"file": relative, "status": "downloaded", "bytes": int(target.stat().st_size)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MarkushGrapher-1 synthetic-training arrow shards with resume-safe manifests.")
    parser.add_argument("--output-root", default="training/molnextr_markush/data/raw/markushgrapher-synthetic-training-source")
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or "")
    parser.add_argument("--train-shards", type=int, default=50)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=int, default=15)
    parser.add_argument("--min-arrow-bytes", type=int, default=1_000_000)
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Sleep this many seconds between sequential downloads when --workers=1.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    shard_count = max(0, min(50, int(args.train_shards)))
    files = [*METADATA_FILES, *TRAIN_FILES[:shard_count]]
    if args.include_test:
        files.extend(TEST_FILES)

    target_root = Path(args.output_root)
    results: list[dict[str, Any]] = []
    worker_count = max(1, int(args.workers))
    if worker_count == 1:
        for index, relative in enumerate(files):
            result = download_one(
                relative,
                target_root=target_root,
                proxy=str(args.proxy or ""),
                retries=int(args.retries),
                retry_delay=int(args.retry_delay),
                min_bytes=0 if not relative.endswith(".arrow") else int(args.min_arrow_bytes),
                resume=not bool(args.no_resume),
                dry_run=bool(args.dry_run),
            )
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
            if float(args.request_delay) > 0 and index < len(files) - 1:
                time.sleep(float(args.request_delay))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    download_one,
                    relative,
                    target_root=target_root,
                    proxy=str(args.proxy or ""),
                    retries=int(args.retries),
                    retry_delay=int(args.retry_delay),
                    min_bytes=0 if not relative.endswith(".arrow") else int(args.min_arrow_bytes),
                    resume=not bool(args.no_resume),
                    dry_run=bool(args.dry_run),
                )
                for relative in files
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)

    failures = [item for item in results if str(item.get("status")) not in {"downloaded", "skipped_existing", "dry_run"}]
    manifest = {
        "schema_version": "markushgrapher1_download_manifest_v1",
        "repo": REPO,
        "output_root": str(target_root),
        "train_shards_requested": shard_count,
        "include_test": bool(args.include_test),
        "workers": worker_count,
        "request_delay_seconds": float(args.request_delay),
        "resume_enabled": not bool(args.no_resume),
        "file_count": len(files),
        "downloaded_or_present": len(results) - len(failures),
        "failure_count": len(failures),
        "results": sorted(results, key=lambda item: str(item.get("file") or "")),
        "policy": {
            "raw_source_only": True,
            "does_not_create_trainable_rows": True,
            "must_pass_capacity_pose_visual_source_leak_and_acceptance_gates_before_training": True,
        },
    }
    target_root.mkdir(parents=True, exist_ok=True)
    (target_root / "download_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
