from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


QUERY_FIELDS = [
    "index",
    "name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "compute_mode",
]


def parse_gpu_csv(stdout: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(QUERY_FIELDS):
            continue
        try:
            index = int(parts[0])
            memory_total_mb = int(float(parts[2]))
            memory_used_mb = int(float(parts[3]))
            utilization_gpu_percent = int(float(parts[4]))
        except ValueError:
            continue
        gpus.append(
            {
                "index": index,
                "name": parts[1],
                "memory_total_mb": memory_total_mb,
                "memory_total_gb": memory_total_mb / 1024.0,
                "memory_used_mb": memory_used_mb,
                "utilization_gpu_percent": utilization_gpu_percent,
                "compute_mode": parts[5],
            }
        )
    return gpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write machine-checkable GPU runtime evidence from nvidia-smi."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--input-csv",
        default="",
        help="Parse a raw nvidia-smi --query-gpu CSV captured outside this Python process.",
    )
    parser.add_argument("--min-gpus", type=int, default=2)
    parser.add_argument("--min-memory-gb", type=float, default=15.0)
    args = parser.parse_args()

    command = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(QUERY_FIELDS),
        "--format=csv,noheader,nounits",
    ]
    input_csv = Path(args.input_csv) if args.input_csv else None
    if input_csv:
        stdout = input_csv.read_text(encoding="utf-8")
        stderr = ""
        returncode = 0
        evidence_source = {
            "mode": "file_backed_nvidia_smi_query_csv",
            "input_csv": str(input_csv),
            "python_process_did_not_call_nvidia_smi": True,
        }
    else:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
        evidence_source = {
            "mode": "python_subprocess_nvidia_smi",
            "input_csv": "",
            "python_process_did_not_call_nvidia_smi": False,
        }
    gpus = parse_gpu_csv(stdout)
    blockers: list[str] = []
    if returncode != 0:
        blockers.append(f"nvidia-smi returned {returncode}")
    if len(gpus) < int(args.min_gpus):
        blockers.append(f"observed GPU count {len(gpus)} < required {int(args.min_gpus)}")
    for gpu in gpus:
        if float(gpu["memory_total_gb"]) < float(args.min_memory_gb):
            blockers.append(
                f"GPU {gpu['index']} memory {float(gpu['memory_total_gb']):.2f} GB < required {float(args.min_memory_gb):.2f} GB"
            )

    report = {
        "schema_version": "gpu_runtime_evidence_v1",
        "timestamp_unix": time.time(),
        "command": command,
        "which": shutil.which("nvidia-smi") or "",
        "evidence_source": evidence_source,
        "returncode": returncode,
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
        "gpus": gpus,
        "gpu_count_observed": len(gpus),
        "min_gpus_required": int(args.min_gpus),
        "min_memory_gb_required": float(args.min_memory_gb),
        "passed": returncode == 0 and not blockers,
        "blockers": blockers,
        "policy": {
            "runtime_gate_evidence_only": True,
            "does_not_start_training": True,
            "formal_runs_require_two_16gb_gpus_or_measured_single_gpu_rationale": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
