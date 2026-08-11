from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShardPaths:
    branch: str
    shard_index: int
    shard_dir: Path
    csv_path: Path
    contracts_dir: Path
    accepted_dir: Path | None = None
    accepted_csv_path: Path | None = None

    @property
    def shard_id(self) -> str:
        return f"s{self.shard_index:03d}"


@dataclass(frozen=True)
class CommandSpec:
    stage: str
    argv: list[str]
    expected_output: Path | None = None
    branch: str = ""
    shard_index: int | None = None
    formal_gate: bool = False

    def is_complete(self) -> bool:
        return self.expected_output is not None and self.expected_output.exists()

    def to_dict(self, *, root: Path | None = None) -> dict[str, Any]:
        def path_text(path: Path | None) -> str:
            if path is None:
                return ""
            if root is None:
                return str(path)
            try:
                return str(path.relative_to(root))
            except ValueError:
                return str(path)

        return {
            "stage": self.stage,
            "branch": self.branch,
            "shard_index": self.shard_index,
            "argv": self.argv,
            "expected_output": path_text(self.expected_output),
            "complete": self.is_complete(),
            "formal_gate": self.formal_gate,
        }

    def run(self, *, cwd: Path, execute: bool, skip_existing: bool = False) -> dict[str, Any]:
        record = self.to_dict(root=cwd)
        record["executed"] = False
        record["returncode"] = None
        if skip_existing and self.is_complete():
            record["skipped_existing"] = True
            return record
        if execute:
            completed = subprocess.run(self.argv, cwd=str(cwd), check=True)
            record["executed"] = True
            record["returncode"] = int(completed.returncode)
        return record


@dataclass(frozen=True)
class GateResult:
    path: Path
    present: bool
    passed: bool
    row_count: int | None = None
    invalid_rows: int | None = None
    failed_rows: int | None = None
    blockers: tuple[str, ...] = ()
    payload: dict[str, Any] | None = None

    @classmethod
    def from_json_file(cls, path: Path) -> "GateResult":
        if not path.exists():
            return cls(path=path, present=False, passed=False, blockers=("missing gate report",))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return cls(path=path, present=True, passed=False, blockers=("gate report is not a JSON object",))
        passed = payload.get("passed")
        if passed is None:
            passed = payload.get("trainable")
        blockers = payload.get("blockers") or payload.get("issues") or []
        if not isinstance(blockers, list):
            blockers = [str(blockers)]
        return cls(
            path=path,
            present=True,
            passed=passed is True,
            row_count=_optional_int(payload.get("row_count")),
            invalid_rows=_optional_int(payload.get("invalid_rows")),
            failed_rows=_optional_int(payload.get("failed_rows")),
            blockers=tuple(str(item) for item in blockers[:20]),
            payload=payload,
        )

    def to_dict(self, *, root: Path | None = None) -> dict[str, Any]:
        path = str(self.path)
        if root is not None:
            try:
                path = str(self.path.relative_to(root))
            except ValueError:
                pass
        return {
            "path": path,
            "present": self.present,
            "passed": self.passed,
            "row_count": self.row_count,
            "invalid_rows": self.invalid_rows,
            "failed_rows": self.failed_rows,
            "blockers": list(self.blockers),
        }


def _optional_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
