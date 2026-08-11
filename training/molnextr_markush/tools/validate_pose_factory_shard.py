from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.pose_factory import validate_pose_factory_shard


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pose-aware generated shard contracts before training.")
    parser.add_argument("--csv", required=True, help="Generated pose-factory CSV shard.")
    parser.add_argument("--output", default="", help="JSON report path. Defaults to <csv>.validation.json")
    parser.add_argument("--skip-image-checks", action="store_true")
    parser.add_argument("--max-issues", type=int, default=200)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    report = validate_pose_factory_shard(
        csv_path,
        check_images=not args.skip_image_checks,
        max_issues=int(args.max_issues),
    )
    output = Path(args.output) if args.output else csv_path.with_suffix(csv_path.suffix + ".validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if not report.trainable:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
