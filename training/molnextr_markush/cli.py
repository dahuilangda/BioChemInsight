#!/usr/bin/env python3
"""Unified CLI for the MolNexTR Markush training toolkit.

Provides a single entry point for training, calibration, and smoke testing.
Each subcommand delegates to the existing tool module's ``main()`` function,
so the CLI is a thin orchestrator — all logic stays in the tools.

Usage::

    # Train the MoE decoder
    python -m training.molnextr_markush.cli train moe --epochs 12 --batch-size 8

    # Train Mask R-CNN detectors
    python -m training.molnextr_markush.cli train wavy-maskrcnn --epochs 20
    python -m training.molnextr_markush.cli train attachment-maskrcnn --epochs 25
    python -m training.molnextr_markush.cli train wavy-unet --epochs 8

    # Calibrate confidence thresholds
    python -m training.molnextr_markush.cli calibrate markush-layout predictions.csv

    # Run smoke tests (fast, no GPU required)
    python -m training.molnextr_markush.cli smoke

    # List all available tools
    python -m training.molnextr_markush.cli list

    # Run any tool by name with passthrough args
    python -m training.molnextr_markush.cli run build_pose_factory_fragment_shard -- --shard 0
"""
from __future__ import annotations

import argparse
import importlib
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOLS_DIR = Path(__file__).resolve().parent / "tools"

# Subcommand → (module_path, description)
TRAIN_COMMANDS = {
    "moe": (
        "training.molnextr_markush.tools.train_moe",
        "Joint LoRA-MoE trainer for the MolNexTR decoder",
    ),
    "wavy-maskrcnn": (
        "training.molnextr_markush.tools.train_wavy_maskrcnn",
        "Mask R-CNN wavy-bond detector",
    ),
    "attachment-maskrcnn": (
        "training.molnextr_markush.tools.train_attachment_maskrcnn",
        "Multi-class attachment-point Mask R-CNN",
    ),
    "wavy-unet": (
        "training.molnextr_markush.tools.train_wavy_unet",
        "U-Net wavy-bond segmentation (Phase 0 gate)",
    ),
    "markush-layout": (
        "training.molnextr_markush.tools.train_markush_layout_expert",
        "Markush layout expert trainer",
    ),
    "fragment-attachment": (
        "training.molnextr_markush.tools.train_fragment_attachment_expert",
        "Fragment attachment expert trainer",
    ),
    "confidence-head": (
        "training.molnextr_markush.tools.train_confidence_head",
        "Confidence head trainer",
    ),
}

CALIBRATE_COMMANDS = {
    "fragment-attachment": (
        "training.molnextr_markush.tools.calibrate_fragment_attachment_expert",
        "Calibrate fragment-attachment confidence thresholds",
    ),
    "markush-layout": (
        "training.molnextr_markush.tools.calibrate_markush_layout_expert",
        "Calibrate markush-layout confidence thresholds",
    ),
}


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #

def _run_tool_main(module_path: str, argv: list[str]) -> int:
    """Import a tool module, patch sys.argv, and call its main()."""
    original_argv = sys.argv
    sys.argv = [module_path.rsplit(".", 1)[-1]] + argv
    try:
        module = importlib.import_module(module_path)
        if hasattr(module, "main"):
            module.main()
            return 0
        print(f"Error: {module_path} has no main() function", file=sys.stderr)
        return 1
    finally:
        sys.argv = original_argv


def _cmd_train(args: argparse.Namespace) -> int:
    spec = TRAIN_COMMANDS.get(args.trainer)
    if spec is None:
        print(f"Unknown trainer: {args.trainer}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(TRAIN_COMMANDS))}", file=sys.stderr)
        return 1
    module_path, _ = spec
    return _run_tool_main(module_path, args.passthrough)


def _cmd_calibrate(args: argparse.Namespace) -> int:
    spec = CALIBRATE_COMMANDS.get(args.target)
    if spec is None:
        print(f"Unknown calibration target: {args.target}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(CALIBRATE_COMMANDS))}", file=sys.stderr)
        return 1
    module_path, _ = spec
    return _run_tool_main(module_path, args.passthrough)


def _cmd_smoke(args: argparse.Namespace) -> int:
    """Run the smoke test suite."""
    smoke_dir = Path(__file__).resolve().parent / "tests" / "smoke"
    if not smoke_dir.exists():
        print(f"Smoke test directory not found: {smoke_dir}", file=sys.stderr)
        return 1
    # Try pytest first, fall back to direct execution
    try:
        import pytest

        return pytest.main(["-v", str(smoke_dir)])
    except ImportError:
        return _run_smoke_direct(smoke_dir)


def _run_smoke_direct(smoke_dir: Path) -> int:
    """Run smoke tests without pytest (plain assert-based)."""
    failed = 0
    passed = 0
    for test_file in sorted(smoke_dir.glob("test_*.py")):
        name = test_file.stem
        print(f"  RUN  {name}", flush=True)
        # Phase 1: load the module (import errors fail the whole file)
        try:
            globals_dict = {"__file__": str(test_file), "__name__": "__main__"}
            exec(compile(test_file.read_text(), str(test_file), "exec"), globals_dict)
        except Exception as exc:
            failed += 1
            print(f"  FAIL {name} (import): {exc}", flush=True)
            continue
        # Phase 2: run each test function independently
        test_funcs = [
            v for k, v in globals_dict.items() if k.startswith("test_") and callable(v)
        ]
        for func in test_funcs:
            try:
                func()
                passed += 1
                print(f"  PASS {name}.{func.__name__}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"  FAIL {name}.{func.__name__}: {exc}", flush=True)
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


def _cmd_list(args: argparse.Namespace) -> int:
    """List all available tools in the tools/ directory."""
    tools = sorted(p.stem for p in TOOLS_DIR.glob("*.py") if not p.stem.startswith("_"))
    print(f"Tools in {TOOLS_DIR.relative_to(ROOT)}:")
    for tool in tools:
        print(f"  {tool}")
    print(f"\nTotal: {len(tools)} tools")
    print(f"\nCLI-accessible trainers: {', '.join(sorted(TRAIN_COMMANDS))}")
    print(f"CLI-accessible calibrators: {', '.join(sorted(CALIBRATE_COMMANDS))}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Generic dispatcher: run any tool by name."""
    tool_path = TOOLS_DIR / f"{args.tool_name}.py"
    if not tool_path.exists():
        print(f"Tool not found: {args.tool_name}", file=sys.stderr)
        print(f"Use 'list' to see available tools", file=sys.stderr)
        return 1
    # Use runpy to execute the tool as __main__
    original_argv = sys.argv
    sys.argv = [str(tool_path)] + args.passthrough
    try:
        runpy.run_path(str(tool_path), run_name="__main__")
    finally:
        sys.argv = original_argv
    return 0


# --------------------------------------------------------------------------- #
# CLI construction
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="molnextr-markush",
        description="Unified CLI for the MolNexTR Markush training toolkit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # train
    train_parser = subparsers.add_parser("train", help="Run a trainer")
    train_parser.add_argument(
        "trainer",
        choices=sorted(TRAIN_COMMANDS),
        help="Which trainer to run",
    )
    train_parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the trainer",
    )
    train_parser.set_defaults(func=_cmd_train)

    # calibrate
    cal_parser = subparsers.add_parser("calibrate", help="Run a calibrator")
    cal_parser.add_argument(
        "target",
        choices=sorted(CALIBRATE_COMMANDS),
        help="Which calibrator to run",
    )
    cal_parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the calibrator",
    )
    cal_parser.set_defaults(func=_cmd_calibrate)

    # smoke
    smoke_parser = subparsers.add_parser("smoke", help="Run the smoke test suite")
    smoke_parser.set_defaults(func=_cmd_smoke)

    # list
    list_parser = subparsers.add_parser("list", help="List all available tools")
    list_parser.set_defaults(func=_cmd_list)

    # run (generic)
    run_parser = subparsers.add_parser("run", help="Run any tool by name")
    run_parser.add_argument("tool_name", help="Tool name (without .py)")
    run_parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the tool",
    )
    run_parser.set_defaults(func=_cmd_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    # Strip leading -- from passthrough args if present
    if hasattr(args, "passthrough") and args.passthrough and args.passthrough[0] == "--":
        args.passthrough = args.passthrough[1:]

    exit_code = args.func(args)
    sys.exit(exit_code or 0)


if __name__ == "__main__":
    main()
