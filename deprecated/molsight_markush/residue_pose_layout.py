"""Compatibility imports for the independently installable Cyclopeptide2D engine.

The canonical implementation lives in packages/cyclopeptide2d. This adapter
keeps existing research entry points working from a source checkout.
"""
from pathlib import Path
import sys

try:
    from cyclopeptide2d.solver import ResidueFrame, PoseProblem, peptide_frames, residue_pose_layout
except ModuleNotFoundError as error:
    if error.name != "cyclopeptide2d":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"packages"/"cyclopeptide2d"/"src"))
    from cyclopeptide2d.solver import ResidueFrame, PoseProblem, peptide_frames, residue_pose_layout

__all__ = ["ResidueFrame", "PoseProblem", "peptide_frames", "residue_pose_layout"]
