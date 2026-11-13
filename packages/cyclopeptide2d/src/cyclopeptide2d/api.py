"""Public depiction and rendering functions."""

from dataclasses import dataclass
import math
import time
from typing import TYPE_CHECKING, Any, Literal

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from .solver import residue_pose_layout

if TYPE_CHECKING:
    from peptidecore import ChemicalRecord


@dataclass
class LayoutResult:
    molecule: Chem.Mol | None
    report: dict[str, Any]

    @property
    def accepted(self) -> bool:
        return bool(self.molecule is not None and self.report.get("qa_pass"))


def validate_input(smiles: str) -> Chem.Mol:
    """Parse a connected molecule without radicals or wildcard atoms."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("invalid_input_smiles")
    if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        raise ValueError("input_contains_radical_electrons")
    if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
        raise ValueError("input_contains_unresolved_atoms")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("input_contains_disconnected_components")
    return molecule


def depict(
    smiles: str, *, seed: int = 0, maxiter: int = 350, time_budget: float | None = None
) -> LayoutResult:
    """Generate a two-dimensional layout and its validation report."""
    return _depict_molecule(
        validate_input(smiles), seed=seed, maxiter=maxiter, time_budget=time_budget
    )


def _depict_molecule(
    molecule: Chem.Mol, *, seed: int, maxiter: int, chemical_record=None, time_budget=None
) -> LayoutResult:
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    if time_budget is not None and (not math.isfinite(time_budget) or time_budget <= 0):
        raise ValueError("time_budget must be finite and positive")
    deadline = None if time_budget is None else time.monotonic() + time_budget
    laid, report = residue_pose_layout(
        molecule, seed=seed, maxiter=maxiter, chemical_record=chemical_record, deadline=deadline
    )
    report.setdefault("qa_pass", False)
    if laid is not None and Chem.MolToSmiles(laid) != Chem.MolToSmiles(molecule):
        raise RuntimeError("layout_changed_molecular_identity")
    report["input_smiles"] = Chem.MolToSmiles(molecule)
    return LayoutResult(laid, report)


def depict_record(
    record: "ChemicalRecord", *, seed: int = 0, maxiter: int = 350, time_budget: float | None = None
) -> LayoutResult:
    """Draw a peptide from registered monomer and connection boundaries."""
    validate_input(record.smiles)
    molecule = record.molecule
    if Chem.MolToSmiles(molecule) != record.smiles:
        raise ValueError("record_structure_mismatch")
    result = _depict_molecule(
        molecule, seed=seed, maxiter=maxiter, chemical_record=record, time_budget=time_budget
    )
    result.report["chemical_record"] = record.as_dict()
    return result


def depict_sequence(
    text: str,
    *,
    input_format: Literal["HELM", "BILN", "FASTA", "sequence"] = "HELM",
    seed: int = 0,
    maxiter: int = 350,
    time_budget: float | None = None,
) -> LayoutResult:
    """Assemble a sequence and draw its registered connection graph."""
    from peptidecore import from_biln, from_fasta, from_helm, from_sequence

    readers = {"HELM": from_helm, "BILN": from_biln, "FASTA": from_fasta, "sequence": from_sequence}
    if input_format not in readers:
        raise ValueError("unsupported_sequence_format")
    return depict_record(
        readers[input_format](text), seed=seed, maxiter=maxiter, time_budget=time_budget
    )


def render_svg(result: LayoutResult, *, width: int = 1000, height: int = 1000) -> str:
    """Render an accepted layout using its existing coordinates."""
    return render_drawing(result, width=width, height=height)[0]


def render_drawing(
    result: LayoutResult, *, width: int = 1000, height: int = 1000
) -> tuple[str, tuple[tuple[float, float], ...]]:
    """Return SVG and atom coordinates in the SVG coordinate system."""
    if not result.accepted:
        raise ValueError("layout_failed_geometry_qa")
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.fixedBondLength = 30
    options.bondLineWidth = 1.6
    options.prepareMolsBeforeDrawing = False
    work = Chem.Mol(result.molecule)
    Chem.WedgeMolBonds(work, work.GetConformer())
    drawer.DrawMolecule(work)
    drawer.FinishDrawing()
    positions = tuple(
        (float(drawer.GetDrawCoords(index).x), float(drawer.GetDrawCoords(index).y))
        for index in range(work.GetNumAtoms())
    )
    return drawer.GetDrawingText(), positions
