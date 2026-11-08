"""Public API with explicit chemical and geometric admission checks."""
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from .solver import residue_pose_layout


@dataclass
class LayoutResult:
    molecule: Chem.Mol | None
    report: dict

    @property
    def accepted(self) -> bool:
        return bool(self.molecule is not None and self.report.get("qa_pass"))


def validate_input(smiles: str) -> Chem.Mol:
    """Parse one connected, resolved, closed-shell molecular graph."""
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


def depict(smiles: str, *, seed: int = 0, maxiter: int = 350) -> LayoutResult:
    """Return a candidate and its QA report; rejected candidates are explicit.

    Coordinates use a 30-unit bond length. The input graph and specified
    stereochemistry are preserved. Unspecified stereochemistry stays unspecified.
    """
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    molecule = validate_input(smiles)
    laid, report = residue_pose_layout(molecule, seed=seed, maxiter=maxiter)
    report.setdefault("qa_pass", False)
    if laid is not None and Chem.MolToSmiles(laid) != Chem.MolToSmiles(molecule):
        raise RuntimeError("layout_changed_molecular_identity")
    report["input_smiles"] = Chem.MolToSmiles(molecule)
    return LayoutResult(laid, report)


def render_svg(result: LayoutResult, *, width: int = 1000, height: int = 1000) -> str:
    """Render an accepted layout; do not regenerate coordinates."""
    if not result.accepted:
        raise ValueError("layout_failed_geometry_qa")
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
    return drawer.GetDrawingText()
