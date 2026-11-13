"""Strict assembly of predicted complete-molecule graphs, without graph repair.

Coordinates are (x, y) image coordinates with y pointing down. Wedges name
their apex explicitly. Double-bond relations refer to *named substituents*,
not to CIP letters, so changing atom ordering cannot silently invert E/Z.
No target labels, atom counts, missing-H search or edge deletion are accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True, slots=True)
class AtomState:
    symbol: str
    charge: int = 0
    isotope: int = 0
    explicit_h: int | None = None
    no_implicit: bool | None = None
    radical_electrons: int = 0
    aromatic: bool | None = None


@dataclass(frozen=True, slots=True)
class ChemicalBond:
    first: int
    second: int
    kind: int  # 0 none, 1 single, 2 double, 3 triple, 4 aromatic


@dataclass(frozen=True, slots=True)
class WedgeBond:
    apex: int
    other: int
    kind: str  # solid, hashed, unknown; apex is the Molfile begin endpoint


@dataclass(frozen=True, slots=True)
class DoubleBondStereo:
    first: int
    second: int
    first_substituent: int
    second_substituent: int
    relation: str  # same/opposite named substituents, geometry, or unknown


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    ok: bool
    smiles: str | None = None
    molblock: str | None = None
    error_code: str | None = None
    error: str | None = None
    num_atoms: int = 0
    num_bonds: int = 0
    unassigned_stereo: tuple[dict, ...] = ()


class _AssemblyError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str) -> None:
    raise _AssemblyError(code, message)


def _index(value: int, count: int) -> int:
    if type(value) is not int or not 0 <= value < count:
        _fail('invalid_index', f'Atom index outside predicted graph: {value!r}')
    return value


def _pair(first: int, second: int, count: int) -> tuple[int, int]:
    _index(first, count); _index(second, count)
    if first == second:
        _fail('self_bond', 'A bond cannot join an atom to itself')
    return tuple(sorted((first, second)))


def _geometric_relation(coords, first, second, left, right) -> str:
    ax, ay = coords[first]; bx, by = coords[second]
    lx, ly = coords[left]; rx, ry = coords[right]
    vx, vy = bx - ax, by - ay
    ux, uy = lx - ax, ly - ay
    wx, wy = rx - bx, ry - by
    left_cross, right_cross = vx * uy - vy * ux, vx * wy - vy * wx
    left_scale = math.hypot(vx, vy) * math.hypot(ux, uy)
    right_scale = math.hypot(vx, vy) * math.hypot(wx, wy)
    if (not left_scale or not right_scale or abs(left_cross) <= 1e-6 * left_scale
            or abs(right_cross) <= 1e-6 * right_scale):
        _fail('degenerate_double_geometry', 'Named double-bond substituents have no clear geometric side')
    return 'same' if left_cross * right_cross > 0 else 'opposite'


def assemble_raw_graph(
    atoms: Sequence[AtomState], coordinates: Sequence[Sequence[float]],
    edges: Sequence[ChemicalBond], *, wedges: Sequence[WedgeBond] = (),
    double_stereo: Sequence[DoubleBondStereo] = (),
    coordinate_system: str = 'image_xy_down',
) -> AssemblyResult:
    """Build only the supplied atoms, states and edges; report invalid graphs.

    Missing stereo cues remain unspecified. ``relation='geometry'`` explicitly
    authorizes deriving the side relation for that double bond from predicted
    coordinates; coordinates alone never specify all potential stereochemistry.
    An explicit same/opposite declaration must agree with the given geometry,
    so the exported 2-D Molfile does not contradict the returned SMILES.
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    try:
        count = len(atoms)
        if not count or len(coordinates) != count:
            _fail('invalid_coordinates', 'Every predicted atom needs one coordinate pair')
        if coordinate_system != 'image_xy_down':
            _fail('coordinate_contract', 'Expected explicit image_xy_down coordinates')
        coords = []
        for point in coordinates:
            if len(point) != 2 or any(isinstance(value, bool) for value in point):
                _fail('invalid_coordinates', 'Expected finite (x,y) pairs')
            pair = tuple(float(value) for value in point)
            if not all(math.isfinite(value) for value in pair):
                _fail('invalid_coordinates', 'Coordinates contain nonfinite values')
            coords.append(pair)

        bonds = {}
        for edge in edges:
            key = _pair(edge.first, edge.second, count)
            if key in bonds:
                _fail('duplicate_bond', f'Repeated predicted bond: {key}')
            if type(edge.kind) is not int or edge.kind not in range(5):
                _fail('invalid_bond_type', 'Chemical bond classes must be 0..4')
            bonds[key] = edge

        wedge_by_pair, apex_kinds = {}, {}
        for wedge in wedges:
            key = _pair(wedge.apex, wedge.other, count)
            if key in wedge_by_pair:
                _fail('duplicate_wedge', f'Repeated or opposite-apex wedge on {key}')
            if wedge.kind not in ('solid', 'hashed', 'unknown'):
                _fail('invalid_wedge', 'Wedge must be solid, hashed, or unknown')
            if key not in bonds or bonds[key].kind != 1:
                _fail('invalid_wedge', 'A wedge must refer to a predicted single bond')
            if wedge.apex in apex_kinds and ((wedge.kind == 'unknown') != (apex_kinds[wedge.apex] == 'unknown')):
                _fail('conflicting_wedges', 'Known and unknown wedges disagree at one apex')
            apex_kinds[wedge.apex] = wedge.kind
            wedge_by_pair[key] = wedge

        molecule = Chem.RWMol()
        declared_states = []
        for state in atoms:
            if not isinstance(state.symbol, str) or not state.symbol or state.symbol == '*':
                _fail('invalid_atom', 'Complete graphs require a concrete element symbol')
            for name, value in (('charge', state.charge), ('isotope', state.isotope),
                                ('radical_electrons', state.radical_electrons)):
                if type(value) is not int or (name != 'charge' and value < 0):
                    _fail('invalid_atom_state', f'Invalid predicted {name}')
            if state.explicit_h is not None and (type(state.explicit_h) is not int or state.explicit_h < 0):
                _fail('invalid_atom_state', 'Explicit H must be a nonnegative integer or unspecified')
            if state.no_implicit is not None and type(state.no_implicit) is not bool:
                _fail('invalid_atom_state', 'no_implicit must be an explicit boolean')
            if state.aromatic is not None and type(state.aromatic) is not bool:
                _fail('invalid_atom_state', 'aromatic must be an explicit boolean')
            symbol = state.symbol[0].upper() + state.symbol[1:]
            atom = Chem.Atom(symbol)
            if atom.GetAtomicNum() <= 0:
                _fail('invalid_atom', 'Unresolved or unsupported element')
            atom.SetFormalCharge(state.charge)
            atom.SetIsotope(state.isotope)
            atom.SetNumRadicalElectrons(state.radical_electrons)
            if state.explicit_h is not None: atom.SetNumExplicitHs(state.explicit_h)
            if state.no_implicit is not None: atom.SetNoImplicit(state.no_implicit)
            if state.aromatic is True or state.symbol.islower(): atom.SetIsAromatic(True)
            declared_states.append((atom.GetFormalCharge(), atom.GetIsotope(), atom.GetNumExplicitHs(), atom.GetNumRadicalElectrons()))
            molecule.AddAtom(atom)

        types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
        for key, edge in bonds.items():
            if edge.kind == 0: continue
            wedge = wedge_by_pair.get(key)
            first, second = (wedge.apex, wedge.other) if wedge is not None else (edge.first, edge.second)
            molecule.AddBond(first, second, types[edge.kind])
            if edge.kind == 4:
                for index in key:
                    if atoms[index].aromatic is False:
                        _fail('aromatic_state_conflict', 'Aromatic bond contradicts explicit nonaromatic atom state')
                    molecule.GetAtomWithIdx(index).SetIsAromatic(True)
        conformer = Chem.Conformer(count)
        conformer.Set3D(False)
        for index, (x, y) in enumerate(coords):
            conformer.SetAtomPosition(index, Point3D(x, -y, 0.0))
        molecule.AddConformer(conformer, assignId=True)
        candidate = molecule.GetMol()
        # Cleanup may rewrite nitro charges or organometallic connectivity.
        # Such repairs are prohibited for this raw-graph contract.
        operations = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUP
        if hasattr(Chem.SanitizeFlags, 'SANITIZE_CLEANUP_ORGANOMETALLICS'):
            operations ^= Chem.SanitizeFlags.SANITIZE_CLEANUP_ORGANOMETALLICS
        Chem.SanitizeMol(candidate, sanitizeOps=operations)
        observed = [(atom.GetFormalCharge(), atom.GetIsotope(), atom.GetNumExplicitHs(), atom.GetNumRadicalElectrons())
                    for atom in candidate.GetAtoms()]
        if observed != declared_states:
            _fail('atom_state_changed', 'Validation would change predicted charge/isotope/H/radical state')
        expected_bonds = {key for key, edge in bonds.items() if edge.kind}
        actual_bonds = {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in candidate.GetBonds()}
        if actual_bonds != expected_bonds or candidate.GetNumAtoms() != count:
            _fail('graph_changed', 'Validation would add, remove or reconnect predicted atoms/bonds')

        # Assign directions after sanitization (which may clear bond dirs).
        direction = {'solid': Chem.BondDir.BEGINWEDGE, 'hashed': Chem.BondDir.BEGINDASH, 'unknown': Chem.BondDir.UNKNOWN}
        expected_tags = {}
        for wedge in wedges:
            if wedge.kind == 'unknown': continue
            probe = Chem.Mol(candidate)
            probe.GetBondBetweenAtoms(wedge.apex, wedge.other).SetBondDir(direction[wedge.kind])
            Chem.AssignChiralTypesFromBondDirs(probe, replaceExistingTags=True)
            tag = probe.GetAtomWithIdx(wedge.apex).GetChiralTag()
            if tag == Chem.ChiralType.CHI_UNSPECIFIED:
                _fail('unresolved_wedge', 'Wedge geometry did not specify its apex stereochemistry')
            previous = expected_tags.setdefault(wedge.apex, tag)
            if previous != tag:
                _fail('conflicting_wedges', 'Wedge directions disagree at one apex')
        for wedge in wedges:
            candidate.GetBondBetweenAtoms(wedge.apex, wedge.other).SetBondDir(direction[wedge.kind])
        Chem.AssignChiralTypesFromBondDirs(candidate, replaceExistingTags=True)
        for apex, expected in expected_tags.items():
            if candidate.GetAtomWithIdx(apex).GetChiralTag() != expected:
                _fail('conflicting_wedges', 'Combined wedge declarations disagree')

        double_pairs = set()
        for stereo in double_stereo:
            key = _pair(stereo.first, stereo.second, count)
            if key in double_pairs:
                _fail('duplicate_double_stereo', 'Repeated double-bond stereo declaration')
            double_pairs.add(key)
            if key not in bonds or bonds[key].kind != 2:
                _fail('invalid_double_stereo', 'Double stereo must reference a predicted double bond')
            left, right = _index(stereo.first_substituent, count), _index(stereo.second_substituent, count)
            if len({stereo.first, stereo.second, left, right}) != 4:
                _fail('invalid_double_stereo', 'Double stereo needs distinct adjacent reference atoms')
            left_bond = candidate.GetBondBetweenAtoms(stereo.first, left)
            right_bond = candidate.GetBondBetweenAtoms(stereo.second, right)
            if left_bond is None or right_bond is None:
                _fail('invalid_double_stereo', 'Named stereo substituents must be bonded to the named endpoints')
            if stereo.relation not in ('same', 'opposite', 'geometry', 'unknown'):
                _fail('invalid_double_stereo', 'Unknown substituent relation')
            bond = candidate.GetBondBetweenAtoms(stereo.first, stereo.second)
            refs = (left, right) if bond.GetBeginAtomIdx() == stereo.first else (right, left)
            bond.SetStereoAtoms(*refs)
            if stereo.relation == 'unknown':
                bond.SetStereo(Chem.BondStereo.STEREOANY)
            else:
                relation = _geometric_relation(coords, stereo.first, stereo.second, left, right)
                if stereo.relation != 'geometry' and relation != stereo.relation:
                    _fail('double_geometry_conflict', 'Predicted double stereo contradicts the 2-D coordinates')
                bond.SetStereo(Chem.BondStereo.STEREOCIS if relation == 'same' else Chem.BondStereo.STEREOTRANS)
        # The legacy cleanup pass reconstructs double stereo from neighboring
        # bond directions. Materialize our explicit relation first, otherwise
        # cleanIt=True silently clears a valid STEREOCIS/STEREOTRANS setting.
        Chem.SetDoubleBondNeighborDirections(candidate)
        Chem.AssignStereochemistry(candidate, cleanIt=True, force=True)
        for apex in expected_tags:
            if candidate.GetAtomWithIdx(apex).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
                _fail('invalid_stereo_apex', 'Specified wedge apex is not a valid stereocenter')
        for stereo in double_stereo:
            if stereo.relation != 'unknown' and candidate.GetBondBetweenAtoms(stereo.first, stereo.second).GetStereo() in (
                    Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY):
                _fail('invalid_double_stereo', 'Declared double stereo is not chemically stereogenic')
        unassigned = tuple({'type': str(item.type), 'centered_on': item.centeredOn, 'specified': str(item.specified)}
                           for item in Chem.FindPotentialStereo(candidate)
                           if str(item.specified) != 'Specified')
        smiles = Chem.MolToSmiles(candidate, canonical=True, isomericSmiles=True)
        # Standard V2000 Kekule serialization preserves aromatic [nH]; writing
        # only type-4 aromatic bonds can lose the H state on RDKit reimport.
        # This changes representation, never predicted graph connectivity.
        molblock = Chem.MolToMolBlock(candidate, includeStereo=True, kekulize=True)
        return AssemblyResult(True, smiles, molblock, num_atoms=count,
                              num_bonds=len(expected_bonds), unassigned_stereo=unassigned)
    except _AssemblyError as exc:
        return AssemblyResult(False, error_code=exc.code, error=str(exc))
    except Exception as exc:
        return AssemblyResult(False, error_code='chemical_validation_failed',
                              error=f'{type(exc).__name__}: {str(exc)[:500]}')
