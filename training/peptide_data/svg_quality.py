"""Drawing checks based on emitted glyph outlines and visible stereo marks."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.svgLib.path import parse_path
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.path import Path as VectorPath
from matplotlib.transforms import Affine2D
from rdkit import Chem
from rdkit.Geometry import Point3D
from scipy.ndimage import distance_transform_edt

from .quality import chemical_identity


class OutlinePen(BasePen):
    def __init__(self):
        super().__init__(None)
        self.vertices, self.codes = [], []
        self.start = None

    def _moveTo(self, point):
        self.start = point
        self.vertices.append(point)
        self.codes.append(VectorPath.MOVETO)

    def _lineTo(self, point):
        self.vertices.append(point)
        self.codes.append(VectorPath.LINETO)

    def _curveToOne(self, first, second, end):
        self.vertices.extend((first, second, end))
        self.codes.extend((VectorPath.CURVE4,) * 3)

    def _qCurveToOne(self, control, end):
        self.vertices.extend((control, end))
        self.codes.extend((VectorPath.CURVE3,) * 2)

    def _closePath(self):
        self.vertices.append(self.start)
        self.codes.append(VectorPath.CLOSEPOLY)

    def _endPath(self):
        pass


@dataclass
class Primitive:
    atoms: tuple[int, ...]
    bond: int | None
    path: VectorPath
    bounds: np.ndarray
    fill: bool
    stroke: float


def primitives(svg):
    root = ET.fromstring(svg)
    output = []
    for element in root.iter():
        if element.tag.rsplit('}', 1)[-1] != 'path':
            continue
        classes = element.get('class', '').split()
        atoms = tuple(int(value[5:]) for value in classes if value.startswith('atom-'))
        bonds = [int(value[5:]) for value in classes if value.startswith('bond-')]
        if not atoms:
            continue
        style = dict(
            field.split(':', 1) for field in element.get('style', '').split(';') if ':' in field
        )
        fill = style.get('fill', element.get('fill', 'black')) != 'none'
        stroke = (
            float(style.get('stroke-width', '1').removesuffix('px'))
            if style.get('stroke', 'none') != 'none' else 0.0
        )
        if not fill and not stroke:
            continue
        bounds = BoundsPen(None)
        outline = OutlinePen()
        parse_path(element.attrib['d'], bounds)
        parse_path(element.attrib['d'], outline)
        if bounds.bounds is None:
            continue
        output.append(Primitive(
            atoms, bonds[0] if bonds else None,
            VectorPath(np.asarray(outline.vertices), outline.codes),
            np.asarray(bounds.bounds), fill, stroke,
        ))
    return root, output


def _ink(primitive, bounds, scale):
    width, height = np.maximum(1, np.ceil((bounds[2:] - bounds[:2]) * scale).astype(int))
    renderer = RendererAgg(int(width), int(height), 72)
    context = renderer.new_gc()
    context.set_foreground((0, 0, 0, 1))
    context.set_linewidth(primitive.stroke * scale)
    context.set_antialiased(True)
    transform = Affine2D().translate(-bounds[0], -bounds[1]).scale(scale)
    renderer.draw_path(
        context, primitive.path, transform,
        (0, 0, 0, 1) if primitive.fill else None,
    )
    return np.asarray(renderer.buffer_rgba())[:, :, 3] >= 128


def _clearance(first, second, scale, margin):
    lower = np.maximum(first.bounds[:2] - first.stroke / 2, second.bounds[:2] - second.stroke / 2)
    upper = np.minimum(first.bounds[2:] + first.stroke / 2, second.bounds[2:] + second.stroke / 2)
    if np.any((lower - upper) * scale > margin):
        return None
    padding = (margin + 1) / scale + max(first.stroke, second.stroke)
    bounds = np.r_[
        np.minimum(first.bounds[:2], second.bounds[:2]) - padding,
        np.maximum(first.bounds[2:], second.bounds[2:]) + padding,
    ]
    sampling = scale * 4
    a, b = _ink(first, bounds, sampling), _ink(second, bounds, sampling)
    if not a.any() or not b.any():
        return 0.0
    return float(distance_transform_edt(~a)[b].min() / 4)


def _stereo_marks(items, positions):
    groups = defaultdict(list)
    for item in items:
        if item.bond is not None:
            groups[item.bond].append(item)
    marks, errors = {}, []
    for index, paths in groups.items():
        filled = [path for path in paths if path.fill]
        lines = [path for path in paths if not path.fill]
        if not filled and len(lines) < 3:
            continue
        atoms = paths[0].atoms
        if len(atoms) != 2:
            errors.append({'bond': index, 'reason': 'invalid_bond_atom_reference'})
            continue
        first, second = positions[list(atoms)]
        axis = second - first
        length = np.linalg.norm(axis)
        if length < 1e-8:
            errors.append({'bond': index, 'reason': 'collapsed_stereo_bond'})
            continue
        axis /= length
        normal = np.array([-axis[1], axis[0]])
        if filled:
            if len(filled) != 1:
                errors.append({'bond': index, 'reason': 'unsupported_filled_bond'})
                continue
            path = filled[0].path
            vertices = path.vertices[np.asarray(path.codes) != VectorPath.CLOSEPOLY]
            if len(vertices) > 1 and np.allclose(vertices[0], vertices[-1]):
                vertices = vertices[:-1]
            if len(vertices) < 3 or any(code not in (
                VectorPath.MOVETO, VectorPath.LINETO, VectorPath.CLOSEPOLY
            ) for code in path.codes):
                errors.append({'bond': index, 'reason': 'unsupported_wedge_outline'})
                continue
            along = (vertices - first) @ axis
            split = (along.min() + along.max()) / 2
            widths = [float(np.ptp(vertices[mask] @ normal))
                      for mask in (along <= split, along > split)]
            width = max(widths)
            if width < 0.2 or min(widths) > 0.5 * width + 0.15:
                errors.append({'bond': index, 'reason': 'ambiguous_wedge_taper'})
                continue
            narrow = atoms[int(widths[0] > widths[1])]
            marks[index] = {'narrow': narrow, 'direction': Chem.BondDir.BEGINWEDGE,
                            'width': width, 'stroke': filled[0].stroke}
        else:
            vertices = [path.path.vertices for path in lines]
            if any(len(points) != 2 for points in vertices):
                continue
            segments = np.asarray(vertices)
            vectors = segments[:, 1] - segments[:, 0]
            widths = np.linalg.norm(vectors, axis=1)
            if np.any(np.abs(vectors @ axis) > 0.15 * np.maximum(widths, 1e-8) + 0.15):
                continue
            along = (segments.mean(axis=1) - first) @ axis
            if np.ptp(along) < max(1.0, 1.5 * float(widths.max())):
                continue
            slope, intercept = np.polyfit(along, widths, 1)
            residual = np.max(np.abs(widths - (slope * along + intercept)))
            if abs(slope) < 0.02 or residual > 0.3:
                errors.append({'bond': index, 'reason': 'ambiguous_hashed_wedge'})
                continue
            marks[index] = {
                'narrow': atoms[0] if slope > 0 else atoms[1],
                'direction': Chem.BondDir.BEGINDASH, 'width': float(widths.max()),
                'stroke': min(path.stroke for path in lines),
                'dash_spacing': float(np.min(np.diff(np.sort(along)))),
            }
    return marks, errors


def _recover_stereo(molecule, positions, marks):
    builder = Chem.RWMol()
    for atom in molecule.GetAtoms():
        copied = Chem.Atom(atom)
        copied.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for prop in ('_CIPCode', '_CIPRank'):
            if copied.HasProp(prop):
                copied.ClearProp(prop)
        builder.AddAtom(copied)
    for bond in molecule.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        mark = marks.get(bond.GetIdx())
        if mark and mark['narrow'] == second:
            first, second = second, first
        builder.AddBond(first, second, bond.GetBondType())
        copied = builder.GetBondWithIdx(bond.GetIdx())
        copied.SetIsAromatic(bond.GetIsAromatic())
        if mark:
            copied.SetBondDir(mark['direction'])
    recovered = builder.GetMol()
    conformer = Chem.Conformer(recovered.GetNumAtoms())
    conformer.Set3D(False)
    for index, (x, y) in enumerate(positions):
        conformer.SetAtomPosition(index, Point3D(float(x), float(-y), 0))
    recovered.AddConformer(conformer, assignId=True)
    Chem.SanitizeMol(recovered)
    Chem.AssignChiralTypesFromBondDirs(recovered, replaceExistingTags=True)
    Chem.AssignStereochemistry(recovered, cleanIt=True, force=True)
    for source in molecule.GetBonds():
        if source.GetStereo() not in (
            Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
            Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS,
        ):
            continue
        bond = recovered.GetBondWithIdx(source.GetIdx())
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        left = min(atom.GetIdx() for atom in recovered.GetAtomWithIdx(first).GetNeighbors()
                   if atom.GetIdx() != second)
        right = min(atom.GetIdx() for atom in recovered.GetAtomWithIdx(second).GetNeighbors()
                    if atom.GetIdx() != first)
        axis = positions[second] - positions[first]
        a, b = positions[left] - positions[first], positions[right] - positions[second]
        cross_a, cross_b = axis[0] * a[1] - axis[1] * a[0], axis[0] * b[1] - axis[1] * b[0]
        if abs(cross_a * cross_b) < 1e-8:
            return None
        bond.SetStereoAtoms(left, right)
        bond.SetStereo(Chem.BondStereo.STEREOCIS if cross_a * cross_b > 0
                       else Chem.BondStereo.STEREOTRANS)
    Chem.AssignStereochemistry(recovered, cleanIt=False, force=True)
    return recovered


def drawing_report(svg, molecule, atom_positions, *, scale=1.0, min_clearance=0.5):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Drawing scale must be finite and positive')
    root, items = primitives(svg)
    positions = np.asarray(atom_positions, dtype=float)
    if positions.shape != (molecule.GetNumAtoms(), 2) or not np.isfinite(positions).all():
        raise ValueError('Invalid drawing atom coordinates')
    glyphs = [item for item in items if item.bond is None]
    bonds = [item for item in items if item.bond is not None]
    collisions = []
    for index, glyph in enumerate(glyphs):
        for other in glyphs[index + 1:] + bonds:
            if set(glyph.atoms) & set(other.atoms):
                continue
            distance = _clearance(glyph, other, scale, min_clearance)
            if distance is not None and distance < min_clearance:
                collisions.append({'atom': glyph.atoms[0], 'other_atoms': list(other.atoms),
                                   'bond': other.bond, 'clearance_pixels': distance})
    marks, errors = _stereo_marks(items, positions)
    expected_bonds = {
        bond.GetIdx(): {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
        for bond in molecule.GetBonds()
    }
    visible_bonds = {item.bond for item in bonds}
    bond_coverage = visible_bonds == set(expected_bonds) and all(
        set(item.atoms) == expected_bonds.get(item.bond) for item in bonds
    )
    recovered = _recover_stereo(molecule, positions, marks) if not errors else None
    stereo_ok = recovered is not None and chemical_identity(recovered) == chemical_identity(molecule)
    expected_centers = dict(Chem.FindMolChiralCenters(molecule, includeUnassigned=True))
    actual_centers = dict(Chem.FindMolChiralCenters(recovered, includeUnassigned=True)) if recovered else {}
    center_errors = [
        {'atom': index, 'expected': label, 'observed': actual_centers.get(index)}
        for index, label in expected_centers.items() if label != actual_centers.get(index)
    ]
    reasons = []
    if collisions:
        reasons.append('glyph_ink_clearance')
    if not stereo_ok:
        reasons.append('visible_stereo_mismatch')
    if not bond_coverage:
        reasons.append('visible_bond_coverage')
    required_labels = {
        atom.GetIdx() for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() != 6 or atom.GetIsotope() or atom.GetFormalCharge()
    }
    label_coverage = required_labels <= {atom for glyph in glyphs for atom in glyph.atoms}
    if not label_coverage:
        reasons.append('visible_atom_label_coverage')
    if any(mark['width'] * scale < 1.2 for mark in marks.values()):
        reasons.append('wedge_width_below_1.2_pixels')
    if any(mark['direction'] == Chem.BondDir.BEGINDASH and (
        mark['stroke'] * scale < 0.6 or mark['dash_spacing'] * scale < 0.8
    ) for mark in marks.values()):
        reasons.append('hashed_wedge_resolution')
    canvas = np.array(list(map(float, re.split(r'[ ,]+', root.attrib['viewBox']))))
    if any(np.any(item.bounds[:2] < canvas[:2]) or np.any(item.bounds[2:] > canvas[:2] + canvas[2:])
           for item in items):
        reasons.append('clipped_drawing')
    return {'accepted': not reasons, 'reasons': reasons, 'glyph_paths': len(glyphs),
            'visible_stereo_roundtrip': stereo_ok, 'stereo_marks': len(marks),
            'visible_bond_coverage': bond_coverage,
            'visible_atom_label_coverage': label_coverage,
            'stereo_errors': errors, 'collisions': collisions,
            'stereo_center_errors': center_errors,
            'minimum_clearance_pixels': min_clearance, 'scale': scale}
