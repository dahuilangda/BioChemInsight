import xml.etree.ElementTree as ET
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from training.peptide_data.review import STYLES, _draw
from training.peptide_data.svg_quality import drawing_report


def draw(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(molecule)
    molecule = rdMolDraw2D.PrepareMolForDrawing(molecule, addChiralHs=False)
    svg, positions = _draw(molecule, STYLES['sans_clean'], (500, 400), vector=True)
    return molecule, svg, positions


@pytest.mark.parametrize('smiles', [
    'N[C@@H](C)C(=O)N[C@H](O)C(=O)O',
    'C/C=C/C(=O)O', 'C/C=C\\C(=O)O',
])
def test_stereo_is_recovered_from_visible_marks_and_coordinates(smiles):
    molecule, svg, positions = draw(smiles)
    report = drawing_report(svg, molecule, positions)
    assert report['accepted'], report
    assert report['visible_stereo_roundtrip']


def test_missing_stereo_marks_are_detected_without_using_source_chiral_tags():
    molecule, svg, positions = draw('N[C@@H](C)C(=O)O')
    root = ET.fromstring(svg)
    for element in list(root):
        if element.tag.endswith('path') and 'bond-' in element.get('class', ''):
            root.remove(element)
    for bond in molecule.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        a, b = positions[first], positions[second]
        ET.SubElement(root, 'path', {
            'class': f'bond-{bond.GetIdx()} atom-{first} atom-{second}',
            'd': f'M {a[0]},{a[1]} L {b[0]},{b[1]}',
            'style': 'fill:none;stroke:#000000;stroke-width:1.6px',
        })
    report = drawing_report(ET.tostring(root, encoding='unicode'), molecule, positions)
    assert not report['visible_stereo_roundtrip']
    assert not report['accepted']


def test_nonincident_glyph_collisions_are_detected_from_ink():
    molecule, svg, positions = draw('NCCO')
    root = ET.fromstring(svg)
    labels = [element for element in root if element.tag.endswith('path')
              and element.get('class', '').startswith('atom-')]
    assert len(labels) >= 2
    labels[-1].set('d', labels[0].get('d'))
    report = drawing_report(ET.tostring(root, encoding='unicode'), molecule, positions)
    assert 'glyph_ink_clearance' in report['reasons']
    assert report['collisions']


def test_missing_nonstereo_bond_is_rejected():
    molecule, svg, positions = draw('NCCO')
    root = ET.fromstring(svg)
    for element in list(root):
        if 'bond-1 ' in element.get('class', ''):
            root.remove(element)
    report = drawing_report(ET.tostring(root, encoding='unicode'), molecule, positions)
    assert not report['visible_bond_coverage']
    assert 'visible_bond_coverage' in report['reasons']


def test_subdividing_a_wedge_edge_preserves_its_visible_stereo():
    from training.peptide_data.svg_quality import OutlinePen
    from fontTools.svgLib.path import parse_path

    molecule, svg, positions = draw('N[C@@H](C)C(=O)N[C@H](O)C(=O)O')
    root = ET.fromstring(svg)
    changed = 0
    for element in root:
        if 'bond-' not in element.get('class', '') or 'fill:none' in element.get('style', ''):
            continue
        pen = OutlinePen()
        parse_path(element.get('d'), pen)
        vertices = np.asarray(pen.vertices[:3])
        midpoint = (vertices[1] + vertices[2]) / 2
        vertices = np.vstack((vertices[:2], midpoint, vertices[2:]))
        element.set('d', 'M ' + ' L '.join(f'{x},{y}' for x, y in vertices) + ' Z')
        changed += 1
    assert changed
    report = drawing_report(ET.tostring(root, encoding='unicode'), molecule, positions)
    assert report['accepted'], report


def test_hashed_wedge_resolution_is_measured_at_the_final_scale():
    molecule, svg, positions = draw('N[C@@H](C)C(=O)N[C@H](O)C(=O)O')
    report = drawing_report(svg, molecule, positions, scale=0.1)
    assert not report['accepted']
    assert set(report['reasons']) & {'hashed_wedge_resolution', 'wedge_width_below_1.2_pixels'}


@pytest.mark.parametrize('scale', [0, -1, np.nan, np.inf])
def test_invalid_scales_are_rejected(scale):
    molecule, svg, positions = draw('NCCO')
    with pytest.raises(ValueError, match='scale'):
        drawing_report(svg, molecule, positions, scale=scale)


FIXTURES = Path(__file__).parent / 'fixtures/drawings'


@pytest.mark.parametrize('record', json.loads((FIXTURES / 'manifest.json').read_text())['records'],
                         ids=lambda record: record['prefix'])
def test_native_review_drawings_expose_actual_ink_and_stereo_defects(record):
    prefix = FIXTURES / record['prefix']
    molecule = Chem.MolFromMolFile(str(prefix.with_suffix('.mol')), removeHs=False)
    report = drawing_report(prefix.with_suffix('.svg').read_text(), molecule,
                            record['atom_positions'])
    assert report['visible_stereo_roundtrip'], report
    assert report['reasons'] == record['expected_reasons']
