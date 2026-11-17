import math
import unittest
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
from utils.pose_refinement import refine_pose, transfer_source_pose


def molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol, canonOrient=False)
    return mol


class PoseRefinementTest(unittest.TestCase):
    def test_noisy_ring_improves_without_rotation_reflection_or_graph_change(self):
        mol = molecule('Oc1ccc2ncccc2c1')
        conf = mol.GetConformer(); rng = np.random.default_rng(18)
        for i, p in enumerate(np.asarray(conf.GetPositions()).copy()):
            theta = .73; noise = rng.normal(0, .04, 2)
            conf.SetAtomPosition(i, (p[0]*math.cos(theta)-p[1]*math.sin(theta)+noise[0],p[0]*math.sin(theta)+p[1]*math.cos(theta)+noise[1],0))
        before = np.asarray(conf.GetPositions())[:,:2].copy()
        graph = Chem.MolToSmiles(mol, isomericSmiles=True)
        result = refine_pose(mol)
        after = np.asarray(conf.GetPositions())[:,:2]
        self.assertTrue(result['accepted'], result)
        self.assertLess(result['objective_after'], result['objective_before'])
        self.assertLess(result['max_shift_bonds'], .32*math.sqrt(2)+1e-6)
        u, _, vt = np.linalg.svd((after-after.mean(0)).T @ (before-before.mean(0)))
        rotation = u @ vt
        self.assertGreater(np.linalg.det(rotation), .999)
        self.assertLess(abs(math.atan2(rotation[0,1], rotation[0,0])), 1e-5)
        self.assertEqual(Chem.MolToSmiles(mol,isomericSmiles=True),graph)
        self.assertLessEqual(result['crossings_after'],result['crossings_before'])

    def test_stereo_and_macrocycle_roundtrip(self):
        for smiles in ['C[C@@H](N)C(=O)O','F/C=C/Cl','C1CCCCCCCCCCC1','O=C1N[C@@H](C)C(=O)NCC(=O)NCC1']:
            with self.subTest(smiles=smiles):
                mol=molecule(smiles); original=Chem.MolToSmiles(mol,isomericSmiles=True)
                result=refine_pose(mol)
                self.assertTrue(result['accepted'],result)
                loaded=Chem.MolFromMolBlock(Chem.MolToMolBlock(mol),removeHs=False)
                self.assertEqual(Chem.MolToSmiles(loaded,isomericSmiles=True),original)

    def test_coordinate_transfer_ignores_atom_order_and_keeps_direction(self):
        source=molecule('Clc1ccccc1CCN')
        conf=source.GetConformer()
        for i,p in enumerate(np.asarray(conf.GetPositions()).copy()):conf.SetAtomPosition(i,(-p[0],-p[1],0))
        target=transfer_source_pose('NCCc1ccccc1Cl',source,minimum_coverage=1)
        self.assertIsNotNone(target)
        mapping=source.GetSubstructMatch(target)
        a=np.asarray(source.GetConformer().GetPositions())[list(mapping),:2]
        b=np.asarray(target.GetConformer().GetPositions())[:,:2]
        a-=a.mean(0);b-=b.mean(0)
        a/=np.linalg.norm(a);b/=np.linalg.norm(b)
        np.testing.assert_allclose(a,b,atol=1e-8)

    def test_geometry_mapping_preserves_isotope_identity_when_bond_orders_differ(self):
        source=molecule('Oc1cc[13cH]cc1')
        for bond in source.GetBonds():
            if bond.IsInRing():bond.SetIsAromatic(False);bond.SetBondType(Chem.BondType.SINGLE)
        for atom in source.GetAtoms():atom.SetIsAromatic(False)
        source.UpdatePropertyCache(strict=False)
        target=transfer_source_pose('Oc1cc[13cH]cc1',source,minimum_coverage=1)
        self.assertIsNotNone(target)
        before=next(a.GetIdx() for a in source.GetAtoms() if a.GetIsotope()==13)
        after=next(a.GetIdx() for a in target.GetAtoms() if a.GetIsotope()==13)
        src=np.asarray(source.GetConformer().GetPositions())[:,:2];dst=np.asarray(target.GetConformer().GetPositions())[:,:2]
        src-=src.mean(0);dst-=dst.mean(0)
        src/=np.linalg.norm(src);dst/=np.linalg.norm(dst)
        np.testing.assert_allclose(src[before],dst[after],atol=1e-8)
        self.assertTrue(any(a.GetIsAromatic() for a in target.GetAtoms()))

    def test_corrected_macrocycle_closure_reuses_image_coordinates(self):
        original=molecule('O=C1N[C@@H](C)CCCNC(=O)CCNCCC1')
        source=Chem.RWMol(original)
        ring=list(Chem.GetSymmSSSR(original))[0]
        source.RemoveBond(ring[0],ring[1])
        source=source.GetMol();Chem.SanitizeMol(source)
        target=transfer_source_pose(Chem.MolToSmiles(original,isomericSmiles=True),source,minimum_coverage=.9)
        self.assertIsNotNone(target)
        mapping=original.GetSubstructMatch(target)
        src=np.asarray(original.GetConformer().GetPositions())[list(mapping),:2];dst=np.asarray(target.GetConformer().GetPositions())[:,:2]
        src-=src.mean(0);dst-=dst.mean(0);src/=np.linalg.norm(src);dst/=np.linalg.norm(dst)
        np.testing.assert_allclose(src,dst,atol=1e-7)

    def test_insufficient_core_is_not_reported_as_source_pose(self):
        self.assertIsNone(transfer_source_pose('CCCCCCCCCCCCCCCCCCN',molecule('c1ccccc1O')))

    def test_macrocycle_backbone_is_fixed_while_side_chain_is_cleaned(self):
        mol=molecule('CC1CCCCCCCCCCC1')
        conf=mol.GetConformer();before=np.asarray(conf.GetPositions()).copy()
        point=conf.GetAtomPosition(0);conf.SetAtomPosition(0,(point.x+.2,point.y+.1,0))
        result=refine_pose(mol)
        self.assertTrue(result['accepted'],result)
        self.assertEqual(result['fixed_macrocycle_atoms'],12)
        np.testing.assert_allclose(np.asarray(conf.GetPositions())[1:],before[1:],atol=1e-10)

    def test_right_angle_linker_is_locally_improved(self):
        mol=molecule('CCC');conf=mol.GetConformer()
        for i,p in enumerate([(0,0,0),(1.5,0,0),(1.5,1.5,0)]):conf.SetAtomPosition(i,p)
        result=refine_pose(mol);points=np.asarray(conf.GetPositions())[:,:2]
        a,b=points[0]-points[1],points[2]-points[1]
        angle=math.degrees(math.acos(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)))
        self.assertTrue(result['accepted'],result)
        self.assertGreater(angle,110)
        self.assertLess(result['max_shift_bonds'],.32*math.sqrt(2)+1e-6)

    def test_missing_and_degenerate_coordinates_do_not_mutate(self):
        self.assertEqual(refine_pose(Chem.MolFromSmiles('CCO'))['status'],'missing_coordinates')
        mol=molecule('CCO');conf=mol.GetConformer()
        for i in range(3):conf.SetAtomPosition(i,(0,0,0))
        self.assertEqual(refine_pose(mol)['status'],'degenerate_coordinates')
        self.assertFalse(refine_pose(mol)['accepted'])

if __name__ == '__main__':unittest.main()
