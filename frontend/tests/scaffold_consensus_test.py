import unittest
from rdkit import Chem
from utils.scaffold_consensus import repair_scaffolds_from_names

class ScaffoldConsensusTest(unittest.TestCase):
    def records(self):
        candidates=[{'ref':'core','smiles':'[1*]COc1ccccc1','structure_type':'markush','image_file':'source.png'}]
        assemblies=[]
        for identifier,fragment,full in [('Q8','*CC','CCOc1ccccc1'),('Q9','*CCC','CCCOc1ccccc1')]:
            candidates.append({'ref':identifier,'molblock_full':Chem.MolToMolBlock(Chem.MolFromSmiles(fragment))})
            assemblies.append({'compound_id':identifier,'scaffold_ref':'core','fragment_refs':[identifier],'variable_positions':['1'],
                'assembly_status':'blocked','blocked_reasons':['source_name_graph_conflict'],
                'independent_name_review':{'status':'conflict','source_name_smiles':full}})
        return assemblies,candidates
    def test_core_is_reconstructed_from_multiple_independent_members(self):
        assemblies,candidates=self.records()
        repaired=repair_scaffolds_from_names(assemblies,candidates)
        self.assertEqual(set(repaired),{'Q8','Q9'})
        self.assertEqual(candidates[0]['smiles'],'[1*]COc1ccccc1')
        self.assertEqual(Chem.MolToSmiles(Chem.MolFromSmiles(candidates[-1]['smiles'])),Chem.MolToSmiles(Chem.MolFromSmiles('[1*]Oc1ccccc1')))
        for assembly in assemblies:
            rendered=Chem.MolFromMolBlock(assembly['assembled_molblock']);positions=rendered.GetConformer().GetPositions()
            self.assertGreater(len({tuple(point.round(3)) for point in positions}),rendered.GetNumAtoms()*.8)
            self.assertEqual(assembly['assembly_status'],'assembled')
            self.assertTrue(assembly['scaffold_recovery']['requires_visual_review'])
    def test_one_member_cannot_rewrite_a_scaffold(self):
        assemblies,candidates=self.records()
        self.assertEqual(repair_scaffolds_from_names(assemblies[:1],candidates),[])
    def test_inconsistent_independent_members_do_not_repair(self):
        assemblies,candidates=self.records()
        assemblies[1]['independent_name_review']['source_name_smiles']='CCCNc1ccccc1'
        self.assertEqual(repair_scaffolds_from_names(assemblies,candidates),[])
    def test_visual_review_context_is_bounded_and_blind_to_prior_verdicts(self):
        from utils.llm_utils import build_review_assembled_structure_prompt
        prompt=build_review_assembled_structure_prompt({'compound_id':'Q8','assembled_smiles':'SECRET_GRAPH','assembled_molblock':'X'*50000,'independent_name_review':{'status':'exact_match'}}, {'ref':'core','molblock':'Y'*50000}, [])
        self.assertLess(len(prompt),8000)
        self.assertNotIn('SECRET_GRAPH',prompt)
        self.assertNotIn('exact_match',prompt)
        self.assertIn('Q8',prompt)

    def test_missing_name_evidence_does_not_promote_an_assembly(self):
        assemblies,candidates=self.records()
        for item in assemblies:item['independent_name_review']={}
        self.assertEqual(repair_scaffolds_from_names(assemblies,candidates),[])

if __name__=='__main__':unittest.main()
