"""Recover an ambiguous scaffold from independent whole names and explicit fragments.

No atom deletion rule or document-specific scaffold is encoded: all proposed
cores come from bounded, attachment-aware graph subtraction and exact reassembly.
"""
from collections import defaultdict
from itertools import product
import hashlib
import re
from rdkit import Chem


def candidate_cores(full, fragments, budget=128):
    options=[]
    for label,fragment in fragments:
        matched = re.fullmatch(r'R?(\d+)', str(label))
        if not matched or int(matched.group(1)) < 1: return {}
        site_number = int(matched.group(1))
        dummies=[a.GetIdx() for a in fragment.GetAtoms() if a.GetAtomicNum()==0]
        if len(dummies)!=1: return {}
        dummy=dummies[0]
        if fragment.GetAtomWithIdx(dummy).GetDegree()!=1: return {}
        query_mol=Chem.Mol(fragment)
        query_mol.GetAtomWithIdx(dummy).SetIsotope(0)
        query_mol.GetAtomWithIdx(dummy).SetAtomMapNum(0)
        query=Chem.MolFromSmarts(Chem.MolToSmiles(query_mol,isomericSmiles=True))
        if query is None:return {}
        # SMILES canonicalization reorders atoms: find the wildcard in query.
        anchors=[a.GetIdx() for a in query.GetAtoms() if a.GetAtomicNum()==0]
        if len(anchors)!=1:return {}
        anchor=anchors[0]
        neighbor=query.GetAtomWithIdx(anchor).GetNeighbors()[0].GetIdx()
        matches=full.GetSubstructMatches(query,uniquify=False,useChirality=True,maxMatches=33)
        if len(matches)>32:return {}
        options.append([(site_number,match[anchor],match[neighbor],set(match)-{match[anchor]}) for match in matches])
    cores={}
    for count,assignment in enumerate(product(*options)):
        if count>=budget:return {}
        removed=set();valid=True
        for _,boundary,neighbor,atoms in assignment:
            if removed&atoms:valid=False;break
            removed.update(atoms)
        if not valid or any(boundary in removed for _,boundary,_,_ in assignment):continue
        remaining=[a.GetIdx() for a in full.GetAtoms() if a.GetIdx() not in removed]
        if not remaining:continue
        # Each fragment must connect to the retained core at its declared site only.
        cuts={(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()) for bond in full.GetBonds() if (bond.GetBeginAtomIdx() in removed)!=(bond.GetEndAtomIdx() in removed)}
        declared={frozenset((boundary,neighbor)) for _,boundary,neighbor,_ in assignment}
        if {frozenset(edge) for edge in cuts}!=declared:continue
        editable=Chem.RWMol(full)
        for index in sorted(removed,reverse=True):editable.RemoveAtom(index)
        index_map={old:new for new,old in enumerate(remaining)}
        for label,boundary,neighbor,_ in assignment:
            atom=Chem.Atom(0);atom.SetIsotope(label)
            added=editable.AddAtom(atom)
            editable.AddBond(index_map[boundary],added,full.GetBondBetweenAtoms(boundary,neighbor).GetBondType())
        try:
            core=editable.GetMol();Chem.SanitizeMol(core)
            if len(Chem.GetMolFrags(core))!=1:continue
            cores[Chem.MolToSmiles(core,isomericSmiles=True)]=core
        except Exception:continue
    return cores


def repair_scaffolds_from_names(assemblies,candidates):
    from utils.markush_assembly import _load_mol,_combine_labeled_substituents
    from utils.markush_text_substituent import normalize_variable_position
    by_ref={item.get('ref'):item for item in candidates}
    groups=defaultdict(list)
    for assembly in assemblies:
        review=assembly.get('independent_name_review') or {}
        full=Chem.MolFromSmiles(str(review.get('source_name_smiles') or ''))
        refs=assembly.get('fragment_refs') or [];labels=assembly.get('variable_positions') or []
        if full is None or not refs or len(refs)!=len(labels):continue
        fragments=[]
        for label,ref in zip(labels,refs):
            candidate=by_ref.get(ref) or {}
            fragment=_load_mol(candidate.get('molblock_full') or candidate.get('molblock') or '')
            if fragment is None:break
            fragments.append((normalize_variable_position('R'+str(label) if str(label).isdigit() else str(label)),fragment))
        else:
            cores=candidate_cores(full,fragments)
            if cores:groups[assembly.get('scaffold_ref')].append((assembly,full,fragments,cores))
    repaired=[]
    for original_ref,members in groups.items():
        # More than one independently named member must constrain the same core.
        if len({Chem.MolToSmiles(full) for _,full,_,_ in members})<2:continue
        shared=set.intersection(*(set(cores) for _,_,_,cores in members))
        if len(shared)!=1:continue
        core_smiles=next(iter(shared));core=Chem.Mol(members[0][3][core_smiles])
        from rdkit.Chem import AllChem
        original=by_ref.get(original_ref) or {}
        from utils.pose_refinement import transfer_source_pose
        reference = _load_mol(original.get('molblock_full') or original.get('molblock') or '')
        aligned = transfer_source_pose(Chem.MolToSmiles(core), reference)
        if aligned is not None: core = aligned
        else: AllChem.Compute2DCoords(core, canonOrient=False)
        if Chem.MolToSmiles(core)==str(original.get('smiles') or ''):continue
        proposals=[]
        for assembly,full,fragments,_ in members:
            smiles,block,reason,notes=_combine_labeled_substituents(core,fragments)
            molecule=Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is None or Chem.MolToSmiles(molecule)!=Chem.MolToSmiles(full):break
            proposals.append((assembly,smiles,block,notes))
        else:
            ref=str(original_ref)+':name_consensus:'+hashlib.sha256(core_smiles.encode()).hexdigest()[:10]
            candidates.append({**original,'ref':ref,'smiles':core_smiles,'molblock':Chem.MolToMolBlock(core),'molblock_full':Chem.MolToMolBlock(core),
                'candidate_source':'independent_name_fragment_consensus','parent_ref':original_ref,
                'consensus_member_count':len(members),'original_decoded_smiles':original.get('smiles') or ''})
            for assembly,smiles,block,notes in proposals:
                assembly.update(scaffold_ref=ref,assembled_smiles=smiles,assembled_molblock=block,assembly_status='assembled',blocked_reasons=[],
                    scaffold_recovery={'method':'independent_name_fragment_consensus','parent_ref':original_ref,'supporting_members':len(members),'requires_visual_review':True})
                assembly['independent_name_review']['status']='exact_match'
                assembly['normalization_notes']=[*(assembly.get('normalization_notes') or []),*notes]
                repaired.append(assembly.get('compound_id'))
    return repaired
