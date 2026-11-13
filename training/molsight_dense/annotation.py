"""Sparse annotation admission through the same strict assembly authority.

This accepts supervised labels and is never called by an image recognizer.
Avoid allocating N² categorical logits just to validate known sparse bonds.
"""
import torch
from dataclasses import asdict
from rdkit import Chem
from training.ocsr_reboot.graph import AtomState,ChemicalBond,WedgeBond,DoubleBondStereo,assemble_raw_graph


def annotation_roundtrip(target,points):
    n=len(target['elements']);points=torch.as_tensor(points).float().tolist();table=Chem.GetPeriodicTable()
    atoms=[AtomState(table.GetElementSymbol(int(target['elements'][i])),
        charge=int(target['charge'][i])-8,isotope=int(target['isotope'][i]),explicit_h=int(target['explicit_h'][i]),
        no_implicit=bool(target['no_implicit'][i]),radical_electrons=int(target['radicals'][i]),aromatic=bool(target['aromatic'][i])) for i in range(n)]
    adjacency=[[] for _ in range(n)];bonds=[];wedges=[];doubles=[]
    indices=torch.triu(target['edges']>0,diagonal=1).nonzero().tolist()
    for i,j in indices:
        bonds.append(ChemicalBond(i,j,int(target['edges'][i,j])));adjacency[i].append(j);adjacency[j].append(i)
    for i,j in (target['wedges']>0).nonzero().tolist():
        wedges.append(WedgeBond(i,j,'solid' if int(target['wedges'][i,j])==1 else 'hashed'))
    for i,j in indices:
        if int(target['double_stereo'][i,j])>0:
            left=[k for k in adjacency[i] if k!=j];right=[k for k in adjacency[j] if k!=i]
            if not left or not right:raise ValueError('Declared double stereo lacks substituents')
            doubles.append(DoubleBondStereo(i,j,min(left),min(right),'geometry'))
    result=assemble_raw_graph(atoms,[[100*x,100*y] for x,y in points],bonds,wedges=wedges,double_stereo=doubles)
    return {**asdict(result),'parse_success':result.ok}
