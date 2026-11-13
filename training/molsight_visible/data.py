"""Visible-unit labels preserve hidden-group membership separately from positions."""
import hashlib,json
from pathlib import Path
import torch
import numpy as np
from rdkit import Chem
from training.molsight_spatial.data import GraphDataset,drawing_quality
from training.molsight_dense.objectives import chemical_targets
from training.ocsr_reboot.protocol import assert_disjoint_splits
from .registry import LABEL_TO_ID,REGISTRY_SHA256
from .model import atom_labels
from training.molsight_dense.drawing import drawing_molecule


def visible_targets(original,visible):
    mapping=list(visible.GetPropsAsDict(includePrivate=True)['_origAtomMapping'])
    n=visible.GetNumAtoms();labels={name:[] for name in ['elements','charge','isotope','hydrogens','radicals','aromatic']};aliases=[]
    for atom,index in zip(visible.GetAtoms(),mapping):
        label=atom.GetProp('atomLabel') if atom.HasProp('atomLabel') else None
        identifier=LABEL_TO_ID.get(label,0)
        if atom.GetAtomicNum()==0 and not identifier:raise ValueError('Unresolved variable group is not a fixed abbreviation')
        aliases.append(identifier)
        values=atom_labels(original.GetAtomWithIdx(index))
        for name in labels:labels[name].append(-100 if identifier else values[name])
    edges=torch.zeros(n,n,dtype=torch.long);joint=edges.clone();defined=edges.clone().fill_(-100)
    for matrix in [edges,joint]:matrix.fill_diagonal_(-100)
    for bond in visible.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        kind={1.:1,2.:2,3.:3,1.5:4}.get(bond.GetBondTypeAsDouble())
        if kind is None:raise ValueError('Unsupported visible bond')
        edges[i,j]=edges[j,i]=kind;joint[i,j]=joint[j,i]=kind
        if bond.GetBondDir() in (Chem.BondDir.BEGINWEDGE,Chem.BondDir.BEGINDASH):
            if kind!=1:raise ValueError('Wedge on non-single annotation')
            direction=1 if bond.GetBondDir()==Chem.BondDir.BEGINWEDGE else 2
            joint[i,j]=4+direction;joint[j,i]=6+direction
        if kind==2:
            value=int(bond.GetStereo() in (Chem.BondStereo.STEREOE,Chem.BondStereo.STEREOZ,Chem.BondStereo.STEREOCIS,Chem.BondStereo.STEREOTRANS))
            defined[i,j]=defined[j,i]=value
    return {**{k:torch.tensor(v,dtype=torch.long) for k,v in labels.items()},'aliases':torch.tensor(aliases,dtype=torch.long),
        'edges':edges,'joint_bonds':joint,'double_defined':defined}


class VisibleDataset(GraphDataset):
    def __init__(self,manifest,split,input_size,augment=False,seed=42,preprocessing='letterbox'):
        self.preprocessing=preprocessing
        self.manifest=Path(manifest);document=json.loads(self.manifest.read_text())
        if document.get('training_release_approved') is not True:raise ValueError('Visible data not admitted')
        if document.get('group_registry_sha256',REGISTRY_SHA256)!=REGISTRY_SHA256:raise ValueError('Visible vocabulary changed')
        assert_disjoint_splits(document['records']);self.records=[r for r in document['records'] if r['split']==split]
        if not self.records:raise ValueError('No visible records')
        self.size,self.augment,self.seed,self.epoch=input_size,augment,seed,0
        for record in self.records:
            if not record.get('visible_graph'):
                drawing_quality(record['coords_pixels'],drawing_molecule(record),record['native_size'])
            elif record['visible_atom_count']!=len(record['coords_pixels']):raise ValueError('Visible count mismatch')
            else:
                xy=np.asarray(record['coords_pixels'])
                if not np.isfinite(xy).all() or (xy<0).any() or (xy>=record['native_size']).any():
                    raise ValueError('Visible supervision lies outside image: '+record['record_key'])
    def get(self,index,*,epoch):
        sample=GraphDataset.get(self,index,epoch=epoch)
        if self.preprocessing=='molsight_native':
            from PIL import Image
            from training.molsight_dense.inputs import native_input
            record=self.records[index]
            with Image.open(record['file_path']) as image:
                sample['image'],sample['coords'],sample['augmentation']=native_input(image,record['coords_pixels'],self.size,
                    seed=sample['augmentation']['seed'],augment=self.augment)
        if not self.records[index].get('visible_graph'):
            sample=chemical_targets(drawing_molecule(self.records[index]),sample)
            sample['aliases']=torch.zeros_like(sample['elements'])
        if len(sample['aliases'])!=len(sample['coords']):raise ValueError('Visible unit/coordinate mismatch')
        return sample
