"""Admitted graph targets and pixel lineage, independent of inference inputs."""
import hashlib,json
from pathlib import Path
import torch
from PIL import Image
from rdkit import Chem
from torch.utils.data import Dataset
from training.ocsr_reboot.protocol import assert_disjoint_splits, canonical_smiles
from .augment import transform


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def drawing_quality(coords, molecule, size):
    """Reject severe atom collisions; this does not prove glyph legibility."""
    points=torch.as_tensor(coords,dtype=torch.float64)
    if points.shape!=(molecule.GetNumAtoms(),2) or not torch.isfinite(points).all():
        raise ValueError('Invalid drawing coordinates')
    if (points<0).any() or (points>=size).any():raise ValueError('Drawing coordinates out of bounds')
    if len(points)<2:return {'minimum_atom_distance_pixels':None,'median_bond_pixels':None}
    distances=torch.cdist(points,points);distances.fill_diagonal_(float('inf'))
    minimum=float(distances.min())
    lengths=[distances[b.GetBeginAtomIdx(),b.GetEndAtomIdx()] for b in molecule.GetBonds()]
    median=float(torch.stack(lengths).median()) if lengths else None
    if minimum<8*size/768 or (median is not None and minimum<.25*median):
        raise ValueError('Severe drawing atom collision')
    return {'minimum_atom_distance_pixels':minimum,'median_bond_pixels':median}


def graph_targets(molecule):
    n=molecule.GetNumAtoms()
    if not 0<n<=512:raise ValueError('Unsupported atom count')
    atoms=list(molecule.GetAtoms())
    result={'elements':[a.GetAtomicNum() for a in atoms], 'charge':[a.GetFormalCharge()+8 for a in atoms],
        'isotope':[a.GetIsotope() for a in atoms],'explicit_h':[a.GetNumExplicitHs() for a in atoms],
        'no_implicit':[int(a.GetNoImplicit()) for a in atoms],'radicals':[a.GetNumRadicalElectrons() for a in atoms],
        'aromatic':[int(a.GetIsAromatic()) for a in atoms]}
    limits={'elements':119,'charge':17,'isotope':513,'explicit_h':9,'no_implicit':2,'radicals':9,'aromatic':2}
    if any(any(not 0<=v<limits[key] for v in values) for key,values in result.items()):raise ValueError('Unsupported atom state')
    if 0 in result['elements']:raise ValueError('Dummy/Markush atoms are outside scope')
    types={Chem.BondType.SINGLE:1,Chem.BondType.DOUBLE:2,Chem.BondType.TRIPLE:3,Chem.BondType.AROMATIC:4}
    edges=torch.zeros(n,n,dtype=torch.long);wedges=torch.zeros_like(edges);double=torch.zeros_like(edges)
    for matrix in [edges,wedges,double]:matrix.fill_diagonal_(-100)
    for bond in molecule.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if bond.GetBondType() not in types:raise ValueError('Unsupported bond type')
        edges[i,j]=edges[j,i]=types[bond.GetBondType()]
        if bond.GetBondDir() in (Chem.BondDir.BEGINWEDGE,Chem.BondDir.BEGINDASH):
            wedges[i,j]=1 if bond.GetBondDir()==Chem.BondDir.BEGINWEDGE else 2
        if bond.GetStereo() in (Chem.BondStereo.STEREOZ,Chem.BondStereo.STEREOE,Chem.BondStereo.STEREOCIS,Chem.BondStereo.STEREOTRANS):
            double[i,j]=double[j,i]=1
    return {**{key:torch.tensor(v,dtype=torch.long) for key,v in result.items()},'edges':edges,'wedges':wedges,'double_stereo':double}


def target_prediction(targets, coordinates):
    """Annotation roundtrip helper, ONLY for data admission/tests, never inference."""
    from .model import SpatialDetector
    n=len(coordinates);out={'objectness':torch.full((1,n),20.),'coords':torch.as_tensor(coordinates).float().unsqueeze(0)}
    for name,count in {**SpatialDetector.STATE_CLASSES,'edges':5,'wedges':3,'double_stereo':2}.items():
        labels=targets[name];shape=(*labels.shape,count);values=torch.full(shape,-20.)
        values.scatter_(-1,labels.clamp_min(0).unsqueeze(-1),20.)
        out[name]=values.unsqueeze(0)
    return out


class GraphDataset(Dataset):
    def __init__(self,manifest,split,input_size,augment=False,seed=42):
        self.manifest=Path(manifest);document=json.loads(self.manifest.read_text())
        if document.get('training_release_approved') is not True:raise ValueError('Graph data not admitted')
        assert_disjoint_splits(document['records'])
        self.records=[r for r in document['records'] if r['split']==split]
        if not self.records:raise ValueError('Empty requested split')
        for record in self.records:
            drawing_quality(record['coords_pixels'],Chem.MolFromSmiles(record['SMILES']),record['native_size'])
        self.size,self.augment,self.seed,self.epoch=input_size,augment,seed,0
    def __len__(self):return len(self.records)
    def set_epoch(self,epoch):self.epoch=epoch
    def __getitem__(self,index):return self.get(index,epoch=self.epoch)
    def get(self,index,*,epoch):
        record=self.records[index]
        if digest(record['file_path'])!=record['image_sha256']:raise ValueError('Image changed after admission')
        path=Path(record['target_path'])
        if digest(path)!=record['target_sha256']:raise ValueError('Graph target changed')
        target=json.loads(path.read_text())
        seed=int.from_bytes(hashlib.sha256(f'{self.seed}:{epoch}:{record["record_key"]}'.encode()).digest()[:8],'little')
        with Image.open(record['file_path']) as image:
            tensor,coords,trace=transform(image,record['coords_pixels'],self.size,seed=seed,augment=self.augment)
        return {'image':tensor,'coords':coords,'smiles':record['SMILES'],'record_key':record['record_key'],
            'domain':record['domain'],'augmentation':trace,**{key:torch.tensor(value,dtype=torch.long) for key,value in target.items()}}
