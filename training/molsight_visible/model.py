"""Predict the visible graph, then expand recognized fixed groups into chemistry."""
import torch
from torch import nn
from torch.nn import functional as F
from training.molsight_dense.model import DenseMolsight,DenseHead,DenseConfig,STATE_CLASSES,CHEMICAL,decode_graph
from .registry import TEMPLATES,REGISTRY_SHA256,template


def install_alias_head(head):
    if 'aliases' in head.states:raise ValueError('Alias classifier already exists')
    layer=nn.Linear(head.config.node_dim,len(TEMPLATES)+1)
    nn.init.zeros_(layer.weight);nn.init.constant_(layer.bias,-2.)
    with torch.no_grad():layer.bias[0]=6.
    head.states['aliases']=layer


def atom_labels(atom):
    return {'elements':atom.GetAtomicNum(),'charge':atom.GetFormalCharge()+8,'isotope':atom.GetIsotope(),
        'hydrogens':atom.GetTotalNumHs(),'radicals':atom.GetNumRadicalElectrons(),'aromatic':int(atom.GetIsAromatic())}


def expand_visible(points,states):
    """Expansion uses only the predicted group ID and its fixed single-port graph.

    Hidden atoms have derived layout positions, not claimed pixel observations.
    Incorrect degree or attachment order fails; edges/H/charge are never searched.
    """
    if not torch.isfinite(points).all() or any(not torch.isfinite(value).all() for value in states.values() if torch.is_tensor(value)):
        raise ValueError('Nonfinite visible-graph prediction')
    aliases=states['aliases'][0].argmax(-1).cpu().tolist()
    if not any(aliases):return points,states,[]
    n=len(aliases);joint=states['joint_bonds'][0].argmax(-1).cpu()
    defined=states['double_defined'][0].argmax(-1).cpu()
    coords=points[0].float().cpu().tolist()
    labels={name:states[name][0].argmax(-1).cpu().tolist() for name in STATE_CLASSES}
    added=[];expansions=[]
    for index,identifier in enumerate(aliases):
        if not identifier:continue
        neighbors=[j for j in range(n) if j!=index and int(joint[index,j])!=0]
        if len(neighbors)!=1:raise ValueError('Fixed single-port group has wrong predicted degree')
        neighbor=neighbors[0]
        if CHEMICAL[int(joint[index,neighbor])]!=1:raise ValueError('Wrong group attachment bond order')
        molecule,port,root=template(identifier)
        origin=torch.tensor(coords[index]);outward=torch.tensor(coords[neighbor])-origin
        if outward.norm()<1e-7:raise ValueError('Degenerate visible group attachment')
        layout=torch.tensor(molecule.GetConformer().GetPositions()[:,:2]).float();layout[:,1]*=-1
        port_direction=layout[port]-layout[root]
        u=port_direction/port_direction.norm();v=outward/outward.norm()
        cosine=(u*v).sum();sine=u[0]*v[1]-u[1]*v[0]
        rotation=torch.stack((torch.stack((cosine,-sine)),torch.stack((sine,cosine))))
        layout=(layout-layout[root])@rotation.T*(outward.norm()/port_direction.norm())+origin
        mapping={root:index};created=[]
        for atom in molecule.GetAtoms():
            if atom.GetIdx()==port:continue
            values=atom_labels(atom)
            if atom.GetIdx()==root:
                for name,value in values.items():labels[name][index]=value
            else:
                new=len(coords);mapping[atom.GetIdx()]=new;coords.append(layout[atom.GetIdx()].tolist());created.append(new)
                for name,value in values.items():labels[name].append(value)
        for bond in molecule.GetBonds():
            first,second=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
            if port in (first,second):continue
            kind={1.:1,2.:2,3.:3,1.5:4}.get(bond.GetBondTypeAsDouble())
            if kind is None:raise ValueError('Unsupported fixed template bond')
            added.append((mapping[first],mapping[second],kind))
        expansions.append({'visible_node':index,'label':TEMPLATES[identifier-1][0],'template':TEMPLATES[identifier-1][1],
            'chemical_root':index,'derived_atoms':created})
    count=len(coords)
    if count>2048:raise ValueError('Expanded chemical graph exceeds capacity')
    expanded_joint=torch.zeros(count,count,dtype=torch.long);expanded_joint[:n,:n]=joint
    expanded_defined=torch.zeros_like(expanded_joint);expanded_defined[:n,:n]=defined
    for i,j,kind in added:expanded_joint[i,j]=expanded_joint[j,i]=kind
    coordinates=torch.tensor(coords)
    minimum=coordinates.amin(0);span=(coordinates.amax(0)-minimum).max().clamp_min(1e-6)
    coordinates=.05+.9*(coordinates-minimum)/span
    device=points.device
    def logits(values,classes):return F.one_hot(torch.as_tensor(values,device=device,dtype=torch.long),classes).float().unsqueeze(0)*40-20
    result={name:logits(values,STATE_CLASSES[name]) for name,values in labels.items()}
    result.update(joint_bonds=logits(expanded_joint,9),double_defined=logits(expanded_defined,2))
    return coordinates.unsqueeze(0).to(device),result,expansions


def decode_visible(points,states):
    try:
        expanded,chemical,expansions=expand_visible(points,states)
        result=decode_graph(expanded,chemical)
        result['visible_count']=points.shape[1];result['expanded_atom_count']=expanded.shape[1]
        result['group_expansions']=expansions
        if expansions and 'raw_graph' in result:
            result['raw_graph']['coordinate_system']='expanded_layout_xy_down_not_hidden_atom_observations'
            result['raw_graph']['observed_visible_points']=points[0].float().cpu().tolist()
        return result
    except ValueError as error:
        return {'parse_success':False,'smiles':None,'error':str(error),'visible_count':points.shape[1]}


class VisibleMolsight(DenseMolsight):
    def __init__(self,checkpoint,vocabulary,config=DenseConfig()):
        nn.Module.__init__(self)
        from training.molsight_spatial.backbone import MolsightSpatial
        initial=MolsightSpatial(checkpoint,vocabulary)
        self.backbone=initial.backbone.requires_grad_(False)
        self.tokenizer=initial.tokenizer;self.configuration=initial.configuration
        self.head=VisibleHead(config)

    @torch.no_grad()
    def predict_image(self,image):
        self.eval();encoded=self.encode(image)
        try:
            points,scores=self.head.peaks(encoded)
            if not points.shape[1]:raise ValueError('No predicted visible units')
            states=self.head.read(encoded,points,image.shape[-1])
            result=decode_visible(points,states)
            result.update(predicted_count=points.shape[1],points=points[0].float().cpu().tolist(),
                elements=states['elements'][0].argmax(-1).cpu().tolist(),
                aliases=states['aliases'][0].argmax(-1).cpu().tolist(),atom_scores=scores.cpu().tolist())
            return result
        except ValueError as error:
            return {'parse_success':False,'smiles':None,'error':str(error),'predicted_count':getattr(error,'count',None)}


class VisibleHead(DenseHead):
    """Keep all existing heads; add large-glyph context through a zero residual."""
    def __init__(self,config):
        super().__init__(config);install_alias_head(self)
        c=config.channels
        self.context=nn.Sequential(nn.Conv2d(c,c,7,padding=12,dilation=4,groups=c),
            nn.GroupNorm(8,c),nn.GELU(),nn.Conv2d(c,c,1))
        nn.init.zeros_(self.context[-1].weight);nn.init.zeros_(self.context[-1].bias)

    def encode(self,image,semantic):
        encoded=super().encode(image,semantic)
        visual=encoded['visual'];visual=visual+self.context(visual)
        return {'visual':visual,'heatmap':self.center(visual),'offset':self.offset(visual).sigmoid()}
