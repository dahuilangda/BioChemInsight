"""MolSight features + pixel centers + explicit bond-tube visual evidence."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from training.molsight_spatial.inference import decode_prediction


# none, single, double, triple, aromatic, solid_i, hashed_i, solid_j, hashed_j
REVERSE=(0,1,2,3,4,7,8,5,6)
CHEMICAL=(0,1,2,3,4,1,1,1,1)
STATE_CLASSES={'elements':119,'charge':17,'isotope':513,'hydrogens':9,'radicals':9,'aromatic':2}


@dataclass(frozen=True)
class DenseConfig:
    feature_dim:int=512
    channels:int=64
    node_dim:int=128
    line_channels:int=12
    line_points:int=9
    pair_chunk:int=2048
    stride:int=2
    max_atoms:int=512
    peak_threshold:float=.3

    def __post_init__(self):
        if any(type(v) is not int or v<1 for k,v in vars(self).items() if k!='peak_threshold'):raise ValueError('Invalid dense dimension')
        if not 0<self.peak_threshold<1:raise ValueError('Invalid fixed peak threshold')


class AtomCapacityError(ValueError):
    def __init__(self,count):
        self.count=count
        super().__init__('Predicted atom count exceeds capacity; no top-k truncation')


def block(inputs,outputs,stride=1):
    return nn.Sequential(nn.Conv2d(inputs,outputs,3,stride=stride,padding=1),
        nn.GroupNorm(math.gcd(8,outputs),outputs),nn.GELU())


def sample_points(features,points):
    grid=(points*2-1).unsqueeze(2)
    return F.grid_sample(features.float(),grid.float(),align_corners=False).squeeze(-1).transpose(1,2)


class BondTube(nn.Module):
    """Sample the interior of each actual image segment, including its flanks.

    All pairs are considered. Chunking controls memory without deleting long
    bonds or assuming that a chemical neighbor must be a geometric nearest one.
    """
    def __init__(self,config):
        super().__init__();self.config=config
        self.project=nn.Conv2d(config.channels,config.line_channels,1)
        self.register_buffer('fractions',torch.linspace(.12,.88,config.line_points))
        self.register_buffer('flanks',torch.tensor([-1.,0.,1.]))
        inputs=config.line_channels*config.line_points*3+config.node_dim*2+5
        self.classifier=nn.Sequential(nn.Linear(inputs,192),nn.GELU(),nn.Linear(192,11))

    def tube_features(self,features,starts,ends,image_size):
        delta=ends-starts
        length=torch.linalg.vector_norm(delta,dim=-1,keepdim=True).clamp_min(1e-6)
        normal=torch.stack((-delta[...,1],delta[...,0]),-1)/length
        width=1.5/image_size
        grid=starts[:,:,None,None,:]+delta[:,:,None,None,:]*self.fractions[None,None,:,None,None]
        grid=grid+normal[:,:,None,None,:]*self.flanks[None,None,None,:,None]*width
        batch,count=starts.shape[:2]
        sampled=sample_points(features,grid.reshape(batch,-1,2))
        return sampled.reshape(batch,count,-1)

    def forward(self,features,points,nodes,image_size):
        if points.shape[0]!=1:raise ValueError('Unpadded single-image graph required')
        n=points.shape[1]
        visual=self.project(features)
        pairs=torch.cartesian_prod(torch.arange(n,device=points.device),torch.arange(n,device=points.device))
        pieces=[]
        for ids in pairs.split(self.config.pair_chunk):
            left,right=ids[:,0],ids[:,1]
            starts,ends=points[:,left],points[:,right]
            delta=ends-starts;distance=torch.linalg.vector_norm(delta,dim=-1,keepdim=True)
            direction=delta/distance.clamp_min(1e-6)
            tube=self.tube_features(visual,starts,ends,image_size)
            geometry=torch.cat((delta,distance,direction),-1)
            inputs=torch.cat((nodes[:,left],nodes[:,right],tube.to(nodes.dtype),geometry.to(nodes.dtype)),-1)
            pieces.append(self.classifier(inputs))
        if not pieces:
            return nodes.new_empty((1,0,0,9)),nodes.new_empty((1,0,0,2))
        scores=torch.cat(pieces,1).reshape(1,n,n,11)
        joint=scores[...,:9]
        # Reverse the wedge apex together with the pair ordering.
        joint=(joint+joint.transpose(1,2)[...,list(REVERSE)])*.5
        defined=(scores[...,9:]+scores[...,9:].transpose(1,2))*.5
        return joint,defined


class DenseHead(nn.Module):
    def __init__(self,config=DenseConfig()):
        super().__init__();self.config=config
        if config.stride!=2:raise ValueError('This dense pixel stem has stride 2')
        c=config.channels
        self.pixel=nn.Sequential(block(3,32,2),block(32,c))
        self.semantic=nn.Conv2d(config.feature_dim,c,1)
        self.fuse=nn.Sequential(block(c*2,c),block(c,c))
        self.center=nn.Conv2d(c,1,1);nn.init.constant_(self.center.bias,-4.6)
        self.offset=nn.Conv2d(c,2,1)
        self.node=nn.Sequential(nn.Linear(c,config.node_dim),nn.GELU(),nn.Linear(config.node_dim,config.node_dim))
        self.states=nn.ModuleDict({name:nn.Linear(config.node_dim,classes) for name,classes in STATE_CLASSES.items()})
        self.bonds=BondTube(config)

    def encode(self,image,semantic):
        pixel=self.pixel(image)
        semantic=F.interpolate(self.semantic(semantic),size=pixel.shape[-2:],mode='bilinear',align_corners=False)
        visual=self.fuse(torch.cat((pixel,semantic),1))
        return {'visual':visual,'heatmap':self.center(visual),'offset':self.offset(visual).sigmoid()}

    def read(self,encoded,points,image_size,with_bonds=True):
        nodes=self.node(sample_points(encoded['visual'],points).to(encoded['visual'].dtype))
        result={name:head(nodes) for name,head in self.states.items()}
        if with_bonds:
            result['joint_bonds'],result['double_defined']=self.bonds(encoded['visual'],points,nodes,image_size)
        return result

    def peaks(self,encoded):
        probability=encoded['heatmap'].float().sigmoid()
        local=F.max_pool2d(probability,3,stride=1,padding=1)
        indices=torch.where((probability[0,0]>=self.config.peak_threshold)&(probability[0,0]==local[0,0]))
        y,x=indices;n=len(x)
        if n>self.config.max_atoms:raise AtomCapacityError(n)
        h,w=probability.shape[-2:]
        offsets=encoded['offset'][0,:,y,x].transpose(0,1).float()
        points=(torch.stack((x,y),-1).float()+offsets)/offsets.new_tensor([w,h])
        return points.unsqueeze(0),probability[0,0,y,x]


def decode_graph(points,states):
    """Encode the predicted joint categorical state for strict raw assembly.

    Total H is an explicit chemical prediction. No H/valence search is used.
    Wedges are part of the same categorical bond as bond order. The separate
    'double defined' prediction is interpreted only on predicted double bonds.
    """
    n=points.shape[1];device=points.device
    chosen=states['joint_bonds'].argmax(-1)
    chemical=torch.tensor(CHEMICAL,device=device)[chosen]
    forward=torch.where((chosen==5)|(chosen==6),chosen-4,0).triu(diagonal=1)
    reverse=torch.where((chosen==7)|(chosen==8),chosen-6,0).triu(diagonal=1)
    wedges=forward+reverse.transpose(1,2)
    defined=states['double_defined'].argmax(-1)*chemical.eq(2)
    def categorical(labels,classes):
        return F.one_hot(labels,classes).float()*40-20
    outputs={'objectness':points.new_full((1,n),20.),'coords':points,
        **{name:states[name] for name in ['elements','charge','isotope','radicals','aromatic']},
        'explicit_h':states['hydrogens'],'no_implicit':categorical(torch.ones((1,n),device=device,dtype=torch.long),2),
        'edges':categorical(chemical,5),'wedges':categorical(wedges,3),'double_stereo':categorical(defined,2)}
    result=decode_prediction(outputs)
    result['representation']='dense_centers_joint_bond_and_apex_total_hydrogen'
    return result


class DenseMolsight(nn.Module):
    def __init__(self,checkpoint,vocabulary,config=DenseConfig()):
        super().__init__()
        from training.molsight_spatial.backbone import MolsightSpatial
        initial=MolsightSpatial(checkpoint,vocabulary)
        self.backbone=initial.backbone.requires_grad_(False)
        self.tokenizer=initial.tokenizer;self.configuration=initial.configuration
        self.head=DenseHead(config)

    def encode(self,image):
        self.backbone.eval()
        with torch.no_grad():semantic=self.backbone.get_image_features(image)
        return self.head.encode(image,semantic)

    @torch.no_grad()
    def predict_image(self,image):
        self.eval();encoded=self.encode(image)
        try:
            points,scores=self.head.peaks(encoded)
            if not points.shape[1]:raise ValueError('No predicted atoms')
            states=self.head.read(encoded,points,image.shape[-1])
            result=decode_graph(points,states)
            result['predicted_count']=points.shape[1]
            result['atom_scores']=scores.cpu().tolist()
            result['points']=points[0].float().cpu().tolist()
            result['elements']=states['elements'][0].argmax(-1).cpu().tolist()
            return result
        except AtomCapacityError as error:
            return {'parse_success':False,'smiles':None,'error':str(error),'predicted_count':error.count,'points':[],'elements':[]}
        except ValueError as error:
            return {'parse_success':False,'smiles':None,'error':str(error),'predicted_count':None}
