"""Independent raw-decode guards without loading a large model or any GPU."""
import torch
from torch import nn
from training.ocsr_reboot.model import WarmStartRecognizer
from training.ocsr_reboot.train import evaluate


class Encoder(nn.Module):
    def forward(self, image): return torch.ones(1,2,4), []


class AR(nn.Module):
    def __init__(self, count, missing=False):
        super().__init__();self.count=count;self.missing=missing
    def decode(self, features, **kwargs):
        assert 'labels' not in kwargs and kwargs['allow_sep'] is False
        if self.missing:return [[]],None,None,[[]]
        return [[torch.tensor([2])]],None,None,[[torch.ones(self.count,4)]]


class Graph(nn.Module):
    def forward(self, hidden, indices):
        n=indices.shape[1];logits=torch.zeros(1,7,n,n);logits[:,1]=10
        if n==2:
            logits[0,:,0,1]=0;logits[0,5,0,1]=10
            logits[0,:,1,0]=0;logits[0,6,1,0]=10
        return {'edges':logits}


class Tokenizer:
    def __init__(self,count):self.count=count
    def sequence_to_smiles(self, tokens):
        assert tokens==[2]
        return {'smiles':'CCO','coords':[[.1,.2]]*self.count,
                'symbols':['C']*self.count,'indices':list(range(self.count))}


def model(count, missing=False):
    value=WarmStartRecognizer.__new__(WarmStartRecognizer);nn.Module.__init__(value)
    value.encoder=Encoder();value.decoder=nn.Module()
    value.decoder.decoder=nn.ModuleDict({'chartok_coords':AR(count,missing),'edges':Graph()})
    value.tokenizer=Tokenizer(count)
    return value


def test_raw_graph_does_not_prune_dense_predictions_or_mutate_mode():
    value=model(24);value.train()
    result=value.predict_image(torch.ones(1,3,8,8))
    assert result['graph_bond_count']==276
    assert all(result['edges'][i][i]==0 for i in range(24))
    assert value.training is True
    assert result['chartok_coords']['_token_sequence']==[2]


def test_stereo_class_symmetrization_respects_reverse_direction():
    result=model(2).predict_image(torch.ones(1,3,8,8))
    assert result['edges']==[[0,5],[6,0]]


def test_missing_eos_is_failure_without_inventing_tokens():
    result=model(3,True).predict_image(torch.ones(1,3,8,8))
    assert result['decode_quality_issue']=='missing_eos'
    assert result['chartok_coords']['_token_sequence']==[]


def test_graph_budget_does_not_discard_sequence_eos():
    result=model(24).predict_image(torch.ones(1,3,8,8),max_atoms=20)
    assert result['decode_quality_issue']=='graph_atom_budget_exceeded'
    assert result['chartok_coords']['_token_sequence']==[2]


def test_sequence_termination_is_independent_of_graph_quality():
    class Stub:
        def predict_image(self,image,**kwargs):
            return {'decode_quality_issue':'graph_issue',
                    'chartok_coords':{'smiles':'CCO','_token_sequence':[2]}}
    record={'record_key':'r','split':'train','category':'complete','SMILES':'CCO'}
    result=evaluate(Stub(),[record],{'r':{'image':torch.ones(3,8,8)}},torch.device('cpu'),'fp32',100)
    assert result['rows'][0]['terminated'] is True
    assert result['rows'][0]['raw_sequence_exact'] is True
    assert result['rows'][0]['graph_decode_issue']=='graph_issue'
