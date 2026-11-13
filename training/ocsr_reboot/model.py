"""Strict, plain pretrained MolNexTR path, without production MoE or repairs."""
from __future__ import annotations
from pathlib import Path
import hashlib
import torch
from torch import nn
from torch.nn import functional as F


def tensor_digest(tensor):
    data=tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


class WarmStartRecognizer(nn.Module):
    def __init__(self, checkpoint, *, input_size=384, activation_checkpointing=True):
        super().__init__()
        from utils.MolNexTR.model import molnextr, _pad_chartok_for_sep
        from utils.MolNexTR.components import Encoder, Decoder
        from utils.MolNexTR.tokenization import get_tokenizer
        payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        self.args=molnextr._get_args(None,payload['args'])
        if self.args.encoder != 'swin_base' or self.args.formats != ['chartok_coords','edges']:
            raise ValueError('Unreviewed pretrained architecture')
        self.args.input_size=input_size
        self.args.use_checkpoint=activation_checkpointing
        self.tokenizers=get_tokenizer(self.args)
        self.encoder=Encoder(self.args,pretrained=False)
        self.args.encoder_dim=self.encoder.n_features
        self.decoder=Decoder(self.args,self.tokenizers)
        self.migrations=[]
        for name,module in [('encoder',self.encoder),('decoder',self.decoder)]:
            original=payload[name]
            states={key.removeprefix('module.'):value for key,value in original.items()}
            if len(states)!=len(original):raise ValueError('Colliding checkpoint keys')
            before={key:tuple(value.shape) for key,value in states.items()}
            states=_pad_chartok_for_sep(module,states)
            for key,value in states.items():
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError(f'Nonfinite pretrained tensor: {name}.{key}')
                if tuple(value.shape)!=before[key]:
                    if name!='decoder' or value.shape[0]!=before[key][0]+1 or value.shape[1:]!=before[key][1:]:
                        raise ValueError('Unreviewed tensor shape migration')
                    self.migrations.append(name+'.'+key)
            module.load_state_dict(states,strict=True)
        if len(self.migrations) not in (0,3):raise ValueError('Unexpected vocabulary migration')
        self.tokenizer=self.tokenizers['chartok_coords']
        self.source_checkpoint=str(Path(checkpoint).resolve())
        self.source_global_step=int(payload.get('global_step',-1))

    def forward(self,image,refs):
        features,hiddens=self.encoder(image)
        return self.decoder(features,hiddens,refs)

    def losses(self,image,refs):
        """FP32 token mean + per-image balanced chemical edge mean.

        No drawing wedge labels are invented from chemical-only annotations.
        Native learned direction logits remain distinct, marginalized for loss.
        """
        from training.molsight_multitask.edges import chemical_edge_sums
        output=self(image,refs)
        logits,targets,_=output['chartok_coords']
        logits=logits.float().clone()
        logits[...,self.tokenizer.sep_id]=torch.finfo(logits.dtype).min
        token=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=0)
        edges,edge_target=output['edges']
        summary=chemical_edge_sums(edges['edges'].float().permute(0,2,3,1),edge_target['edges'],positive_weight=.5)
        if int(summary['images'])!=image.shape[0]:raise ValueError('Incomplete graph supervision')
        edge=summary['loss_sum']/summary['images']
        return {'token_nll':token,'chemical_edge':edge,'loss':token+edge}

    @torch.no_grad()
    def predict_image(self,image,*,max_tokens=4096,max_atoms=512):
        """Image-only decode; callers must not pass targets or gold atom counts."""
        if image.ndim!=4 or image.shape[0]!=1:raise ValueError('Use one image per diagnostic decode')
        if max_tokens < 1 or max_atoms < 1: raise ValueError('Invalid decode budget')
        was_training=self.training
        self.eval()
        try:
            features,_=self.encoder(image)
            outputs,_,_,hidden=self.decoder.decoder['chartok_coords'].decode(
                features,beam_size=1,n_best=1,max_length=max_tokens,allow_sep=False)
            if not outputs or not outputs[0]:
                return {'chartok_coords':{'smiles':'','symbols':[],'coords':[],'indices':[],
                    '_token_sequence':[]},'edges':[],'decode_quality_issue':'missing_eos'}
            sequence=outputs[0][0].tolist()
            decoded=self.tokenizer.sequence_to_smiles(sequence)
            decoded['_token_sequence']=sequence
            result={'chartok_coords':decoded,'edges':[],
                    'graph_scope':'unrepaired native directed seven-class head'}
            count=len(decoded['indices'])
            if count>max_atoms:
                result['decode_quality_issue']='graph_atom_budget_exceeded'
                return result
            if not count: return result
            states=hidden[0][0].unsqueeze(0)
            indices=torch.tensor([decoded['indices']],device=states.device,dtype=torch.long)
            if indices.min()<0 or indices.max()>=states.shape[1]:
                result['decode_quality_issue']='graph_hidden_alignment_failure'
                return result
            logits=self.decoder.decoder['edges'](states,indices)['edges'][0].float().permute(1,2,0)
            if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite raw graph logits')
            probabilities=logits.softmax(-1)
            reverse=[0,1,2,3,4,6,5]
            symmetric=(probabilities+probabilities.transpose(0,1)[...,reverse])*.5
            matrix=symmetric.argmax(-1)
            matrix.fill_diagonal_(0)
            # No bond-count clipping, valence pruning or native chemical repair.
            result['edges']=matrix.cpu().tolist()
            result['graph_bond_count']=int(torch.triu(matrix.ne(0),diagonal=1).sum())
            return result
        finally:
            self.train(was_training)
