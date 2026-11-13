"""Self-contained dense model artifacts with strict inference-source bindings."""
import json,shutil
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from training.molsight_spatial.data import digest
from training.molsight_spatial.artifact import runtime_sources
from training.ocsr_reboot.checkpoint import load_checkpoint_strict
from .model import DenseMolsight,DenseHead,DenseConfig


def inference_sources(provenance):
    result={**runtime_sources(provenance),'training/molsight_dense/model.py':provenance['relative_source_sha256']['training/molsight_dense/model.py']}
    path='training/molsight_dense/inputs.py'
    if path in provenance['relative_source_sha256']:result[path]=provenance['relative_source_sha256'][path]
    if 'visible_registry_sha256' in provenance:
        for name in ['model.py','registry.py']:
            path='training/molsight_visible/'+name;result[path]=provenance['relative_source_sha256'][path]
    if 'proposal_validity' in provenance:
        path='training/molsight_proposals/model.py';result[path]=provenance['relative_source_sha256'][path]
    return result


def export_artifact(checkpoint,out):
    checkpoint=Path(checkpoint);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    shutil.copy2(checkpoint,out/'model.pth');shutil.copy2(payload['config']['vocabulary'],out/'vocabulary.json')
    metadata={'schema':'molsight_dense_artifact_v1','model_sha256':digest(out/'model.pth'),'model_class':payload['model_class'],
        'vocabulary_sha256':digest(out/'vocabulary.json'),'step':payload['step'],
        'input_size':payload['config']['input_size'],'inference_sources':inference_sources(payload['provenance']),
        'preprocessing':payload['config'].get('preprocessing','letterbox'),
        'release_state':'experimental_requires_complete_small_and_cyclic_molecular_acceptance'}
    if metadata['vocabulary_sha256']!=payload['provenance']['vocabulary_sha256']:raise ValueError('Vocabulary changed')
    (out/'artifact.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def load_artifact(directory,device='cpu'):
    from molsight.model import MolsightModel
    from molsight.tokenizer import CharTokenizer
    directory=Path(directory);meta=json.loads((directory/'artifact.json').read_text())
    if meta['schema']!='molsight_dense_artifact_v1':raise ValueError('Wrong architecture artifact')
    if digest(directory/'model.pth')!=meta['model_sha256'] or digest(directory/'vocabulary.json')!=meta['vocabulary_sha256']:raise ValueError('Corrupt artifact')
    payload=torch.load(directory/'model.pth',map_location='cpu',mmap=True,weights_only=False)
    provenance=payload['provenance']
    if meta.get('preprocessing','letterbox')!=payload['config'].get('preprocessing','letterbox'):raise ValueError('Changed preprocessing contract')
    if meta['inference_sources']!=inference_sources(provenance) or meta['input_size']!=payload['config']['input_size']:raise ValueError('Altered artifact configuration')
    if meta['vocabulary_sha256']!=provenance['vocabulary_sha256']:raise ValueError('Wrong vocabulary')
    root=Path(__file__).resolve().parents[2]
    for relative,expected in meta['inference_sources'].items():
        if digest(root/relative)!=expected:raise ValueError('Inference source changed: '+relative)
    if payload['model_class']=='training.molsight_proposals.model.ProposalMolsight':
        from training.molsight_proposals.model import ProposalMolsight,ProposalHead
        model_type=ProposalMolsight
    elif payload['model_class']=='training.molsight_visible.model.VisibleMolsight':
        from training.molsight_visible.model import VisibleMolsight,VisibleHead
        model_type=VisibleMolsight
    elif payload['model_class']=='training.molsight_dense.model.DenseMolsight':model_type=DenseMolsight
    else:raise ValueError('Unknown dense model class')
    if meta.get('model_class',payload['model_class'])!=payload['model_class']:raise ValueError('Changed model class')
    model=model_type.__new__(model_type);nn.Module.__init__(model)
    model.tokenizer=CharTokenizer(directory/'vocabulary.json')
    configuration=dict(provenance['backbone_configuration']);configuration.update(use_checkpoint=False,compile_layers=False,resume=True)
    model.configuration=configuration
    model.backbone=MolsightModel(SimpleNamespace(**configuration),model.tokenizer)
    head_type=(ProposalHead if model_type.__name__=='ProposalMolsight' else VisibleHead if model_type is not DenseMolsight else DenseHead)
    model.head=head_type(DenseConfig(**provenance['dense_configuration']))
    load_checkpoint_strict(directory/'model.pth',model,expected_config=payload['config'],expected_provenance=provenance,
        expected_step=meta['step'],restore_rng=False)
    return model.requires_grad_(False).eval().to(device),meta
