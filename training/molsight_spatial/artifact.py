"""Self-contained model/vocabulary artifacts with strict runtime bindings."""
import json,shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from training.ocsr_reboot.checkpoint import load_checkpoint_strict
from .data import digest
from .backbone import MolsightSpatial
from .model import SpatialDetector,SpatialDetectorConfig


def runtime_sources(provenance):
    """Bind inference dependencies, not unrelated preparation/training drivers."""
    spatial={'model.py','backbone.py','inference.py','augment.py','data.py'}
    shared={'graph.py','protocol.py','checkpoint.py'}
    return {key:value for key,value in provenance['relative_source_sha256'].items()
        if (key.startswith('training/molsight_spatial/') and Path(key).name in spatial)
        or (key.startswith('training/ocsr_reboot/') and Path(key).name in shared)
        or key.startswith('training/molsight_runtime/molsight/')}


def export_artifact(checkpoint,out):
    checkpoint=Path(checkpoint);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    config,provenance=payload['config'],payload['provenance']
    shutil.copy2(checkpoint,out/'model.pth');shutil.copy2(config['vocabulary'],out/'vocabulary.json')
    metadata={'schema_version':'molsight_spatial_artifact_v2','model_sha256':digest(out/'model.pth'),
        'vocabulary_sha256':digest(out/'vocabulary.json'),'input_size':config['input_size'],
        'objectness_threshold':config['objectness_threshold'],'step':payload['step'],
        'backbone_configuration':provenance['backbone_configuration'],'detector_configuration':provenance['detector_configuration'],
        'relative_source_sha256':runtime_sources(provenance),
        'release_state':'experimental_requires_three_domain_real_image_acceptance'}
    if metadata['vocabulary_sha256']!=provenance['vocabulary_sha256']:raise ValueError('Vocabulary changed before export')
    (out/'artifact.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def load_artifact(directory,device='cpu'):
    from molsight.model import MolsightModel
    from molsight.tokenizer import CharTokenizer
    directory=Path(directory);meta=json.loads((directory/'artifact.json').read_text())
    if meta['schema_version'] not in ['molsight_spatial_artifact_v1','molsight_spatial_artifact_v2']:raise ValueError('Unknown artifact schema')
    if digest(directory/'model.pth')!=meta['model_sha256'] or digest(directory/'vocabulary.json')!=meta['vocabulary_sha256']:
        raise ValueError('Artifact integrity mismatch')
    repo=Path(__file__).resolve().parents[2]
    for relative,expected in meta['relative_source_sha256'].items():
        if digest(repo/relative)!=expected:raise ValueError('Runtime source differs from artifact: '+relative)
    payload=torch.load(directory/'model.pth',map_location='cpu',mmap=True,weights_only=False)
    config,provenance=payload['config'],payload['provenance']
    expected_sources=(runtime_sources(provenance) if meta['schema_version'].endswith('_v2') else provenance['relative_source_sha256'])
    if meta['relative_source_sha256']!=expected_sources:raise ValueError('Artifact runtime bindings differ from checkpoint')
    if any(meta[key]!=config[key] for key in ['input_size','objectness_threshold']):raise ValueError('Artifact decoding configuration differs from checkpoint')
    if any(meta[key]!=json.loads(json.dumps(provenance[key])) for key in ['backbone_configuration','detector_configuration']):raise ValueError('Artifact architecture differs from checkpoint')
    if meta['vocabulary_sha256']!=provenance['vocabulary_sha256']:raise ValueError('Artifact vocabulary differs from checkpoint')
    model=MolsightSpatial.__new__(MolsightSpatial);nn.Module.__init__(model)
    model.tokenizer=CharTokenizer(directory/'vocabulary.json')
    configuration=dict(meta['backbone_configuration']);configuration.update(use_checkpoint=False,compile_layers=False,resume=True)
    model.configuration=configuration;model.backbone=MolsightModel(SimpleNamespace(**configuration),model.tokenizer)
    model.detector=SpatialDetector(SpatialDetectorConfig(**meta['detector_configuration']))
    load_checkpoint_strict(directory/'model.pth',model,expected_config=payload['config'],expected_provenance=payload['provenance'],expected_step=meta['step'],restore_rng=False)
    return model.requires_grad_(False).eval().to(device),meta
