"""Explicit two-head MolSight policy: preserve complete AR outputs, recover failures visually.

This reports its selected branch. It never chooses a candidate using a reference
molecule, and its combined accuracy must not be reported as spatial-only accuracy.
"""
import argparse,io,json,hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from rdkit import Chem
from training.ocsr_reboot.protocol import canonical_smiles


def complete_ar(text,terminated):
    if not terminated:return None
    from molsight.structure import parse_structure,StructureError
    from molsight.abbreviations import resolve_abbreviations
    try:
        resolved=resolve_abbreviations(parse_structure(text))
        document=resolved.document
        if resolved.unresolved or document.annotations or document.repeat or document.variation_records:return None
        return canonical_smiles(Chem.MolToSmiles(document.molecule,canonical=True,isomericSmiles=True))
    except (StructureError,ValueError,RuntimeError):return None


def cyclic_peptide_graph(text,min_ring_size=6):
    molecule=Chem.MolFromSmiles(text)
    amides=molecule.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=[OX1])[NX3]'))
    units=molecule.GetSubstructMatches(Chem.MolFromSmarts('[NX3,OX2]-[CX4]-[CX3](=[OX1])-[NX3,OX2]'))
    # A cyclic urea has two C(=O)-N matches sharing ONE carbonyl. Counting
    # matches routes ordinary small molecules into the peptide specialist.
    # Require actual repeated backbone units too: a glutarimide or barbiturate
    # has multiple carbonyls but is not a cyclic peptide specialist input.
    return any(len(ring)>=min_ring_size and len({c for c,o,n in amides if c in ring and n in ring})>=2
        and len({n for c,o,n in amides if c in ring and n in ring})>=2
        and len({alpha for n,alpha,c,o,z in units if n in ring and alpha in ring and c in ring and z in ring})>=2
        for ring in molecule.GetRingInfo().AtomRings())


def peptide_graph(text):
    molecule=Chem.MolFromSmiles(text)
    units=molecule.GetSubstructMatches(Chem.MolFromSmarts('[NX3]-[CX4]-[CX3](=[OX1])-[NX3]'))
    return len({carbonyl for nitrogen,alpha,carbonyl,oxygen,next_n in units})>=2 or cyclic_peptide_graph(text)


def select_prediction(ar_text,terminated,visual=None,policy='preserve_ar'):
    if policy not in ['preserve_ar','cyclic_visible','peptide_visible']:raise ValueError('Unknown branch policy')
    if policy in ['cyclic_visible','peptide_visible'] and visual and visual.get('parse_success'):
        try:
            text=canonical_smiles(visual['smiles'])
            recognized=peptide_graph(text) if policy=='peptide_visible' else cyclic_peptide_graph(text)
            if recognized:return {'branch':'visible_graph','smiles':text,'parse_success':True,'reason':'complete_visible_peptide_specialist'}
        except ValueError:pass
    ar=complete_ar(ar_text,terminated)
    if ar is not None:return {'branch':'molsight_ar','smiles':ar,'parse_success':True,'reason':'complete_AR_preserved'}
    if visual and visual.get('parse_success'):
        try:text=canonical_smiles(visual['smiles'])
        except ValueError:text=None
        if text is not None:return {'branch':'visible_graph','smiles':text,'parse_success':True,'reason':'AR_failed_visible_graph_complete'}
    return {'branch':None,'smiles':None,'parse_success':False,'reason':'both_branches_failed'}


@torch.no_grad()
def predict_file(model,metadata,path,policy='preserve_ar',graph_sizes=None):
    from molsight.dataset import get_transforms
    from molsight.inference import predict_images
    from training.molsight_dense.inputs import native_input
    from training.molsight_spatial.augment import transform
    path=Path(path);encoded=path.read_bytes();device=next(model.parameters()).device
    image=Image.open(io.BytesIO(encoded)).convert('RGB')
    def predict_visual():
        if graph_sizes is not None:
            from training.molsight_multiscale import predict_file as predict_scales
            return predict_scales(model,metadata,path,graph_sizes)
        prepare=native_input if metadata.get('preprocessing')=='molsight_native' else transform
        tensor,_,_=prepare(image,np.empty((0,2)),metadata['input_size'],augment=False)
        return model.predict_image(tensor.unsqueeze(0).to(device))
    visual=None
    if policy in ['cyclic_visible','peptide_visible']:
        # Visual failures must never take down a request whose AR branch would
        # have succeeded: degrade to visual=None and let AR carry the answer.
        try:
            visual=predict_visual()
        except Exception as error:
            visual={'parse_success':False,'smiles':None,'error':f'{type(error).__name__}: {error}'}
        selected=select_prediction('',False,visual,policy)
        if selected['parse_success'] and selected['reason']=='complete_visible_peptide_specialist':
            return {**selected,'ar_prediction':None,'ar_terminated':None,'visual_prediction':visual,
                'input_sha256':hashlib.sha256(encoded).hexdigest(),'policy':policy,'graph_sizes':graph_sizes,'oracle_selection':False,'optimizer_updates':0}
    official=get_transforms(SimpleNamespace(input_size=512,encoder='efficientvit'),augment=False,rotate=False)
    ar_image=official(image=np.asarray(image),keypoints=[])['image']
    model.eval();model.backbone.max_len=2048
    ar=predict_images(model.backbone,ar_image.unsqueeze(0).to(device))
    selected=select_prediction(ar['smiles'][0],ar['terminated'][0],visual,policy)
    if not selected['parse_success'] and visual is None:
        try:
            visual=predict_visual()
        except Exception as error:
            visual={'parse_success':False,'smiles':None,'error':f'{type(error).__name__}: {error}'}
        selected=select_prediction(ar['smiles'][0],ar['terminated'][0],visual,policy)
    return {**selected,'ar_prediction':ar['smiles'][0],'ar_terminated':bool(ar['terminated'][0]),'visual_prediction':visual,
        'input_sha256':hashlib.sha256(encoded).hexdigest(),'policy':policy,'graph_sizes':graph_sizes,'oracle_selection':False,
        'optimizer_updates':0}


if __name__=='__main__':
    from training.molsight_dense.artifact import load_artifact
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['artifact','image','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--policy',choices=['preserve_ar','cyclic_visible','peptide_visible'],default='preserve_ar')
    parser.add_argument('--graph-sizes',nargs='+',type=int,help='Optional fixed graph resolution cascade; primary must match artifact')
    parser.add_argument('--inference-policy',type=Path);args=parser.parse_args()
    model,meta=load_artifact(args.artifact,args.device)
    if args.inference_policy:
        policy=json.loads(args.inference_policy.read_text())
        if policy['artifact_sha256']!=hashlib.sha256((args.artifact/'artifact.json').read_bytes()).hexdigest():raise ValueError('Policy/artifact mismatch')
        model.head.config=replace(model.head.config,peak_threshold=policy['threshold'])
    result=predict_file(model,meta,args.image,args.policy,args.graph_sizes)
    result['inference_policy_sha256']=hashlib.sha256(args.inference_policy.read_bytes()).hexdigest() if args.inference_policy else None
    args.out.write_text(json.dumps(result,indent=2)+'\n')
