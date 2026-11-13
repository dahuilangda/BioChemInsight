"""Image-only tensor-to-graph decoding; no reference counts or coordinate input."""
from dataclasses import asdict
import math
import torch
from rdkit import Chem
from training.ocsr_reboot.graph import AtomState, ChemicalBond, WedgeBond, DoubleBondStereo, assemble_raw_graph
from .model import SpatialDetector


def decode_prediction(outputs, *, objectness_threshold=.5, config=None):
    raw = {}
    try:
        if not math.isfinite(objectness_threshold) or not 0 < objectness_threshold < 1:
            raise ValueError('Invalid objectness threshold')
        def finite(item):
            if isinstance(item, torch.Tensor) and not torch.isfinite(item).all():
                raise ValueError('Nonfinite network prediction')
            if isinstance(item, dict):
                for value in item.values(): finite(value)
            if isinstance(item, (list, tuple)):
                for value in item: finite(value)
        finite(outputs)
        objectness = outputs['objectness']
        if objectness.ndim != 2 or objectness.shape[0] != 1:
            raise ValueError('Expected single-image query predictions')
        q = objectness.shape[1]
        sizes = {'coords': (1,q,2), **{name:(1,q,count) for name,count in SpatialDetector.STATE_CLASSES.items()},
                 **{name:(1,q,q,count) for name,count in [('edges',5),('wedges',3),('double_stereo',2)]}}
        for name, shape in sizes.items():
            if tuple(outputs[name].shape) != shape:
                raise ValueError('Invalid prediction shape for '+name)
        chosen = torch.where(objectness[0].sigmoid() > objectness_threshold)[0]
        raw['selected_query_indices'] = chosen.cpu().tolist()
        if not len(chosen):
            raise ValueError('No predicted atoms')
        states = {name: outputs[name][0,chosen].argmax(-1).cpu().tolist() for name in SpatialDetector.STATE_CLASSES}
        if any(not 1 <= number <= 118 for number in states['elements']):
            raise ValueError('Unsupported predicted element')
        coords = outputs['coords'][0,chosen].float().cpu().tolist()
        if any(not 0 <= v <= 1 for point in coords for v in point):
            raise ValueError('Coordinates outside normalized image')
        atoms = [AtomState(Chem.GetPeriodicTable().GetElementSymbol(states['elements'][i]),
            charge=states['charge'][i]-8, isotope=states['isotope'][i], explicit_h=states['explicit_h'][i],
            no_implicit=bool(states['no_implicit'][i]), radical_electrons=states['radicals'][i],
            aromatic=bool(states['aromatic'][i])) for i in range(len(chosen))]
        def pair_logits(name):
            return outputs[name][0][chosen[:,None],chosen[None,:]].float()
        chemical = pair_logits('edges').softmax(-1)
        bonds = ((chemical+chemical.transpose(0,1))*.5).argmax(-1).cpu()
        wedge = pair_logits('wedges').argmax(-1).cpu()
        stereo = pair_logits('double_stereo').softmax(-1)
        specified = ((stereo+stereo.transpose(0,1))*.5).argmax(-1).cpu()
        edges, wedges, doubles = [], [], []
        n = len(atoms)
        for i in range(n):
            for j in range(i+1,n):
                kind = int(bonds[i,j])
                if kind: edges.append(ChemicalBond(i,j,kind))
                forward, reverse = int(wedge[i,j]), int(wedge[j,i])
                if forward and reverse:
                    raise ValueError('Conflicting opposite-apex wedge predictions')
                if forward or reverse:
                    if kind != 1: raise ValueError('Wedge cue on a non-single bond')
                    apex, other, direction = (i,j,forward) if forward else (j,i,reverse)
                    wedges.append(WedgeBond(apex,other,'solid' if direction==1 else 'hashed'))
                if int(specified[i,j]):
                    if kind != 2: raise ValueError('Double stereo cue on a non-double bond')
                    left = [k for k in range(n) if k != j and k != i and int(bonds[i,k])]
                    right = [k for k in range(n) if k != i and k != j and int(bonds[j,k])]
                    if not left or not right: raise ValueError('Missing predicted double-bond substituents')
                    doubles.append(DoubleBondStereo(i,j,min(left),min(right),'geometry'))
        raw.update(atoms=[asdict(a) for a in atoms], coordinates=coords,
                   edges=[asdict(e) for e in edges],wedges=[asdict(w) for w in wedges],double_stereo=[asdict(d) for d in doubles])
        # RDKit's 2-D stereo routines use absolute near-zero bond tolerances.
        # Normalized image coordinates are not Molfile drawing units. A fixed,
        # orientation-preserving unit conversion must not change chemistry.
        raw['coordinate_system']='normalized_image_xy_down'
        raw['molfile_coordinate_scale']=100.
        result = assemble_raw_graph(atoms,[[100*x,100*y] for x,y in coords],edges,wedges=wedges,double_stereo=doubles)
        return {**asdict(result),'parse_success':result.ok,'raw_graph':raw}
    except (ValueError, KeyError, IndexError, RuntimeError) as error:
        return {'ok':False,'parse_success':False,'smiles':None,'molblock':None,
                'error_code':'invalid_network_graph','error':str(error),'raw_graph':raw}
