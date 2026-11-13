"""Create contracted visible-group supervision and prove full-chemistry roundtrip."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from rdkit import Chem,RDLogger
from rdkit.Chem import rdAbbreviations
from rdkit.Chem.Draw import rdMolDraw2D
from training.molsight_spatial.data import digest
from training.ocsr_reboot.protocol import canonical_smiles,assert_disjoint_splits
from training.molsight_visible.registry import condense_visible,REGISTRY_SHA256,TEMPLATES
from training.molsight_visible.data import visible_targets
from training.molsight_visible.model import decode_visible
from training.molsight_dense.model import STATE_CLASSES
from training.molsight_dense.drawing import drawing_molecule


def ideal(target,coords):
    from training.molsight_visible.admission import visible_annotation_roundtrip
    return visible_annotation_roundtrip(target,coords)


def render(record,out):
    if digest(record['file_path'])!=record['image_sha256']:raise ValueError('Parent image changed')
    original=drawing_molecule(record);conf=Chem.Conformer(original.GetNumAtoms());conf.Set3D(False)
    for i,(x,y) in enumerate(record['coords_pixels']):conf.SetAtomPosition(i,(x,-y,0.))
    original.AddConformer(conf)
    # Assign visible stereo on the full chemical graph BEFORE contraction.
    # Two different abbreviations are both atomic-number-zero placeholders;
    # reassigning stereo on that lossy graph can erase a real stereocenter.
    for bond in original.GetBonds():bond.SetBondDir(Chem.BondDir.NONE)
    Chem.SetDoubleBondNeighborDirections(original);Chem.WedgeMolBonds(original,original.GetConformer())
    visible=condense_visible(original)
    if not any(a.GetAtomicNum()==0 for a in visible.GetAtoms()):return None
    target=visible_targets(original,visible)
    depiction=Chem.Mol(visible);Chem.Kekulize(depiction,clearAromaticFlags=False)
    size=record['native_size'];drawer=rdMolDraw2D.MolDraw2DCairo(size,size);options=drawer.drawOptions()
    options.prepareMolsBeforeDrawing=False;options.useBWAtomPalette();options.padding=.07
    median=record.get('drawing_quality',{}).get('median_bond_pixels') or 20.
    options.fixedFontSize=int(np.clip(median*.55,8,52));options.bondLineWidth=float(np.clip(median*.035,.7,4.))
    drawer.DrawMolecule(depiction);drawer.FinishDrawing()
    coords=[[drawer.GetDrawCoords(i).x,drawer.GetDrawCoords(i).y] for i in range(visible.GetNumAtoms())]
    from training.molsight_dense.admission_geometry import assert_native_crop_contains_points
    import io
    from PIL import Image
    with Image.open(io.BytesIO(drawer.GetDrawingText())) as rendered:crop=assert_native_crop_contains_points(rendered,coords)
    observed=np.asarray(coords)
    if not np.isfinite(observed).all() or (observed<0).any() or (observed>=size).any():
        raise ValueError('Observed visible anchor outside rendered canvas')
    result=ideal(target,np.array(coords)/size)
    if not result['parse_success'] or canonical_smiles(result['smiles'])!=canonical_smiles(record['SMILES']):
        raise ValueError('Expanded chemistry/stereo roundtrip mismatch: '+str(result.get('error')))
    key=hashlib.sha256((record['record_key']+':visible-groups-v1:'+REGISTRY_SHA256).encode()).hexdigest()
    image=out/'images'/(key+'.png');image.parent.mkdir(exist_ok=True);image.write_bytes(drawer.GetDrawingText())
    labels=image.with_suffix('.json');labels.write_text(json.dumps({k:v.tolist() for k,v in target.items()})+'\n')
    return {**record,'record_key':key,'file_path':str(image),'image_sha256':digest(image),'coords_pixels':coords,
        'target_path':str(labels),'target_sha256':digest(labels),'visible_graph':True,'visible_atom_count':len(coords),
        'chemical_atom_count':Chem.RemoveHs(original).GetNumAtoms(),'group_registry_sha256':REGISTRY_SHA256,
        'original_atom_mapping':list(visible.GetPropsAsDict(includePrivate=True)['_origAtomMapping']),
        'parent_record_key':record['record_key'],'parent_image_sha256':record['image_sha256'],'native_crop_admission':crop,
        'style':{'font_size':options.fixedFontSize,'bond_width':options.bondLineWidth,'font_backend':'RDKit bundled default','nonchemical_legend':''}}


def main(args):
    RDLogger.DisableLog('rdApp.*');torch.set_num_threads(2);args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    parent=json.loads(args.manifest.read_text());records=list(parent['records']);existing={r['record_key'] for r in records};seen=set();rejected=[];added=0
    for row in parent['records']:
        hydrogen_child=args.include_hydrogen_views and row.get('source')=='explicit_stereocenter_H_view'
        if row['split']!='train' or (row.get('parent_record_key') and not hydrogen_child) or row['record_key'] in seen:continue
        expected=hashlib.sha256((row['record_key']+':visible-groups-v1:'+REGISTRY_SHA256).encode()).hexdigest()
        if expected in existing:continue
        seen.add(row['record_key'])
        try:
            item=render(row,args.out)
            if item:records.append(item);added+=1
        except (ValueError,RuntimeError) as error:rejected.append({'record_key':row['record_key'],'error':str(error)})
        if len(seen)%100==0:print(json.dumps({'checked':len(seen),'added':added,'rejected':len(rejected)}),flush=True)
    if not added:raise ValueError('No visible abbreviation examples admitted')
    document={**parent,'records':records,'split_audit':assert_disjoint_splits(records),'parent_manifest':str(args.manifest),
        'parent_manifest_sha256':digest(args.manifest),'group_registry_sha256':REGISTRY_SHA256,
        'visible_generator_sha256':digest(__file__),'visible_rejections':rejected,'training_release_approved':True}
    (args.out/'manifest.json').write_text(json.dumps(document,indent=2)+'\n')
    print(json.dumps({'added':added,'counts':dict(Counter(r['split']+':'+r['domain'] for r in records)),'rejected':len(rejected)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--include-hydrogen-views',action='store_true')
    main(parser.parse_args())
