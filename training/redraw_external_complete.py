"""Keep external chemistry/layout; replace unverified pixel-frame supervision.

External keypoints are used only as a relative conformer. Both the new PNG and
its atom coordinates come from the same RDKit drawing. Original assets remain.
"""
import argparse,hashlib,io,json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from rdkit import Chem,RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
from training.molsight_dense.drawing import drawing_molecule
from training.molsight_dense.annotation import annotation_roundtrip
from training.molsight_dense.admission_geometry import assert_native_crop_contains_points
from training.molsight_spatial.data import digest,graph_targets,drawing_quality
from training.ocsr_reboot.protocol import canonical_smiles,assert_disjoint_splits


def render(record,out):
    mol=drawing_molecule(record);xy=np.asarray(record['coords_pixels'],float);xy[:,1]*=-1;xy-=xy.mean(0)
    lengths=[np.linalg.norm(xy[b.GetBeginAtomIdx()]-xy[b.GetEndAtomIdx()]) for b in mol.GetBonds()]
    if len(xy)!=mol.GetNumAtoms():raise ValueError('Invalid relative source geometry')
    divisor=np.median(lengths) if lengths else max(float(np.ptp(xy,axis=0).max()),1.)
    xy*=1.5/divisor;conf=Chem.Conformer(len(xy));conf.Set3D(False)
    for i,(x,y) in enumerate(xy):conf.SetAtomPosition(i,(float(x),float(y),0.))
    mol.RemoveAllConformers();mol.AddConformer(conf)
    for bond in mol.GetBonds():bond.SetBondDir(Chem.BondDir.NONE)
    Chem.SetDoubleBondNeighborDirections(mol);Chem.WedgeMolBonds(mol,mol.GetConformer())
    targets=graph_targets(mol);seed=int(hashlib.sha256(record['record_key'].encode()).hexdigest()[:16],16);rng=np.random.default_rng(seed)
    native=record['native_size'];drawer=rdMolDraw2D.MolDraw2DCairo(native,native);options=drawer.drawOptions()
    options.prepareMolsBeforeDrawing=False;options.useBWAtomPalette();options.padding=.07
    median=record['drawing_quality']['median_bond_pixels'];options.fixedFontSize=int(np.clip(median*rng.uniform(.35,.65),8,48))
    options.bondLineWidth=float(np.clip(median*rng.uniform(.025,.05),.7,4.))
    font=Path('/usr/share/fonts/truetype/liberation')/('LiberationSerif-Regular.ttf' if seed%2 else 'LiberationSans-Regular.ttf')
    options.fontFile=str(font)
    legend=['(I)','Compound 12','Example 7'][seed%3] if seed%4==0 else ''
    drawer.DrawMolecule(mol,legend=legend);drawer.FinishDrawing()
    points=np.asarray([[drawer.GetDrawCoords(i).x,drawer.GetDrawCoords(i).y] for i in range(mol.GetNumAtoms())])
    png=drawer.GetDrawingText()
    with Image.open(io.BytesIO(png)) as image:crop=assert_native_crop_contains_points(image,points)
    quality=drawing_quality(points,mol,native);result=annotation_roundtrip(targets,points/native)
    if not result['parse_success'] or canonical_smiles(result['smiles'])!=canonical_smiles(record['SMILES']):raise ValueError('Redrawn chemical/stereo roundtrip failed')
    key=hashlib.sha256((record['record_key']+':aligned-pixel-render-v1').encode()).hexdigest();image=out/'images'/(key+'.png')
    image.write_bytes(png);target=image.with_suffix('.json');target.write_text(json.dumps({k:v.tolist() for k,v in targets.items()})+'\n')
    return {**record,'record_key':key,'file_path':str(image),'image_sha256':digest(image),'coords_pixels':points.tolist(),
        'target_path':str(target),'target_sha256':digest(target),'drawing_quality':quality,'native_crop_admission':crop,
        'external_source_record_key':record['record_key'],'external_source_image_sha256':record['image_sha256'],
        'coordinate_scope':'measured_from_same_new_drawing_after_finish','source':'redrawn_external_chemistry',
        'original_external_points_used_only_for_relative_layout':True,
        'style':{'font_size':options.fixedFontSize,'font_file':str(font),'font_sha256':digest(font),'bond_width':options.bondLineWidth,'nonchemical_legend':legend}}


def main(a):
    RDLogger.DisableLog('rdApp.*');a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);(a.out/'images').mkdir()
    parent=json.loads(a.manifest.read_text());records=[];rejected=[];count=0
    for row in parent['records']:
        if row['split']=='train' and row.get('source')=='molgrapher_synthetic_300k' and not row.get('parent_record_key'):
            try:records.append(render(row,a.out));count+=1
            except (ValueError,RuntimeError) as error:rejected.append({'record_key':row['record_key'],'error':str(error)})
        else:records.append(row)
        if count and count%250==0:print(json.dumps({'redrawn':count,'rejected':len(rejected)}),flush=True)
    document={**parent,'records':records,'split_audit':assert_disjoint_splits(records),'pixel_frame_correction':{
        'redrawn':count,'rejections':rejected,'parent_manifest_sha256':digest(a.manifest),'generator_sha256':digest(__file__),
        'source_pixel_keypoints_not_used_as_training_labels':True,'real_evaluation_images_unchanged':True}}
    (a.out/'manifest.json').write_text(json.dumps(document,indent=2)+'\n');print(json.dumps({'redrawn':count,'rejected':len(rejected)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
