"""Depict observed stereocenter H atoms as explicit graph nodes, not ghost C/N.

These are new views of existing training identities. Validation is untouched.
"""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from rdkit import Chem,RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
from training.molsight_dense.drawing import drawing_molecule
from training.molsight_spatial.data import digest,graph_targets,drawing_quality,target_prediction
from training.molsight_spatial.inference import decode_prediction
from training.ocsr_reboot.protocol import canonical_smiles,assert_disjoint_splits


def render(record,out):
    if digest(record['file_path'])!=record['image_sha256']:raise ValueError('Parent image changed')
    molecule=drawing_molecule(record)
    indices=[a.GetIdx() for a in molecule.GetAtoms() if a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED and a.GetTotalNumHs()>0]
    if not indices:return None
    xy=np.asarray(record['coords_pixels'],float);xy[:,1]*=-1;xy-=xy.mean(0)
    lengths=[np.linalg.norm(xy[b.GetBeginAtomIdx()]-xy[b.GetEndAtomIdx()]) for b in molecule.GetBonds()]
    xy*=1.5/np.median(lengths);conf=Chem.Conformer(len(xy));conf.Set3D(False)
    for i,(x,y) in enumerate(xy):conf.SetAtomPosition(i,(float(x),float(y),0.))
    molecule.RemoveAllConformers();molecule.AddConformer(conf)
    molecule=Chem.AddHs(molecule,addCoords=True,onlyOnAtoms=indices)
    if molecule.GetNumAtoms()>512:raise ValueError('Visible hydrogen capacity')
    molecule.GetConformer().Set3D(False)
    for bond in molecule.GetBonds():bond.SetBondDir(Chem.BondDir.NONE)
    Chem.SetDoubleBondNeighborDirections(molecule);Chem.WedgeMolBonds(molecule,molecule.GetConformer())
    text=Chem.MolToSmiles(molecule,canonical=True,isomericSmiles=True);order=json.loads(molecule.GetProp('_smilesAtomOutputOrder'))
    molecule=Chem.RenumberAtoms(molecule,order);targets=graph_targets(molecule)
    size=record['native_size'];drawer=rdMolDraw2D.MolDraw2DCairo(size,size);options=drawer.drawOptions()
    options.prepareMolsBeforeDrawing=False;options.useBWAtomPalette();options.padding=.07
    median=record.get('drawing_quality',{}).get('median_bond_pixels') or 24.
    options.fixedFontSize=int(np.clip(median*.5,8,48))
    options.bondLineWidth=float(np.clip(median*.035,.7,3.5))
    font=Path('/usr/share/fonts/truetype/liberation')/('LiberationSerif-Regular.ttf' if int(hashlib.sha256(record['record_key'].encode()).hexdigest()[:8],16)%2 else 'LiberationSans-Regular.ttf')
    if not font.exists():raise ValueError('Required drawing font missing')
    options.fontFile=str(font)
    legend='(II)' if int(hashlib.sha256(record['record_key'].encode()).hexdigest()[:8],16)%4==0 else ''
    drawer.DrawMolecule(molecule,legend=legend);drawer.FinishDrawing()
    coords=np.array([[drawer.GetDrawCoords(i).x,drawer.GetDrawCoords(i).y] for i in range(molecule.GetNumAtoms())])
    from training.molsight_dense.admission_geometry import assert_native_crop_contains_points
    import io
    from PIL import Image
    with Image.open(io.BytesIO(drawer.GetDrawingText())) as rendered:crop=assert_native_crop_contains_points(rendered,coords)
    if not np.isfinite(coords).all() or (coords<0).any() or (coords>=size).any():raise ValueError('Observed H bounds')
    quality=drawing_quality(coords,molecule,size)
    from training.molsight_dense.annotation import annotation_roundtrip
    result=annotation_roundtrip(targets,coords/size)
    if not result['parse_success'] or canonical_smiles(result['smiles'])!=canonical_smiles(record['SMILES']):raise ValueError('Visible H stereo roundtrip')
    key=hashlib.sha256((record['record_key']+':visible-stereocenter-H-v2').encode()).hexdigest();image=out/'images'/(key+'.png')
    image.write_bytes(drawer.GetDrawingText());target=image.with_suffix('.json');target.write_text(json.dumps({k:v.tolist() for k,v in targets.items()})+'\n')
    return {**record,'record_key':key,'SMILES':text,'coords_pixels':coords.tolist(),'file_path':str(image),'image_sha256':digest(image),
        'target_path':str(target),'target_sha256':digest(target),'drawing_quality':quality,'visible_hydrogens':True,
        'chemical_atom_count':Chem.RemoveHs(molecule).GetNumAtoms(),'parent_record_key':record['record_key'],
        'parent_image_sha256':record['image_sha256'],'source':'explicit_stereocenter_H_view','observed_H_count':sum(a.GetAtomicNum()==1 for a in molecule.GetAtoms()),
        'style':{'font_size':options.fixedFontSize,'bond_width':options.bondLineWidth,'font_file':str(font),'font_sha256':digest(font),'nonchemical_legend':legend},
        'atom_order':'canonical_SMILES_with_observed_H','native_crop_admission':crop}


def main(a):
    RDLogger.DisableLog('rdApp.*');a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);(a.out/'images').mkdir()
    parent=json.loads(a.manifest.read_text());added=[];rejected=Counter()
    for row in parent['records']:
        if row['split']!='train' or row.get('parent_record_key') or row.get('visible_hydrogens'):continue
        try:
            item=render(row,a.out)
            if item:added.append(item)
        except (ValueError,RuntimeError) as error:rejected[str(error)]+=1
    records=parent['records']+added
    result={**parent,'records':records,'split_audit':assert_disjoint_splits(records),'visible_H_views':{
        'added':len(added),'by_domain':dict(Counter(r['domain'] for r in added)),'rejections':dict(rejected),
        'parent_manifest_sha256':digest(a.manifest),'generator_sha256':digest(__file__),'new_views_not_new_identities':True}}
    (a.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['visible_H_views']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
