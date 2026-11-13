"""Render chemically identical training views with Kekule rings and literal methyl labels.

Validation images are kept byte-identical. All labels come from the actual
rotated drawing, not from an imagined sequence or projected source coordinates.
"""
import argparse,json,hashlib,math
from collections import Counter
from pathlib import Path
import numpy as np
from rdkit import Chem,RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
from training.molsight_spatial.data import digest,graph_targets,target_prediction,drawing_quality
from training.molsight_spatial.inference import decode_prediction
from training.ocsr_reboot.protocol import canonical_smiles,assert_disjoint_splits


def render(record,out,variant):
    seed=int.from_bytes(hashlib.sha256((record['record_key']+str(variant)).encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed);mol=Chem.MolFromSmiles(record['SMILES'])
    if digest(record['file_path'])!=record['image_sha256']:raise ValueError('Parent image changed')
    xy=np.array(record['coords_pixels'],float);xy[:,1]*=-1;xy-=xy.mean(0)
    angle=float(rng.uniform(-math.pi,math.pi));matrix=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    xy=xy@matrix.T
    conformer=Chem.Conformer(len(xy));conformer.Set3D(False)
    for i,(x,y) in enumerate(xy):conformer.SetAtomPosition(i,(float(x),float(y),0.))
    mol.AddConformer(conformer)
    for bond in mol.GetBonds():bond.SetBondDir(Chem.BondDir.NONE)
    Chem.SetDoubleBondNeighborDirections(mol);Chem.WedgeMolBonds(mol,mol.GetConformer())
    targets=graph_targets(mol);depiction=Chem.Mol(mol)
    kekule=variant%2==0
    if kekule:Chem.Kekulize(depiction,clearAromaticFlags=False)
    size=record['native_size'];drawer=rdMolDraw2D.MolDraw2DCairo(size,size);options=drawer.drawOptions()
    options.prepareMolsBeforeDrawing=False;options.useBWAtomPalette();options.padding=.07
    # Patent lettering varies independently of chemical identity. Alternate
    # installed serif/sans fonts; record exact bytes for reproducibility.
    font=Path('/usr/share/fonts/truetype/liberation')/('LiberationSerif-Regular.ttf' if variant%2 else 'LiberationSans-Regular.ttf')
    if not font.is_file():raise ValueError('Required depiction font missing: '+str(font))
    options.fontFile=str(font)
    parent_bond=record.get('drawing_quality',{}).get('median_bond_pixels') or 20.
    options.fixedFontSize=int(np.clip(parent_bond*rng.uniform(.35,.75),8,56))
    options.bondLineWidth=float(np.clip(parent_bond*rng.uniform(.025,.055),.7,4.))
    methyl_labels={}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum()==6 and atom.GetDegree()==1 and atom.GetTotalNumHs()==3 and not atom.GetIsotope() and not atom.GetFormalCharge():
            if rng.random()<.5:
                label='Me' if variant%2 else 'CH3';options.atomLabels[atom.GetIdx()]=label;methyl_labels[atom.GetIdx()]=label
    legend=['(I)','(II)','Compound 12','Example 7'][variant%4]
    drawer.DrawMolecule(depiction,legend=legend);drawer.FinishDrawing()
    coords=[[drawer.GetDrawCoords(i).x,drawer.GetDrawCoords(i).y] for i in range(len(xy))]
    quality=drawing_quality(coords,mol,size)
    if not np.isfinite(coords).all() or (np.asarray(coords)<0).any() or (np.asarray(coords)>=size).any():
        raise ValueError('Styled observed atom outside canvas')
    result=decode_prediction(target_prediction(targets,np.array(coords)/size))
    if not result['parse_success'] or canonical_smiles(result['smiles'])!=canonical_smiles(record['SMILES']):
        raise ValueError('Styled visible stereochemistry failed roundtrip: '+str(result.get('error')))
    key=hashlib.sha256((record['record_key']+':patent-type-style-v2:'+str(variant)).encode()).hexdigest()
    image=out/'images'/(key+'.png');image.parent.mkdir(exist_ok=True);image.write_bytes(drawer.GetDrawingText())
    target=image.with_suffix('.json');target.write_text(json.dumps({k:v.tolist() for k,v in targets.items()})+'\n')
    return {**record,'record_key':key,'file_path':str(image),'image_sha256':digest(image),'coords_pixels':coords,
        'target_path':str(target),'target_sha256':digest(target),'drawing_quality':quality,
        'style':{'rotation_radians':angle,'kekule':kekule,'font_size':options.fixedFontSize,'bond_width':options.bondLineWidth,'literal_methyl_labels':methyl_labels,
            'font_file':str(font),'font_sha256':digest(font),'nonchemical_legend':legend},
        'parent_record_key':record['record_key'],'parent_image_sha256':record['image_sha256']}


def main(args):
    RDLogger.DisableLog('rdApp.*');args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    parent=json.loads(args.manifest.read_text());records=list(parent['records']);rejected=[]
    for row in parent['records']:
        if row['split']!='train':continue
        for variant in range(args.views):
            try:records.append(render(row,args.out,variant))
            except ValueError as error:rejected.append({'parent':row['record_key'],'variant':variant,'error':str(error)})
        if len(records)%100<args.views:print(json.dumps({'views':len(records),'rejected':len(rejected)}),flush=True)
    document={**parent,'records':records,'parent_manifest':str(args.manifest),'parent_manifest_sha256':digest(args.manifest),
        'split_audit':assert_disjoint_splits(records),'style_rejections':rejected,'style_generator_sha256':digest(__file__),
        'training_release_approved':True,'views_are_not_independent_identities':True}
    (args.out/'manifest.json').write_text(json.dumps(document,indent=2)+'\n')
    print(json.dumps({'counts':dict(Counter(r['split']+':'+r['domain'] for r in records)),'rejected':len(rejected),'validation_images_changed':0}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--views',type=int,default=2)
    main(parser.parse_args())
