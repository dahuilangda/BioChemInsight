"""Prepare a component-separated, three-domain coordinate-verified pilot.

Training records are synthetic drawings of admitted TRAIN identities. Validation
uses only admitted VALIDATION identities. This is development data, not real-
patent generalization evidence. Dataset sizes are explicit CLI parameters.
"""
import argparse,hashlib,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from training.ocsr_reboot.protocol import assert_disjoint_splits,canonical_smiles
from .data import digest,graph_targets,target_prediction,drawing_quality
from .inference import decode_prediction


def render(row,domain,out,index,size,relative_typography=False):
    mol=Chem.MolFromSmiles(row['SMILES'])
    if mol is None or mol.GetNumAtoms()>512:raise ValueError('Invalid or oversized molecule')
    origins={'admitted_image':row['image_sha256']}
    if digest(row['file_path'])!=row['image_sha256']:raise ValueError('Source image changed')
    if domain=='small_molecule':rdDepictor.Compute2DCoords(mol,forceRDKit=True,useRingTemplates=True)
    else:
        drawing_path=Path(row['file_path']).parents[1]/'drawing.mol'
        drawing=Chem.MolFromMolFile(str(drawing_path),removeHs=False)
        if drawing is None:raise ValueError('Invalid source drawing')
        text=Chem.MolToSmiles(drawing,canonical=True,isomericSmiles=True)
        if text!=canonical_smiles(row['SMILES']):raise ValueError('Source drawing disagrees with admitted chemical identity')
        order=json.loads(drawing.GetProp('_smilesAtomOutputOrder'))
        mol=Chem.MolFromSmiles(text)
        if len(order)!=mol.GetNumAtoms():raise ValueError('Drawing atom count differs')
        def indexed(value):
            value=Chem.Mol(value)
            for k,atom in enumerate(value.GetAtoms()):atom.SetAtomMapNum(k+1)
            return Chem.MolToSmiles(value,canonical=True,isomericSmiles=True)
        if indexed(Chem.RenumberAtoms(drawing,order))!=indexed(mol):raise ValueError('Indexed source atom/stereo correspondence differs')
        row={**row,'source_label_smiles':row['SMILES'],'SMILES':text}
        conf=Chem.Conformer(mol.GetNumAtoms());conf.Set3D(False)
        for i,k in enumerate(order):conf.SetAtomPosition(i,drawing.GetConformer().GetAtomPosition(k))
        mol.AddConformer(conf,assignId=True)
        origins.update(drawing_mol=digest(drawing_path))
    for bond in mol.GetBonds():bond.SetBondDir(Chem.BondDir.NONE)
    Chem.SetDoubleBondNeighborDirections(mol)
    Chem.WedgeMolBonds(mol,mol.GetConformer())
    target=graph_targets(mol)
    drawer=rdMolDraw2D.MolDraw2DCairo(size,size)
    options=drawer.drawOptions();options.prepareMolsBeforeDrawing=False;options.useBWAtomPalette()
    options.fixedFontSize=14 if index%2 else 16;options.bondLineWidth=1.5 if index%2 else 2
    if relative_typography:
        positions=np.asarray(mol.GetConformer().GetPositions())[:,:2]
        lengths=[np.linalg.norm(positions[b.GetBeginAtomIdx()]-positions[b.GetEndAtomIdx()]) for b in mol.GetBonds()]
        estimated=(float(np.median(lengths))*size*.84/max(float(np.ptp(positions,axis=0).max()),1e-6) if lengths else 24.)
        options.fixedFontSize=int(np.clip(estimated*.5,8,48));options.bondLineWidth=float(np.clip(estimated*.035,.7,3.5))
        font=Path('/usr/share/fonts/truetype/liberation')/('LiberationSerif-Regular.ttf' if index%2 else 'LiberationSans-Regular.ttf')
        options.fontFile=str(font)
    options.padding=.06
    drawer.DrawMolecule(mol)
    coords=[[drawer.GetDrawCoords(i).x,drawer.GetDrawCoords(i).y] for i in range(mol.GetNumAtoms())]
    quality=drawing_quality(coords,mol,size)
    drawer.FinishDrawing()
    from training.molsight_dense.annotation import annotation_roundtrip
    predicted=annotation_roundtrip(target,np.array(coords)/size)
    if not predicted['parse_success'] or canonical_smiles(predicted['smiles'])!=canonical_smiles(row['SMILES']):
        raise ValueError('Visible graph/stereo roundtrip mismatch: '+str(predicted.get('error')))
    directory=out/'images';directory.mkdir(exist_ok=True)
    key=hashlib.sha256((row['record_key']+':spatial-v2-render').encode()).hexdigest()
    image=directory/(key+'.png');image.write_bytes(drawer.GetDrawingText())
    target_path=directory/(key+'.json');target_path.write_text(json.dumps({k:v.tolist() for k,v in target.items()})+'\n')
    return {'record_key':key,'component':row['component'],'split':row['split'],'domain':domain,
        'category':'complete' if domain=='small_molecule' else 'peptide','file_path':str(image),
        'image_sha256':digest(image),'SMILES':row['SMILES'],'coords_pixels':coords,
        'target_path':str(target_path),'target_sha256':digest(target_path),'source_record_key':row['record_key'],
        'source_file_path':row['file_path'],'source_hashes':origins,'synthetic_render':True,
        'peptide_stratum':row.get('peptide_stratum'),'native_size':size,'drawing_quality':quality,
        'style':{'font_size':options.fixedFontSize,'bond_width':options.bondLineWidth,'relative_typography':relative_typography}}


def candidates(manifest,split,desired,max_atoms=512):
    path=manifest['files'][split]['path']
    if digest(path)!=manifest['files'][split]['sha256']:raise ValueError('Arrow release changed')
    table=ipc.open_file(pa.memory_map(path)).read_all()
    pools=defaultdict(list);seen=set()
    fields=['record_key','component','category','SMILES','file_path','image_sha256','peptide_stratum']
    for batch in table.select(fields).to_batches(max_chunksize=1024):
        for row in batch.to_pylist():
            row['split']=split
            if row['category'] not in ('complete','peptide'):continue
            if row['category']=='peptide':
                stratum=row['peptide_stratum'] or ''
                domain='linear_peptide' if stratum.startswith('linear') or stratum=='branched' else 'cyclic_peptide'
                pools[domain+':'+stratum].append(row);continue
            mol=Chem.MolFromSmiles(row['SMILES'])
            if mol is None or not 1<=mol.GetNumAtoms()<=max_atoms or any(a.GetAtomicNum()==0 for a in mol.GetAtoms()):continue
            try:identity=canonical_smiles(row['SMILES'])
            except ValueError:continue
            if identity in seen:continue
            seen.add(identity)
            atoms=mol.GetNumAtoms()
            flags=['size:'+('small' if atoms<25 else 'medium' if atoms<50 else 'large' if atoms<120 else 'macro' if atoms<=256 else 'very_large')]
            if any(a.GetIsotope() for a in mol.GetAtoms()):flags.append('isotope')
            if any(a.GetFormalCharge() for a in mol.GetAtoms()):flags.append('charge')
            if '.' in row['SMILES']:flags.append('salt')
            if '@' in row['SMILES']:flags.append('chiral')
            if '/' in row['SMILES'] or '\\' in row['SMILES']:flags.append('double_stereo')
            if any(a.GetAtomicNum()==53 for a in mol.GetAtoms()):flags.append('iodine')
            if any(a.GetAtomicNum()==35 for a in mol.GetAtoms()):flags.append('bromine')
            for flag in flags:
                pool=pools['small_molecule:'+flag];pool.append(row)
                if len(pool)>desired*4:pool.sort(key=lambda r:r['record_key']);del pool[desired*3:]
    grouped=defaultdict(list)
    for key,pool in sorted(pools.items()):
        pool.sort(key=lambda r:r['record_key']);grouped[key.split(':')[0]].append(pool)
    result={}
    for domain,lists in grouped.items():
        ordered=[];keys=set()
        for index in range(max(map(len,lists))):
            for pool in lists:
                if index<len(pool) and pool[index]['record_key'] not in keys:
                    ordered.append(pool[index]);keys.add(pool[index]['record_key'])
        result[domain]=ordered
    return result


def main(args):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.source.read_text());records=[];rejected=Counter();identities=set()
    for split,count in [('train',args.train_per_domain),('validation',args.validation_per_domain)]:
        pools=candidates(manifest,split,count,args.max_atoms)
        for domain in ['small_molecule','linear_peptide','cyclic_peptide']:
            accepted=0
            for row in pools[domain]:
                try:
                    identity=canonical_smiles(row['SMILES'])
                    if identity in identities:continue
                    item=render(row,domain,out,len(records),args.size)
                    records.append(item);identities.add(identity);accepted+=1
                    print(json.dumps({'split':split,'domain':domain,'accepted':accepted,'atoms':len(item['coords_pixels'])}),flush=True)
                    if accepted>=count:break
                except Exception as error:rejected[type(error).__name__+': '+str(error)[:100]]+=1
            if accepted!=count:raise RuntimeError(f'Insufficient admitted {split}/{domain}: {accepted}/{count}; {dict(rejected)}')
    splits=assert_disjoint_splits(records)
    document={'schema_version':'molsight_spatial_data_v1','training_release_approved':True,
        'source_manifest':str(args.source),'source_manifest_sha256':digest(args.source),'records':records,
        'split_audit':splits,'rejections':dict(rejected),'source_code_sha256':{str(p):digest(p) for p in Path(__file__).parent.glob('*.py')},
        'limitations':['Development pilot; all new drawings synthetic, real patent benchmark stays reserved.','No claim of real cyclic-peptide generalization or sufficient domain scale.']}
    (out/'manifest.json').write_text(json.dumps(document,indent=2)+'\n')
    print(json.dumps({'manifest':str(out/'manifest.json'),'counts':dict(Counter((r['split']+':'+r['domain']) for r in records))}))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-atoms',type=int,default=512)
    parser.add_argument('--train-per-domain',type=int,default=32);parser.add_argument('--validation-per-domain',type=int,default=8);parser.add_argument('--size',type=int,default=768)
    main(parser.parse_args())
