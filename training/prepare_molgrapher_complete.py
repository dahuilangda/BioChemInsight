"""Admit existing synthetic images using their measured keypoints, never redraw.

Reject explicit separately drawn H (current SMILES parser removes those atoms),
variables, inconsistent MOL/SMILES, non-similarity coordinate maps and any
stereochemical roundtrip error. Public real test molecules are exclusion-only.
"""
import argparse,hashlib,io,json
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from rdkit import Chem,RDLogger
from training.molsight_spatial.data import digest,graph_targets,drawing_quality,target_prediction
from training.molsight_spatial.inference import decode_prediction
from training.ocsr_reboot.protocol import canonical_smiles,assert_disjoint_splits


def admit(row,out,include_hydrogens=False):
    molecule=Chem.MolFromMolBlock(row['mol'],removeHs=False)
    if molecule is None:raise ValueError('invalid_mol')
    # MOL parsing turns visible wedge directions into tetrahedral atom tags;
    # restore the original recorded wedges, never choose new drawing bonds.
    Chem.ReapplyMolBlockWedging(molecule)
    if any(a.GetAtomicNum()==0 or (a.GetAtomicNum()==1 and not include_hydrogens) for a in molecule.GetAtoms()):raise ValueError('variable_or_separate_H')
    if not 5<=molecule.GetNumAtoms()<=192:raise ValueError('size')
    peptide=Chem.MolFromSmarts('[NX3]-[CX4]-[CX3](=[OX1])-[NX3]')
    if len(molecule.GetSubstructMatches(peptide))>=2:raise ValueError('peptide_outside_small_molecule_stratum')
    if canonical_smiles(Chem.MolToSmiles(molecule))!=canonical_smiles(row['smiles']):raise ValueError('reference_disagreement')
    points=np.asarray(row['keypoints']).reshape(-1,3)
    if len(points)!=molecule.GetNumAtoms() or not (points[:,2]==2).all():raise ValueError('keypoint_count_or_visibility')
    xy=points[:,:2].astype(float);layout=molecule.GetConformer().GetPositions()[:,:2]
    design=np.c_[layout,np.ones(len(layout))];fit=np.linalg.lstsq(design,xy,rcond=None)[0]
    error=float(np.linalg.norm(design@fit-xy,axis=1).max())
    scales=np.linalg.svd(fit[:2],compute_uv=False)
    if error>1.5 or scales[-1]<=0 or scales[0]/scales[-1]>1.02 or np.linalg.det(fit[:2])>=0:
        raise ValueError('keypoints_not_same_drawing_geometry')
    with Image.open(io.BytesIO(row['image']['bytes'])) as image:
        width,height=image.size
        if width!=height:raise ValueError('nonsquare_source')
    if (xy<0).any() or (xy>=width).any():raise ValueError('bounds')
    text=Chem.MolToSmiles(molecule,canonical=True,isomericSmiles=True)
    order=json.loads(molecule.GetProp('_smilesAtomOutputOrder'))
    ordered=Chem.RenumberAtoms(molecule,order);xy=xy[order]
    options=Chem.SmilesParserParams();options.removeHs=not include_hydrogens
    parsed=Chem.MolFromSmiles(text,options)
    if [a.GetAtomicNum() for a in parsed.GetAtoms()]!=[a.GetAtomicNum() for a in ordered.GetAtoms()]:raise ValueError('atom_mapping')
    targets=graph_targets(ordered);quality=drawing_quality(xy,ordered,width)
    from training.molsight_dense.annotation import annotation_roundtrip
    result=annotation_roundtrip(targets,xy/width)
    if not result['parse_success'] or canonical_smiles(result['smiles'])!=canonical_smiles(text):raise ValueError('image_stereo_roundtrip')
    key='molgrapher:'+str(row['id']);stem=hashlib.sha256(key.encode()).hexdigest();image=out/'images'/(stem+'.png')
    image.write_bytes(row['image']['bytes']);target=image.with_suffix('.json');target.write_text(json.dumps({k:v.tolist() for k,v in targets.items()})+'\n')
    return {'record_key':key,'component':canonical_smiles(text),'SMILES':text,'domain':'small_molecule','split':'train',
        'file_path':str(image),'image_sha256':digest(image),'native_size':width,'coords_pixels':xy.tolist(),
        'target_path':str(target),'target_sha256':digest(target),'drawing_quality':quality,
        'source':'molgrapher_synthetic_300k','source_id':row['id'],'source_mol_sha256':hashlib.sha256(row['mol'].encode()).hexdigest(),
        'keypoint_geometry_max_error_pixels':error,'coordinate_scope':'source_measured_visible_atoms',
        'visible_hydrogens':any(a.GetAtomicNum()==1 for a in ordered.GetAtoms()),
        'chemical_atom_count':Chem.RemoveHs(ordered).GetNumAtoms()}


def main(args):
    RDLogger.DisableLog('rdApp.*');args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False);(args.out/'images').mkdir()
    parent=json.loads(args.manifest.read_text());blocked={canonical_smiles(r['SMILES']) for r in parent['records']}
    blocked.update(canonical_smiles(r['SMILES']) for r in json.loads(args.exclude_pairs.read_text()))
    records=[];rejected=Counter();scanned=0;shards={}
    for path in sorted(args.source.glob('*.parquet')):
        shard_start=len(records)
        shards[str(path)]=digest(path)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                scanned+=1
                try:
                    identity=canonical_smiles(row['smiles'])
                    if identity in blocked:raise ValueError('excluded_or_duplicate_identity')
                    item=admit(row,args.out,args.include_visible_hydrogens);blocked.add(identity);records.append(item)
                except (ValueError,RuntimeError,IndexError) as error:rejected[str(error)]+=1
                if len(records)>=args.count or (args.shard_quota and len(records)-shard_start>=args.shard_quota):break
            if len(records)>=args.count or (args.shard_quota and len(records)-shard_start>=args.shard_quota):break
        if len(records)>=args.count:break
    if len(records)<args.count:raise ValueError('Insufficient admitted examples')
    all_records=parent['records']+records
    result={**parent,'records':all_records,'split_audit':assert_disjoint_splits(all_records),
        'parent_manifest_sha256':digest(args.manifest),'external_admission':{'added_identities':len(records),'scanned':scanned,
        'rejections':dict(rejected),'shard_sha256':shards,'test_exclusion_sha256':digest(args.exclude_pairs),
        'generator_sha256':digest(__file__),'annotation_authority_sha256':digest('training/molsight_dense/annotation.py'),
        'visible_hydrogens_allowed':args.include_visible_hydrogens,'per_shard_admission_quota':args.shard_quota,
        'test_used_for_training':False,'synthetic_not_real':True}}
    (args.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['external_admission']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','source','exclude-pairs','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--count',type=int,default=2000);p.add_argument('--shard-quota',type=int,default=0)
    p.add_argument('--include-visible-hydrogens',action='store_true');main(p.parse_args())
