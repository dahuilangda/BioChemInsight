"""Create a MolSight pilot corpus from joint residue poses, with named skips.

No circular/legacy fallback. This pilot is not a validated training release:
recognition accuracy and patent-domain style still require evaluation.
"""
from __future__ import annotations
import inspect
import argparse,csv,hashlib,json,math,random,signal,time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from rdkit import Chem
from residue_pose_layout import residue_pose_layout
from cyclopeptide2d import validate_input
from style_calibration import draw_molecule,ALIGNED
from render_via_engine import apply_patent_bw_palette
from render_stage11 import finish
from wedge_spec import plan_wedge_marks


class LayoutTimeout(Exception): pass

def _timeout(signum,frame): raise LayoutTimeout('layout_time_budget')


def validate_training_molecule(smiles):
    """Closed-shell, connected peptide inputs only; parsing is insufficient."""
    return validate_input(smiles)


def render_one(job):
    idx,smiles,out,seed,budget=job
    started=time.monotonic();record={'index':idx,'input_smiles':smiles,'seed':seed}
    signal.signal(signal.SIGALRM,_timeout);signal.alarm(budget)
    try:
        mol=validate_training_molecule(smiles)
        canonical=Chem.MolToSmiles(mol)
        laid,report=residue_pose_layout(mol,seed=seed)
        record['layout']=report
        if laid is None or not report.get('qa_pass'):
            record.update(status='skipped',reason='layout_constraint_failure')
        else:
            render=Chem.Mol(laid)
            np.random.seed(seed)
            image=draw_molecule(render,ALIGNED,random.Random(seed),opts_hook=apply_patent_bw_palette)
            if Chem.MolToSmiles(render)!=canonical:
                raise ValueError('renderer_changed_input_stereochemistry')
            bl=float(np.median([np.linalg.norm(np.array(render.GetConformer().GetAtomPosition(b.GetBeginAtomIdx()))-
                                                  np.array(render.GetConformer().GetAtomPosition(b.GetEndAtomIdx())))
                                   for b in render.GetBonds()]))
            marks,wedge_report=plan_wedge_marks(render,font_px=float(ALIGNED.get('fixed_font_size') or 16),
                page_scale=float(ALIGNED['fixed_bond_length']/bl))
            record['wedge_plan']=wedge_report
            if not wedge_report.get('combined_valid'):
                raise ValueError('unproven_rendered_stereo_marks')
            path=Path(out)/'images'/f'{idx:06d}.png'
            if not finish(image,random.Random(seed+811),str(path)):
                record.update(status='skipped',reason='ink_density_rejected')
            else:
                record.update(status='rendered_pilot',file_path=str(path.resolve()),SMILES=canonical,
                    image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    except LayoutTimeout:
        record.update(status='skipped',reason='layout_timeout')
    except Exception as exc:
        record.update(status='skipped',reason=type(exc).__name__+':'+str(exc))
    finally:
        signal.alarm(0)
        record['seconds']=time.monotonic()-started
    return record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True,help='CSV with SMILES column')
    ap.add_argument('--out',required=True)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--limit',type=int,default=8)
    ap.add_argument('--seed',type=int,default=20261108)
    ap.add_argument('--budget-seconds',type=int,default=180)
    args=ap.parse_args();out=Path(args.out);(out/'images').mkdir(parents=True,exist_ok=True)
    with open(args.input) as fh: source=list(csv.DictReader(fh))
    jobs=[(i,r['SMILES'],str(out),args.seed+i,args.budget_seconds)
          for i,r in enumerate(source[:args.limit])]
    records=[]
    with Pool(args.workers) as pool:
        for record in pool.imap_unordered(render_one,jobs):
            records.append(record)
            (out/f"record_{record['index']:06d}.json").write_text(json.dumps(record,indent=2)+'\n')
            print(record['index'],record['status'],record.get('reason',''),round(record['seconds'],2),flush=True)
    records.sort(key=lambda r:r['index'])
    kept=[r for r in records if r['status']=='rendered_pilot']
    with (out/'train_680k.csv').open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=['file_path','SMILES']);writer.writeheader()
        writer.writerows({k:r[k] for k in ['file_path','SMILES']} for r in kept)
    manifest={'engine':'residue_pose_layout','input':args.input,'jobs':len(jobs),'rendered':len(kept),
        'skipped':len(records)-len(kept),'training_release_approved':False,
        'remaining_validation':['patent style and glyph review','MolSight recognition evaluation','broader chemistry/seed coverage'],
        'engine_sha256':hashlib.sha256(Path(inspect.getsourcefile(residue_pose_layout)).read_bytes()).hexdigest(),
        'records':records}
    (out/'render_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('pilot kept',len(kept),'/',len(jobs),flush=True)

if __name__=='__main__':main()
