"""Add the largest source-bound cyclic peptides without touching a frozen pilot."""
import argparse,json,signal
from collections import Counter
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
from rdkit import Chem,RDLogger
from training.ocsr_reboot.protocol import canonical_smiles,assert_disjoint_splits
from .prepare import render
from .data import digest


def main(args):
    RDLogger.DisableLog('rdApp.*')
    original=json.loads(args.manifest.read_text());records=list(original['records'])
    source_path=Path(original['source_manifest']);source=json.loads(source_path.read_text())
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    seen={canonical_smiles(r['SMILES']) for r in records};rejected=Counter()
    for split in ['train','validation']:
        arrow=source['files'][split]['path']
        if digest(arrow)!=source['files'][split]['sha256']:raise ValueError('Source Arrow changed')
        table=ipc.open_file(pa.memory_map(arrow)).read_all()
        selected=table.filter(pc.equal(table['peptide_stratum'],'head_to_tail_patent_large'))
        rows=selected.select(['record_key','component','category','SMILES','file_path','image_sha256','peptide_stratum']).to_pylist()
        print(json.dumps({'split':split,'candidate_count':len(rows),'stage':'sorting_sizes'}),flush=True)
        rows.sort(key=lambda r:(-Chem.MolFromSmiles(r['SMILES']).GetNumAtoms(),r['record_key']))
        count=0
        for row in rows:
            identity=canonical_smiles(row['SMILES'])
            if identity in seen:continue
            row['split']=split
            try:
                def expired(*_):raise TimeoutError('Bounded source graph correspondence timed out')
                previous=signal.signal(signal.SIGALRM,expired);signal.alarm(10)
                print(json.dumps({'split':split,'candidate':row['record_key'],'atoms':Chem.MolFromSmiles(row['SMILES']).GetNumAtoms()}),flush=True)
                result=render(row,'cyclic_peptide',out,len(records),768)
                records.append(result);seen.add(identity);count+=1
                print(json.dumps({'split':split,'largest_atoms':len(result['coords_pixels'])}),flush=True)
                if count==2:break
            except Exception as error:rejected[str(error)[:120]]+=1
            finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        if count!=2:raise ValueError('Could not admit largest cyclic records: '+str(rejected))
    original.update(records=records,split_audit=assert_disjoint_splits(records),
        parent_manifest_sha256=digest(args.manifest),large_extension_rejections=dict(rejected),
        source_code_sha256={str(p):digest(p) for p in Path(__file__).parent.glob('*.py')})
    (out/'manifest.json').write_text(json.dumps(original,indent=2)+'\n')
    print(json.dumps({'manifest':str(out/'manifest.json'),'counts':dict(Counter(r['split']+':'+r['domain'] for r in records)),'max_atoms':max(len(r['coords_pixels']) for r in records)}))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);main(parser.parse_args())
