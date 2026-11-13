"""Freeze additional real patent peptide candidates, never training data."""
import argparse,hashlib,json,re
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq
from rdkit import Chem,RDLogger
from training.ocsr_reboot.protocol import canonical_smiles
from training.molsight_spatial.data import digest
from training.molsight_hybrid import peptide_graph,cyclic_peptide_graph


def macro_peptide_candidate(text):
    return cyclic_peptide_graph(text,min_ring_size=12)


def main(a):
    RDLogger.DisableLog('rdApp.*');a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);(a.out/'images').mkdir()
    manifests=[json.loads(p.read_text()) for p in a.exclude_manifest]
    blocked={canonical_smiles(r['SMILES']) for m in manifests for r in m['records']}
    blocked.update(canonical_smiles(r['SMILES']) for r in json.loads(a.exclude_pairs.read_text()))
    rows=[];counts=Counter();seen=set()
    for batch in pq.ParquetFile(a.parquet).iter_batches(batch_size=128):
        for r in batch.to_pylist():
            counts['all_source_images']+=1
            try:text=canonical_smiles(r['SMILES'])
            except ValueError:counts['invalid_or_variable_label']+=1;continue
            if not peptide_graph(text):continue
            counts['peptide_candidates_before_exclusion']+=1
            if text in blocked:counts['excluded_identity']+=1;continue
            if text in seen:counts['duplicate_identity']+=1;continue
            seen.add(text);cyclic=cyclic_peptide_graph(text);counts['cyclic_candidates']+=int(cyclic)
            key='uspto10k:'+r['image_id'];stem=hashlib.sha256(key.encode()).hexdigest();p=a.out/'images'/(stem+'.png');p.write_bytes(r['image']['bytes'])
            rows.append({'record_key':key,'component':'uspto10k:'+re.sub(r'-C\d+$','',r['image_id']),
                'file_path':str(p),'image_sha256':digest(p),'SMILES':text,'domain':'real_peptide_candidate','category':'complete',
                'peptide_like_candidate':True,'cyclic_peptide_like_candidate':cyclic,'source_image_id':r['image_id'],
                'source':'Keylab/OCSR-Benchmarks/USPTO-10K','source_revision':'2fdcdc59067acb36d84ddc82c1ace24c91d5ad71',
                'atoms':Chem.MolFromSmiles(text).GetNumAtoms()})
    if not rows:raise ValueError('Empty additional benchmark')
    (a.out/'pairs.json').write_text(json.dumps(rows,indent=2)+'\n')
    macro=[row for row in rows if macro_peptide_candidate(row['SMILES'])]
    (a.out/'macro_pairs.json').write_text(json.dumps(macro,indent=2)+'\n')
    plan={'counts':dict(counts),'admitted':len(rows),'source_sha256':digest(a.parquet),'pairs_sha256':digest(a.out/'pairs.json'),
        'exclusion_manifest_sha256':[digest(p) for p in a.exclude_manifest],'exclude_pairs_sha256':digest(a.exclude_pairs),
        'candidate_selector_sha256':digest('training/molsight_hybrid.py'),'generator_sha256':digest(__file__),
        'scope':'Real patent image candidates selected structurally; not manually verified peptide annotations. No training use.',
        'historical_pretraining_overlap':'unknown','family_separation_from_training':'unproven',
        'macro_subset':{'records':len(macro),'components':len({r['component'] for r in macro}),
            'pairs_sha256':digest(a.out/'macro_pairs.json'),'criterion':'ring >=12, >=2 distinct ring amide carbonyls and >=2 ring backbone alphaC motifs'}}
    (a.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parquet','exclude-pairs','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--exclude-manifest',type=Path,action='append',required=True);main(p.parse_args())
