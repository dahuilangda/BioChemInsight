"""Add all eligible admitted TRAIN peptides with measured redraw coordinates."""
import argparse,json
from pathlib import Path
from collections import Counter
import pyarrow as pa,pyarrow.compute as pc,pyarrow.ipc as ipc
from rdkit import RDLogger
from training.molsight_spatial.prepare import render
from training.molsight_spatial.data import digest
from training.ocsr_reboot.protocol import canonical_smiles,assert_disjoint_splits


def main(a):
    RDLogger.DisableLog('rdApp.*');a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    parent=json.loads(a.manifest.read_text());release=json.loads(a.source.read_text());source=Path(release['files']['train']['path'])
    if digest(source)!=release['files']['train']['sha256']:raise ValueError('Admitted Arrow source changed')
    seen={canonical_smiles(r['SMILES']) for r in parent['records']};blocked={canonical_smiles(r['SMILES']) for r in json.loads(a.exclude_pairs.read_text())}
    arrow=ipc.open_file(pa.memory_map(str(source)));records=list(parent['records']);counts=Counter();rejected=Counter()
    for batch_index in range(arrow.num_record_batches):
        batch=arrow.get_batch(batch_index);batch=batch.filter(pc.equal(batch.column(batch.schema.get_field_index('category')),'peptide'))
        fields=['record_key','component','category','SMILES','file_path','image_sha256','peptide_stratum']
        for row in batch.select(fields).to_pylist():
            identity=canonical_smiles(row['SMILES'])
            if identity in seen or identity in blocked:continue
            domain='linear_peptide' if (row['peptide_stratum'] or '').startswith('linear') or row['peptide_stratum']=='branched' else 'cyclic_peptide'
            try:
                result=render({**row,'split':'train'},domain,a.out,len(records),1024,relative_typography=True)
                records.append(result);seen.add(identity);counts[domain]+=1
            except (ValueError,RuntimeError) as error:rejected[str(error)]+=1
            if sum(counts.values()) and sum(counts.values())%100==0:print(json.dumps({'added':dict(counts),'rejected':sum(rejected.values())}),flush=True)
    result={**parent,'records':records,'split_audit':assert_disjoint_splits(records),'peptide_extension':{
        'added':dict(counts),'rejections':dict(rejected),'admitted_source_manifest_sha256':digest(a.source),
        'parent_manifest_sha256':digest(a.manifest),'reserved_exclusion_sha256':digest(a.exclude_pairs),
        'generator_sha256':digest(__file__),'render_authority_sha256':digest('training/molsight_spatial/prepare.py'),
        'annotation_authority_sha256':digest('training/molsight_dense/annotation.py'),'all_new_drawings_synthetic':True}}
    (a.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['peptide_extension']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','source','exclude-pairs','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
