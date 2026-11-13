from collections import Counter
import pyarrow as pa
from molsight_multitask.replay import DomainReplaySampler


class Data:
    split='train'
    max_len=2048
    specification={'files':{'train':{'sha256':'frozen'}}}
    def __init__(self):
        self.table=pa.Table.from_pylist([
            {'category': category,'peptide_stratum':group,'input_size':size,'token_ids':tokens}
            for category,group,size,tokens,count in [
                ('complete',None,512,[1,3,2],8),('peptide','linear',512,[1,3,2],8),
                ('peptide','cyclic',512,[1,4,2],8),('attachment',None,512,[1,3,2],8),
                ('peptide','bad_resolution',768,[1,3,2],8),('peptide','bad_vocab',512,[1,99,2],8)]
            for _ in range(count)])
    def _open(self):pass


def sample(dataset,category,**kwargs):
    return DomainReplaySampler(dataset,category,vocabulary_size=5,updates=100,batch_size=1,world_size=2,**kwargs)


def test_domain_replay_is_native_compatible_and_resumes_exactly():
    data=Data();rows=data.table.to_pylist();batches=list(sample(data,'peptide'))
    assert list(sample(data,'peptide',start_update=30))==batches[30:]
    counts=Counter(rows[b[0]]['peptide_stratum'] for b in batches)
    assert set(counts)=={'linear','cyclic'} and all(30<count<70 for count in counts.values())
    assert all(rows[b[0]]['input_size']==512 for b in batches)


def test_replay_ranks_share_definition_without_duplicate_images():
    data=Data();first,second=sample(data,'complete',rank=0),sample(data,'complete',rank=1)
    assert first.definition_sha256==second.definition_sha256
    assert all(set(left).isdisjoint(right) for left,right in zip(first,second,strict=True))
    assert sample(data,'attachment').definition_sha256!=first.definition_sha256
