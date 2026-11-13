import copy
import pytest
from torch import nn
from molsight_multitask.train import parameter_groups, validate_configuration


def config():
    return dict(manifest='manifest', checkpoint='checkpoint', train_sidecar='train', validation_sidecar='val',
        out='out', updates=500, warmup=50, batch_size=1, accumulation=8, lr=7e-6, encoder_lr=5e-7,
        head_lr=7e-5, edge_weight=0.1, edge_warmup=50, weights={'complete':.25,'peptide':.75},
        peptide_group_weights={},stratum_weights={},seed=42, workers=3, eval_every=125, save_every=125,
        eval_batch_size=1, max_len=2048, max_grad_norm=1.,weight_decay=1e-4,retention_weight=1.,
        retention_supervised_weight=.25,retention_temperature=2.,retention_batch_size=1,retention_tolerance=.03,retention_domains={'complete':.5,'peptide':.35,'attachment':.15},excluded_domains=['markush','attachment'])


@pytest.mark.parametrize('key,value', [('updates', 0),('warmup',501),('edge_weight',0),
    ('encoder_lr',-1),('head_lr',float('nan')),('workers',-1),('batch_size',1.5)])
def test_invalid_experiment_cannot_reach_cuda_training(key,value):
    values=config();values[key]=value
    with pytest.raises(ValueError):validate_configuration(values)


def test_parameter_groups_cover_old_and_new_trainable_weights_once():
    values=config();validate_configuration(values)
    model=nn.Module();model.image_encoder=nn.Linear(2,2);model.decoder=nn.Module()
    model.decoder.old=nn.Linear(2,2);model.decoder.edge_predictor=nn.Linear(4,7)
    groups=parameter_groups(model,values,False)
    assert {g['name']:g['lr'] for g in groups}=={'decoder':7e-6,'encoder':5e-7,'chemical_edge_head':7e-5}
    identifiers=[id(p) for g in groups for p in g['params']]
    assert len(identifiers)==len(set(identifiers))==len(list(model.parameters()))
    assert all(g['lr']==0 for g in parameter_groups(model,values,True))
    model.image_encoder.requires_grad_(False)
    assert 'encoder' not in {g['name'] for g in parameter_groups(model,values,False)}
