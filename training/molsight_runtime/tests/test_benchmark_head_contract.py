import pytest
from benchmark import checkpoint_formats


def test_character_checkpoint_retains_original_behavior():
    assert checkpoint_formats({}, {'decoder.blocks.0.weight': None}) == ['char']


def test_trained_graph_head_is_kept_for_strict_loading():
    state={'decoder.edge_predictor.mlp.0.weight':None}
    checkpoint={'plan':{'configuration':{'formats':['char','edges']}}}
    assert checkpoint_formats(checkpoint,state)==['char','edges']


def test_contradictory_head_metadata_is_rejected():
    with pytest.raises(ValueError,match='contradict'):
        checkpoint_formats({'plan':{'configuration':{'formats':['char']}}},
                           {'decoder.edge_predictor.mlp.0.weight':None})


def test_unconfigured_coordinate_head_cannot_be_silently_dropped():
    with pytest.raises(ValueError,match='Coordinate'):
        checkpoint_formats({}, {'decoder.loc_predictor.weight':None})
