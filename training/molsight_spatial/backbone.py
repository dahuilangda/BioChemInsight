"""Retain the actual MolSight model and decoder; initialize only the spatial head."""
from types import SimpleNamespace
from pathlib import Path
import json
import torch
from torch import nn


class MolsightSpatial(nn.Module):
    def __init__(self, checkpoint, vocabulary, detector_config=None):
        super().__init__()
        from molsight.model import MolsightModel
        from molsight.tokenizer import CharTokenizer
        from molsight.checkpoint import load_parameters, model_state
        from .model import SpatialDetector, SpatialDetectorConfig
        initial = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        self.tokenizer = CharTokenizer(vocabulary)
        if initial['vocabulary'] != self.tokenizer.stoi:
            raise ValueError('MolSight vocabulary must remain byte/ID compatible')
        configuration = dict(initial['plan']['configuration'])
        configuration.update(use_checkpoint=True, compile_layers=False, resume=True)
        self.backbone = MolsightModel(SimpleNamespace(**configuration), self.tokenizer)
        load_parameters(self.backbone, model_state(initial), source_vocabulary=initial['vocabulary'],
                        target_vocabulary=self.tokenizer.stoi)
        # The old teacher-forced edge head is retained intact, but unused here.
        for name in ['edge_predictor', 'loc_predictor']:
            head = getattr(self.backbone.decoder, name, None)
            if head is not None:
                head.requires_grad_(False)
        self.detector = SpatialDetector(detector_config or SpatialDetectorConfig())
        self.configuration = configuration

    def forward(self, images, *, dn_points=None):
        features = self.backbone.get_image_features(images)
        return self.detector(features, dn_points=dn_points)

    @torch.no_grad()
    def predict_image(self, image, *, objectness_threshold=.5):
        from .inference import decode_prediction
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError('One image required; no teacher inputs are accepted')
        previous = self.training
        self.eval()
        try:
            return decode_prediction(self(image), objectness_threshold=objectness_threshold)
        finally:
            self.train(previous)
