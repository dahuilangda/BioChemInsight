"""Image-only decoding with optional aligned graph predictions."""

import torch
from torch.nn.utils.rnn import pad_sequence

from .decoding import GreedyDecoder
from .model import get_edge_prediction
from .policy import frozen_forward, policy_mode, sample_completions
from .tokenizer import EOS_ID, PAD_ID


def predict_images(model, images, *, formats=("char",)):
    policy = getattr(model, "module", model)
    with policy_mode(policy, adapters=None), frozen_forward(images.device):
        predictions, intermediate = sample_completions(
            policy,
            images,
            None,
            GreedyDecoder(temperature=0, eot=EOS_ID),
            1,
        )
        predictions["terminated"] = (
            intermediate["raw_tokens"].eq(EOS_ID).any(dim=1).tolist()
        )
        predictions.update(edges=None, edge_scores=None, coords=None)
        if not set(formats) & {"edges", "coords"}:
            return predictions
        lengths = [len(indices) for indices in predictions["indices"]]
        count = len(lengths)
        if not max(lengths, default=0):
            for name in ("edges", "edge_scores", "coords"):
                if name in formats or name == "edge_scores" and "edges" in formats:
                    predictions[name] = [[] for _ in range(count)]
            return predictions
        hidden = intermediate["hidden_states"]
        features = intermediate["image_features"]
        indices = (
            pad_sequence(
                [
                    torch.tensor(values, dtype=torch.long)
                    for values in predictions["indices"]
                ],
                batch_first=True,
                padding_value=PAD_ID,
            ).to(hidden.device)
            + policy.sample_begin
        )
        if "edges" in formats:
            if policy.decoder.edge_predictor is None:
                raise ValueError("The checkpoint has no edge predictor")
            logits = policy.decoder.edge_predictor(hidden, indices)
            predictions["edges"], predictions["edge_scores"] = get_edge_prediction(
                logits.softmax(dim=-1),
                lengths,
            )
        if "coords" in formats:
            predictor = policy.decoder.loc_predictor
            if predictor is None:
                raise ValueError("The checkpoint has no coordinate predictor")
            positions = None
            if hasattr(policy.decoder, "embed_cos"):
                length = hidden.shape[1]
                positions = (
                    policy.decoder.embed_cos[:, :length].to(features.dtype),
                    policy.decoder.embed_sin[:, :length].to(features.dtype),
                )
            for block in getattr(predictor, "loc_blocks", ()):
                hidden = block(hidden, features, position_embeddings=positions)
            coordinates = predictor(hidden, indices)[0]
            predictions["coords"] = [
                coordinates[index, :length].tolist()
                for index, length in enumerate(lengths)
            ]
    return predictions
