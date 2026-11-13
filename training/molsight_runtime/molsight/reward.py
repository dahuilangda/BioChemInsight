from collections import Counter

import torch
from rdkit import Chem, DataStructs

from molsight.structure import StructureError, parse_structure


def _pairs(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("Prediction and target counts differ")
    for index, (prediction, target) in enumerate(zip(predictions, targets)):
        try:
            gold = parse_structure(target)
        except StructureError as error:
            raise ValueError(f"Invalid target structure at index {index}") from error
        try:
            predicted = parse_structure(prediction)
        except StructureError:
            predicted = None
        yield predicted, gold


def _label_agreement(predicted, gold):
    first, second = Counter(predicted.semantic_labels), Counter(gold.semantic_labels)
    total = first.total() + second.total()
    return 2 * (first & second).total() / total if total else 1.0


def tanimoto_reward_fn(pred_smiles_list, gt_smiles_list):
    rewards = []
    for predicted, gold in _pairs(pred_smiles_list, gt_smiles_list):
        similarity = 0.0
        if predicted is not None:
            similarity = DataStructs.FingerprintSimilarity(
                Chem.RDKFingerprint(predicted.molecule),
                Chem.RDKFingerprint(gold.molecule),
            )
            similarity *= _label_agreement(predicted, gold)
        rewards.append(similarity)
    return torch.tensor(rewards, dtype=torch.float32) * 0.4


def stereochem_reward_fn(pred_list, gt_list):
    rewards = []
    for predicted, gold in _pairs(pred_list, gt_list):
        exact = predicted is not None and predicted.equivalent(gold)
        rewards.append(float(exact))
    return torch.tensor(rewards, dtype=torch.float32) * 0.6
