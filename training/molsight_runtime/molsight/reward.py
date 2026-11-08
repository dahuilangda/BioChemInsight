import torch
from rdkit import Chem, DataStructs


def _pairs(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("Prediction and target counts differ")
    for index, (prediction, target) in enumerate(zip(predictions, targets)):
        gold = Chem.MolFromSmiles(target) if isinstance(target, str) and target else None
        if gold is None or gold.GetNumAtoms() == 0:
            raise ValueError(f"Invalid target structure at index {index}")
        predicted = Chem.MolFromSmiles(prediction) if isinstance(prediction, str) and prediction else None
        yield predicted, gold


def tanimoto_reward_fn(pred_smiles_list, gt_smiles_list):
    rewards = []
    for predicted, gold in _pairs(pred_smiles_list, gt_smiles_list):
        similarity = 0.0
        if predicted is not None and predicted.GetNumAtoms():
            similarity = DataStructs.FingerprintSimilarity(
                Chem.RDKFingerprint(predicted), Chem.RDKFingerprint(gold)
            )
        rewards.append(similarity)
    return torch.tensor(rewards, dtype=torch.float32) * 0.4


def stereochem_reward_fn(pred_list, gt_list):
    rewards = []
    for predicted, gold in _pairs(pred_list, gt_list):
        exact = predicted is not None and predicted.GetNumAtoms() > 0
        if exact:
            exact = Chem.MolToSmiles(predicted, isomericSmiles=True) == Chem.MolToSmiles(
                gold, isomericSmiles=True
            )
        rewards.append(float(exact))
    return torch.tensor(rewards, dtype=torch.float32) * 0.6
