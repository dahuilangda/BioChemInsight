import json
import argparse
import numpy as np
import multiprocessing
import pandas as pd

import rdkit
from rdkit import Chem, DataStructs

rdkit.RDLogger.DisableLog('rdApp.*')
from SmilesPE.pretokenizer import atomwise_tokenizer
from molsight.structure import StructureError, parse_structure


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_file', type=str, default="data/real/USPTO.csv")
    parser.add_argument('--pred_file', type=str, default="osra-2.2.1/prediction_USPTO.csv")
    parser.add_argument('--pred_field', type=str, default='SMILES')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--keep_main', action='store_true')
    args = parser.parse_args()
    return args


def canonicalize_smiles(smiles, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True):
    if type(smiles) is not str or smiles == '':
        return '', False
    if ignore_cistrans:
        smiles = smiles.replace('/', '').replace('\\', '')
    if replace_rgroup:
        tokens = atomwise_tokenizer(smiles)
        if ''.join(tokens) != smiles:
            return '', False
        for j, token in enumerate(tokens):
            if token.startswith('[R') and token.endswith(']') and token[2:-1].isdigit():
                tokens[j] = f'[{token[2:-1]}*]'
        smiles = ''.join(tokens)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not mol.GetNumAtoms():
            return '', False
        return Chem.MolToSmiles(mol, isomericSmiles=not ignore_chiral), True
    except (ValueError, RuntimeError):
        return '', False


def convert_smiles_to_canonsmiles(
        smiles_list, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True, num_workers=16):
    if len(smiles_list) == 0:
        return [], 0.0
    if num_workers == 1:
        results = [canonicalize_smiles(s, ignore_chiral, ignore_cistrans, replace_rgroup) for s in smiles_list]
    else:
        with multiprocessing.Pool(num_workers) as p:
            results = p.starmap(canonicalize_smiles,
                            [(smiles, ignore_chiral, ignore_cistrans, replace_rgroup) for smiles in smiles_list],
                            chunksize=128)
    canon_smiles, success = zip(*results)
    return list(canon_smiles), np.mean(success)


def _keep_main_molecule(smiles, debug=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            num_atoms = [m.GetNumAtoms() for m in frags]
            main_mol = frags[np.argmax(num_atoms)]
            smiles = Chem.MolToSmiles(main_mol)
    except Exception as e:
        pass
    return smiles


def keep_main_molecule(smiles, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        results = p.map(_keep_main_molecule, smiles, chunksize=128)
    return results


def tanimoto_similarity(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        tanimoto = DataStructs.FingerprintSimilarity(fp1, fp2)
        return tanimoto
    except:
        return 0


def compute_tanimoto_similarities(gold_smiles, pred_smiles, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        similarities = p.starmap(tanimoto_similarity, [(gs, ps) for gs, ps in zip(gold_smiles, pred_smiles)])
    return similarities

def has_stereochemistry(smiles: str) -> bool:
    return any(c in smiles for c in ['@', '/', '\\'])


def _evaluate_structure_pair(pair):
    gold_text, prediction_text = pair
    gold = parse_structure(gold_text)
    try:
        prediction = parse_structure(prediction_text)
    except StructureError:
        prediction = None
    row = {
        "parse_success": prediction is not None,
        "exact": prediction is not None and gold.equivalent(prediction),
        "ignore_cistrans": prediction is not None and gold.without_stereo().equivalent(prediction.without_stereo()),
        "ignore_chiral_and_cistrans": prediction is not None and gold.without_stereo(atoms=True).equivalent(prediction.without_stereo(atoms=True)),
        "has_chiral": "@" in gold.base,
        "has_cistrans": "/" in gold.base or "\\" in gold.base,
        "tanimoto": 0.0,
    }
    roles = {kind for kind, _ in gold.semantic_labels}
    if roles & {"a", "r", "c", "rc", "sgroup", "repeat", "variation"}:
        row["category"] = "markush"
    elif "d" in roles or any(atom.GetAtomicNum() == 0 for atom in gold.molecule.GetAtoms()):
        row["category"] = "attachment"
    else:
        row["category"] = "complete"
    if prediction is not None:
        row["tanimoto"] = DataStructs.FingerprintSimilarity(
            Chem.RDKFingerprint(gold.molecule), Chem.RDKFingerprint(prediction.molecule)
        )
    return row


class SmilesEvaluator:
    def __init__(self, gold_smiles, num_workers=16, tanimoto=False):
        self.gold_smiles = list(gold_smiles)
        if not self.gold_smiles:
            raise ValueError("Evaluation requires nonempty, valid gold structures")
        for index, value in enumerate(self.gold_smiles):
            try:
                parse_structure(value)
            except StructureError as error:
                raise ValueError(f"Invalid gold structure at index {index}") from error
        self.num_workers = num_workers
        self.tanimoto = tanimoto

    def evaluate(self, pred_smiles, include_details=False):
        predictions = list(pred_smiles)
        if len(predictions) != len(self.gold_smiles):
            raise ValueError("Prediction and gold counts differ")
        pairs = list(zip(self.gold_smiles, predictions))
        if self.num_workers == 1:
            rows = list(map(_evaluate_structure_pair, pairs))
        else:
            with multiprocessing.get_context("spawn").Pool(self.num_workers) as pool:
                rows = pool.map(_evaluate_structure_pair, pairs, chunksize=128)
        scores = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("parse_success", "exact", "ignore_cistrans", "ignore_chiral_and_cistrans")
        }
        for name, condition in (
            ("stereo", lambda row: row["has_chiral"] or row["has_cistrans"]),
            ("chiral", lambda row: row["has_chiral"]),
            ("cistrans", lambda row: row["has_cistrans"]),
        ):
            selected = [row["exact"] for row in rows if condition(row)]
            scores[f"acc_in_{name}_mols"] = float(np.mean(selected)) if selected else -1.0
        scores["categories"] = {}
        for category in sorted({row["category"] for row in rows}):
            selected = [row for row in rows if row["category"] == category]
            scores["categories"][category] = {
                "count": len(selected),
                "exact": float(np.mean([row["exact"] for row in selected])),
                "parse_success": float(np.mean([row["parse_success"] for row in selected])),
            }
        if self.tanimoto:
            scores["tanimoto"] = float(np.mean([row["tanimoto"] for row in rows]))
        if include_details:
            scores["exact_match"] = [row["exact"] for row in rows]
        return scores


if __name__ == "__main__":
    args = get_args()
    gold_df = pd.read_csv(args.gold_file)
    pred_df = pd.read_csv(args.pred_file)

    if gold_df['image_id'].duplicated().any() or pred_df['image_id'].duplicated().any():
        raise ValueError('image_id values must be unique')

    if len(pred_df) != len(gold_df):
        print(f"Pred ({len(pred_df)}) and Gold ({len(gold_df)}) have different lengths!")

    # Re-order pred_df to have the same order with gold_df
    image2goldidx = {image_id: idx for idx, image_id in enumerate(gold_df['image_id'])}
    image2predidx = {image_id: idx for idx, image_id in enumerate(pred_df['image_id'])}
    for image_id in gold_df['image_id']:
        # If image_id doesn't exist in pred_df, add an empty prediction.
        if image_id not in image2predidx:
            #pred_df = pred_df.append({'image_id': image_id, args.pred_field: ""}, ignore_index=True)
            pred_df = pd.concat([pred_df, pd.DataFrame({'image_id': [image_id], args.pred_field: [""]})], ignore_index=True)
    image2predidx = {image_id: idx for idx, image_id in enumerate(pred_df['image_id'])}
    pred_df = pred_df.reindex([image2predidx[image_id] for image_id in gold_df['image_id']])

    evaluator = SmilesEvaluator(gold_df['SMILES'], args.num_workers, True)
    scores = evaluator.evaluate(pred_df[args.pred_field], include_details=True)
    exact_match = None
    if 'exact_match' in scores:
        exact_match = scores.pop('exact_match')
    print(json.dumps(scores, indent=4))
    '''out_df = pd.DataFrame(gold_df['file_path'])
    out_df['gold'] = evaluator.gold_smiles_canon
    out_df['pred'] = evaluator.pred_smiles_canon
    if exact_match is not None:
        out_df['exact_match'] = exact_match
    out_df.to_csv('runs/exp_5/evaluation_USPTO.csv', index=False)'''
