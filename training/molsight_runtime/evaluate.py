import json
import argparse
import numpy as np
import multiprocessing
import pandas as pd

import rdkit
from rdkit import Chem, DataStructs

rdkit.RDLogger.DisableLog('rdApp.*')
from SmilesPE.pretokenizer import atomwise_tokenizer
from molsight.chemistry import postprocess_smiles


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


class SmilesEvaluator(object):
    def __init__(self, gold_smiles, num_workers=16, tanimoto=False):
        self.gold_smiles = list(gold_smiles)
        if not self.gold_smiles or any(not canonicalize_smiles(s)[1] for s in self.gold_smiles):
            raise ValueError("Evaluation requires nonempty, valid gold structures")
        self.num_workers = num_workers
        self.tanimoto = tanimoto
        self.gold_smiles_canon, _ = convert_smiles_to_canonsmiles(gold_smiles,
                                                                     ignore_chiral=False, ignore_cistrans=False,
                                                                     num_workers=num_workers)
        self.gold_smiles_ignore_cistrans, _ = convert_smiles_to_canonsmiles(gold_smiles,
                                                                     ignore_cistrans=True,
                                                                     num_workers=num_workers)
        self.gold_smiles_ignore_chiral_and_cistrans, _ = convert_smiles_to_canonsmiles(gold_smiles,
                                                                   ignore_chiral=True, ignore_cistrans=True,
                                                                   num_workers=num_workers)
        self.gold_smiles_ignore_cistrans = self._replace_empty(self.gold_smiles_ignore_cistrans)
        self.gold_smiles_ignore_chiral_and_cistrans = self._replace_empty(self.gold_smiles_ignore_chiral_and_cistrans)

    def _replace_empty(self, smiles_list):
        """Replace empty SMILES in the gold, otherwise it will be considered correct if both pred and gold is empty."""
        return [smiles if smiles is not None and type(smiles) is str and smiles != "" else "<empty>"
                for smiles in smiles_list]

    def evaluate(self, pred_smiles, include_details=False):
        pred_smiles = list(pred_smiles)
        if len(pred_smiles) != len(self.gold_smiles):
            raise ValueError("Prediction and gold counts differ")
        results = {"parse_success": float(np.mean([canonicalize_smiles(s)[1] for s in pred_smiles]))}
        if self.tanimoto:
            results['tanimoto'] = np.mean(compute_tanimoto_similarities(self.gold_smiles, pred_smiles))
        self.pred_smiles_canon, _ = convert_smiles_to_canonsmiles(pred_smiles,
                                                            ignore_chiral=False, ignore_cistrans=False,
                                                            num_workers=self.num_workers)
        results['exact'] = np.mean(np.array(self.gold_smiles_canon) == np.array(self.pred_smiles_canon))
        if include_details:
            results['exact_match'] = (np.array(self.gold_smiles_canon) == np.array(self.pred_smiles_canon)).tolist()
        
        # Ignore double bond cis/trans
        pred_smiles_ignore_cistrans, _ = convert_smiles_to_canonsmiles(pred_smiles,
                                                                ignore_cistrans=True,
                                                                num_workers=self.num_workers)
        results['ignore_cistrans'] = np.mean(np.array(self.gold_smiles_ignore_cistrans) == np.array(pred_smiles_ignore_cistrans))

        # Ignore chirality (Graph exact match)
        pred_smiles_ignore_chiral_and_cistrans, _ = convert_smiles_to_canonsmiles(pred_smiles,
                                                              ignore_chiral=True, ignore_cistrans=True,
                                                              num_workers=self.num_workers)
        results['ignore_chiral_and_cistrans'] = np.mean(np.array(self.gold_smiles_ignore_chiral_and_cistrans) == np.array(pred_smiles_ignore_chiral_and_cistrans))
        
        # Evaluate on molecules with chiral centers
        stereo = np.array([[g, p] for g, p in zip(self.gold_smiles_canon, self.pred_smiles_canon) if has_stereochemistry(g)])
        results['acc_in_stereo_mols'] = np.mean(stereo[:, 0] == stereo[:, 1]) if len(stereo) > 0 else -1
        
        # Evaluate on molecules with chiral centers
        chiral = np.array([[g, p] for g, p in zip(self.gold_smiles_canon, self.pred_smiles_canon) if '@' in g])
        results['acc_in_chiral_mols'] = np.mean(chiral[:, 0] == chiral[:, 1]) if len(chiral) > 0 else -1

        # Evaluate on molecules with cistrans
        cistrans = np.array([[g, p] for g, p in zip(self.gold_smiles_canon, self.pred_smiles_canon) if '/' in g or '\\' in g])
        results['acc_in_cistrans_mols'] = np.mean(cistrans[:, 0] == cistrans[:, 1]) if len(cistrans) > 0 else -1
        return results


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