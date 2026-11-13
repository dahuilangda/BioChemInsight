import ast
import os
from pathlib import Path
import cv2
import time
import random
import re
import string
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from SmilesPE.pretokenizer import atomwise_tokenizer

from .indigo import Indigo
from .indigo.renderer import IndigoRenderer

from .augment import CropWhite
from .utils import normalize_nodes
from .constants import RGROUP_SYMBOLS, SUBSTITUTIONS, ELEMENTS, COLORS
from .tokenizer import PAD_ID
from .structure import parse_structure

cv2.setNumThreads(0)

INDIGO_HYGROGEN_PROB = 0.2
INDIGO_FUNCTIONAL_GROUP_PROB = 0.8
INDIGO_CONDENSED_PROB = 0.5
INDIGO_RGROUP_PROB = 0.5
INDIGO_COMMENT_PROB = 0.3
INDIGO_DEARMOTIZE_PROB = 0.8
INDIGO_COLOR_PROB = 0.2


def get_transforms(args, augment=True, rotate=True, debug=False):
    input_size = args.input_size
    trans_list = []
    if rotate:
        trans_list.append(A.SafeRotate(limit=45, border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.5))
    trans_list.append(CropWhite(pad=5))
    trans_list.append(A.LongestMaxSize(max_size=input_size))
    trans_list.append(A.PadIfNeeded(min_height=input_size, min_width=input_size, border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255)))
    if augment:
        if not debug:
            trans_list += [
                A.Downscale(scale_range=(0.4, 0.8), p=0.5),
                A.Blur(p=0.5),
                A.GaussNoise(std_range=(0.05, 0.1), p=0.5),
                A.SaltAndPepper(amount=(0.005, 0.01), p=0.5)
            ]
        else:
            trans_list += [
                A.Downscale(scale_range=(0.4, 0.8), p=1),
                A.Blur(p=1),
                A.GaussNoise(std_range=(0.05, 0.1), p=1),
                A.SaltAndPepper(amount=(0.005, 0.01), p=1)
            ]

    if not debug:
        trans_list.append(A.ToGray(p=1))
        mean = [0.48145466, 0.4578275, 0.40821073] if args.encoder == 'vitdet' else [0.485, 0.456, 0.406]
        std = [0.26862954, 0.26130258, 0.27577711] if args.encoder == 'vitdet' else [0.229, 0.224, 0.225]
        trans_list += [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    return A.Compose(trans_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def add_functional_group(indigo, mol, debug=False):
    if random.random() > INDIGO_FUNCTIONAL_GROUP_PROB:
        return mol
    # Delete functional group and add a pseudo atom with its abbrv
    substitutions = [sub for sub in SUBSTITUTIONS]
    random.shuffle(substitutions)
    for sub in substitutions:
        query = indigo.loadSmarts(sub.smarts)
        matcher = indigo.substructureMatcher(mol)
        matched_atoms_ids = set()
        for match in matcher.iterateMatches(query):
            if random.random() < sub.probability or debug:
                atoms = []
                atoms_ids = set()
                for item in query.iterateAtoms():
                    atom = match.mapAtom(item)
                    atoms.append(atom)
                    atoms_ids.add(atom.index())
                if len(matched_atoms_ids.intersection(atoms_ids)) > 0:
                    continue
                abbrv = random.choice(sub.abbrvs)
                superatom = mol.addAtom(abbrv)
                for atom in atoms:
                    for nei in atom.iterateNeighbors():
                        if nei.index() not in atoms_ids:
                            if nei.symbol() == 'H':
                                # indigo won't match explicit hydrogen, so remove them explicitly
                                atoms_ids.add(nei.index())
                            else:
                                superatom.addBond(nei, nei.bond().bondOrder())
                for id in atoms_ids:
                    mol.getAtom(id).remove()
                matched_atoms_ids = matched_atoms_ids.union(atoms_ids)
    return mol


def add_explicit_hydrogen(indigo, mol):
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append((atom, hs))
        except:
            continue
    if len(atoms) > 0 and random.random() < INDIGO_HYGROGEN_PROB:
        atom, hs = random.choice(atoms)
        for i in range(hs):
            h = mol.addAtom('H')
            h.addBond(atom, 1)
    return mol


def add_rgroup(indigo, mol, smiles):
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append(atom)
        except:
            continue
    if len(atoms) > 0 and '*' not in smiles:
        if random.random() < INDIGO_RGROUP_PROB:
            atom_idx = random.choice(range(len(atoms)))
            atom = atoms[atom_idx]
            atoms.pop(atom_idx)
            symbol = random.choice(RGROUP_SYMBOLS)
            r = mol.addAtom(symbol)
            r.addBond(atom, 1)
    return mol


def get_rand_symb():
    symb = random.choice(ELEMENTS)
    if random.random() < 0.1:
        symb += random.choice(string.ascii_lowercase)
    if random.random() < 0.1:
        symb += random.choice(string.ascii_uppercase)
    if random.random() < 0.1:
        symb = f'({gen_rand_condensed()})'
    return symb


def get_rand_num():
    if random.random() < 0.9:
        if random.random() < 0.8:
            return ''
        else:
            return str(random.randint(2, 9))
    else:
        return '1' + str(random.randint(2, 9))


def gen_rand_condensed():
    tokens = []
    for i in range(5):
        if i >= 1 and random.random() < 0.8:
            break
        tokens.append(get_rand_symb())
        tokens.append(get_rand_num())
    return ''.join(tokens)


def add_rand_condensed(indigo, mol):
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append(atom)
        except:
            continue
    if len(atoms) > 0 and random.random() < INDIGO_CONDENSED_PROB:
        atom = random.choice(atoms)
        symbol = gen_rand_condensed()
        r = mol.addAtom(symbol)
        r.addBond(atom, 1)
    return mol


def generate_output_smiles(indigo, mol):
    # TODO: if using mol.canonicalSmiles(), explicit H will be removed
    smiles = mol.smiles()
    mol = indigo.loadMolecule(smiles)
    if '*' in smiles:
        part_a, part_b = smiles.split(' ', maxsplit=1)
        part_b = re.search(r'\$.*\$', part_b).group(0)[1:-1]
        symbols = [t for t in part_b.split(';') if len(t) > 0]
        output = ''
        cnt = 0
        for i, c in enumerate(part_a):
            if c != '*':
                output += c
            else:
                output += f'[{symbols[cnt]}]'
                cnt += 1
        return mol, output
    else:
        if ' ' in smiles:
            # special cases with extension
            smiles = smiles.split(' ')[0]
        return mol, smiles


def add_comment(indigo):
    if random.random() < INDIGO_COMMENT_PROB:
        indigo.setOption('render-comment', str(random.randint(1, 20)) + random.choice(string.ascii_letters))
        indigo.setOption('render-comment-font-size', random.randint(40, 60))
        indigo.setOption('render-comment-alignment', random.choice([0, 0.5, 1]))
        indigo.setOption('render-comment-position', random.choice(['top', 'bottom']))
        indigo.setOption('render-comment-offset', random.randint(2, 30))


def add_color(indigo, mol):
    if random.random() < INDIGO_COLOR_PROB:
        indigo.setOption('render-coloring', True)
    if random.random() < INDIGO_COLOR_PROB:
        indigo.setOption('render-base-color', random.choice(list(COLORS.values())))
    if random.random() < INDIGO_COLOR_PROB:
        if random.random() < 0.5:
            indigo.setOption('render-highlight-color-enabled', True)
            indigo.setOption('render-highlight-color', random.choice(list(COLORS.values())))
        if random.random() < 0.5:
            indigo.setOption('render-highlight-thickness-enabled', True)
        for atom in mol.iterateAtoms():
            if random.random() < 0.1:
                atom.highlight()
    return mol


def get_graph(mol, image, shuffle_nodes=False, pseudo_coords=False):
    mol.layout()
    coords, symbols = [], []
    index_map = {}
    atoms = [atom for atom in mol.iterateAtoms()]
    if shuffle_nodes:
        random.shuffle(atoms)
    for i, atom in enumerate(atoms):
        if pseudo_coords:
            x, y, z = atom.xyz()
        else:
            x, y = atom.coords()
        coords.append([x, y])
        symbols.append(atom.symbol())
        index_map[atom.index()] = i
    if pseudo_coords:
        coords = normalize_nodes(np.array(coords))
        h, w, _ = image.shape
        coords[:, 0] = coords[:, 0] * w
        coords[:, 1] = coords[:, 1] * h
    n = len(symbols)
    edges = np.zeros((n, n), dtype=int)
    for bond in mol.iterateBonds():
        s = index_map[bond.source().index()]
        t = index_map[bond.destination().index()]
        # 1/2/3/4 : single/double/triple/aromatic
        edges[s, t] = bond.bondOrder()
        edges[t, s] = bond.bondOrder()
        if bond.bondStereo() in [5, 6]:
            edges[s, t] = bond.bondStereo()
            edges[t, s] = 11 - bond.bondStereo()
    graph = {
        'coords': coords,
        'symbols': symbols,
        'edges': edges,
        'num_atoms': len(symbols)
    }
    return graph


def generate_indigo_image(smiles, mol_augment=True, default_option=False, shuffle_nodes=False, pseudo_coords=False,
                          include_condensed=True, debug=False):
    indigo = Indigo()
    renderer = IndigoRenderer(indigo)
    indigo.setOption('render-output-format', 'png')
    indigo.setOption('render-background-color', '1,1,1')
    indigo.setOption('render-stereo-style', 'none')
    indigo.setOption('render-label-mode', 'hetero')
    indigo.setOption('render-font-family', 'Arial')
    if not default_option:
        thickness = random.uniform(0.5, 2)  # limit the sum of the following two parameters to be smaller than 4
        indigo.setOption('render-relative-thickness', thickness)
        indigo.setOption('render-bond-line-width', random.uniform(1, 4 - thickness))
        if random.random() < 0.5:
            indigo.setOption('render-font-family', random.choice(['Arial', 'Times', 'Courier', 'Helvetica']))
        indigo.setOption('render-label-mode', random.choice(['hetero', 'terminal-hetero']))
        indigo.setOption('render-implicit-hydrogens-visible', random.choice([True, False]))
        if random.random() < 0.1:
            indigo.setOption('render-stereo-style', 'old')
        if random.random() < 0.2:
            indigo.setOption('render-atom-ids-visible', True)

    try:
        mol = indigo.loadMolecule(smiles)
        if mol_augment:
            if random.random() < INDIGO_DEARMOTIZE_PROB:
                mol.dearomatize()
            else:
                mol.aromatize()
            smiles = mol.canonicalSmiles()
            add_comment(indigo)
            mol = add_explicit_hydrogen(indigo, mol)
            mol = add_rgroup(indigo, mol, smiles)
            if include_condensed:
                mol = add_rand_condensed(indigo, mol)
            if not ('/' in smiles or '\\' in smiles):   # avoid losing stereo information
                mol = add_functional_group(indigo, mol, debug)
            mol = add_color(indigo, mol)
            mol, smiles = generate_output_smiles(indigo, mol)

        buf = renderer.renderToBuffer(mol)
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)  # decode buffer to image
        # img = np.repeat(np.expand_dims(img, 2), 3, axis=2)  # expand to RGB
        graph = get_graph(mol, img, shuffle_nodes, pseudo_coords)
        success = True
    except Exception:
        if debug:
            raise Exception
        img = None
        graph = {}
        success = False
    return img, smiles, graph, success


class PubchemDataset(Dataset):
    def __init__(self, args, df, tokenizer):
        super().__init__()
        self.df = df
        self.args = args
        self.tokenizer = tokenizer
        assert 'SMILES' in df.columns
        self.smiles = df['SMILES'].values
        self.transform = get_transforms(
            args, augment=getattr(args, "augment", False),
            rotate=getattr(args, "augment", False), debug=args.debug,
        )
        self.debug = args.debug

    def __len__(self):
        return len(self.df)

    def image_transform(self, image, coords=[], renormalize=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # .astype(np.float32)
        augmented = self.transform(image=image, keypoints=coords)
        image = augmented['image']
        if len(coords) > 0:
            coords = np.array(augmented['keypoints'])
            if renormalize:
                coords = normalize_nodes(coords, flip_y=False)
            else:
                _, height, width = image.shape
                coords[:, 0] = coords[:, 0] / width
                coords[:, 1] = coords[:, 1] / height
            coords = np.array(coords).clip(0, 1)
            return image, coords
        return image

    def __getitem__(self, idx):
        return self.getitem(idx)

    def getitem(self, idx):
        ref = {}
        image, smiles, graph, success = generate_indigo_image(
            self.smiles[idx], mol_augment=self.args.mol_augment, default_option=self.args.default_option,
            shuffle_nodes=self.args.shuffle_nodes, include_condensed=self.args.include_condensed)
        if not success:
            raise ValueError(f"Structure rendering failed at sample {idx}")
        image, coords = self.image_transform(image, graph['coords'])
        graph['coords'] = coords

        ref['edges'] = torch.tensor(graph['edges'])
        self._process_chartok_coords(idx, ref, smiles, graph['coords'], graph['edges'], mask_ratio=self.args.mask_ratio)

        ref['idx'] = idx
        ref['image'] = image

        ref['smiles'] = smiles

        return ref

    def _process_chartok_coords(self, idx, ref, smiles, coords=None, edges=None, mask_ratio=0):
        max_len = self.args.max_len
        tokenizer = self.tokenizer
        if not isinstance(smiles, str) or not smiles:
            raise ValueError(f"Missing structure target at sample {idx}")
        label, indices = tokenizer.smiles_to_sequence(smiles, mask_ratio=mask_ratio)
        if len(label) > max_len:
            raise ValueError(f'Target length {len(label)} exceeds max_len {max_len} at sample {idx}')
        ref['label'] = torch.LongTensor(label)
        indices = [i for i in indices if i < max_len]
        ref['atom_indices'] = torch.LongTensor(indices)

        if edges is not None:
            ref['edges'] = torch.tensor(edges)[:len(indices), :len(indices)]
        else:
            if 'edges' in self.df.columns:
                edge_list = ast.literal_eval(self.df.loc[idx, 'edges'])
                n = len(indices)
                edges = torch.zeros((n, n), dtype=torch.long)
                for u, v, t in edge_list:
                    if u < n and v < n:
                        if t <= 4:
                            edges[u, v] = t
                            edges[v, u] = t
                        else:
                            edges[u, v] = t
                            edges[v, u] = 11 - t
                ref['edges'] = edges
            else:
                ref['edges'] = torch.ones(len(indices), len(indices), dtype=torch.long) * (-100)

        if coords is not None:
            ref['coords'] = torch.tensor(coords)
        else:
            ref['coords'] = torch.ones(len(indices), 2) * (-100)


class USPTODataset(PubchemDataset):

    def __init__(self, args, df, tokenizer):
        super().__init__(args, df, tokenizer)
        assert 'file_path' in df.columns
        self.file_paths = [
            str(Path(path) if Path(path).is_absolute() else Path(args.data_path or '.') / path)
            for path in df['file_path']
        ]
        self.has_coords = 'node_coords' in df.columns

    def getitem(self, idx):
        ref = {}
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f'Image cannot be decoded: {file_path}')
        h, w, _ = image.shape

        if self.has_coords:
            coords = np.array(ast.literal_eval(self.df.loc[idx, 'node_coords']))
            if not os.environ.get('MOLSIGHT_KEEP_PIXEL_COORDS'):
                # mol-block convention: bbox-normalize + flip y. Pixel-truth
                # annotations (molgrapher keypoints) must skip this — the
                # stretch decouples coords from the actual image layout.
                coords = normalize_nodes(coords)
            coords[:, 0] = coords[:, 0] * w
            coords[:, 1] = coords[:, 1] * h
            image, coords = self.image_transform(image, coords, renormalize=True)
        else:
            image = self.image_transform(image)
            coords = None

        smiles = self.smiles[idx]
        self._process_chartok_coords(idx, ref, smiles, coords, mask_ratio=0)

        ref['idx'] = idx
        ref['image'] = image

        ref['smiles'] = smiles

        return ref

class StakerDataset(USPTODataset):
    def __init__(self, args, df, tokenizer):
        super().__init__(args, df, tokenizer)
        self.transform = get_transforms(args, augment=False, rotate=False, debug=args.debug)

def is_atom_token(token):
    return token.isalpha() or token.startswith("[") or token == '*'

def process_exsmiles(exsmiles: str) -> str:
    return parse_structure(exsmiles).model_text


class MolParser7MDataset(Dataset):
    def __init__(self, args, parquet, tokenizer):
        super().__init__()
        self.parquet = parquet
        self.args = args
        self.tokenizer = tokenizer
        self.transform = get_transforms(
            args, augment=getattr(args, "augment", False),
            rotate=getattr(args, "augment", False), debug=args.debug,
        )
        self.max_len = args.max_len
        self.mask_ratio = args.mask_ratio

    def __len__(self):
        return len(self.parquet)

    def __getitem__(self, idx):
        return self.getitem(idx)

    def getitem(self, idx):
        ref = {}
        data = self.parquet[idx]
        image = data.get('image', None)
        if image is None:
            raise ValueError(f'Missing image at sample {idx}')
        image = np.array(image)

        image = self.transform(image=image, keypoints=[])['image']

        smiles = process_exsmiles(data['SMILES'])
        label, indices = self.tokenizer.smiles_to_sequence(smiles, mask_ratio=self.mask_ratio)
        if len(label) > self.max_len:
            raise ValueError(f'Target length {len(label)} exceeds max_len {self.max_len} at sample {idx}')
        ref['label'] = torch.LongTensor(label)
        indices = [i for i in indices if i < self.max_len]
        ref['atom_indices'] = torch.LongTensor(indices)

        ref['edges'] = torch.ones(len(indices), len(indices), dtype=torch.long) * (-100)
        ref['coords'] = torch.ones(len(indices), 2) * (-100)

        ref['idx'] = idx
        ref['image'] = image
        ref['smiles'] = smiles
        return ref


class HybridDataset(Dataset):
    def __init__(self, args, train_data: Dict, tokenizer):
        super().__init__()
        self.all_datasets = []
        for name in train_data.keys():
            if name == "pubchem":
                self.all_datasets.append(PubchemDataset(args, train_data[name], tokenizer))
            elif name == "uspto":
                self.all_datasets.append(USPTODataset(args, train_data[name], tokenizer))
            elif name == "molparser7m":
                self.all_datasets.append(MolParser7MDataset(args, train_data[name], tokenizer))
            elif name == 'stereo':
                self.all_datasets.append(USPTODataset(args, train_data[name], tokenizer))
            elif name == 'staker':
                self.all_datasets.append(StakerDataset(args, train_data[name], tokenizer))
            else:
                raise ValueError(f"Unknown dataset name: {name}")

    def __len__(self):
        return sum([len(dataset) for dataset in self.all_datasets])

    def __getitem__(self, idx):
        for dataset in self.all_datasets:
            if idx < len(dataset):
                return dataset[idx]
            idx -= len(dataset)
        raise IndexError("Index out of range")


class ValDataset(USPTODataset):
    def __init__(self, args, df, tokenizer):
        super().__init__(args, df, tokenizer)
        self.transform = get_transforms(args, rotate=False, augment=False)

    def getitem(self, idx):
        ref = {}
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f'Image cannot be decoded: {file_path}')
        image = self.image_transform(image)

        smiles = self.smiles[idx]
        self._process_chartok_coords(idx, ref, smiles, mask_ratio=0)

        ref['idx'] = idx
        ref['image'] = image
        return ref


def collate_fn(batch: List[Dict[str, torch.Tensor]], input_ids_padding: int=PAD_ID) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    required = {'idx', 'image', 'label', 'atom_indices', 'coords', 'edges'}
    keys = set(batch[0])
    if not required <= keys or any(set(row) != keys for row in batch):
        raise ValueError("Batch samples have inconsistent fields")
    for row in batch:
        if any(row[key] is None for key in required):
            raise ValueError(f"Missing batch field at sample {row['idx']}")
        count = len(row['atom_indices'])
        if row['coords'].shape != (count, 2) or row['edges'].shape != (count, count):
            raise ValueError(f"Atom annotations are not aligned at sample {row['idx']}")
    batch = {key: [row[key] for row in batch] for key in batch[0]}

    batch['idx'] = torch.LongTensor(batch['idx'])

    batch["image"] = torch.stack(batch["image"])

    batch["label"] = torch.nn.utils.rnn.pad_sequence(batch["label"], batch_first=True, padding_value=input_ids_padding)
    batch["atom_indices"] = torch.nn.utils.rnn.pad_sequence(batch["atom_indices"], batch_first=True, padding_value=input_ids_padding)

    # [num_atoms, 2]
    batch["coords"] = torch.nn.utils.rnn.pad_sequence(batch["coords"], batch_first=True, padding_value=-100)

    # [num_atoms, num_atoms]
    max_len = max([len(edges) for edges in batch["edges"]])
    batch['edges'] = torch.stack(
        [F.pad(edges, (0, max_len - len(edges), 0, max_len - len(edges)), value=-100) for edges in batch["edges"]],
        dim=0)

    return batch

