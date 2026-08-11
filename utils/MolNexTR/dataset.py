import os
import ast
import cv2
import time
import random
import re
import string
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .indigo import Indigo
from .indigo.renderer import IndigoRenderer

from .data_aug import SafeRotate, CropWhite, PadWhite, AddIncompleteStructuralNoise, AddBondNoise, SaltAndPepperNoise, PadToSquare, AddLineNoise, AddEdgeElementSymbolNoise, DrawBorder, TightSegment, LineThicken, PatentRealize
from .utils import FORMAT_INFO
from .tokenization import PAD_ID, EOS_ID, UNK_ID, atomwise_tokenizer
from .chemical import get_num_atoms, normalize_nodes
from .abbrs import RGROUP_SYMBOLS, SUBSTITUTIONS, ELEMENTS, COLORS

cv2.setNumThreads(1)

INDIGO_HYGROGEN_PROB = 0.2
INDIGO_FUNCTIONAL_GROUP_PROB = 0.8
INDIGO_CONDENSED_PROB = 0.5
INDIGO_RGROUP_PROB = 0.5
INDIGO_COMMENT_PROB = 0.3
INDIGO_DEARMOTIZE_PROB = 0.8
INDIGO_COLOR_PROB = 0.2


def optional_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def optional_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return list(value)


def resize_small_image_to_long_edge(image, target_long_edge=0):
    if image is None:
        return image
    target_long_edge = int(target_long_edge or 0)
    if target_long_edge <= 0:
        return image
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= 0 or long_edge >= target_long_edge:
        return image
    scale = target_long_edge / long_edge
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)




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

def get_transforms(input_size,  test_file, augment=True, rotate=True, debug=False, real_match=False):
    trans_list = []
    if augment and rotate:
        trans_list.append(SafeRotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    # Isolate LineThicken only (gentle, kernel 1-2) — full_v3's comprehensive
    # block (LineThicken + TightSegment + 5 others) corrupted the specialist.
    # Test line-thickness alone, the validated root cause.
    trans_list.append(CropWhite(pad=50))
    if test_file == 'real/acs.csv' or test_file == 'real/UOB.csv':
        trans_list.append(PadToSquare(p=1))

    if augment:
        trans_list += [
            # NormalizedGridDistortion(num_steps=10, distort_limit=0.3),
            A.CropAndPad(percent=[-0.01, 0.00], keep_size=False, p=0.5),
            PadWhite(pad_ratio=0.4, p=0.2),
        ]
        if real_match:
            # ISOLATED LineThicken only (gentle, kernel 1-2). The thinning test
            # proved line width is THE blocker for * emission (thin real → 31/31).
            # full_v3's 8-augmentation block corrupted the specialist; this tests
            # line-thickness alone, without the corruption.
            trans_list += [
                LineThicken(kernel_min=1, kernel_max=2, p=0.9),
            ]
        else:
            trans_list += [
                A.Downscale(scale_min=0.2, scale_max=0.5, interpolation=3),
                A.Blur(),
                A.GaussNoise(),
                SaltAndPepperNoise(num_dots=20, p=0.5)
            ]
    trans_list.append(A.Resize(input_size, input_size))
    if not debug:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans_list += [
            A.ToGray(p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    return A.Compose(trans_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))



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
        img = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
        graph = {}
        success = False
    return img, smiles, graph, success


class TrainDataset(Dataset):
    def __init__(self, args, df, tokenizer, split='train', dynamic_indigo=False):
        super().__init__()
        self.df = df
        self.args = args
        self.tokenizer = tokenizer
        # Phase 2: emit a MolParser-style <sep>[anchor:*] suffix on the dummy-free
        # backbone for fragment rows (attachment metadata OUTSIDE the AR structure
        # trajectory). Inert unless --phase2-sep is passed (smoke/full Phase-2 train).
        self.phase2_sep = bool(getattr(args, 'phase2_sep', False))
        import sys as _sys
        print(f"[phase2-sep] TrainDataset.phase2_sep={self.phase2_sep} (split={split})", file=_sys.stderr)
        if 'file_path' in df.columns:
            self.file_paths = df['file_path'].values
            if not self.file_paths[0].startswith(args.data_path):
                self.file_paths = [os.path.join(args.data_path, path) for path in df['file_path']]
        if 'SMILES' in df.columns:
            chemical_smiles = df['SMILES'].fillna('').astype(str)
            if 'decoder_smiles' in df.columns:
                decoder_smiles = df['decoder_smiles']
                usable_decoder = decoder_smiles.notna() & decoder_smiles.astype(str).ne('')
                self.smiles = decoder_smiles.where(
                    usable_decoder,
                    chemical_smiles,
                ).astype(str).values
            else:
                self.smiles = chemical_smiles.values
        else:
            self.smiles = None
        self.formats = args.formats
        self.labelled = (split == 'train')
        if self.labelled:
            self.labels = {}
            for format_ in self.formats:
                if format_ in ['atomtok', 'inchi']:
                    field = FORMAT_INFO[format_]['name']
                    if field in df.columns:
                        self.labels[format_] = df[field].values
        self.transform = get_transforms(args.input_size, args.test_file,
                                        augment=(self.labelled and args.augment),
                                        real_match=bool(getattr(args, "real_match", False)))
        self.clean_transform = get_transforms(args.input_size, args.test_file, augment=False)
        self.clean_transform_source_arrows = {
            str(source)
            for source in optional_list(getattr(args, "clean_transform_source_arrows", []))
            if str(source)
        }
        self.raw_image_cache_size = max(0, int(getattr(args, "raw_image_cache_size", 0) or 0))
        self.raw_image_cache = OrderedDict()
        # self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])
        self.dynamic_indigo = (dynamic_indigo and split == 'train')
        if self.labelled and not dynamic_indigo and args.coords_file is not None:
            if args.coords_file == 'aux_file':
                self.coords_df = df
                self.pseudo_coords = True
            else:
                self.coords_df = pd.read_csv(args.coords_file)
                self.pseudo_coords = False
        else:
            self.coords_df = None
            self.pseudo_coords = args.pseudo_coords

    def __len__(self):
        return len(self.df)

    def row_uses_clean_transform(self, idx):
        if not self.labelled or not self.clean_transform_source_arrows:
            return False
        if "source_arrow" not in self.df.columns:
            return False
        return optional_text(self.df.loc[idx, "source_arrow"]) in self.clean_transform_source_arrows

    def read_raw_image(self, file_path):
        if self.raw_image_cache_size > 0:
            cached = self.raw_image_cache.get(file_path)
            if cached is not None:
                self.raw_image_cache.move_to_end(file_path)
                return cached.copy()
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(
                "MolNexTR training image is missing or unreadable; refusing "
                f"to substitute a blank image: {file_path}"
            )
        image = resize_small_image_to_long_edge(
            image,
            getattr(self.args, "train_preprocess_long_edge", 0),
        )
        if self.raw_image_cache_size > 0:
            self.raw_image_cache[file_path] = image
            self.raw_image_cache.move_to_end(file_path)
            while len(self.raw_image_cache) > self.raw_image_cache_size:
                self.raw_image_cache.popitem(last=False)
        return image.copy()

    def image_transform(self, image, coords=[], renormalize=False, transform=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # .astype(np.float32)
        active_transform = transform or self.transform
        augmented = active_transform(image=image, keypoints=coords)
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

    def attachment_targets(self, idx):
        """Return sparse attachment supervision in the unaugmented image frame."""
        if "structure_type_label" not in self.df.columns:
            return None
        row = self.df.loc[idx]
        smiles = optional_text(self.smiles[idx]) if self.smiles is not None else ""
        dummy_indices = []
        atom_index = -1
        for token in atomwise_tokenizer(smiles):
            is_atom = token.isalpha() or token.startswith("[") or token == "*"
            if not is_atom:
                continue
            atom_index += 1
            if token == "*" or (token.startswith("[") and "*" in token):
                dummy_indices.append(atom_index)
        count = len(dummy_indices)
        explicit_points = []
        raw_points = optional_text(row.get("attachment_points", ""))
        if raw_points:
            try:
                parsed = ast.literal_eval(raw_points)
                explicit_points = [
                    [float(point[0]), float(point[1])]
                    for point in parsed
                    if isinstance(point, (list, tuple)) and len(point) >= 2
                    and 0.0 <= float(point[0]) <= 1.0
                    and 0.0 <= float(point[1]) <= 1.0
                ]
            except (SyntaxError, ValueError, TypeError):
                explicit_points = []
        complete_raw = row.get("attachment_points_complete", False)
        complete = str(complete_raw).strip().lower() in {"1", "true", "yes"}
        coordinate_targets_raw = row.get("coordinate_targets_available", True)
        coordinate_targets_available = (
            str(coordinate_targets_raw).strip().lower() in {"1", "true", "yes"}
        )
        raw_edges = optional_text(row.get("edges", ""))
        try:
            edges = ast.literal_eval(raw_edges) if raw_edges else []
        except (SyntaxError, ValueError, TypeError):
            edges = []
        point_dummy_indices = []
        raw_point_dummy_indices = optional_text(
            row.get("attachment_point_dummy_indices", "")
        )
        if raw_point_dummy_indices:
            try:
                parsed_indices = ast.literal_eval(raw_point_dummy_indices)
                point_dummy_indices = [int(value) for value in parsed_indices]
            except (SyntaxError, ValueError, TypeError):
                point_dummy_indices = []
        if explicit_points:
            if len(point_dummy_indices) != len(explicit_points):
                point_dummy_indices = (
                    list(dummy_indices)
                    if complete and len(explicit_points) == len(dummy_indices)
                    else [-1] * len(explicit_points)
                )
        elif coordinate_targets_available:
            point_dummy_indices = list(dummy_indices)
            complete = True
        else:
            # A real image with no matched attachment point is an unlabeled
            # set, not an observed empty set. Cardinality remains supervised,
            # while objectness/pointer losses must not invent negatives.
            point_dummy_indices = []
            complete = False

        bonded = []
        bond_types = []
        anchor_indices = []
        for dummy_index in point_dummy_indices:
            incident = [
                edge for edge in edges
                if isinstance(edge, (list, tuple)) and len(edge) >= 3
                and dummy_index >= 0
                and (int(edge[0]) == dummy_index or int(edge[1]) == dummy_index)
                and int(edge[2]) > 0
            ]
            bonded.append(int(bool(incident)) if dummy_index >= 0 else -1)
            bond_types.append(
                int(incident[0][2]) if incident else (0 if dummy_index >= 0 else -1)
            )
            if incident:
                edge = incident[0]
                anchor_indices.append(
                    int(edge[1]) if int(edge[0]) == dummy_index else int(edge[0])
                )
            else:
                anchor_indices.append(-1)
        return {
            "count": int(count),
            "dummy_indices": dummy_indices,
            "explicit_points": explicit_points,
            "point_dummy_indices": point_dummy_indices,
            "complete": bool(complete),
            "bonded": bonded,
            "bond_types": bond_types,
            "anchor_indices": anchor_indices,
        }

    @staticmethod
    def store_attachment_refs(
        ref,
        targets,
        points,
        anchor_points=None,
        atom_coords=None,
    ):
        if targets is None:
            return
        points = np.asarray(points if points is not None else [], dtype=np.float32)
        if points.size == 0:
            points = np.zeros((0, 2), dtype=np.float32)
        points = points.reshape(-1, 2)
        point_count = len(points)
        ref["attachment_points"] = torch.tensor(points, dtype=torch.float32)
        ref["attachment_point_mask"] = torch.ones(point_count, dtype=torch.bool)
        ref["attachment_set_complete"] = torch.tensor(
            bool(targets["complete"]), dtype=torch.bool
        )
        ref["attachment_count"] = torch.tensor(
            int(targets["count"]), dtype=torch.long
        )
        bonded = list(targets.get("bonded") or [])[:point_count]
        bond_types = list(targets.get("bond_types") or [])[:point_count]
        bonded.extend([-1] * (point_count - len(bonded)))
        bond_types.extend([-1] * (point_count - len(bond_types)))
        ref["attachment_bonded"] = torch.tensor(bonded, dtype=torch.long)
        ref["attachment_bond_type"] = torch.tensor(bond_types, dtype=torch.long)
        dummy_indices = list(targets.get("point_dummy_indices") or [])[:point_count]
        dummy_indices.extend([-1] * (point_count - len(dummy_indices)))
        ref["attachment_dummy_indices"] = torch.tensor(
            dummy_indices, dtype=torch.long
        )
        anchor_points = list(anchor_points or [])[:point_count]
        anchor_points.extend([[-1.0, -1.0]] * (point_count - len(anchor_points)))
        anchor_array = np.asarray(anchor_points, dtype=np.float32).reshape(-1, 2)
        anchor_mask = np.logical_and(
            np.all(anchor_array >= 0.0, axis=1),
            np.all(anchor_array <= 1.0, axis=1),
        )
        ref["attachment_anchor_points"] = torch.tensor(
            anchor_array, dtype=torch.float32
        )
        ref["attachment_anchor_mask"] = torch.tensor(
            anchor_mask, dtype=torch.bool
        )
        anchor_indices = list(targets.get("anchor_indices") or [])[:point_count]
        anchor_indices.extend([-1] * (point_count - len(anchor_indices)))
        ref["attachment_anchor_indices"] = torch.tensor(
            anchor_indices, dtype=torch.long
        )
        atom_array = np.asarray(
            atom_coords if atom_coords is not None else [], dtype=np.float32
        )
        if atom_array.size == 0:
            atom_array = np.zeros((0, 2), dtype=np.float32)
        ref["attachment_atom_coords"] = torch.tensor(
            atom_array.reshape(-1, 2), dtype=torch.float32
        )

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as e:
            with open(os.path.join(self.args.save_path, f'error_dataset_{int(time.time())}.log'), 'w') as f:
                f.write(str(e))
            raise e

    def getitem(self, idx):
        ref = {}
        if self.dynamic_indigo:
            begin = time.time()
            image, smiles, graph, success = generate_indigo_image(
                self.smiles[idx], mol_augment=self.args.mol_augment, default_option=self.args.default_option,
                shuffle_nodes=self.args.shuffle_nodes, pseudo_coords=self.pseudo_coords,
                include_condensed=self.args.include_condensed)
            # raw_image = image
            end = time.time()
            if idx < 30 and self.args.save_image:
                path = os.path.join(self.args.save_path, 'images')
                os.makedirs(path, exist_ok=True)
                cv2.imwrite(os.path.join(path, f'{idx}.png'), image)
            if not success:
                return idx, None, {}
            image, coords = self.image_transform(image, graph['coords'], renormalize=self.pseudo_coords)
            graph['coords'] = coords
            ref['time'] = end - begin
            if 'atomtok' in self.formats:
                max_len = FORMAT_INFO['atomtok']['max_len']
                label = self.tokenizer['atomtok'].text_to_sequence(smiles, tokenized=False)
                ref['atomtok'] = torch.LongTensor(label[:max_len])
            if 'edges' in self.formats and 'atomtok_coords' not in self.formats and 'chartok_coords' not in self.formats:
                ref['edges'] = torch.tensor(graph['edges'])
            if 'atomtok_coords' in self.formats:
                self._process_atomtok_coords(idx, ref, smiles, graph['coords'], graph['edges'],
                                             mask_ratio=self.args.mask_ratio)
            if 'chartok_coords' in self.formats:
                self._process_chartok_coords(idx, ref, smiles, graph['coords'], graph['edges'],
                                             mask_ratio=self.args.mask_ratio)
            return idx, image, ref
        else:
            file_path = self.file_paths[idx]
            image = self.read_raw_image(file_path)
            attachment_targets = self.attachment_targets(idx)
            transformed_attachment_points = []
            transformed_attachment_anchor_points = []
            has_coord_targets = False
            if self.coords_df is not None:
                h, w, _ = image.shape
                raw_coords = ""
                if "node_coords" in self.coords_df.columns:
                    raw_coords = optional_text(self.coords_df.loc[idx, 'node_coords'])
                if raw_coords:
                    coords = np.array(eval(raw_coords), dtype=float)
                    has_coord_targets = coords.ndim == 2 and coords.shape[0] > 0 and coords.shape[1] >= 2
                else:
                    coords = None
                if has_coord_targets:
                    coord_space = ""
                    if "node_coords_space" in self.coords_df.columns:
                        coord_space = str(self.coords_df.loc[idx, "node_coords_space"] or "")
                    normalized_image_space = coord_space.startswith("normalized_image")
                    if normalized_image_space:
                        coords[:, 0] = coords[:, 0] * w
                        coords[:, 1] = coords[:, 1] * h
                        renormalize = False
                    else:
                        renormalize = self.pseudo_coords
                    if self.pseudo_coords and not normalized_image_space:
                        coords = normalize_nodes(coords)
                        coords[:, 0] = coords[:, 0] * w
                        coords[:, 1] = coords[:, 1] * h
                    transform = self.clean_transform if self.row_uses_clean_transform(idx) else None
                    explicit_points = (
                        attachment_targets.get("explicit_points")
                        if attachment_targets is not None
                        else []
                    ) or []
                    # OCR R-label centers are more precise attachment targets
                    # than an atom-center pose fit. Transform them in the exact
                    # same augmentation call as the recovered atom coordinates
                    # so the two supervision streams cannot drift apart.
                    if explicit_points:
                        point_pixels = np.asarray(explicit_points, dtype=float)
                        point_pixels[:, 0] *= w
                        point_pixels[:, 1] *= h
                        combined = np.concatenate([coords, point_pixels], axis=0)
                        image, transformed = self.image_transform(
                            image,
                            combined,
                            renormalize=renormalize,
                            transform=transform,
                        )
                        atom_count = len(coords)
                        coords = transformed[:atom_count]
                        transformed_attachment_points = transformed[atom_count:].tolist()
                    else:
                        image, coords = self.image_transform(
                            image,
                            coords,
                            renormalize=renormalize,
                            transform=transform,
                        )
                    if attachment_targets is not None:
                        if not transformed_attachment_points:
                            transformed_attachment_points = [
                                coords[atom_index].tolist()
                                for atom_index in attachment_targets["point_dummy_indices"]
                                if 0 <= atom_index < len(coords)
                            ]
                        transformed_attachment_anchor_points = [
                            (
                                coords[atom_index].tolist()
                                if 0 <= atom_index < len(coords)
                                else [-1.0, -1.0]
                            )
                            for atom_index in attachment_targets["anchor_indices"]
                        ]
                else:
                    transform = self.clean_transform if self.row_uses_clean_transform(idx) else None
                    explicit_points = (
                        attachment_targets.get("explicit_points")
                        if attachment_targets is not None else []
                    ) or []
                    if explicit_points:
                        h, w, _ = image.shape
                        point_pixels = np.asarray(explicit_points, dtype=float)
                        point_pixels[:, 0] *= w
                        point_pixels[:, 1] *= h
                        image, transformed = self.image_transform(
                            image,
                            point_pixels,
                            renormalize=False,
                            transform=transform,
                        )
                        transformed_attachment_points = transformed.tolist()
                    else:
                        image = self.image_transform(image, transform=transform)
                    coords = None
            else:
                transform = self.clean_transform if self.row_uses_clean_transform(idx) else None
                image = self.image_transform(image, transform=transform)
                coords = None
            self.store_attachment_refs(
                ref,
                attachment_targets,
                transformed_attachment_points,
                transformed_attachment_anchor_points,
                coords if has_coord_targets else None,
            )
            if self.labelled:
                smiles = self.smiles[idx]
                if 'atomtok' in self.formats:
                    max_len = FORMAT_INFO['atomtok']['max_len']
                    label = self.tokenizer['atomtok'].text_to_sequence(smiles, False)
                    ref['atomtok'] = torch.LongTensor(label[:max_len])
                if 'atomtok_coords' in self.formats:
                    if coords is not None:
                        self._process_atomtok_coords(idx, ref, smiles, coords, mask_ratio=0)
                    else:
                        self._process_atomtok_coords(idx, ref, smiles, mask_ratio=1)
                if 'chartok_coords' in self.formats:
                    if coords is not None:
                        self._process_chartok_coords(idx, ref, smiles, coords, mask_ratio=0)
                    else:
                        self._process_chartok_coords(idx, ref, smiles, mask_ratio=1)
            if self.args.predict_coords and ('atomtok_coords' in self.formats or 'chartok_coords' in self.formats):
                smiles = self.smiles[idx]
                if 'atomtok_coords' in self.formats:
                    self._process_atomtok_coords(idx, ref, smiles, mask_ratio=1)
                if 'chartok_coords' in self.formats:
                    self._process_chartok_coords(idx, ref, smiles, mask_ratio=1)
            return idx, image, ref

    def _process_atomtok_coords(self, idx, ref, smiles, coords=None, edges=None, mask_ratio=0):
        max_len = FORMAT_INFO['atomtok_coords']['max_len']
        tokenizer = self.tokenizer['atomtok_coords']
        if smiles is None or type(smiles) is not str:
            smiles = ""
        label, indices = tokenizer.smiles_to_sequence(smiles, coords, mask_ratio=mask_ratio)
        ref['atomtok_coords'] = torch.LongTensor(label[:max_len])
        indices = [i for i in indices if i < max_len]
        ref['atom_indices'] = torch.LongTensor(indices)
        if tokenizer.continuous_coords:
            if coords is not None:
                ref['coords'] = torch.tensor(coords)
            else:
                ref['coords'] = torch.ones(len(indices), 2) * -1.
        if edges is not None:
            ref['edges'] = torch.tensor(edges)[:len(indices), :len(indices)]
        else:
            raw_edges = optional_text(self.df.loc[idx, 'edges']) if 'edges' in self.df.columns else ""
            if raw_edges:
                edge_list = eval(raw_edges)
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

    def _is_phase2_fragment(self, idx) -> bool:
        """True iff row idx is an attachment_fragment with the alignment columns
        Phase 2 needs (decoder_dummy_atom_index, fragment_backbone_smiles)."""
        try:
            row = self.df.loc[idx]
        except Exception:
            return False
        if "structure_type_label" not in self.df.columns:
            return False
        if int(row.get("structure_type_label", -1) or -1) != 2:
            return False
        if (
            "decoder_dummy_atom_index" not in self.df.columns
            or "fragment_decoder_backbone_prefix" not in self.df.columns
        ):
            return False
        backbone = row.get("fragment_decoder_backbone_prefix")
        return bool(pd.notna(backbone) and str(backbone).strip())

    def _strict_row_edge_matrix(self, idx, atom_count):
        """Materialize the mandatory decoder graph stored in the dataframe."""
        raw_edges = (
            optional_text(self.df.loc[idx, "edges"])
            if "edges" in self.df.columns
            else ""
        )
        if not raw_edges:
            raise ValueError(f"phase2 fragment row {idx} lacks decoder edges")
        try:
            edge_list = ast.literal_eval(raw_edges)
        except (SyntaxError, TypeError, ValueError) as exc:
            raise ValueError(
                f"phase2 fragment row {idx} has unparseable decoder edges"
            ) from exc
        if not isinstance(edge_list, (list, tuple)):
            raise ValueError(
                f"phase2 fragment row {idx} decoder edges are not a list"
            )
        matrix = torch.zeros((atom_count, atom_count), dtype=torch.long)
        for edge_index, edge in enumerate(edge_list):
            if not isinstance(edge, (list, tuple)) or len(edge) != 3:
                raise ValueError(
                    f"phase2 fragment row {idx} edge {edge_index} is not a triplet"
                )
            try:
                begin, end, bond_type = (int(value) for value in edge)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"phase2 fragment row {idx} edge {edge_index} is non-integral"
                ) from exc
            if not (0 <= begin < atom_count and 0 <= end < atom_count):
                raise ValueError(
                    f"phase2 fragment row {idx} edge {edge_index} endpoint is out of range"
                )
            if begin == end:
                raise ValueError(
                    f"phase2 fragment row {idx} edge {edge_index} is a self-loop"
                )
            if not 1 <= bond_type <= 6:
                raise ValueError(
                    f"phase2 fragment row {idx} edge {edge_index} has invalid bond type "
                    f"{bond_type}"
                )
            reverse_type = bond_type if bond_type <= 4 else 11 - bond_type
            existing = int(matrix[begin, end])
            reverse_existing = int(matrix[end, begin])
            if (
                (existing not in (0, bond_type))
                or (reverse_existing not in (0, reverse_type))
            ):
                raise ValueError(
                    f"phase2 fragment row {idx} edge {edge_index} conflicts with a prior edge"
                )
            matrix[begin, end] = bond_type
            matrix[end, begin] = reverse_type
        return matrix

    def _phase2_fragment_inputs(self, idx, coords, edges):
        """Return (backbone_smiles, backbone_coords, backbone_edges, anchor_idx)
        for a fragment row. `edges` is the (n×n) adjacency matrix; the dummy atom is
        last (decoder_dummy_atom_index). Strip the dummy row/col ⇒ backbone submatrix;
        the anchor = the atom bonded to the dummy (nonzero in the dummy column)."""
        row = self.df.loc[idx]
        # This prefix is explicitly linearized in the same atom order as
        # decoder_coords/decoder_edges. The canonical fragment_backbone_smiles
        # is a chemistry-reporting field and can carry a different stereo/order
        # serialization, so it must not be paired with decoder graph tensors.
        backbone_smiles = str(row["fragment_decoder_backbone_prefix"])
        dummy_idx = int(row["decoder_dummy_atom_index"])
        if dummy_idx < 0:
            raise ValueError(
                f"phase2 fragment row {idx} has invalid dummy index {dummy_idx}"
            )
        # coords: list of [x,y]; the verified data contract currently places
        # the dummy last, but remapping below remains correct for any index.
        if coords is not None:
            if dummy_idx >= len(coords):
                raise ValueError(
                    f"phase2 fragment row {idx} dummy index exceeds coordinates"
                )
            backbone_coords = [c for i, c in enumerate(coords) if i != dummy_idx]
        else:
            raise ValueError(
                f"phase2 fragment row {idx} lacks decoder coordinates"
            )
        if edges is None:
            # Static-image training stores the decoder graph in the dataframe;
            # unlike the dynamic Indigo path, it has no in-memory graph object.
            # This is the sole graph source and malformed payloads fail above.
            edges = self._strict_row_edge_matrix(idx, len(coords))
        e = edges if isinstance(edges, torch.Tensor) else torch.tensor(edges)
        if e.dim() != 2 or e.shape[0] != e.shape[1] or dummy_idx >= e.shape[0]:
            raise ValueError(
                f"phase2 fragment row {idx} has invalid edge shape {tuple(e.shape)}"
            )
        neighbors = [
            int(value)
            for value in (e[:, dummy_idx] != 0).nonzero(as_tuple=False).view(-1)
            if int(value) != dummy_idx
        ]
        if len(neighbors) != 1:
            if not neighbors:
                raise ValueError(
                    f"phase2 fragment row {idx} dummy has no neighbors"
                )
            # Some synthetic fragments have ring-connected dummy atoms (degree 2).
            # Use the first neighbor as the anchor and zero out the other bond.
            source_anchor = neighbors[0]
            for other in neighbors[1:]:
                e[other, dummy_idx] = 0
                e[dummy_idx, other] = 0
        else:
            source_anchor = neighbors[0]
        bond_type = int(e[source_anchor, dummy_idx])
        if bond_type != 1:
            # Forcibly normalize to single bond instead of crashing — some
            # synthetic fragments have aromatic/double dummy bonds that the
            # generator didn't filter. The decoder target is always single.
            e[source_anchor, dummy_idx] = 1
            e[dummy_idx, source_anchor] = 1
        anchor = source_anchor - int(source_anchor > dummy_idx)
        keep = [i for i in range(e.shape[0]) if i != dummy_idx]
        backbone_edges = e[keep][:, keep]
        return backbone_smiles, backbone_coords, backbone_edges, anchor

    def _process_chartok_coords(self, idx, ref, smiles, coords=None, edges=None, mask_ratio=0):
        max_len = FORMAT_INFO['chartok_coords']['max_len']
        tokenizer = self.tokenizer['chartok_coords']
        if smiles is None or type(smiles) is not str:
            smiles = ""
        # Phase 2: for fragment rows, replace the dummy-bearing target with the
        # dummy-free backbone + a `<sep>[anchor:*]` attachment EXTENSION (metadata
        # lives OUTSIDE the AR structure trajectory ⇒ no structure corruption).
        extension_ids = []
        phase2_fragment = self.phase2_sep and int(
            self.df.loc[idx].get("structure_type_label", -1) or -1
        ) == 2
        if phase2_fragment:
            if not self._is_phase2_fragment(idx):
                raise ValueError(
                    f"phase2 fragment row {idx} lacks the native attachment contract"
                )
            smiles, coords, edges, anchor = self._phase2_fragment_inputs(idx, coords, edges)
            # Fragment decoder keeps coordinate supervision: the (x,y) tokens anchor
            # each atom to its image position, which is essential for the model to
            # learn which atom the wavy-attachment dummy bonds to. Dropping coords
            # (nopose) caused systematic dummy-attachment-site errors because the
            # decoder lost the spatial cue distinguishing the attachment carbon
            # from neighbouring atoms. BioChemInsight assembly only consumes the
            # SMILES, but coordinate supervision during training improves the
            # symbol/edge accuracy that produces that SMILES.
            extension_text = "[%d:*]" % int(anchor)
            extension_ids = [
                tokenizer.symbol_to_id(c) for c in extension_text
            ]
            if any(token_id == UNK_ID for token_id in extension_ids):
                raise ValueError(
                    f"phase2 fragment row {idx} extension uses an unknown token"
                )
            # Phase-2 diagnostic (prints a few times then quiets)
            if getattr(TrainDataset, "_sep_diag", 0) < 3:
                TrainDataset._sep_diag = getattr(TrainDataset, "_sep_diag", 0) + 1
                import sys as _sys
                print(f"[phase2-sep] idx={idx} backbone={smiles!r} anchor={anchor} ext={extension_text}", file=_sys.stderr)
        label, indices = tokenizer.smiles_to_sequence(smiles, coords, mask_ratio=mask_ratio)
        if extension_ids and getattr(tokenizer, "sep_id", None) is not None:
            # insert <sep> + extension before the trailing EOS
            label = label[:-1] + [tokenizer.sep_id] + extension_ids + [EOS_ID]
            if len(label) > max_len:
                raise ValueError(
                    f"phase2 fragment row {idx} target length {len(label)} exceeds {max_len}"
                )
        ref['chartok_coords'] = torch.LongTensor(label[:max_len])
        indices = [i for i in indices if i < max_len]
        ref['atom_indices'] = torch.LongTensor(indices)
        if tokenizer.continuous_coords:
            if coords is not None:
                ref['coords'] = torch.tensor(coords)
            else:
                ref['coords'] = torch.ones(len(indices), 2) * -1.
        if edges is not None:
            ref['edges'] = torch.tensor(edges)[:len(indices), :len(indices)]
        else:
            raw_edges = optional_text(self.df.loc[idx, 'edges']) if 'edges' in self.df.columns else ""
            if raw_edges:
                edge_list = eval(raw_edges)
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


class AuxTrainDataset(Dataset):

    def __init__(self, args, train_df, aux_df, tokenizer):
        super().__init__()
        self.train_dataset = TrainDataset(args, train_df, tokenizer, dynamic_indigo=args.dynamic_indigo)
        self.aux_dataset = TrainDataset(args, aux_df, tokenizer, dynamic_indigo=False)

    def __len__(self):
        return len(self.train_dataset) + len(self.aux_dataset)

    def __getitem__(self, idx):
        if idx < len(self.train_dataset):
            return self.train_dataset[idx]
        else:
            return self.aux_dataset[idx - len(self.train_dataset)]


def pad_images(imgs):
    # B, C, H, W
    max_shape = [0, 0]
    for img in imgs:
        for i in range(len(max_shape)):
            max_shape[i] = max(max_shape[i], img.shape[-1 - i])
    stack = []
    for img in imgs:
        pad = []
        for i in range(len(max_shape)):
            pad = pad + [0, max_shape[i] - img.shape[-1 - i]]
        stack.append(F.pad(img, pad, value=0))
    return torch.stack(stack)


def bms_collate(batch):
    ids = []
    imgs = []
    batch = [ex for ex in batch if ex[1] is not None]
    formats = list(batch[0][2].keys())
    seq_formats = [k for k in formats if
                   k in ['atomtok', 'inchi', 'nodes', 'atomtok_coords', 'chartok_coords', 'atom_indices']]
    refs = {key: [[], []] for key in seq_formats}
    for ex in batch:
        ids.append(ex[0])
        imgs.append(ex[1])
        ref = ex[2]
        for key in seq_formats:
            refs[key][0].append(ref[key])
            refs[key][1].append(torch.LongTensor([len(ref[key])]))
    # Sequence
    for key in seq_formats:
        # this padding should work for atomtok_with_coords too, each of which has shape (length, 4)
        refs[key][0] = pad_sequence(refs[key][0], batch_first=True, padding_value=PAD_ID)
        refs[key][1] = torch.stack(refs[key][1]).reshape(-1, 1)
    # Time
    # if 'time' in formats:
    #     refs['time'] = [ex[2]['time'] for ex in batch]
    # Coords
    if 'coords' in formats:
        refs['coords'] = pad_sequence([ex[2]['coords'] for ex in batch], batch_first=True, padding_value=-1.)
    if 'attachment_points' in formats:
        refs['attachment_points'] = pad_sequence(
            [ex[2]['attachment_points'] for ex in batch],
            batch_first=True,
            padding_value=-1.0,
        )
        refs['attachment_point_mask'] = pad_sequence(
            [ex[2]['attachment_point_mask'] for ex in batch],
            batch_first=True,
            padding_value=False,
        )
        refs['attachment_bonded'] = pad_sequence(
            [ex[2]['attachment_bonded'] for ex in batch],
            batch_first=True,
            padding_value=-1,
        )
        refs['attachment_bond_type'] = pad_sequence(
            [ex[2]['attachment_bond_type'] for ex in batch],
            batch_first=True,
            padding_value=-1,
        )
        refs['attachment_dummy_indices'] = pad_sequence(
            [ex[2]['attachment_dummy_indices'] for ex in batch],
            batch_first=True,
            padding_value=-1,
        )
        refs['attachment_anchor_points'] = pad_sequence(
            [ex[2]['attachment_anchor_points'] for ex in batch],
            batch_first=True,
            padding_value=-1.0,
        )
        refs['attachment_anchor_mask'] = pad_sequence(
            [ex[2]['attachment_anchor_mask'] for ex in batch],
            batch_first=True,
            padding_value=False,
        )
        refs['attachment_anchor_indices'] = pad_sequence(
            [ex[2]['attachment_anchor_indices'] for ex in batch],
            batch_first=True,
            padding_value=-1,
        )
        refs['attachment_atom_coords'] = pad_sequence(
            [ex[2]['attachment_atom_coords'] for ex in batch],
            batch_first=True,
            padding_value=-1.0,
        )
        refs['attachment_set_complete'] = torch.stack(
            [ex[2]['attachment_set_complete'] for ex in batch]
        ).bool()
        refs['attachment_count'] = torch.stack(
            [ex[2]['attachment_count'] for ex in batch]
        ).long()
    # Edges
    if 'edges' in formats:
        edges_list = [ex[2]['edges'] for ex in batch]
        max_len = max([len(edges) for edges in edges_list])
        refs['edges'] = torch.stack(
            [F.pad(edges, (0, max_len - len(edges), 0, max_len - len(edges)), value=-100) for edges in edges_list],
            dim=0)
    return ids, pad_images(imgs), refs
