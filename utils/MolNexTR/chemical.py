""" chemical rules"""
import copy
import itertools
import traceback
import numpy as np
import multiprocessing
import rdkit
from rdkit.Chem import rdFMCS, rdDepictor
import rdkit.Chem as Chem
rdkit.RDLogger.DisableLog('rdApp.*')
from SmilesPE.pretokenizer import atomwise_tokenizer
from .abbrs import RGROUP_SYMBOLS, ABBREVIATIONS, VALENCES, FORMULA_REGEX,SUBSTITUTIONS
import difflib
import re


class FunctionalGroupExpansionError(ValueError):
    """Raised when a predicted Markush/abbreviation atom cannot be expanded exactly."""


NUMBERED_DUMMY_ALIAS_RE = re.compile(r"^\[(\d+)\*\]$")


def get_smiles_stereo_list(smiles):
    pat = re.compile(r'\[C@[\w\d]*\]|\[C@@[\w\d]*\]')
    lst = []
    for m in pat.finditer(smiles):
        if '@@' in m.group():
            lst.append('@@')
        else:
            lst.append('@')
    return lst

def flip_stereo_in_smiles(smiles, flip_indices):
    pat = re.compile(r'\[C@[\w\d]*\]|\[C@@[\w\d]*\]')
    matches = list(pat.finditer(smiles))
    assert len(matches) >= max(flip_indices, default=-1) + 1, "索引越界"
    smiles_new = smiles
    offset = 0
    for idx in flip_indices:
        m = matches[idx]
        start, end = m.start() + offset, m.end() + offset
        orig = smiles_new[start:end]
        if '@@' in orig:
            flipped = orig.replace('@@', '@')
        else:
            flipped = orig.replace('@', '@@')
        smiles_new = smiles_new[:start] + flipped + smiles_new[end:]
        offset += len(flipped) - len(orig)
    return smiles_new

def chirality_sign(coords):
    p0, p1, p2 = coords[:3]
    v1 = np.array([p1.x - p0.x, p1.y - p0.y])
    v2 = np.array([p2.x - p0.x, p2.y - p0.y])
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    return np.sign(cross)

def align_chirality(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        # 1. Find MCS
        res = rdFMCS.FindMCS([mol1, mol2],
                             atomCompare=rdFMCS.AtomCompare.CompareElements,
                             bondCompare=rdFMCS.BondCompare.CompareOrder)
        mcs_mol = Chem.MolFromSmarts(res.smartsString)
        match1 = mol1.GetSubstructMatch(mcs_mol)
        match2 = mol2.GetSubstructMatch(mcs_mol)
        if not match1 or not match2:
            print("MCS映射失败，输出原始smiles2")
            return smiles2
        #print(f"mol1 MCS atoms: {match1}")
        #print(f"mol2 MCS atoms: {match2}")

        # 2. 2D coords for spatial arrangement comparison
        rdDepictor.Compute2DCoords(mol1)
        rdDepictor.Compute2DCoords(mol2)
        coords1 = [mol1.GetConformer().GetAtomPosition(i) for i in match1]
        coords2 = [mol2.GetConformer().GetAtomPosition(i) for i in match2]
        sign1 = chirality_sign(coords1)
        sign2 = chirality_sign(coords2)
        #print(f"mol1骨架排列: {sign1}，mol2骨架排列: {sign2}")
        is_mirror = (sign1 != sign2)

        # 3. Find chiral centers
        chiral1 = list(Chem.FindMolChiralCenters(mol1, includeUnassigned=False, includeCIP=True))
        chiral2 = list(Chem.FindMolChiralCenters(mol2, includeUnassigned=False, includeCIP=True))
        #print(f"mol1 chiral centers: {chiral1}")
        #print(f"mol2 chiral centers: {chiral2}")

        if len(chiral1) == 0:
            # 没有绝对手性，按SMILES手性符号逐一对齐
            stereo1 = get_smiles_stereo_list(smiles1)
            stereo2 = get_smiles_stereo_list(smiles2)
            #print("smiles1手性符号顺序:", stereo1)
            #print("smiles2手性符号顺序:", stereo2)
            flip_indices = []
            for i, (s1, s2) in enumerate(zip(stereo1, stereo2)):
                target = s1 if not is_mirror else ('@@' if s1 == '@' else '@')
                if s2 != target:
                    flip_indices.append(i)
            #print("需要flip的手性符号索引:", flip_indices)
            output_smiles2 = flip_stereo_in_smiles(smiles2, flip_indices)
            #print("对齐后smiles2:", output_smiles2)
            return output_smiles2
        if len(chiral1) != 0 and len(chiral2) != len(chiral1):
            #print("mol1,2有多个手性中心单没对齐，直接输出原始smiles2")
            return smiles2

        # 正常对齐手性（绝对R/S对齐）
        mol2_edit = Chem.RWMol(mol2)
        chiral1_dict = dict(chiral1)
        chiral2_dict = dict(chiral2)
        for i, idx1 in enumerate(match1):
            idx2 = match2[i]
            if idx1 in chiral1_dict:
                ref_chirality = chiral1_dict[idx1]
                if is_mirror:
                    ref_chirality = {'R':'S','S':'R'}.get(ref_chirality, ref_chirality)
                rs2 = chiral2_dict.get(idx2, None)
                #print(f"mol1原子{idx1}({ref_chirality}) <-> mol2原子{idx2}({rs2})")
                if rs2 is None:
                    pass
                elif ref_chirality == rs2:
                    pass
                else:
                    atom = mol2_edit.GetAtomWithIdx(idx2)
                    tag = atom.GetChiralTag()
                    if tag == Chem.CHI_TETRAHEDRAL_CW:
                        atom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)
                    elif tag == Chem.CHI_TETRAHEDRAL_CCW:
                        atom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)

        mol2_new = mol2_edit.GetMol()
        Chem.SanitizeMol(mol2_new)
        smiles2_new = Chem.MolToSmiles(mol2_new, isomericSmiles=True)
        #print("手性对齐后的smiles2:", smiles2_new)
        return smiles2_new

    except Exception as e:
        print(f"发生异常，输出原始smiles2: {e}")
        return smiles2
    







def is_valid_mol(s, format_='atomtok'):
    if format_ == 'atomtok':
        mol = Chem.MolFromSmiles(s)
    elif format_ == 'inchi':
        if not s.startswith('InChI=1S'):
            s = f"InChI=1S/{s}"
        mol = Chem.MolFromInchi(s)
    else:
        raise NotImplemented
    return mol is not None


def _convert_smiles_to_inchi(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        inchi = Chem.MolToInchi(mol)
    except:
        inchi = None
    return inchi


def convert_smiles_to_inchi(smiles_list, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        inchi_list = p.map(_convert_smiles_to_inchi, smiles_list, chunksize=128)
    n_success = sum([x is not None for x in inchi_list])
    r_success = n_success / len(inchi_list)
    inchi_list = [x if x else 'InChI=1S/H2O/h1H2' for x in inchi_list]
    return inchi_list, r_success


def merge_inchi(inchi1, inchi2):
    replaced = 0
    inchi1 = copy.deepcopy(inchi1)
    for i in range(len(inchi1)):
        if inchi1[i] == 'InChI=1S/H2O/h1H2':
            inchi1[i] = inchi2[i]
            replaced += 1
    return inchi1, replaced


def _get_num_atoms(smiles):
    try:
        return Chem.MolFromSmiles(smiles).GetNumAtoms()
    except:
        return 0


def get_num_atoms(smiles, num_workers=16):
    if type(smiles) is str:
        return _get_num_atoms(smiles)
    with multiprocessing.Pool(num_workers) as p:
        num_atoms = p.map(_get_num_atoms, smiles)
    return num_atoms


def normalize_nodes(nodes, flip_y=True):
    x, y = nodes[:, 0], nodes[:, 1]
    minx, maxx = min(x), max(x)
    miny, maxy = min(y), max(y)
    x = (x - minx) / max(maxx - minx, 1e-6)
    if flip_y:
        y = (maxy - y) / max(maxy - miny, 1e-6)
    else:
        y = (y - miny) / max(maxy - miny, 1e-6)
    return np.stack([x, y], axis=1)


def _verify_chirality(mol, coords, symbols, edges, debug=False):
    try:
        n = mol.GetNumAtoms()
        
        # Make a temp mol to find chiral centers
        mol_tmp = mol.GetMol()
        
        Chem.SanitizeMol(mol_tmp)
        #print(f"n: {n}", f"mol_tmp: {mol_tmp}")

        # chiral_centers = Chem.FindMolChiralCenters(
        #     mol_tmp, includeUnassigned=True, includeCIP=False, useLegacyImplementation=False)
        # chiral_center_ids = [idx for idx, _ in chiral_centers]
        # print(f"chiral_center_ids: {chiral_center_ids}")  # List[Tuple[int, any]] -> List[int]

        # if not chiral_center_ids:
        # # symbols 是一个原子符号列表（如 ['[F3C]', '[C@]', ...]）
        chiral_tags = ['[C@]', '[C@@]', '[C@H]', '[C@@H]']
        chiral_center_ids = [i for i, sym in enumerate(symbols) if any(tag in sym for tag in chiral_tags)]
        if debug:
            print(f"chiral_center_ids (from symbols): {chiral_center_ids}")
        
        # correction to clear pre-condition violation (for some corner cases)
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                bond.SetBondDir(Chem.BondDir.NONE)

        # Create conformer from 2D coordinate
        conf = Chem.Conformer(n)
        conf.Set3D(True)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
        mol.AddConformer(conf)
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistryFrom3D(mol)
        # NOTE: seems that only AssignStereochemistryFrom3D can handle double bond E/Z
        # So we do this first, remove the conformer and add back the 2D conformer for chiral correction

        mol.RemoveAllConformers()
        conf = Chem.Conformer(n)
        conf.Set3D(False)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
        mol.AddConformer(conf)

        # Magic, inferring chirality from coordinates and BondDir. DO NOT CHANGE.
        Chem.SanitizeMol(mol)
        Chem.AssignChiralTypesFromBondDirs(mol)
        Chem.AssignStereochemistry(mol, force=True)

        # Second loop to reset any wedge/dash bond to be starting from the chiral center)
        for i in chiral_center_ids:
            for j in range(n):
                if edges[i][j] == 5:
                    # assert edges[j][i] == 6
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINWEDGE)
                elif edges[i][j] == 6:
                    # assert edges[j][i] == 5
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINDASH)
            Chem.AssignChiralTypesFromBondDirs(mol)
            Chem.AssignStereochemistry(mol, force=True)

        # reset chiral tags for non-carbon atom
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        mol = mol.GetMol()

    except Exception as e:
        if debug:
            raise e
        pass
    return mol


def _parse_tokens(tokens: list):
    """
    Parse tokens of condensed formula into list of pairs `(elt, num)`
    where `num` is the multiplicity of the atom (or nested condensed formula) `elt`
    Used by `_parse_formula`, which does the same thing but takes a formula in string form as input
    """
    elements = []
    i = 0
    j = 0
    while i < len(tokens):
        if tokens[i] == '(':
            while j < len(tokens) and tokens[j] != ')':
                j += 1
            elt = _parse_tokens(tokens[i + 1:j])
        else:
            elt = tokens[i]
        j += 1
        if j < len(tokens) and tokens[j].isnumeric():
            num = int(tokens[j])
            j += 1
        else:
            num = 1
        elements.append((elt, num))
        i = j
    return elements


def _parse_formula(formula: str):
    """
    Parse condensed formula into list of pairs `(elt, num)`
    where `num` is the subscript to the atom (or nested condensed formula) `elt`
    Example: "C2H4O" -> [('C', 2), ('H', 4), ('O', 1)]
    """
    tokens = FORMULA_REGEX.findall(formula)
    # if ''.join(tokens) != formula:
    #     tokens = FORMULA_REGEX_BACKUP.findall(formula)
    return _parse_tokens(tokens)


def _expand_carbon(elements: list):
    """
    Given list of pairs `(elt, num)`, output single list of all atoms in order,
    expanding carbon sequences (CaXb where a > 1 and X is halogen) if necessary
    Example: [('C', 2), ('H', 4), ('O', 1)] -> ['C', 'H', 'H', 'C', 'H', 'H', 'O'])
    """
    expanded = []
    i = 0
    while i < len(elements):
        elt, num = elements[i]
        # expand carbon sequence
        if elt == 'C' and num > 1 and i + 1 < len(elements):
            next_elt, next_num = elements[i + 1]
            quotient, remainder = next_num // num, next_num % num
            for _ in range(num):
                expanded.append('C')
                for _ in range(quotient):
                    expanded.append(next_elt)
            for _ in range(remainder):
                expanded.append(next_elt)
            i += 2
        # recurse if `elt` itself is a list (nested formula)
        elif isinstance(elt, list):
            new_elt = _expand_carbon(elt)
            for _ in range(num):
                expanded.append(new_elt)
            i += 1
        # simplest case: simply append `elt` `num` times
        else:
            for _ in range(num):
                expanded.append(elt)
            i += 1
    return expanded


def _expand_abbreviation(abbrev):
    """
    Expand abbreviation into its SMILES; also converts [Rn] to [n*]
    Used in `_condensed_formula_list_to_smiles` when encountering abbrev. in condensed formula
    """
    if abbrev in ABBREVIATIONS:
        return ABBREVIATIONS[abbrev].smiles
    if abbrev in RGROUP_SYMBOLS or (abbrev[0] == 'R' and abbrev[1:].isdigit()):
        if abbrev[1:].isdigit():
            return f'[{abbrev[1:]}*]'
        return '*'
    return f'[{abbrev}]'


def _get_bond_symb(bond_num):
    """
    Get SMILES symbol for a bond given bond order
    Used in `_condensed_formula_list_to_smiles` while writing the SMILES string
    """
    if bond_num == 0:
        return '.'
    if bond_num == 1:
        return ''
    if bond_num == 2:
        return '='
    if bond_num == 3:
        return '#'
    return ''


def _condensed_formula_list_to_smiles(formula_list, start_bond, end_bond=None, direction=None):
    """
    Converts condensed formula (in the form of a list of symbols) to smiles
    Input:
    `formula_list`: e.g. ['C', 'H', 'H', 'N', ['C', 'H', 'H', 'H'], ['C', 'H', 'H', 'H']] for CH2N(CH3)2
    `start_bond`: # bonds attached to beginning of formula
    `end_bond`: # bonds attached to end of formula (deduce automatically if None)
    `direction` (1, -1, or None): direction in which to process the list (1: left to right; -1: right to left; None: deduce automatically)
    Returns:
    `smiles`: smiles corresponding to input condensed formula
    `bonds_left`: bonds remaining at the end of the formula (for connecting back to main molecule); should equal `end_bond` if specified
    `num_trials`: number of trials
    `success` (bool): whether conversion was successful
    """
    # `direction` not specified: try left to right; if fails, try right to left
    if direction is None:
        num_trials = 1
        for dir_choice in [1, -1]:
            smiles, bonds_left, trials, success = _condensed_formula_list_to_smiles(formula_list, start_bond, end_bond, dir_choice)
            num_trials += trials
            if success:
                return smiles, bonds_left, num_trials, success
        return None, None, num_trials, False
    assert direction == 1 or direction == -1

    def dfs(smiles, bonds_left, cur_idx, add_idx):
        """
        `smiles`: SMILES string so far
        `cur_idx`: index (in list `formula`) of current atom (i.e. atom to which subsequent atoms are being attached)
        `cur_flat_idx`: index of current atom in list of atom tokens of SMILES so far
        `bonds_left`: bonds remaining on current atom for subsequent atoms to be attached to
        `add_idx`: index (in list `formula`) of atom to be attached to current atom
        `add_flat_idx`: index of atom to be added in list of atom tokens of SMILES so far
        Note: "atom" could refer to nested condensed formula (e.g. CH3 in CH2N(CH3)2)
        """
        num_trials = 1
        # end of formula: return result
        if (direction == 1 and add_idx == len(formula_list)) or (direction == -1 and add_idx == -1):
            if end_bond is not None and end_bond != bonds_left:
                return smiles, bonds_left, num_trials, False
            return smiles, bonds_left, num_trials, True

        # no more bonds but there are atoms remaining: conversion failed
        if bonds_left <= 0:
            return smiles, bonds_left, num_trials, False
        to_add = formula_list[add_idx]  # atom to be added to current atom

        if isinstance(to_add, list):  # "atom" added is a list (i.e. nested condensed formula): assume valence of 1
            if bonds_left > 1:
                # "atom" added does not use up remaining bonds of current atom
                # get smiles of "atom" (which is itself a condensed formula)
                add_str, val, trials, success = _condensed_formula_list_to_smiles(to_add, 1, None, direction)
                if val > 0:
                    add_str = _get_bond_symb(val + 1) + add_str
                num_trials += trials
                if not success:
                    return smiles, bonds_left, num_trials, False
                # put smiles of "atom" in parentheses and append to smiles; go to next atom to add to current atom
                result = dfs(smiles + f'({add_str})', bonds_left - 1, cur_idx, add_idx + direction)
            else:
                # "atom" added uses up remaining bonds of current atom
                # get smiles of "atom" and bonds left on it
                add_str, bonds_left, trials, success = _condensed_formula_list_to_smiles(to_add, 1, None, direction)
                num_trials += trials
                if not success:
                    return smiles, bonds_left, num_trials, False
                # append smiles of "atom" (without parentheses) to smiles; it becomes new current atom
                result = dfs(smiles + add_str, bonds_left, add_idx, add_idx + direction)
            smiles, bonds_left, trials, success = result
            num_trials += trials
            return smiles, bonds_left, num_trials, success

        # atom added is a single symbol (as opposed to nested condensed formula)
        for val in VALENCES.get(to_add, [1]):  # try all possible valences of atom added
            add_str = _expand_abbreviation(to_add)  # expand to smiles if symbol is abbreviation
            if bonds_left > val:  # atom added does not use up remaining bonds of current atom; go to next atom to add to current atom
                if cur_idx >= 0:
                    add_str = _get_bond_symb(val) + add_str
                result = dfs(smiles + f'({add_str})', bonds_left - val, cur_idx, add_idx + direction)
            else:  # atom added uses up remaining bonds of current atom; it becomes new current atom
                if cur_idx >= 0:
                    add_str = _get_bond_symb(bonds_left) + add_str
                result = dfs(smiles + add_str, val - bonds_left, add_idx, add_idx + direction)
            trials, success = result[2:]
            num_trials += trials
            if success:
                return result[0], result[1], num_trials, success
            if num_trials > 10000:
                break
        return smiles, bonds_left, num_trials, False

    cur_idx = -1 if direction == 1 else len(formula_list)
    add_idx = 0 if direction == 1 else len(formula_list) - 1
    return dfs('', start_bond, cur_idx, add_idx)


def get_smiles_from_symbol(symbol, mol, atom, bonds):
    """
    Convert symbol (abbrev. or condensed formula) to smiles
    If condensed formula, determine parsing direction and num. bonds on each side using coordinates
    """
    if symbol in ABBREVIATIONS:
        return ABBREVIATIONS[symbol].smiles
    if len(symbol) > 20:
        return None

    total_bonds = int(sum([bond.GetBondTypeAsDouble() for bond in bonds]))
    formula_list = _expand_carbon(_parse_formula(symbol))
    smiles, bonds_left, num_trails, success = _condensed_formula_list_to_smiles(formula_list, total_bonds, None)
    if success:
        return smiles
    return None


def _replace_functional_group(smiles):
    smiles = smiles.replace('<unk>', 'C')
    for i, r in enumerate(RGROUP_SYMBOLS):
        symbol = f'[{r}]'
        if symbol in smiles:
            if r[0] == 'R' and r[1:].isdigit():
                smiles = smiles.replace(symbol, f'[{int(r[1:])}*]')
            else:
                smiles = smiles.replace(symbol, '*')
    # For unknown tokens (i.e. rdkit cannot parse), replace them with [{isotope}*], where isotope is an identifier.
    tokens = atomwise_tokenizer(smiles)
    new_tokens = []
    mappings = {}  # isotope : symbol
    isotope = 50
    for token in tokens:
        if token[0] == '[':
            parsed_atom = Chem.AtomFromSmiles(token)
            if parsed_atom is None:
                while f'[{isotope}*]' in smiles or f'[{isotope}*]' in new_tokens:
                    isotope += 1
                placeholder = f'[{isotope}*]'
                mappings[isotope] = token[1:-1]
                new_tokens.append(placeholder)
                continue
        new_tokens.append(token)
    smiles = ''.join(new_tokens)
    return smiles, mappings


def convert_smiles_to_mol(smiles):
    if smiles is None or smiles == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return None
    return mol


BOND_TYPES = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}


def _num_swaps_to_interconvert(orders):
    n = len(orders)
    seen = [False] * n
    nswaps = 0
    for i in range(n):
        if not seen[i]:
            j = i
            while orders[j] != i:
                j = orders[j]
                if j >= n:
                    raise ValueError("_num_swaps_to_interconvert: index outside range")
                seen[j] = True
                nswaps += 1
    return nswaps

    
def _expand_functional_group(mol, mappings, debug=True):
    def _need_expand(mol, mappings):
        return any([len(Chem.GetAtomAlias(atom)) > 0 for atom in mol.GetAtoms()]) or len(mappings) > 0

    if _need_expand(mol, mappings):
        mol_w = Chem.RWMol(mol)
        num_atoms = mol_w.GetNumAtoms()

        # 重置所有原子的自由基电子
        for atom in mol_w.GetAtoms():
            atom.SetNumRadicalElectrons(0)

        atoms_to_remove = []

        for i in range(num_atoms):
            atom = mol_w.GetAtomWithIdx(i)
            if atom.GetSymbol() != '*':
                continue

            symbol = Chem.GetAtomAlias(atom)
            isotope = atom.GetIsotope()
            if isotope > 0 and isotope in mappings:
                symbol = mappings[isotope]

            if not (isinstance(symbol, str) and len(symbol) > 0):
                continue

            # 裸 '*' 是 SMILES 通配符附件点（attachment point，atomic num 0，无缩写别名），
            # 必须原样保留，不能当成可展开的官能团缩写——否则 get_smiles_from_symbol('*')
            # 解析失败并抛 functional_group_expansion_failed:*（MoE specialist 的 fragment/
            # markush 输出正是裸 '*'，这是导致组装产出空 SMILES 的根因）。与 R-group 标记同理。
            if symbol == '*':
                continue

            # R-group markers are graph attachment atoms, not abbreviations.
            # MolNexTR emits both textual R1/R2 labels and RDKit isotope-dummy
            # spellings such as [1*]. The latter already parsed correctly in
            # _atom_from_predicted_symbol; expanding its MolBlock alias as a
            # condensed formula would discard an otherwise valid graph.
            if symbol in RGROUP_SYMBOLS or NUMBERED_DUMMY_ALIAS_RE.fullmatch(symbol):
                continue

            bonds = atom.GetBonds()
            sub_smiles = get_smiles_from_symbol(symbol, mol_w, atom, bonds)

            # 从 SMILES 获得官能团分子
            mol_r = convert_smiles_to_mol(sub_smiles)
            if mol_r is None:
                raise FunctionalGroupExpansionError(
                    f"functional_group_expansion_failed:{symbol}"
                )

            # ====== 记录原始键信息 & 可能受影响的手性中心 ======
            bond_infos = []
            chiral_centers_affected = set()
            bonds_list = list(bonds)

            for bond in bonds_list:
                adj_idx = bond.GetOtherAtomIdx(i)
                bond_infos.append(
                    (
                        adj_idx,
                        int(round(bond.GetBondTypeAsDouble())),
                        bond.GetBondDir()
                    )
                )
                adj_atom = mol_w.GetAtomWithIdx(adj_idx)
                if adj_atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                    chiral_centers_affected.add(adj_idx)

            # ====== 类 molzip_like：连接前标记手性中心邻居顺序 ======
            chiral_mark_dict = {}  # {chiral_idx: mark_name}
            for chiral_idx in chiral_centers_affected:
                chiral_atom = mol_w.GetAtomWithIdx(chiral_idx)
                tag = chiral_atom.GetChiralTag()
                if tag not in (
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                ):
                    continue

                mark_name = f"__expand_chiral_mark_{chiral_idx}"
                chiral_mark_dict[chiral_idx] = mark_name

                neighbors_before = list(chiral_atom.GetNeighbors())
                order = 0
                for nbr in neighbors_before:
                    nbr.SetIntProp(mark_name, order)
                    order += 1

                if debug:
                    print(f"  Marking neighbor order of chiral center {chiral_idx} (before expansion)")
                    for idx_nb, nbr in enumerate(neighbors_before):
                        if nbr.HasProp(mark_name):
                            print(
                                f"    Neighbor {nbr.GetIdx()}: order {nbr.GetIntProp(mark_name)} "
                                f"(the {idx_nb}-th in GetNeighbors order)"
                            )

            if debug:
                print(f"Expanding functional group {symbol} (atom {i})")
                print(f"  bond_infos: {bond_infos}")
                print(f"  chiral_centers_affected: {chiral_centers_affected}")

            # ====== 断开 * 与主体之间的所有键，并用自由基“记”键阶 ======
            adjacent_indices = [bond.GetOtherAtomIdx(i) for bond in bonds_list]
            for adjacent_idx in adjacent_indices:
                mol_w.RemoveBond(i, adjacent_idx)

            adjacent_atoms = [mol_w.GetAtomWithIdx(adj_idx) for adj_idx in adjacent_indices]
            for adjacent_atom, bond in zip(adjacent_atoms, bonds_list):
                adjacent_atom.SetNumRadicalElectrons(int(bond.GetBondTypeAsDouble()))

            bonding_atoms_w = adjacent_indices  # 主体侧连接点

            if debug:
                print(f"  Main molecule connection points (bonding_atoms_w): {bonding_atoms_w}")

            # ====== 分析 sub_smiles 中的连接点顺序（官能团侧） ======
            sub_smiles_atoms = []
            if sub_smiles:
                try:
                    temp_mol = Chem.MolFromSmiles(sub_smiles)
                    if temp_mol:
                        for atm in temp_mol.GetAtoms():
                            if atm.GetNumRadicalElectrons() > 0:
                                sub_smiles_atoms.append(atm.GetIdx())
                        for atm in temp_mol.GetAtoms():
                            if atm.GetSymbol() == '*':
                                sub_smiles_atoms.append(atm.GetIdx())
                except Exception as e:
                    if debug:
                        print(f"  Failed to parse sub_smiles: {e}")

            bonding_atoms_r = []

            # 方法 1：按 sub_smiles 中自由基顺序确定连接点
            if sub_smiles and len(sub_smiles_atoms) > 0:
                base_idx = mol_w.GetNumAtoms()
                for star_idx in sub_smiles_atoms:
                    # star_idx 是 mol_r 中的原子 index
                    bonding_atoms_r.append(base_idx + star_idx)

            if len(bonding_atoms_r) == 0:
                raise FunctionalGroupExpansionError(
                    f"missing_functional_group_attachment_point:{symbol}"
                )

            if debug:
                print(f"  Functional group connection points (bonding_atoms_r estimated): {bonding_atoms_r}")
                print(f"  sub_smiles: {sub_smiles}")
                print(f"  sub_smiles_atoms: {sub_smiles_atoms}")

            # ====== Combine 主体与官能团 ======
            combo = Chem.CombineMols(mol_w, mol_r)
            mol_w = Chem.RWMol(combo)

            # ====== 决定最终配对的 target_atoms（官能团侧连接点） ======
            target_atoms = []
            if len(bonding_atoms_r) == len(bonding_atoms_w):
                target_atoms = bonding_atoms_r
                if debug:
                    print(f"  Connection points count matches, matching in order: {bonding_atoms_w} -> {target_atoms}")
            elif len(bonding_atoms_r) >= len(bonding_atoms_w):
                target_atoms = bonding_atoms_r[:len(bonding_atoms_w)]
                if debug:
                    print(f"  More functional group connection points, taking first {len(bonding_atoms_w)}: {target_atoms}")
            else:
                raise FunctionalGroupExpansionError(
                    "functional_group_attachment_count_mismatch:"
                    f"{symbol}:main={len(bonding_atoms_w)}:group={len(bonding_atoms_r)}"
                )

            # ====== 加键 + 继承方向 + 传递手性标记 ======
            for info, target_idx in zip(bond_infos, target_atoms):
                adj_idx, order_val, bond_dir = info
                order_val = max(1, min(3, order_val))
                mol_w.GetAtomWithIdx(adj_idx).SetNumRadicalElectrons(order_val)

                # 避免重复加键
                existing_bond = mol_w.GetBondBetweenAtoms(adj_idx, target_idx)
                if existing_bond is None:
                    mol_w.AddBond(
                        adj_idx,
                        target_idx,
                        order=BOND_TYPES.get(order_val, Chem.rdchem.BondType.SINGLE),
                    )

                new_bond = mol_w.GetBondBetweenAtoms(adj_idx, target_idx)
                if new_bond is not None and bond_dir != Chem.BondDir.NONE:
                    new_bond.SetBondDir(bond_dir)

                # ====== 核心修正：从 dummy(*) 继承手性邻居顺序标记 ======
                # i 是当前正在展开的 '*' 原子索引
                dummy_atom = mol_w.GetAtomWithIdx(i)
                for chiral_idx, mark_name in chiral_mark_dict.items():
                    if dummy_atom.HasProp(mark_name):
                        adj_order = dummy_atom.GetIntProp(mark_name)
                        target_atom = mol_w.GetAtomWithIdx(target_idx)
                        target_atom.SetIntProp(mark_name, adj_order)
                        if debug:
                            print(
                                f"  Transferring mark: dummy atom {i} (order {adj_order}, chiral center {chiral_idx})"
                                f" -> new atom {target_idx}"
                            )

            # ====== 连接后恢复手性（和 molzip_like 相同思想） ======
            for chiral_idx in chiral_centers_affected:
                if chiral_idx not in chiral_mark_dict:
                    continue

                mark_name = chiral_mark_dict[chiral_idx]
                chiral_atom = mol_w.GetAtomWithIdx(chiral_idx)
                tag = chiral_atom.GetChiralTag()

                if tag not in (
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                ):
                    continue

                neighbors_after = list(chiral_atom.GetNeighbors())
                orders_after = []
                all_have_mark = True
                for nbr in neighbors_after:
                    if not nbr.HasProp(mark_name):
                        all_have_mark = False
                        break
                    orders_after.append(nbr.GetIntProp(mark_name))

                if all_have_mark and len(orders_after) > 0:
                    if debug:
                        print(f"  Chiral center {chiral_idx}: neighbor order after connection {orders_after}")
                    try:
                        if set(orders_after) == set(range(len(orders_after))):
                            nswaps = _num_swaps_to_interconvert(orders_after)
                            if debug:
                                print(f"  Number of swaps: {nswaps}")
                            if nswaps % 2 == 1:
                                if tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                                    chiral_atom.SetChiralTag(
                                        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
                                    )
                                    if debug:
                                        print(f"  Flipping chiral center {chiral_idx}: CW -> CCW")
                                elif tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                                    chiral_atom.SetChiralTag(
                                        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
                                    )
                                    if debug:
                                        print(f"  Flipping chiral center {chiral_idx}: CCW -> CW")
                        else:
                            if debug:
                                print(
                                    f"  Warning: Neighbor marks of chiral center {chiral_idx} are not a valid permutation: {orders_after}"
                                )
                    except Exception as e:
                        if debug:
                            print(f"  Error calculating number of swaps (chiral center {chiral_idx}): {e}")
                else:
                    if debug:
                        missing = [
                            nbr.GetIdx()
                            for nbr in neighbors_after
                            if not nbr.HasProp(mark_name)
                        ]
                        print(
                            f"  Chiral center {chiral_idx}: Some neighbors lack marks, cannot determine chirality change (neighbors missing marks: {missing})"
                        )

            # 清除临时自由基
            for atm_idx in bonding_atoms_w:
                mol_w.GetAtomWithIdx(atm_idx).SetNumRadicalElectrons(0)
            for atm_idx in bonding_atoms_r:
                if 0 <= atm_idx < mol_w.GetNumAtoms():
                    mol_w.GetAtomWithIdx(atm_idx).SetNumRadicalElectrons(0)

            # 局部 sanitize（不强制重算立体化学）
            try:
                Chem.SanitizeMol(mol_w)
            except Exception as e:
                if debug:
                    print(f"Warning: Failed to sanitize after expanding {symbol}: {e}")

            # 记录要删除的 '*' 原子
            atoms_to_remove.append(i)

        # ====== 删除所有 * 原子（从大 index 开始） ======
        atoms_to_remove = sorted(set(atoms_to_remove), reverse=True)
        for idx in atoms_to_remove:
            if idx < mol_w.GetNumAtoms():
                mol_w.RemoveAtom(idx)

        # 清理临时手性标记属性
        for atom in mol_w.GetAtoms():
            for prop in list(atom.GetPropNames()):
                if prop.startswith("__expand_chiral_mark"):
                    atom.ClearProp(prop)

        # 最终 sanitize
        try:
            Chem.SanitizeMol(mol_w)
        except Exception as e:
            if debug:
                print("Warning: Failed to final sanitize after expanding functional groups:", e)

        smiles = Chem.MolToSmiles(mol_w, isomericSmiles=True)
        mol = mol_w.GetMol()
    else:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        mol = mol

    return smiles, mol



def _atom_from_predicted_symbol(symbol):
    bracketed = symbol.startswith("[") and symbol.endswith("]")
    normalized = symbol[1:-1] if bracketed else symbol
    if not bracketed and normalized in RGROUP_SYMBOLS:
        atom = Chem.Atom("*")
        if normalized[0] == 'R' and normalized[1:].isdigit():
            atom.SetIsotope(int(normalized[1:]))
        atom.SetProp('atomLabel', normalized)
        return atom
    if not bracketed and normalized in ABBREVIATIONS:
        atom = Chem.Atom("*")
        atom.SetProp('atomLabel', normalized)
        return atom
    parsed_atom = None
    try:
        parsed_atom = Chem.AtomFromSmiles(symbol)
    except Exception:
        parsed_atom = None
    if parsed_atom is not None:
        parsed_atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        return parsed_atom
    if normalized in RGROUP_SYMBOLS:
        atom = Chem.Atom("*")
        if normalized[0] == 'R' and normalized[1:].isdigit():
            atom.SetIsotope(int(normalized[1:]))
        atom.SetProp('atomLabel', normalized)
        return atom
    atom = Chem.Atom("*")
    atom.SetProp('atomLabel', normalized)
    return atom


def _scaled_graph_coords(coords, image=None):
    if image is None:
        return coords
    height, width, _ = image.shape
    ratio = width / height
    return [[x * ratio * 10, y * 10] for x, y in coords]


def _set_graph_2d_conformer(mol, coords):
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(False)
    for i, (x, y) in enumerate(coords):
        conf.SetAtomPosition(i, (x, 1 - y, 0))
    mol.AddConformer(conf, assignId=True)
    return mol


# Bond-order drop preference: lower = remove first when repairing an over-valent
# atom. Single/wedge bonds are the most disposable; aromatic/double/triple are
# preserved in preference. Matches the valence accounting in
# _max_bond_valence_weight above (aromatic counts 1.0 — chemically correct, so
# valid fused-ring junction atoms are never wrongly flagged).
_BOND_DROP_RANK = {1: 0, 5: 1, 6: 1, 4: 2, 2: 3, 3: 4}


def _max_bond_valence_weight(edge_value: int) -> float:
    v = int(edge_value)
    if v == 2:
        return 2.0
    if v == 3:
        return 3.0
    if v in (1, 4, 5, 6):
        return 1.0
    return 0.0


def _extract_element(symbol) -> str:
    """Extract the periodic-table element name from a predicted atom symbol.

    Handles bracketed ([C@H], [nH], [13*], [SiH2]) and bare (C, c, Cl, Si)
    forms. Returns the uppercase element name, or "" for wildcards/unknowns.
    """
    text = str(symbol or "").strip()
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1]
        if inner.endswith("*"):
            return ""
        # Try uppercase first (e.g. [C@H] → C, [SiH2] → Si, [NH+] → N)
        m = re.match(r"([A-Z][a-z]?)", inner)
        if m:
            return m.group(1).upper()
        # Try lowercase aromatic (e.g. [nH] → N, [o] → O, [s] → S)
        m = re.match(r"([a-z])", inner)
        if m:
            return m.group(1).upper()
        return ""
    if text in ("*", "") or text.startswith("R"):
        return ""
    if len(text) == 1:
        return text.upper()
    return text.upper()


def _valence_cap_for_symbol(symbol: str) -> float:
    """Max permitted valence for a predicted atom symbol."""
    element = _extract_element(symbol)
    if not element:
        return 1.0  # wildcard/attachment point
    allowed = VALENCES.get(element)
    if not allowed:
        return 99.0
    return float(max(allowed))


def _is_attachment_symbol(symbol) -> bool:
    text = str(symbol or "").strip()
    if not text:
        return False
    if text.startswith("[") and text.endswith("]") and text[1:-1].endswith("*"):
        return True
    return text == "*" or text.startswith("R")


def repair_hypervalent_edges(symbols, edges):
    """Drop bonds from over-valent atoms until the graph sanitizes.

    Valid graphs are returned unchanged. For over-valent graphs, removes
    the lowest-priority bond (dummy bonds first, then lower bond orders)
    on each flagged atom until rdkit accepts the molecule.
    """
    n = len(symbols)
    if n == 0 or edges is None:
        return edges

    code_to_type = {
        1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC, 5: Chem.BondType.SINGLE, 6: Chem.BondType.SINGLE,
    }

    def build_mol():
        mol = Chem.RWMol()
        for s in symbols:
            try:
                atom = _atom_from_predicted_symbol(s)
            except Exception:
                atom = Chem.Atom(0) if s == "*" else Chem.Atom(6)
            mol.AddAtom(atom)
        for i in range(n):
            for j in range(i + 1, n):
                bt = code_to_type.get(int(edges[i][j]))
                if bt is not None:
                    mol.AddBond(i, j, bt)
        return mol

    # Fast path: valid graph returned unchanged.
    try:
        Chem.SanitizeMol(build_mol())
        return edges
    except Exception:
        pass

    # Repair loop: remove the weakest-priority bond from each over-valent atom.
    # ONLY act on explicit-valence violations ("greater than permitted"); other
    # sanitize failures (kekulization, aromaticity, radical) are left untouched
    # so we don't over-repair chemically-different problems.
    edges = [list(row) for row in edges]
    for _ in range(2 * n + 4):
        try:
            Chem.SanitizeMol(build_mol())
            break
        except Exception as e:
            msg = str(e)
        if "valence" not in msg.lower() and "greater than permitted" not in msg.lower():
            break
        am = re.search(r"atom\s*#?\s*(\d+)", msg)
        if not am:
            break
        worst = int(am.group(1))
        if worst >= n:
            break
        # Remove the lowest-priority bond (dummy first, then low bond order).
        # Orphaning a dummy is acceptable — it keeps the SMILES valid.
        best_key, best_drop = None, None
        for j in range(n):
            if j == worst:
                continue
            a, b = (worst, j) if worst < j else (j, worst)
            code = int(edges[a][b])
            if not code:
                continue
            attachment = 0 if _is_attachment_symbol(symbols[j]) else 1
            rank = _BOND_DROP_RANK.get(code, 0)
            key = (attachment, rank)
            if best_drop is None or key < best_drop:
                best_drop, best_key = key, (a, b)
        if best_key is None:
            break
        a, b = best_key
        edges[a][b] = 0
        edges[b][a] = 0
    return edges


def re_bond_orphan_dummies(symbols, edges, coords):
    """Re-attach orphaned dummy atoms (degree 0 after hypervalence repair) to the
    nearest backbone atom that has remaining valence.

    ``repair_hypervalent_edges`` removes bonds from over-valent atoms, preferring
    dummy bonds. When a dummy's only bond is removed, it becomes a disconnected
    component (.[n*]). This function re-bonds each such orphan to the nearest
    non-dummy atom that still has valence room, restoring graph connectivity.

    No-op when there are no orphans, or when coords are unavailable (returns the
    edges unchanged). Only adds single bonds (dummy attachment points are single-
    bond by convention). Never bonds to an atom that would become hypervalent.
    """
    n = len(symbols)
    if n == 0 or edges is None or coords is None:
        return edges
    if len(coords) < n:
        return edges

    edges = [list(row) for row in edges]

    def _degree(atom_index):
        return sum(1 for v in edges[atom_index] if int(v) > 0)

    for i in range(n):
        if not _is_attachment_symbol(symbols[i]):
            continue
        if _degree(i) > 0:
            continue  # already bonded, not an orphan

        # Find nearest backbone atom with valence room.
        best_j = -1
        best_dist = float("inf")
        xi, yi = float(coords[i][0]), float(coords[i][1])
        for j in range(n):
            if j == i:
                continue
            if _is_attachment_symbol(symbols[j]):
                continue
            used = sum(_max_bond_valence_weight(v) for v in edges[j])
            cap_j = _valence_cap_for_symbol(symbols[j])
            # Leave room for implicit H on common elements (rdkit adds H to
            # unfilled C/N/O/S/P/B/Si). Bracketed spellings ([C@H], [nH], etc.)
            # are normalized via the same element-extraction as _valence_cap.
            elem_j = _extract_element(symbols[j])
            needs_h_room = elem_j in ("C", "N", "O", "S", "P", "B", "SI")
            room_needed = 2.0 if needs_h_room else 1.0
            if used + room_needed > cap_j + 1e-6:
                continue
            xj, yj = float(coords[j][0]), float(coords[j][1])
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            if d < best_dist:
                best_dist = d
                best_j = j

        if best_j >= 0:
            a, b = (i, best_j) if i < best_j else (best_j, i)
            edges[a][b] = 1
            edges[b][a] = 1

    return edges


def _convert_graph_to_smiles(coords, symbols, edges, image=None, debug=False):
    mol = Chem.RWMol()
    n = len(symbols)
    ids = []
    for i in range(n):
        symbol = symbols[i]
        atom = _atom_from_predicted_symbol(symbol)

        if atom.GetSymbol() == '*':
            atom.SetProp('molFileAlias', symbol)

        idx = mol.AddAtom(atom)
        assert idx == i
        ids.append(idx)

    # Repair over-valent graphs before building the molecule.
    edges = repair_hypervalent_edges(symbols, edges)

    # Re-attach orphaned dummies (degree 0 after repair) to nearest atom with valence room.
    edges = re_bond_orphan_dummies(symbols, edges, coords)

    for i in range(n):
        for j in range(i + 1, n):
            if edges[i][j] == 1:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
            elif edges[i][j] == 2:
                mol.AddBond(ids[i], ids[j], Chem.BondType.DOUBLE)
            elif edges[i][j] == 3:
                mol.AddBond(ids[i], ids[j], Chem.BondType.TRIPLE)
            elif edges[i][j] == 4:
                mol.AddBond(ids[i], ids[j], Chem.BondType.AROMATIC)
            elif edges[i][j] == 5:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINWEDGE)
            elif edges[i][j] == 6:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINDASH)

    try:
        pred_smiles = rdkit.Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception as e:
        pred_smiles = '<invalid>'
    pred_molblock = ''
    success = False
    quality_issue = ''
    try:
        coords = _scaled_graph_coords(coords, image=image)
        graph_mol = _set_graph_2d_conformer(mol.GetMol(), coords)
        # MolBlock is the graph output and must preserve predicted 2D pose,
        # including Markush dummy/R-group atoms, before later text expansion.
        pred_molblock = Chem.MolToMolBlock(graph_mol, kekulize=False)

        mol = _verify_chirality(mol, coords, symbols, edges, debug)
        smiles1 = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        #print(f"after_chirality_SMILES: {smiles1}")
        pred_smiles, mol = _expand_functional_group(mol, {}, debug)
        success = True
    except Exception as e:
        quality_issue = str(e) or e.__class__.__name__
        if debug:
            print(traceback.format_exc())
        success = False

    if debug:
        return pred_smiles, pred_molblock, mol, success, quality_issue
    return pred_smiles, pred_molblock, success, quality_issue


def convert_graph_to_smiles(coords, symbols, edges, images=None, num_workers=16):
    if images is None:
        args_zip = zip(coords, symbols, edges)
    else:
        args_zip = zip(coords, symbols, edges, images)

    if num_workers <= 1:
        results = itertools.starmap(_convert_graph_to_smiles, args_zip)
        results = list(results)
    else:
        with multiprocessing.Pool(num_workers) as p:
            results = p.starmap(_convert_graph_to_smiles, args_zip, chunksize=128)

    smiles_list, molblock_list, success, quality_issues = zip(*results)
    r_success = np.mean(success)
    return smiles_list, molblock_list, r_success, quality_issues


def _postprocess_smiles(smiles, coords=None, symbols=None, edges=None, molblock=False, debug=False):
    if type(smiles) is not str or smiles == '':
        return '', False
    mol = None
    pred_molblock = ''
    quality_issue = ''
    try:
        pred_smiles = smiles
        pred_smiles, mappings = _replace_functional_group(pred_smiles)
        if coords is not None and symbols is not None and edges is not None:
            pred_smiles = pred_smiles.replace('@', '').replace('/', '').replace('\\', '')
            mol = Chem.RWMol(Chem.MolFromSmiles(pred_smiles, sanitize=False))
            mol = _verify_chirality(mol, coords, symbols, edges, debug)
        else:
            mol = Chem.MolFromSmiles(pred_smiles, sanitize=False)
        # pred_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        if molblock:
            pred_molblock = Chem.MolToMolBlock(mol)
        pred_smiles, mol = _expand_functional_group(mol, mappings)
        success = True
    except Exception as e:
        quality_issue = str(e) or e.__class__.__name__
        if debug:
            print(traceback.format_exc())
        pred_smiles = ''
        pred_molblock = ''
        success = False
    if debug:
        return pred_smiles, pred_molblock, mol, success, quality_issue
    return pred_smiles, pred_molblock, success, quality_issue


def postprocess_smiles(smiles, coords=None, symbols=None, edges=None, molblock=False, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        if coords is not None and symbols is not None and edges is not None:
            results = p.starmap(_postprocess_smiles, zip(smiles, coords, symbols, edges), chunksize=128)
        else:
            results = p.map(_postprocess_smiles, smiles, chunksize=128)
    smiles_list, molblock_list, success, quality_issues = zip(*results)
    r_success = np.mean(success)
    return smiles_list, molblock_list, r_success, quality_issues


def _keep_main_molecule(smiles, debug=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            num_atoms = [m.GetNumAtoms() for m in frags]
            main_mol = frags[np.argmax(num_atoms)]
            smiles = Chem.MolToSmiles(main_mol)
    except Exception as e:
        if debug:
            print(traceback.format_exc())
    return smiles


def keep_main_molecule(smiles, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        results = p.map(_keep_main_molecule, smiles, chunksize=128)
    return results
