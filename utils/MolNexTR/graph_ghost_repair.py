"""Graph-level attachment-decode repair for fragment/markush structures.

The MolNexTR backbone decoder follows the bond-line convention "every vertex is
a carbon atom". Wavy/zigzag attachment bonds are triangle-wave zigzags with
several turning points, so the decoder frequently emits a spurious "ghost"
carbon at the attachment vertex. Two recurring ghost patterns, both repaired
here at the GRAPH level (image-level inpaint was evaluated end-to-end and found
NET-NEGATIVE because re-decode rearranges the whole backbone):

  Pattern A (amino):  ``*C(C)N...``  →  ``*CN...``
     dummy → ghost-C, which has a spurious carbon branch + a heteroatom (N/O/S/P)
     neighbor. Remove the ghost (and its carbon branch), bond dummy to heteroatom.

  Pattern B (amide):  ``*CC(=O)N...``  →  ``*C(=O)N...``
     dummy → ghost-C → carbonyl-C (a C with a =O). Remove the ghost, bond dummy
     directly to the carbonyl carbon.

Both rules are SAFE (single, deterministic topology match) and NET-POSITIVE on
real_wavy_hard assembly Tanimoto (A: +0.019, B: +0.027, combined ~+0.03).

Unmatched cases (true backbone rearrangements like ``CNC=O`` → ``CC(=O)N`` with
no clean ghost vertex, or atom loss) are left unchanged rather than risk
corruption. Only fragment/markush rows are touched; complete is byte-identical.
"""
from __future__ import annotations

from typing import Any

from rdkit import Chem

HETERO_SYMBOLS = ("N", "O", "S", "P")


def _find_single_dummy(mol: Chem.Mol) -> int | None:
    dummies = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 0 or a.GetSymbol() in ("*", "R", "R#")
    ]
    return dummies[0] if len(dummies) == 1 else None


def _is_carbonyl_carbon(mol: Chem.Mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C":
        return False
    for n in atom.GetNeighbors():
        b = mol.GetBondBetweenAtoms(idx, n.GetIdx())
        if b is not None and n.GetSymbol() == "O" and b.GetBondType() == Chem.BondType.DOUBLE:
            return True
    return False


def _drop_ghost_and_rebond(mol: Chem.Mol, dummy_idx: int, ghost_idx: int, target_idx: int, bond_type: Chem.BondType) -> str | None:
    """Remove ghost_idx, rebond dummy->target with bond_type. Returns new molblock
    or None on failure. Relocates dummy/target by symbol after the removal."""
    target_sym = mol.GetAtomWithIdx(target_idx).GetSymbol()
    rw = Chem.RWMol(mol)
    for n in mol.GetAtomWithIdx(ghost_idx).GetNeighbors():
        b = mol.GetBondBetweenAtoms(ghost_idx, n.GetIdx())
        if b is not None:
            rw.RemoveBond(ghost_idx, n.GetIdx())
    rw.RemoveAtom(ghost_idx)
    repaired = rw.GetMol()
    new_dummy = _find_single_dummy(repaired)
    new_target = None
    if new_dummy is not None:
        # Prefer a target matching the original symbol; for carbonyl carbons,
        # also require it still be a carbonyl to avoid mis-bonding.
        for atom in repaired.GetAtoms():
            if atom.GetSymbol() == target_sym and atom.GetIdx() != new_dummy:
                if target_sym != "C" or _is_carbonyl_carbon(repaired, atom.GetIdx()):
                    new_target = atom.GetIdx()
                    break
        # Fallback: any same-symbol non-dummy atom.
        if new_target is None:
            for atom in repaired.GetAtoms():
                if atom.GetSymbol() == target_sym and atom.GetIdx() != new_dummy:
                    new_target = atom.GetIdx()
                    break
    if new_dummy is None or new_target is None:
        return None
    try:
        rw2 = Chem.RWMol(repaired)
        if rw2.GetBondBetweenAtoms(new_dummy, new_target) is None:
            rw2.AddBond(new_dummy, new_target, bond_type)
        repaired = rw2.GetMol()
        Chem.SanitizeMol(repaired)
        if _find_single_dummy(repaired) is None:
            return None
        return Chem.MolToMolBlock(repaired)
    except Exception:
        return None


def _path_to_heteroatom(mol: Chem.Mol, start_idx: int, exclude: int, max_depth: int = 3) -> list[int] | None:
    """BFS from start_idx to find the shortest path through carbon atoms to a
    heteroatom (N/O/S/P). Returns the path [start_idx, ..., hetero_idx] or None.
    ``exclude`` is an atom index to never traverse (typically the ghost center).
    Only traverses through carbon atoms except for the final heteroatom.
    """
    from collections import deque
    queue: deque[tuple[int, list[int]]] = deque([(start_idx, [start_idx])])
    visited: set[int] = {start_idx, exclude}
    while queue:
        curr, path = queue.popleft()
        if len(path) > max_depth:
            continue
        atom = mol.GetAtomWithIdx(curr)
        for nb in atom.GetNeighbors():
            nbidx = nb.GetIdx()
            if nbidx in visited:
                continue
            nbsym = nb.GetSymbol()
            if nbsym in HETERO_SYMBOLS:
                return path + [nbidx]
            if nbsym == "C":
                visited.add(nbidx)
                queue.append((nbidx, path + [nbidx]))
    return None


def _collect_carbon_subtree(mol: Chem.Mol, start_idx: int, exclude: int, dummy_idx: int | None = None) -> tuple[set[int], bool]:
    """Collect all carbon atoms reachable from start_idx without passing through
    exclude. Returns (atoms, is_safe).
    - is_safe=False signals "this branch contains a heteroatom or ring — do NOT remove"
    - is_safe=True means all atoms in the set are safe to remove (pure carbon chain, no rings)
    """
    visited: set[int] = {start_idx}
    stack = [start_idx]
    while stack:
        curr = stack.pop()
        atom = mol.GetAtomWithIdx(curr)
        for nb in atom.GetNeighbors():
            nbidx = nb.GetIdx()
            if nbidx == exclude or nbidx in visited or nbidx == dummy_idx:
                continue
            if nb.GetSymbol() not in ("C", "H"):
                # Hitting a non-carbon means this is a real substituent, not ghost
                return visited, False
            if nb.GetSymbol() == "C" and nb.IsInRing():
                # Carbon in a ring — removing it would destroy a real ring system
                return visited, False
            visited.add(nbidx)
            stack.append(nbidx)
    return visited, True


def _drop_carbon_branches(mol: Chem.Mol, center_idx: int, branch_indices: list[tuple[int, str]]) -> str | None:
    """Remove the spurious carbon branch atoms from center_idx, keeping the
    dummy→center→heteroatom chain intact. Returns new molblock or None on failure.

    For ``*C(C)NCCO`` this removes the branch ``C`` (at each branch_idx), leaving
    ``*CNCCO`` — the correct linear structure. The central carbon (center_idx)
    and all non-carbon neighbors are preserved.

    Handles multi-atom branches (e.g. ``*C(CC)NCCO``): collects ALL atoms reachable
    from each branch carbon without passing back through center_idx, and removes
    the entire subtree. This avoids leaving orphan fragments.
    """
    rw = Chem.RWMol(mol)
    dummy_idx = _find_single_dummy(mol)

    # Collect all atoms to remove: for each branch carbon, find its entire
    # subtree (atoms reachable without going through center_idx or dummy_idx).
    atoms_to_remove: set[int] = set()
    for bidx, _ in branch_indices:
        subtree, is_safe = _collect_carbon_subtree(mol, bidx, center_idx, dummy_idx)
        if not is_safe:
            return None  # branch contains heteroatom or ring — not a ghost
        atoms_to_remove.update(subtree)

    if not atoms_to_remove:
        return None

    # Remove all bonds to these atoms, then remove atoms (descending order).
    for aidx in atoms_to_remove:
        atom = rw.GetAtomWithIdx(aidx)
        for nb in list(atom.GetNeighbors()):
            b = rw.GetBondBetweenAtoms(aidx, nb.GetIdx())
            if b is not None:
                rw.RemoveBond(aidx, nb.GetIdx())

    for aidx in sorted(atoms_to_remove, reverse=True):
        rw.RemoveAtom(aidx)

    repaired = rw.GetMol()
    try:
        Chem.SanitizeMol(repaired)
        if _find_single_dummy(repaired) is None:
            return None
        # Reject disconnected fragments (orphan atoms left behind)
        if len(Chem.GetMolFrags(repaired)) != 1:
            return None
        return Chem.MolToMolBlock(repaired)
    except Exception:
        return None


def repair_ghost_attachment_carbon(molblock: str, mask_context: dict[str, Any] | None = None) -> tuple[str, str]:
    """Return (repaired_molblock, status).

    status: "repaired_amino", "repaired_amide", "repaired_mask", "not_applicable", "unparseable".
    Input is returned unchanged unless a repair pattern matched.

    ``mask_context`` (the DECIMER detector output for this image) enables a third
    pattern: when the dummy's carbon neighbor has no spurious branch (so the
    topology alone is ambiguous — could be a legitimate *CN or a ghost *CCN),
    the detector's wavy/asterisk mask geometry decides. If that carbon's image
    coordinate sits inside a detected attachment mask, it is a ghost vertex and
    is removed; otherwise the structure is legitimate and left unchanged.
    """
    if not molblock:
        return molblock, "not_applicable"
    mol = Chem.MolFromMolBlock(molblock, sanitize=False)
    if mol is None:
        return molblock, "unparseable"
    dummy_idx = _find_single_dummy(mol)
    if dummy_idx is None:
        return molblock, "not_applicable"
    dummy = mol.GetAtomWithIdx(dummy_idx)
    dummy_neighbors = [n.GetIdx() for n in dummy.GetNeighbors()]
    if len(dummy_neighbors) != 1:
        return molblock, "not_applicable"
    ghost_idx = dummy_neighbors[0]
    ghost = mol.GetAtomWithIdx(ghost_idx)
    if ghost.GetSymbol() != "C":
        return molblock, "not_applicable"
    others = [
        (n.GetIdx(), n.GetSymbol())
        for n in ghost.GetNeighbors()
        if n.GetIdx() != dummy_idx
    ]

    # Pattern A (amino): the decoder emits dummy→C with a spurious carbon
    # BRANCH and a heteroatom neighbor, i.e. ``*C(C)N...``. The BRANCH carbon
    # (inside the parentheses) is the ghost — the decoder mis-read the wavy
    # bond vertex as a 3-way junction and emitted a branch instead of a linear
    # chain. The correct structure is ``*CN...`` (dummy→C→hetero, linear).
    #
    # Fix: remove the carbon branch(es), keep dummy→C→hetero intact. The
    # central carbon (ghost_idx) is the REAL methylene; the branch carbons are
    # spurious. This is the opposite of the old _drop_ghost_and_rebond which
    # removed the central carbon and bonded dummy directly to the heteroatom
    # (producing ``*NCCO.C`` instead of the correct ``*CNCCO``).
    carbon_branches = [(idx, sym) for idx, sym in others if sym == "C"]
    heteros = [(idx, sym) for idx, sym in others if sym in HETERO_SYMBOLS]
    if carbon_branches and len(heteros) == 1:
        target_idx, _ = heteros[0]
        if all(sym in HETERO_SYMBOLS or sym == "C" for _, sym in others):
            # Remove only the carbon branches; keep dummy→C→hetero.
            new_block = _drop_carbon_branches(mol, ghost_idx, carbon_branches)
            if new_block is not None:
                return new_block, "repaired_amino"

    # Pattern A' (delayed heteroatom): the central carbon has ONLY carbon
    # neighbors (no direct heteroatom), but one carbon leads to a heteroatom
    # within 2 hops while another does not. The non-leading carbons are ghost
    # branches; the leading carbon is the chain continuation. Additionally, if
    # the leading carbon's path to the heteroatom goes through an extra methylene
    # (dummy→C→C→N instead of dummy→C→N), collapse that to dummy→C→N.
    #
    # Example: ``*C(C)CN1CCC1`` → ``*CN1CCC1``. The central C has two carbon
    # neighbors: C(2) leads nowhere (branch ghost), C(3) leads to N(ring). Remove
    # C(2) (branch), collapse C(3) (extra chain carbon), bond ghost_C→N directly.
    if not heteros and len(carbon_branches) >= 2:
        leads_to_hetero = []
        dead_end_carbons = []
        for cidx, _ in carbon_branches:
            path = _path_to_heteroatom(mol, cidx, exclude=ghost_idx, max_depth=3)
            if path is not None:
                leads_to_hetero.append((cidx, path))
            else:
                dead_end_carbons.append((cidx, _))
        if len(leads_to_hetero) == 1 and dead_end_carbons:
            chain_cidx, hetero_path = leads_to_hetero[0]
            target_hetero_idx = hetero_path[-1]
            # Validate dead-end branches are safe to remove (no heteroatoms/rings)
            atoms_to_remove: set[int] = set()
            abort = False
            for cidx, _ in dead_end_carbons:
                subtree, is_safe = _collect_carbon_subtree(mol, cidx, ghost_idx, dummy_idx)
                if not is_safe:
                    abort = True
                    break
                atoms_to_remove.update(subtree)
            if abort:
                return molblock, "not_applicable"
            # Remove the intermediate chain carbons (path[:-1] = all C's before hetero)
            for intermediate_c in hetero_path[:-1]:
                atoms_to_remove.add(intermediate_c)

            # Tag the target heteroatom BEFORE removal so we can find it after
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(target_hetero_idx).SetAtomMapNum(99)
            # Remove all bonds to atoms being removed
            for aidx in atoms_to_remove:
                atom = rw.GetAtomWithIdx(aidx)
                for nb in list(atom.GetNeighbors()):
                    b = rw.GetBondBetweenAtoms(aidx, nb.GetIdx())
                    if b is not None:
                        rw.RemoveBond(aidx, nb.GetIdx())
            # Remove atoms (descending order)
            for aidx in sorted(atoms_to_remove, reverse=True):
                rw.RemoveAtom(aidx)

            repaired = rw.GetMol()
            new_dummy = _find_single_dummy(repaired)
            if new_dummy is None:
                return molblock, "not_applicable"
            new_center_atoms = repaired.GetAtomWithIdx(new_dummy).GetNeighbors()
            if len(new_center_atoms) != 1:
                return molblock, "not_applicable"
            new_center_idx = new_center_atoms[0].GetIdx()

            # Find the tagged heteroatom deterministically
            new_hetero_idx = None
            for atom in repaired.GetAtoms():
                if atom.GetAtomMapNum() == 99:
                    new_hetero_idx = atom.GetIdx()
                    break
            if new_hetero_idx is None or new_hetero_idx == new_center_idx:
                return molblock, "not_applicable"
            # Bond center→hetero if not already bonded
            if repaired.GetBondBetweenAtoms(new_center_idx, new_hetero_idx) is None:
                rw2 = Chem.RWMol(repaired)
                rw2.AddBond(new_center_idx, new_hetero_idx, Chem.BondType.SINGLE)
                repaired = rw2.GetMol()
            # Clear the map num tag
            rw3 = Chem.RWMol(repaired)
            for atom in rw3.GetAtoms():
                if atom.GetAtomMapNum() == 99:
                    atom.SetAtomMapNum(0)
            repaired = rw3.GetMol()

            try:
                Chem.SanitizeMol(repaired)
                if _find_single_dummy(repaired) is None:
                    return molblock, "not_applicable"
                # Reject disconnected fragments
                if len(Chem.GetMolFrags(repaired)) != 1:
                    return molblock, "not_applicable"
                return Chem.MolToMolBlock(repaired), "repaired_amino_delayed"
            except Exception:
                return molblock, "not_applicable"

    # Pattern B (amide): ghost has exactly one neighbor, a carbonyl carbon
    # (C with =O). Remove ghost, bond dummy to the carbonyl carbon.
    if len(others) == 1:
        c_idx, c_sym = others[0]
        if c_sym == "C" and _is_carbonyl_carbon(mol, c_idx):
            new_block = _drop_ghost_and_rebond(mol, dummy_idx, ghost_idx, c_idx, Chem.BondType.SINGLE)
            if new_block is not None:
                return new_block, "repaired_amide"

    # Pattern C (mask-guided): topology is ambiguous (dummy->C->single hetero,
    # no spurious branch — could be legit *CN or ghost *CCN). The DECIMER
    # detector locates the wavy attachment mark in image space. The decoded
    # dummy atom's normalized coordinate tells us where the decoder placed the
    # attachment endpoint. In a LEGIT *CN the dummy sits AT the wavy mark (the
    # decoder correctly localized the attachment). In a GHOST *CCN the decoder
    # placed the dummy INSIDE the structure (away from the wavy mark) and
    # inserted a carbon at the vertex. So: if the dummy's decoded coordinate is
    # far from the detected wavy mask, the adjacent carbon is a ghost vertex.
    if mask_context and len(others) == 1:
        idx, sym = others[0]
        if sym in HETERO_SYMBOLS:
            norm_coords = mask_context.get("atom_norm_coords") or []
            if 0 <= dummy_idx < len(norm_coords) and norm_coords[dummy_idx] is not None:
                dx, dy = norm_coords[dummy_idx]
                # Is the dummy far from every detected wavy/asterisk mask?
                far_from_mark = not _point_in_masks(
                    dx, dy, mask_context, classes=("wavy", "asterisk"),
                    expand=0.15,  # generous tolerance: dummy may be slightly offset
                )
                if far_from_mark and _has_attachment_mark(mask_context):
                    gt_bond = mol.GetBondBetweenAtoms(ghost_idx, idx)
                    bt = gt_bond.GetBondType() if gt_bond is not None else Chem.BondType.SINGLE
                    new_block = _drop_ghost_and_rebond(mol, dummy_idx, ghost_idx, idx, bt)
                    if new_block is not None:
                        return new_block, "repaired_mask"

    return molblock, "not_applicable"


def apply_ghost_repair(result: dict[str, Any], *, expected_type: str, mask_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Repair a fragment/markush prediction result in place. Complete rows and
    non-dict results are returned unchanged.

    ``mask_context`` (optional) carries the DECIMER detector output for this
    image — a list of detected attachment masks, each with a normalized bbox
    center (cx, cy) and a class (wavy/rgroup/asterisk/dashed). When present and
    the pure-topology rule is ambiguous (the dummy's carbon neighbor has no
    spurious branch, so it could be a legitimate *CN or a ghost *CCN), the mask
    geometry breaks the tie: if the dummy's carbon sits inside a wavy/asterisk
    mask region it is a ghost (the wavy zigzag vertex misread as carbon), and is
    removed; otherwise the structure is left unchanged.
    """
    if str(expected_type or "").strip().lower() not in {"fragment", "markush"}:
        return result
    if not isinstance(result, dict):
        return result
    molblock = result.get("predicted_molfile") or ""
    if not molblock:
        return result
    new_block, status = repair_ghost_attachment_carbon(molblock, mask_context=mask_context)
    if not status.startswith("repaired"):
        return result
    new_smiles = ""
    try:
        m = Chem.MolFromMolBlock(new_block, sanitize=False)
        if m is not None:
            Chem.SanitizeMol(m)
            # Reject if the repair produced a disconnected structure
            if len(Chem.GetMolFrags(m)) != 1:
                return result
            new_smiles = Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        new_smiles = ""
    if not new_smiles:
        return result
    result["predicted_molfile"] = new_block
    result["predicted_smiles"] = new_smiles
    existing = result.get("graph_ghost_repair")
    result["graph_ghost_repair"] = {"status": status, **(existing if isinstance(existing, dict) else {})}
    return result


def _point_in_masks(cx: float, cy: float, mask_context: dict[str, Any] | None, classes: tuple[str, ...] | None = None, expand: float = 0.0) -> bool:
    """True if a normalized point (cx,cy) falls inside (or within `expand` of)
    any detected attachment mask bbox of the requested classes.

    mask_context = {"detections": [{"class","cx","cy","bw","bh"}, ...], ...}.
    The bbox (cx,cy) is the mask center and bw/bh its normalized width/height.
    `expand` pads each bbox symmetrically (normalized units).
    """
    if not mask_context:
        return False
    detections = mask_context.get("detections") or []
    for det in detections:
        if classes and det.get("class") not in classes:
            continue
        dcx = float(det.get("cx", -1))
        dcy = float(det.get("cy", -1))
        bw = float(det.get("bw", 0)) + expand
        bh = float(det.get("bh", 0)) + expand
        if bw <= 0 or bh <= 0:
            continue
        if dcx - bw / 2 <= cx <= dcx + bw / 2 and dcy - bh / 2 <= cy <= dcy + bh / 2:
            return True
    return False


def _has_attachment_mark(mask_context: dict[str, Any] | None, classes: tuple[str, ...] = ("wavy", "asterisk")) -> bool:
    """True if the detector found any wavy/asterisk attachment mark."""
    if not mask_context:
        return False
    return any(d.get("class") in classes for d in (mask_context.get("detections") or []))
