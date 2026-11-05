"""V3000 molblock bridge: normalize LINKNODE placement, delegate enumeration.

RDKit (verified on 2026.09.1) reads LINKNODE specs only in their inline
CTfile form `M  V30 LINKNODE minrep maxrep nbonds (inatom outatom)*nbonds`
placed directly after the BOND block. Block-wrapped forms (BEGIN/END
LINKNODE) and specs parked in other spec-legal positions are skipped with
a warning, so the variation is silently lost. This module is the
normalizing pass: it rewrites every LINKNODE spec into the one position
every reader accepts and hands the result to RDKit's production
enumerator (rdMolEnumerator) instead of re-implementing repeat-unit
expansion. Behaviour and the RDKit quirks motivating it are documented in
research/v3000_linknode_reference_impls.md (section 2.3, locally
reproduced) and research/v3000_linknode_spec.md.

Scope: single-CTAB template molblocks. Position-variation bonds
(ENDPTS=/ATTACH=) need no normalization - RDKit stores them as bond
properties wherever they appear - and are enumerated by the same call.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolEnumerator

_PREFIX = "M  V30 "


class V3000ParseError(ValueError):
    """A LINKNODE spec that is malformed (as opposed to spec-legal but
    unsupported, which becomes a named note instead of an error)."""


def _logical_records(molblock: str) -> list[tuple[bool, str]]:
    """Split the molblock into logical lines as (is_v30, text) records.

    A trailing '-' splices the following physical 'M  V30 ' lines onto the
    current one (CTfile continuation rule); a non-prefixed line breaks the
    splice and the pending text is flushed as its own record.
    """
    records: list[tuple[bool, str]] = []
    pending: str | None = None
    for raw in molblock.splitlines():
        if not raw.startswith(_PREFIX):
            if pending is not None:
                records.append((True, pending))
                pending = None
            records.append((False, raw))
            continue
        body = raw[len(_PREFIX):]
        pending = (pending or "") + body
        if pending.rstrip().endswith("-"):
            pending = pending.rstrip()[:-1]
            continue
        records.append((True, pending))
        pending = None
    if pending is not None:
        records.append((True, pending))
    return records


def _linknode_line(tokens: list[str], notes: list[str]) -> str | None:
    """Grammar-check one LINKNODE arg list; return its canonical line.

    Malformed specs raise V3000ParseError with the named fault. Spec-legal
    specs that RDKit cannot enumerate (nbonds != 2; nbonds=1 attachment
    points are allowed by the CTfile spec) are dropped with a named note
    rather than passed on to be silently discarded by RDKit.
    """
    try:
        args = [int(t) for t in tokens]
    except ValueError:
        raise V3000ParseError(
            f"LINKNODE has non-integer fields: {tokens}") from None
    if len(args) < 5 or len(args) != 3 + 2 * args[2]:
        raise V3000ParseError(
            f"LINKNODE field count does not match nbonds: {tokens}")
    minrep, maxrep, nbonds = args[:3]
    if minrep < 1 or maxrep < minrep:
        raise V3000ParseError(
            f"LINKNODE repeat interval [{minrep}, {maxrep}] is invalid")
    if nbonds != 2:
        notes.append(f"LINKNODE {tokens} dropped: RDKit's enumerator supports "
                     f"nbonds=2 only (spec-legal nbonds={nbonds} form)")
        return None
    (in1, out1), (in2, out2) = zip(args[3::2], args[4::2])
    if in1 != in2:
        raise V3000ParseError(
            f"LINKNODE tuples start at different atoms: {tokens}")
    if out1 == out2:
        raise V3000ParseError(f"LINKNODE anchors coincide: {tokens}")
    return _PREFIX + "LINKNODE " + " ".join(str(a) for a in args)


def normalize_linknodes(molblock: str) -> tuple[str, list[str]]:
    """Rewrite the molblock with every LINKNODE spec inline after END BOND.

    Returns (normalized text, notes). Non-V3000 text and files without
    LINKNODE specs pass through unchanged; already-inline specs are
    re-emitted at the same spot, so the pass is idempotent. Placement
    matters because RDKit only reads LINKNODE lines in the window directly
    after the BOND block. When a rebuild happens (LINKNODE specs present),
    continuation splices are flattened into single physical lines - the
    logical content is identical.
    """
    notes: list[str] = []
    links: list[str] = []
    kept: list[str] = []
    end_bond_at: int | None = None
    section: str | None = None
    for is_v30, text in _logical_records(molblock):
        if not is_v30:
            kept.append(text)
            continue
        tokens = text.split()
        head = tokens[0].upper() if tokens else ""
        if head == "BEGIN" and len(tokens) > 1:
            section = tokens[1].upper()
            if section != "LINKNODE":      # wrapper line itself is dropped
                kept.append(_PREFIX + text)
            continue
        if head == "END" and len(tokens) > 1:
            ending = tokens[1].upper()
            if section == "LINKNODE" and ending == "LINKNODE":
                section = None
                continue
            section = None
            kept.append(_PREFIX + text)
            if ending == "BOND":
                end_bond_at = len(kept) - 1
            continue
        if section == "LINKNODE":          # bare args inside the block form
            line = _linknode_line(tokens, notes)
            if line:
                links.append(line)
            continue
        if head == "LINKNODE":             # already inline somewhere
            line = _linknode_line(tokens[1:], notes)
            if line:
                links.append(line)
            continue
        kept.append(_PREFIX + text)
    if not links:
        return molblock, notes
    if end_bond_at is None:
        raise V3000ParseError(
            "LINKNODE specs present but the CTAB has no BOND block")
    out = kept[:end_bond_at + 1] + links + kept[end_bond_at + 1:]
    return "\n".join(out), notes


def enumerate_molblock(molblock: str) -> tuple[list[Chem.Mol], list[str]]:
    """Template variants for a molblock, via RDKit's enumerator.

    Pipeline: normalize LINKNODE placement, parse with RDKit, enumerate,
    sanitize each variant. The parse itself is tolerant (sanitize=False)
    and keeps explicit hydrogens (removeHs=False) because dropping them
    renumbers atoms and desyncs the atom indices stored inside the
    LINKNODE/ENDPTS properties. Returns (variants, notes); notes carry the
    named reason whenever the enumerator was skipped or a variant could
    not be sanitized. Raises V3000ParseError when RDKit cannot parse the
    molblock at all.
    """
    normalized, notes = normalize_linknodes(molblock)
    base = Chem.MolFromMolBlock(normalized, sanitize=False, removeHs=False)
    if base is None:
        raise V3000ParseError("RDKit rejected the molblock")
    has_variation = any(b.HasProp("_MolFileBondEndPts")
                        for b in base.GetBonds()) \
        or _PREFIX + "LINKNODE" in normalized
    if not has_variation:
        notes.append("no LINKNODE or position-variation bonds; "
                     "template used as-is")
        return [base], notes
    try:
        bundle = rdMolEnumerator.Enumerate(base)
    except (ValueError, RuntimeError) as exc:
        # RDKit signals malformed specs and unsupported shapes (nbonds=1,
        # multi-fragment templates) through these two exception families
        notes.append(f"rdMolEnumerator failed: {type(exc).__name__}: "
                     f"{str(exc)[:120]}")
        return [base], notes
    variants = [bundle.GetMol(i) for i in range(bundle.Size())]
    # the parse is tolerant (sanitize=False), so perceived aromaticity and
    # implicit Hs are missing until each variant is sanitized; variants that
    # cannot sanitize stay unsanitized for fragment completion, where the
    # skill's gate reports the concrete fault
    for i, mol in enumerate(variants):
        try:
            Chem.SanitizeMol(mol)
        except (Chem.MolSanitizeException, ValueError, RuntimeError) as exc:
            notes.append(f"variant {i + 1} kept unsanitized "
                         f"({type(exc).__name__}: {str(exc)[:80]})")
    if not variants:
        # nothing enumerable (e.g. only ATTACH=ALL bonds, a multi-centre
        # representation rather than a variation): keep the template itself
        notes.append("rdMolEnumerator produced no variants (ATTACH=ALL bonds "
                     "are a multi-centre representation, not an enumeration); "
                     "template used as-is")
        return [base], notes
    notes.append(f"rdMolEnumerator produced {len(variants)} variants; "
                 "fragment lists apply per variant in order")
    return variants, notes
