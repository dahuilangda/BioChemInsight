"""Unit tests for the skills package — no model calls; chemistry only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

from skills.peptide_notation.skill import _assemble, _validate_units, \
    _find_amine_and_carboxyl

NLE = {"code": "Nle", "fragment_smiles": "N[C@H](C=O)CCCC"}
PAL = {"code": "4Pal", "fragment_smiles": "N[C@H](C=O)Cc1ccncc1"}
TRP = {"code": "W", "fragment_smiles": "N[C@@H](C=O)Cc1c[nH]c2ccccc12"}
SAR = {"code": "Sar", "fragment_smiles": "N(C)CC=O"}
ACPC = {"code": "ACPC", "fragment_smiles": "N1CC[C@H](C=O)C1"}


def _parsed(units, ring_size, caps=None, terminal=None):
    return {"units": units, "ring_size": ring_size, "caps": caps or [],
            "terminal": terminal}


def test_macrocycle_graph_assembly():
    parsed = _parsed([NLE, PAL, TRP, SAR, ACPC], ring_size=5)
    units, failures = _validate_units(parsed)
    assert not failures
    smiles, err, notes = _assemble(units, parsed)
    assert smiles, err
    assert not notes
    mol = Chem.MolFromSmiles(smiles)
    ring_sizes = sorted({len(r) for r in mol.GetRingInfo().AtomRings()},
                        reverse=True)
    # macrocycle through ACPC's pyrrolidine (the long way around) fuses
    # with the 5-ring; both must be present
    assert ring_sizes[0] >= 17 and 5 in ring_sizes


def test_ring_nitrogen_residue_in_cycle():
    parsed = _parsed([ACPC, NLE], ring_size=2)
    units, _ = _validate_units(parsed)
    smiles, err, notes = _assemble(units, parsed)
    assert smiles, err
    # ACPC's ring N is the backbone amine: the macrocycle must pass it
    mol = Chem.MolFromSmiles(smiles)
    assert any(len(r) >= 6 for r in mol.GetRingInfo().AtomRings())
    # cyclo(ACPC-Nle) composition: both residues, two water losses
    assert rdMolDescriptors.CalcMolFormula(mol) == "C12H20N2O2"
    # ACPC's ring N is the backbone amine: in-ring and acylated
    assert any(a.GetSymbol() == "N" and a.IsInRing() and a.GetDegree() >= 2
               and any(n.GetSymbol() == "C" and any(
                   b.GetBondTypeAsDouble() == 2 for b in n.GetBonds())
                   for n in a.GetNeighbors())
               for a in mol.GetAtoms())


def test_cap_and_terminal():
    gly = {"code": "G", "fragment_smiles": "NCC=O"}
    ala = {"code": "A", "fragment_smiles": "N[C@@H](C)C=O"}
    parsed = _parsed([gly, ala], ring_size=1, caps=[
        {"code": "ClAc", "fragment_smiles": "ClCC=O"}], terminal=None)
    units, _ = _validate_units(parsed)
    smiles, err, notes = _assemble(units, parsed)
    # tail-less cap-only compound: ring closes, cap acylates the head amine
    assert smiles or "backbone" in err


def test_fragment_shape_gate():
    parsed = _parsed([{"code": "bad", "fragment_smiles": "CCC"}], ring_size=1)
    units, failures = _validate_units(parsed)
    assert not units and failures       # no amine-first skeleton -> rejected


def test_variant_sites_parsing():
    import asyncio
    from skills.registry import load_skills, get
    load_skills()
    handler = get("variant_sites").handler
    r = asyncio.run(handler({"hgvs": [
        "NM_000546.6:c.743G>A", "NP_000537.3:p.Arg248Trp"]}))
    assert r["ok"] and len(r["sites"]) == 2
    assert r["sites"][0]["start"] == 743
    assert r["sites"][1]["alt"] == "W"   # hgvs normalizes to one letter


def test_ledger_dedup_and_roundtrip(tmp_path):
    from services.document_ledger import (DocumentLedger, FigureEntity,
                                          Contradiction)
    ledger = DocumentLedger("t")
    ledger.add_figure(FigureEntity(entity_id="f1", page=1, bbox=[0, 0, 1, 1]))
    c = Contradiction(kind="id-conflict", entity_id="f1", source_a="a",
                      source_b="b", detail_a="x", detail_b="y")
    ledger.add_contradiction(c)
    ledger.add_contradiction(c)
    assert len(ledger.contradictions) == 1
    path = str(tmp_path / "ledger.json")
    ledger.save(path)
    loaded = DocumentLedger.load(path)
    assert len(loaded.contradictions) == 1


def test_sequence_assembler_cross_page():
    import asyncio
    from skills.registry import load_skills, get
    load_skills()
    handler = get("sequence_assembler").handler
    seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
    cut = 60
    pages = [
        {"page": 1, "text": f"SEQ ID NO: 7\n<211> LENGTH: {len(seq)}\nFEATURE\nsource\n"
                            f"    {seq[:cut]}\n"},
        {"page": 2, "text": f"-continued\n    {seq[cut:]}\n"},
    ]
    r = asyncio.run(handler({"pages": pages}))
    rows = [x for x in r["sequences"] if x["seq_id"] == 7]
    assert rows and rows[0]["cross_page"]
    assert rows[0]["length"] == len(seq)
    assert rows[0]["length_ok"] is True
    assert rows[0]["sequence"] == seq


def test_markush_assemble_plain_fragments():
    import asyncio
    from skills.registry import load_skills, get
    load_skills()
    handler = get("markush_assemble").handler
    # multi-atom fragments must declare their attachment site explicitly
    r = asyncio.run(handler({
        "template_smiles": "c1ccc([*:1])cc1",
        "fragments": {"1": ["[*:1]C", "[*:1]CO", "[*:1]C(=O)O"]},
    }))
    assert r["ok"] and len(r["members"]) == 3
    formulas = {m["formula"] for m in r["members"]}
    assert formulas == {"C7H8", "C7H8O", "C7H6O2"}
    keys = {m["inchikey"] for m in r["members"]}
    assert len(keys) == 3
    r = asyncio.run(handler({
        "template_smiles": "c1ccc([*:1])cc1",
        "fragments": {"1": ["C", "CO"]},
    }))
    assert r["ok"] and len(r["members"]) == 1  # single-atom C attaches; CO is refused
    assert any("no explicit attachment site" in g for g in r["gaps"])

    # activity joins run through explicit molecular identity only
    r2 = asyncio.run(handler({
        "template_smiles": "c1ccc([*:1])cc1", "fragments": {"1": ["C", "CO"]},
        "activity_rows": [
            {"compound_id": "m1", "ic50": "12 nM",
             "structure_smiles": "Cc1ccccc1"},
            {"compound_id": "m2", "ic50": "30 nM",
             "structure_smiles": "c1ccccc1"},
        ]}))
    joined = {j["compound_id"]: j for j in r2["activity_join"]}
    assert joined["m1"]["matched"] is True
    assert joined["m1"]["activity"]["ic50"] == "12 nM"
    assert joined["m2"]["matched"] is False   # benzene is not a member


def _review_activity_fixture(monkeypatch, source, cells):
    import asyncio
    import json
    from skills.activity_tables.skill import parse
    from utils import llm_utils
    reference = {"line_start": 1, "line_end": len(source.splitlines())}
    observations = [{
        "compound_id": compound, "id_kind": "compound", "assay_name": endpoint,
        "endpoint": endpoint, "target": None, "conditions": None, "source_kind": "unknown", "raw": raw,
        "value": value, "operator": operator, "unit": unit, "evidence": [reference],
    } for compound, endpoint, raw, value, operator, unit in cells]
    responses = iter([
        {"status": "complete", "reason": "", "observations": observations, "source_issues": []},
        {"status": "accepted", "issues": [], "coverage": [reference], "question_decisions": []},
    ])
    def run(**kwargs):
        return kwargs["parser"](json.dumps(next(responses)))
    monkeypatch.setattr(llm_utils, "run_text_json_task", run)
    return asyncio.run(parse({"markdown": source}))


def test_activity_tables_header_semantics(monkeypatch):
    md = ("<table><tr><td>SEQ ID NO.</td><td>IC50 (TNFR1)</td><td>EC50</td></tr>"
          "<tr><td>53</td><td>12 nM</td><td>&lt;0.5 nM</td></tr>"
          "<tr><td>Compound 88</td><td>&gt;100 uM</td><td>3.4 uM</td></tr></table>")
    result = _review_activity_fixture(monkeypatch, md, [
        ("53", "IC50 (TNFR1)", "12 nM", 12, "", "nM"),
        ("53", "EC50", "<0.5 nM", 0.5, "<", "nM"),
        ("88", "IC50 (TNFR1)", ">100 uM", 100, ">", "uM"),
        ("88", "EC50", "3.4 uM", 3.4, "", "uM"),
    ])
    assert result["ok"] and set(result["assay_types"]) == {"IC50 (TNFR1)", "EC50"}
    by_id = {row["compound_id"]: row for row in result["rows"]}
    assert by_id["53"]["assays"]["IC50 (TNFR1)"]["value"] == 12
    assert by_id["53"]["assays"]["EC50"]["operator"] == "<"
    assert by_id["88"]["assays"]["IC50 (TNFR1)"]["operator"] == ">"


# --------------------------------------------------------------------------
# V3000 LINKNODE pathway (skills/markush_assemble/v3000.py)
#
# CTfile link atom line (three-source verified):
#   M  V30 LINKNODE minrep maxrep nbonds (inatom outatom)*nbonds
# spec '1 2 2 2 1 2 4' -> minrep=1 maxrep=2 nbonds=2,
# pairs (inatom=2,outatom=1)(inatom=2,outatom=4): repeat atom = 2,
# anchors = 1 and 4. RDKit 2026.09 only reads the inline form placed
# directly after the BOND block, so the v3000 module rewrites block-wrapped
# specs into that position and delegates to rdMolEnumerator.
# --------------------------------------------------------------------------

LINKNODE_MOLBLOCK = "\n".join(
    ["", "  BioChemInsight fixture", "",
     "  0  0  0     0  0            999 V3000",
     "M  V30 BEGIN CTAB", "M  V30 COUNTS 5 4 0 0 0", "M  V30 BEGIN ATOM",
     "M  V30 1 C 0 0 0 0", "M  V30 2 C 0 0 0 0", "M  V30 3 C 0 0 0 0",
     "M  V30 4 C 0 0 0 0",
     "M  V30 5 R 0 0 0 1 VAL=1",   # RDKit's V3000 spelling of [*:1]
     "M  V30 END ATOM", "M  V30 BEGIN BOND",
     "M  V30 1 1 1 2", "M  V30 2 1 2 4", "M  V30 3 1 4 3",
     "M  V30 4 1 1 5",
     "M  V30 END BOND",
     "M  V30 BEGIN LINKNODE", "M  V30 1 2 2 2 1 2 4", "M  V30 END LINKNODE",
     "M  V30 END CTAB", "M  END", ""])


def test_v3000_linknode_block_to_inline():
    from skills.markush_assemble import v3000
    text, notes = v3000.normalize_linknodes(LINKNODE_MOLBLOCK)
    assert notes == []
    lines = text.splitlines()
    at = lines.index("M  V30 END BOND")
    assert lines[at + 1] == "M  V30 LINKNODE 1 2 2 2 1 2 4"   # moved inline
    assert lines[at + 2] == "M  V30 END CTAB"
    assert not any("LINKNODE" == ln[6:].split()[0].upper()
                   for ln in lines[at + 2:] if ln.startswith("M  V30 "))
    text2, _ = v3000.normalize_linknodes(text)
    assert text2 == text                                      # idempotent


def test_v3000_linknode_spec_variants():
    from skills.markush_assemble import v3000
    variants, notes = v3000.enumerate_molblock(LINKNODE_MOLBLOCK)
    assert any("2 variants" in n for n in notes)
    # repeat counts r in [minrep..maxrep] = [1..2]: r=1 identity, r=2 one copy
    assert [m.GetNumAtoms() for m in variants] == [5, 6]
    for mol in variants:                     # no valence/kekulization faults
        Chem.SanitizeMol(mol)
        dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        assert len(dummies) == 1 and dummies[0].GetAtomMapNum() == 1
    assert {Chem.MolToSmiles(m) for m in variants} == \
        {"CCCC[*:1]", "CCCCC[*:1]"}


def test_v3000_linknode_spec_benchmarks():
    # samples A-D from research/v3000_linknode_spec.md section 6; the
    # expected SMILES are RDKit enumerator_catch.cpp assertions, quoted in
    # the spec doc and re-verified locally on RDKit 2026.09.1
    from skills.markush_assemble import v3000
    variants, notes = v3000.enumerate_molblock(SAMPLE_A)
    assert not [n for n in notes if n.startswith("rdMolEnumerator failed")]
    assert {Chem.MolToSmiles(m) for m in variants} == {
        "OC1CCCC1", "OC1CCCCC1O", "OC1CCCCC(O)C1O", "OC1CCCCC(O)C(O)C1O"}
    variants_b, _ = v3000.enumerate_molblock(SAMPLE_B)
    assert len(variants_b) == 12             # 3 x 4 repeat-count product
    smiles_b = {Chem.MolToSmiles(m) for m in variants_b}
    assert {"OC1CCC(F)C1", "OC1CCC(F)CC(O)C1O",
            "OC1CCC(F)C(F)C(F)CC1O"} <= smiles_b
    variants_c, _ = v3000.enumerate_molblock(SAMPLE_C)
    assert len(variants_c) == 6              # 3 x 2, cascading link atoms
    smiles_c = {Chem.MolToSmiles(m) for m in variants_c}
    assert {"COC1CCC(F)C1C", "COC1CCC(F)C(C)C1OC",
            "COC1CCC(F)C(C)C(C)C1OC"} <= smiles_c
    variants_d, _ = v3000.enumerate_molblock(SAMPLE_D)
    assert {Chem.MolToSmiles(m) for m in variants_d} == {
        "Cc1cnc2c(c1)CC(O)C2", "Cc1cnc2c(c1)CC(O)C(O)C2",
        "Cc1cnc2c(c1)CC(O)C(O)C(O)C2",
        "Cc1ccc2c(n1)CC(O)C2", "Cc1ccc2c(n1)CC(O)C(O)C2",
        "Cc1ccc2c(n1)CC(O)C(O)C(O)C2",
        "Cc1ccnc2c1CC(O)C2", "Cc1ccnc2c1CC(O)C(O)C2",
        "Cc1ccnc2c1CC(O)C(O)C(O)C2"}


def test_v3000_attach_all_is_named_gap():
    # sample E (ferrocene): ATTACH=ALL is a multi-centre representation, so
    # the enumerator yields nothing and the template passes through
    from skills.markush_assemble import v3000
    variants, notes = v3000.enumerate_molblock(SAMPLE_E)
    assert len(variants) == 1
    assert any("produced no variants" in n for n in notes)


def test_v3000_malformed_linknode_raises_named():
    from skills.markush_assemble import v3000
    bad = LINKNODE_MOLBLOCK.replace("M  V30 1 2 2 2 1 2 4",
                                    "M  V30 0 2 2 2 1 2 4")   # minrep=0
    with pytest.raises(v3000.V3000ParseError, match="repeat interval"):
        v3000.normalize_linknodes(bad)
    nbonds1 = LINKNODE_MOLBLOCK.replace("M  V30 1 2 2 2 1 2 4",
                                        "M  V30 1 2 1 2 1")   # spec-legal,
    text, notes = v3000.normalize_linknodes(nbonds1)          # unsupported
    assert any("nbonds=1" in n and "dropped" in n for n in notes)
    assert "M  V30 LINKNODE" not in text


def test_markush_assemble_molblock_linknode():
    import asyncio
    from skills.registry import load_skills, get
    load_skills()
    handler = get("markush_assemble").handler
    r = asyncio.run(handler({
        "molblock": LINKNODE_MOLBLOCK,
        "fragments": {"1": ["[*:1]C", "[*:1]CO"], "2": ["[*:2]C", "[*:2]CO"]},
    }))
    assert r["ok"] and len(r["members"]) >= 2
    # r=1 -> pentane/pentan-1-ol, r=2 -> hexane/hexan-1-ol
    assert {m["formula"] for m in r["members"]} == \
        {"C5H12", "C5H12O", "C6H14", "C6H14O"}
    for m in r["members"]:
        assert m["formula"] and len(m["inchikey"]) == 27 and m["exact_mass"] > 0


SAMPLE_A = "\n".join(
    ["", "  Mrv2007 06222005102D", "",
     "  0  0  0     0  0            999 V3000",
     "M  V30 BEGIN CTAB", "M  V30 COUNTS 6 6 0 0 0", "M  V30 BEGIN ATOM",
     "M  V30 1 C 8.25 12.1847 0 0", "M  V30 2 C 6.9164 12.9547 0 0",
     "M  V30 3 C 6.9164 14.4947 0 0", "M  V30 4 C 9.5836 14.4947 0 0",
     "M  V30 5 C 9.5836 12.9547 0 0", "M  V30 6 O 8.25 10.6447 0 0",
     "M  V30 END ATOM", "M  V30 BEGIN BOND",
     "M  V30 1 1 1 2", "M  V30 2 1 2 3", "M  V30 3 1 4 5",
     "M  V30 4 1 1 5", "M  V30 5 1 3 4", "M  V30 6 1 1 6",
     "M  V30 END BOND", "M  V30 LINKNODE 1 4 2 1 2 1 5",
     "M  V30 END CTAB", "M  END", ""])

SAMPLE_B = "\n".join(
    ["", "  Mrv2014 07072016412D", "",
     "  0  0  0     0  0            999 V3000",
     "M  V30 BEGIN CTAB", "M  V30 COUNTS 7 7 0 0 0", "M  V30 BEGIN ATOM",
     "M  V30 1 C 8.25 12.1847 0 0", "M  V30 2 C 6.9164 12.9547 0 0",
     "M  V30 3 C 7.2366 14.4611 0 0", "M  V30 4 C 8.7681 14.622 0 0",
     "M  V30 5 C 9.3945 13.2151 0 0", "M  V30 6 O 8.25 10.6447 0 0",
     "M  V30 7 F 9.5382 15.9557 0 0",
     "M  V30 END ATOM", "M  V30 BEGIN BOND",
     "M  V30 1 1 1 2", "M  V30 2 1 2 3", "M  V30 3 1 4 5",
     "M  V30 4 1 1 5", "M  V30 5 1 3 4", "M  V30 6 1 1 6", "M  V30 7 1 4 7",
     "M  V30 END BOND", "M  V30 LINKNODE 1 3 2 1 2 1 5",
     "M  V30 LINKNODE 1 4 2 4 3 4 5",
     "M  V30 END CTAB", "M  END", ""])

SAMPLE_C = "\n".join(
    ["", "  Mrv2014 07082008052D", "",
     "  0  0  0     0  0            999 V3000",
     "M  V30 BEGIN CTAB", "M  V30 COUNTS 9 9 0 0 0", "M  V30 BEGIN ATOM",
     "M  V30 1 C 8.25 12.1847 0 0", "M  V30 2 C 6.9164 12.9547 0 0",
     "M  V30 3 C 7.2366 14.4611 0 0", "M  V30 4 C 8.7681 14.622 0 0",
     "M  V30 5 C 9.3945 13.2151 0 0", "M  V30 6 O 8.25 10.6447 0 0",
     "M  V30 7 F 9.5382 15.9557 0 0", "M  V30 8 C 9.5837 9.8747 0 0",
     "M  V30 9 C 10.9008 12.8949 0 0",
     "M  V30 END ATOM", "M  V30 BEGIN BOND",
     "M  V30 1 1 1 2", "M  V30 2 1 2 3", "M  V30 3 1 4 5",
     "M  V30 4 1 1 5", "M  V30 5 1 3 4", "M  V30 6 1 1 6",
     "M  V30 7 1 4 7", "M  V30 8 1 6 8", "M  V30 9 1 5 9",
     "M  V30 END BOND", "M  V30 LINKNODE 1 3 2 1 2 1 5",
     "M  V30 LINKNODE 1 2 2 5 1 5 4",
     "M  V30 END CTAB", "M  END", ""])

SAMPLE_D = "\n".join(
    ["", "  Mrv2014 12212013392D", "",
     "  0  0  0     0  0            999 V3000",
     "M  V30 BEGIN CTAB", "M  V30 COUNTS 12 12 0 0 0", "M  V30 BEGIN ATOM",
     "M  V30 1 C -6 4.7484 0 0", "M  V30 2 C -7.3337 3.9784 0 0",
     "M  V30 3 N -7.3337 2.4383 0 0", "M  V30 4 C -6 1.6683 0 0",
     "M  V30 5 C -4.6663 2.4383 0 0", "M  V30 6 C -4.6663 3.9784 0 0",
     "M  V30 7 C -5.8773 0.0617 0 0", "M  V30 8 C -3.2136 0.2013 0 0",
     "M  V30 9 C -4.5052 -0.6374 0 0", "M  V30 10 O -4.4246 -2.1753 0 0",
     "M  V30 11 * -6 4.235 0 0", "M  V30 12 C -4.845 6.2355 0 0",
     "M  V30 END ATOM", "M  V30 BEGIN BOND",
     "M  V30 1 1 1 2", "M  V30 2 2 2 3", "M  V30 3 1 3 4",
     "M  V30 4 2 4 5", "M  V30 5 1 5 6", "M  V30 6 2 1 6",
     "M  V30 7 1 8 9", "M  V30 8 1 7 9", "M  V30 9 1 5 8",
     "M  V30 10 1 7 4", "M  V30 11 1 9 10",
     "M  V30 12 1 11 12 ENDPTS=(3 1 2 6) ATTACH=ANY",
     "M  V30 END BOND", "M  V30 LINKNODE 1 3 2 9 7 9 8",
     "M  V30 END CTAB", "M  END", ""])

# sample E transcribed from spec doc section 6; an extra blank header line
# was inserted so the counts line (carrying the V3000 stamp) sits on line 4
SAMPLE_E = "\n".join(
    ["ACCLDraw10102310482D", "", "",
     "  0  0  0     0  0            999 V3000",
     "M  V30 BEGIN CTAB", "M  V30 COUNTS 13 12 0 0 0", "M  V30 BEGIN ATOM",
     "M  V30 1 C 8.0843 -7.4691 0 0 CHG=-1 CFG=3",
     "M  V30 2 C 6.5925 -7.6259 0 0", "M  V30 3 C 5.9824 -6.2556 0 0",
     "M  V30 4 C 7.0971 -5.2519 0 0", "M  V30 5 C 8.3962 -6.0019 0 0",
     "M  V30 6 * 11.6957 -7.5993 0 0", "M  V30 7 C 10.9375 -6.542 0 0",
     "M  V30 8 C 10.6256 -8.0092 0 0 CHG=-1 CFG=3",
     "M  V30 9 C 12.4293 -6.3852 0 0", "M  V30 10 C 13.0394 -7.7556 0 0",
     "M  V30 11 C 11.9246 -8.7592 0 0", "M  V30 12 * 7.1957 -6.5993 0 0",
     "M  V30 13 Fe 9.3436 -7.165 0 0 CHG=2",
     "M  V30 END ATOM", "M  V30 BEGIN BOND",
     "M  V30 1 1 1 2", "M  V30 2 2 2 3", "M  V30 3 1 3 4",
     "M  V30 4 2 4 5", "M  V30 5 1 5 1", "M  V30 6 1 7 8",
     "M  V30 7 2 7 9", "M  V30 8 1 9 10", "M  V30 9 2 10 11",
     "M  V30 10 1 8 11",
     "M  V30 11 9 13 12 ENDPTS=(5 1 5 4 3 2) ATTACH=ALL",
     "M  V30 12 9 13 6 ENDPTS=(5 7 8 9 10 11) ATTACH=ALL",
     "M  V30 END BOND", "M  V30 END CTAB", "M  END", ""])


# --------------------------------------------------------------------------
# P0 regression assertions (review_services / review_skills)
# --------------------------------------------------------------------------

def test_verify_rejects_empty_smiles():
    # MolFromSmiles("") returns a 0-atom valid Mol; an empty SMILES must be
    # rejected up front, not verified
    from services.structure_api import _verify_structure_impl
    for smi in ("", "   "):
        r = _verify_structure_impl(None, smi, None, None, None)
        assert r["verdict"] == "rejected"
        assert r["signals"]["rdkit_valid"] is False
        assert "empty SMILES" in r["contradictions"]


def test_verify_sequence_anchor_cyclo_notation():
    # patent cyclo() anchor must rebuild through the grammar and agree with
    # the decode; the old path passed the raw string to a list parameter and
    # always reported "grammar could not build"
    from services.structure_api import _verify_structure_impl
    from utils.sequence_parser import rebuild_from_anchor
    built = rebuild_from_anchor("cyclo(Nle-4Pal-PipOH-W)")
    assert built and Chem.MolFromSmiles(built)
    r = _verify_structure_impl(None, built, None, None,
                               "cyclo(Nle-4Pal-PipOH-W)")
    seq = r["signals"]["sequence_agreement"]
    assert seq["agree"] is True and seq["tanimoto"] >= 0.99


def test_lexicon_corrupt_json_fails_fast(tmp_path, monkeypatch):
    from skills import residue_lexicon
    from skills.peptide_notation import skill as pn_skill
    path = tmp_path / "residue_lexicon.json"
    path.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(residue_lexicon, "LEXICON_PATH", str(path))
    with pytest.raises(residue_lexicon.LexiconCorruptError):
        residue_lexicon.lexicon_snapshot()
    # the live write path must refuse too — not seed an empty lexicon over
    # the damaged file
    with pytest.raises(residue_lexicon.LexiconCorruptError):
        pn_skill._lexicon_pin("X", {"fragment_smiles": "NCC=O"})
    assert path.read_text(encoding="utf-8") == "{broken"


def test_markush_unknown_label_named_gap():
    import asyncio
    from skills.registry import load_skills, get
    load_skills()
    handler = get("markush_assemble").handler
    r = asyncio.run(handler({
        "template_smiles": "c1ccc([*:1])cc1",
        "fragments": {"1": ["C"], "9": ["CO"]},   # "9" is not on the template
    }))
    assert r["ok"] and len(r["members"]) == 1
    assert any("fragment label '9' not present in template" in g
               for g in r["gaps"])


def test_seed_free_acid_requires_carboxyl_tail():
    from skills.seed_lexicon import free_acid
    assert free_acid("N1CC[C@@H](O)C1C(=O)") == "N1CC[C@@H](O)C1C(=O)O"
    assert free_acid("NCC=O") == "NCC(=O)O"   # bare =O must not give C=OO
    with pytest.raises(ValueError, match="carboxyl"):
        free_acid("N[C@@H](C(N)=O)CS")         # thioether tail, no carboxyl


# --------------------------------------------------------------------------
# P1 regression assertions (review_skills / review_services / review_utils_frontend)
# --------------------------------------------------------------------------

def test_model_api_fault_named_not_retried(monkeypatch):
    # an LLM transport failure must surface with its real cause in the
    # attempt trail, not masquerade as "model returned no parseable JSON"
    # while burning correction rounds
    from skills.peptide_notation import skill as pn
    def _dead_client():
        raise ConnectionError("gateway down")
    monkeypatch.setattr(pn, "_client", _dead_client)
    with pytest.raises(pn.ModelCallError, match="LLM API call failed"):
        pn._ask_model("cyclo(W-W-W)", "", "")
    out = pn.read_and_build("cyclo(W-W-W)")
    assert out["smiles"] is None
    assert "LLM API call failed" in out["attempts"][-1]["failure"]
    assert any("model call failed" in g["reason"] for g in out["gaps"])
    assert len(out["attempts"]) == 1          # no correction rounds burned


def test_ring_size_out_of_range_becomes_named_note():
    # silent clamp violated the "nothing silently overwritten" contract
    parsed = _parsed([NLE, PAL, TRP], ring_size=99)
    units, failures = _validate_units(parsed)
    assert not failures
    smiles, err, notes = _assemble(units, parsed)
    assert smiles, err
    assert any("ring_size 99 out of range" in n for n in notes)


def test_stitch_failure_carries_reason():
    from skills.markush_assemble.skill import _stitch
    mol = Chem.MolFromSmiles("c1ccc([*:1])cc1")
    merged, err = _stitch(mol, {"1": "C(N)(N)(N)(N)N"})   # unparseable fragment
    assert merged is None and "does not parse" in err


def test_sequence_assembler_requires_page_numbers():
    import asyncio
    from skills.registry import load_skills, get
    load_skills()
    r = asyncio.run(get("sequence_assembler").handler(
        {"pages": [{"text": "SEQ ID NO: 7\nACDEFGHIKLMNP"}]}))
    assert not r["ok"] and "pages[0] missing integer 'page'" in r["error"]


def test_peptide_notation_empty_input_named():
    import asyncio
    from skills.registry import load_skills, get
    load_skills()
    r = asyncio.run(get("peptide_notation").handler({}))
    assert r["smiles"] is None and r["attempts"] == []
    assert r["gaps"][0]["reason"] == "empty notation input"


def test_activity_unitless_value_is_named(monkeypatch):
    md = ("<table><tr><td>Compound</td><td>IC50</td></tr>"
          "<tr><td>9</td><td>12.5</td></tr></table>")
    result = _review_activity_fixture(monkeypatch, md, [("9", "IC50", "12.5", 12.5, "", None)])
    value = result["rows"][0]["assays"]["IC50"]
    assert value["unit"] is None and value["unit_missing"] is True
    assert any("unit" in gap for gap in result["gaps"])


def test_activity_identifiers_do_not_depend_on_word_order(monkeypatch):
    result = _review_activity_fixture(monkeypatch, "IC50 12.5 nM cpd 5\ncpd 7 EC50 3 nM\n", [
        ("5", "IC50", "12.5 nM", 12.5, "", "nM"),
        ("7", "EC50", "3 nM", 3, "", "nM"),
    ])
    assert [row["compound_id"] for row in result["rows"]] == ["5", "7"]


def test_rdkit_import_failure_is_named(monkeypatch):
    # an rdkit import failure is an environment fault: it propagates as
    # ImportError instead of reporting every SMILES as invalid
    import services.structure_api as api
    monkeypatch.setitem(sys.modules, "rdkit", None)
    monkeypatch.setitem(sys.modules, "rdkit.Chem", None)
    monkeypatch.setitem(sys.modules, "rdkit.RDLogger", None)
    with pytest.raises(ImportError):
        api._rdkit_ok("CCO")


def test_find_sequence_by_id_digit_exact():
    # "3" must not substring-match "53"; a label with no registered entity
    # stays unmatched (no page-proximity guessing)
    from services.document_ledger import DocumentLedger, SequenceEntity
    ledger = DocumentLedger("t")
    ledger.add_sequence(SequenceEntity(entity_id="s53", page=1,
                                       codes=["ACDEF"], seq_id="SEQ ID NO: 53"))
    ledger.add_sequence(SequenceEntity(entity_id="s3", page=1,
                                       codes=["ACDEFK"], seq_id="3"))
    hit3 = ledger._find_sequence_by_id("SEQ ID NO: 3", 1)
    assert hit3 is not None and hit3.entity_id == "s3"
    hit53 = ledger._find_sequence_by_id("53", 1)
    assert hit53 is not None and hit53.entity_id == "s53"
    assert ledger._find_sequence_by_id("99", 1) is None


def test_ledger_cache_bounded_and_document_id_required():
    from services import structure_api
    with pytest.raises(ValueError, match="document_id is required"):
        structure_api._ledger_for_payload({})
    for i in range(structure_api._LEDGERS_MAX + 5):
        structure_api._ledger_for_payload({"document_id": f"doc{i}"})
    assert len(structure_api._LEDGERS) <= structure_api._LEDGERS_MAX
    assert "doc0" not in structure_api._LEDGERS            # oldest dropped
    assert f"doc{structure_api._LEDGERS_MAX + 4}" in structure_api._LEDGERS


def test_heavy_harness_endpoints_off_event_loop():
    # sync def endpoints run in FastAPI's worker thread pool; serial OCR/LLM
    # calls inside them must not block the event loop (parse-sequence precedent)
    import inspect
    from services import structure_api
    for name in ("ocr_page", "identify_compound_id", "link_activity",
                 "harness_parse_page"):
        fn = getattr(structure_api, name)
        assert not inspect.iscoroutinefunction(fn), f"{name} still async"


def test_sidechain_exotic_fragments_stereo_consistent():
    # the one-letter path (_SIDECHAIN) and the codes path
    # (EXOTIC_RESIDUE_FRAGMENTS) must build identical molecules for I/T
    from utils.sequence_parser import (_SIDECHAIN, EXOTIC_RESIDUE_FRAGMENTS,
                                       build_cyclopeptide_smiles,
                                       sequence_to_smiles)
    for code in ("I", "T"):
        assert _SIDECHAIN[code] == EXOTIC_RESIDUE_FRAGMENTS[code]
    for seq in ("WIG", "WTG"):
        one = sequence_to_smiles(seq, cyclic=True)
        codes = build_cyclopeptide_smiles(list(seq))
        assert one and codes
        assert one == codes, seq


def test_nmeg_defined_once_in_units():
    # no 'SARCOSINE' sentinel in the fragments dict, no per-consumer patch:
    # sarcosine lives once in EXOTIC_RESIDUE_UNITS with 'Sar' as an alias
    from utils.sequence_parser import (EXOTIC_RESIDUE_FRAGMENTS,
                                       EXOTIC_RESIDUE_UNITS,
                                       build_cyclopeptide_smiles)
    assert "NMeG" not in EXOTIC_RESIDUE_FRAGMENTS
    assert EXOTIC_RESIDUE_UNITS["NMeG"] == "N(C)CC(=O)"
    assert build_cyclopeptide_smiles(["Nle", "NMeG", "W"]) == \
        build_cyclopeptide_smiles(["Nle", "Sar", "W"])


def test_ring_digit_9_collision_renamed(monkeypatch):
    # a fragment opening digit 9 while the macrocycle digit 9 is still open
    # must be renamed — the old identity remap mis-paired the closures and
    # welded a bogus N-N bond into the product
    import utils.sequence_parser as sp
    monkeypatch.setitem(sp.EXOTIC_RESIDUE_UNITS, "NineRing", "N9CCCCC9C(=O)")
    smiles = sp.build_cyclopeptide_smiles(["W", "NineRing", "G"])
    assert smiles
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    ring_sizes = sorted({len(r) for r in mol.GetRingInfo().AtomRings()})
    assert 6 in ring_sizes                    # the fragment's own ring intact
    assert max(ring_sizes) >= 9               # the macrocycle intact (was 6)
    assert not any(a.GetSymbol() == "N" and
                   any(n.GetSymbol() == "N" for n in a.GetNeighbors())
                   for a in mol.GetAtoms())   # no mis-paired N-N bond


# --- render_via_engine: external-engine 2D layout into the ALIGNED style ---

_SEQ53_SMILES = (
    "CCCC[C@@H]1NC(=O)[C@@H]2CC[C@@H](C2)NC(=O)[C@H]([C@H](C)c2c[nH]c3c(OC)"
    "cccc23)NC(=O)CN(C)C(=O)[C@H](Cc2ccc(OCC(=O)O)cc2)NC(=O)[C@H](Cc2ccc(C("
    "N)=O)cc2)NC(=O)[C@](C)(CO)NC(=O)[C@H](Cc2c[nH]c3cc(Cl)ccc23)NC(=O)[C@H"
    "](Cc2ccc(OCC(=O)O)cc2)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H]2C[C@@H](O)CN2C(="
    "O)[C@H](Cc2c[nH]c3ccccc23)NC(=O)[C@H]2[C@H](O)CCCN2C(=O)[C@H](Cc2ccncc"
    "2)NC1=O"
)
