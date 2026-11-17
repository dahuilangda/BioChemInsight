"""Skill registration and peptide operation contracts."""

import asyncio

import pytest

from skills import registry
from skills.cross_validate import skill


@pytest.fixture
def empty_registry(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})


async def _handler(payload):
    return {"ok": True, "value": payload}


def test_duplicate_registration_does_not_replace_handler(empty_registry):
    registry.register("read_sequence", "model", "Read a sequence")(_handler)
    original = registry.get("read_sequence")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("read_sequence", "chemistry", "Different operation")(_handler)
    assert registry.get("read_sequence") is original


@pytest.mark.parametrize("name", ["", "ReadSequence", "read-sequence", "序列", 1])
def test_skill_names_are_stable_identifiers(empty_registry, name):
    with pytest.raises(ValueError):
        registry.register(name, "model", "Read a sequence")(_handler)


def test_handler_must_be_async(empty_registry):
    with pytest.raises(TypeError):
        registry.register("read_sequence", "model", "Read a sequence")(
            lambda value: value
        )
    assert registry.get("read_sequence") is None


def test_catalog_does_not_expose_mutable_contracts(empty_registry):
    schema = {"type": "object", "required": ["sequence"]}
    registry.register("read_sequence", "model", "Read a sequence", input_schema=schema)(
        _handler
    )
    schema["required"].clear()
    catalog = registry.listing()
    assert catalog[0]["input_schema"]["required"] == ["sequence"]
    catalog[0]["input_schema"]["required"].clear()
    assert registry.get("read_sequence").input_schema["required"] == ["sequence"]


def test_residue_decoder_calls_graph_function(monkeypatch):
    observed = []
    monkeypatch.setattr(
        skill,
        "decode_residue_codes",
        lambda smiles: (observed.append(smiles) or ["G", "A", "S"], True, "matched"),
    )
    result = asyncio.run(skill.smiles_to_sequence({"smiles": "CCO"}))
    assert observed == ["CCO"]
    assert result["ok"] and result["codes"] == ["G", "A", "S"]
    assert (
        not result["stereochemistry_verified"]
        and not result["source_identity_verified"]
    )


@pytest.mark.parametrize(
    "payload", [{}, {"smiles": None}, {"smiles": ""}, {"smiles": 3}]
)
def test_invalid_smiles_does_not_call_decoder(monkeypatch, payload):
    def forbidden(_):
        pytest.fail("invalid input reached the graph decoder")

    monkeypatch.setattr(skill, "decode_residue_codes", forbidden)
    assert asyncio.run(skill.smiles_to_sequence(payload))["ok"] is False


@pytest.mark.parametrize(
    "codes", [None, [], "GAS", ["G", None, "S"], ["G", "", "S"], ["G", "A"]]
)
def test_invalid_residue_input_does_not_reach_assembly(monkeypatch, codes):
    def forbidden(_):
        pytest.fail("invalid input reached graph assembly")

    monkeypatch.setattr(skill, "build_cyclopeptide_smiles", forbidden)
    assert asyncio.run(skill.sequence_to_smiles({"codes": codes}))["ok"] is False


def test_residue_agreement_does_not_claim_source_verification(monkeypatch):
    monkeypatch.setattr(
        skill, "compare_with_anchor", lambda smiles, codes: {"ok": True, "agree": True}
    )
    result = asyncio.run(skill.cross_validate({"smiles": "CCO", "codes": ["G"]}))
    assert result["agree"]
    assert result["validation_scope"] == "residue_codes"
    assert (
        not result["stereochemistry_verified"]
        and not result["source_identity_verified"]
    )


def test_registered_catalog_has_input_contracts():
    registry.load_skills()
    catalog = registry.listing()
    assert len(catalog) == 13
    assert all(item["input_schema"].get("type") == "object" for item in catalog)


@pytest.mark.parametrize(
    "sequence", ["ACD-EFG", "ACD EFG", "acdefg", "ACD3EFG", "ＡＣＤ", None, 3]
)
def test_numbering_does_not_rewrite_source_sequences(monkeypatch, sequence):
    from skills.antibody_numbering import skill as numbering

    def forbidden():
        pytest.fail("Invalid sequence reached numbering model")

    monkeypatch.setattr(numbering, "_model", forbidden)
    assert not asyncio.run(numbering.number_chain({"sequence": sequence}))["ok"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"hgvs": []},
        {"hgvs": 3},
        {"hgvs": [None]},
        {"hgvs": [""]},
        {"hgvs": "p.Ala1Gly", "hgvs_string": "p.Ala1Val"},
    ],
)
def test_invalid_variant_input_is_not_reported_as_success(payload):
    from skills.variant_sites.skill import parse_variants

    assert not asyncio.run(parse_variants(payload))["ok"]


def test_empty_numbering_result_is_not_success(monkeypatch):
    from types import SimpleNamespace

    from skills.antibody_numbering import skill as numbering

    monkeypatch.setattr(
        numbering, "_model", lambda: SimpleNamespace(number=lambda _: {})
    )
    assert not asyncio.run(numbering.number_chain({"sequence": "ACDEFGHIKLMNP"}))["ok"]


@pytest.mark.parametrize(
    "error_name,status",
    [
        ("ModelEvidenceError", 422),
        ("ModelTimeoutError", 504),
        ("ModelProviderError", 502),
        ("ModelCapacityError", 503),
        ("ModelContractError", 502),
        ("ModelDownstreamVerificationError", 502),
    ],
)
def test_skill_api_distinguishes_evidence_transport_and_contract_errors(
    monkeypatch, error_name, status
):
    import json
    from types import SimpleNamespace

    from services.structure_api import skill_invoke
    from utils import model_harness

    exception = getattr(model_harness, error_name)("test failure")

    async def fail(payload):
        raise exception

    monkeypatch.setattr(registry, "load_skills", lambda: None)
    monkeypatch.setattr(registry, "get", lambda name: SimpleNamespace(handler=fail))
    response = skill_invoke("test", {})
    body = json.loads(response.body)
    assert response.status_code == status
    assert body["ok"] is False and body["error_type"] == exception.error_type


def test_skill_api_preserves_source_review_details(monkeypatch):
    import json
    from types import SimpleNamespace

    from services.structure_api import skill_invoke
    from utils.model_harness import ModelSourceAmbiguityError

    details = {
        "review": {"issues": [{"description": "Conflicting IC50 values", "page": 4}]}
    }

    async def fail(payload):
        raise ModelSourceAmbiguityError("Unresolved", details=details)

    monkeypatch.setattr(registry, "load_skills", lambda: None)
    monkeypatch.setattr(registry, "get", lambda name: SimpleNamespace(handler=fail))
    response = skill_invoke("test", {})
    assert response.status_code == 422
    assert json.loads(response.body)["details"] == details
