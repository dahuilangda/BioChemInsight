"""Local parsing and remote identity lookup are separate, auditable operations."""
from __future__ import annotations
import asyncio
from skills.registry import register


def _name(payload):
    name = payload.get('name')
    if not isinstance(name, str) or not name.strip() or len(name) > 4096:
        raise ValueError('A source-supported chemical name is required')
    return name.strip()


@register('pubchem_lookup', 'external', 'Resolve a source-supported chemical name through cached, globally throttled PubChem PUG-REST. Returns ambiguity/cooldown explicitly; does not choose an attachment site.')
async def pubchem_lookup(payload):
    from utils.pubchem_client import pubchem_lookup_name_details
    try: name = _name(payload)
    except ValueError as exc: return {'ok': False, 'status': 'invalid', 'error': str(exc)}
    evidence = str(payload.get('evidence') or '').strip()
    if not evidence:
        return {'ok': False, 'status': 'blocked', 'error': 'Source-name evidence required before external lookup'}
    result = await asyncio.to_thread(pubchem_lookup_name_details, name)
    return {'ok': result['status'] == 'found', **result, 'operation': 'pubchem_lookup',
            'input_evidence': evidence, 'purpose': str(payload.get('purpose') or 'identity_resolution')}


@register('opsin_resolve', 'chemistry', 'Parse a source-supported chemical name locally. intent=fragment yields an explicit attachment dummy; intent=compound requires a closed molecule. No network lookup.')
async def opsin_resolve(payload):
    from rdkit import Chem
    from utils.series_member_synthesis import _opsin_attachment_smiles, parse_attachment_fragment_smiles, parse_full_molecule_smiles
    try: name = _name(payload)
    except ValueError as exc: return {'ok': False, 'status': 'invalid', 'error': str(exc)}
    intent = payload.get('intent', 'compound')
    if intent not in {'compound', 'fragment'}:
        return {'ok': False, 'status': 'invalid', 'error': 'intent must be compound or fragment'}
    def run():
        if intent == 'fragment':
            molecule = parse_attachment_fragment_smiles(_opsin_attachment_smiles(name))
        else:
            try:
                from utils.opsin_local import name_to_smiles as py2opsin
                molecule = parse_full_molecule_smiles(py2opsin(name))
            except Exception: molecule = None
        if molecule is None:
            return {'ok': False, 'status': 'unresolved', 'operation': 'opsin_resolve', 'name': name, 'intent': intent}
        return {'ok': True, 'status': 'resolved', 'operation': 'opsin_resolve', 'name': name, 'intent': intent,
                'smiles': Chem.MolToSmiles(molecule), 'source': 'local_opsin',
                'attachment_sites': sum(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms())}
    return await asyncio.to_thread(run)


@register('peptide_graph_validate', 'chemistry', 'Decompose a peptide atomic graph into typed monomers and connections, reassemble exactly, retain unknown modifications and ambiguity. Does not replace MolSight output or flatten to a one-letter sequence.')
async def peptide_graph_validate(payload):
    def run():
        import sys
        from pathlib import Path
        source = Path(__file__).resolve().parents[2] / 'packages' / 'peptidecore' / 'src'
        if str(source) not in sys.path: sys.path.insert(0, str(source))
        from peptidecore import decompose_smiles
        text = str(payload.get('smiles') or '')
        if not text: return {'ok': False, 'status': 'invalid', 'error': 'Atomic peptide structure required'}
        try:
            result = decompose_smiles(text, timeout=min(5.0, max(.1, float(payload.get('timeout', 1.0)))), max_solutions=4, max_states=4000)
            details = result.as_dict()
            return {'ok': result.unique, 'status': result.status, 'operation': 'peptide_graph_validate',
                'validation': details, 'image_identity_verified': False,
                'note': 'Exact monomer/connection round trip checks the candidate graph, not its identity with the source image.'}
        except Exception as exc:
            return {'ok': False, 'status': 'unresolved', 'error': f'{type(exc).__name__}: {str(exc)[:200]}'}
    return await asyncio.to_thread(run)

# Publish contracts for tool discovery; examples and document-specific names are absent.
from skills.registry import get as _get_skill
_get_skill('pubchem_lookup').input_schema = {'type': 'object', 'required': ['name', 'evidence'], 'properties': {
    'name': {'type': 'string', 'minLength': 1}, 'evidence': {'type': 'string', 'minLength': 1}, 'purpose': {'type': 'string'}}}
_get_skill('opsin_resolve').input_schema = {'type': 'object', 'required': ['name'], 'properties': {
    'name': {'type': 'string', 'minLength': 1}, 'intent': {'enum': ['compound', 'fragment']}}}
_get_skill('peptide_graph_validate').input_schema = {'type': 'object', 'required': ['smiles'], 'properties': {
    'smiles': {'type': 'string', 'minLength': 1}, 'timeout': {'type': 'number', 'minimum': .1, 'maximum': 5}}}
