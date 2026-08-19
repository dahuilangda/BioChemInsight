"""Series member structure synthesis.

A series-range structure row (a scaffold drawn once and labelled with a range
identifier) declares members that the document defines individually through
text: substituent descriptions in member tables or full chemical names in
characterization sections. This module turns that textual evidence into
verified member structures.

Two evidence paths produce a member structure:
- name path: a full chemical name stated for the member converts to a complete
  molecule;
- substituent path: a substituent description converts to a fragment with one
  dummy attachment atom and assembles onto the scaffold's dummy site, which
  preserves the drawn scaffold pose.

Every produced structure is verified chemically, and the member set must agree
on a substantial common core before any row is emitted. Members without
verified structure evidence are simply not emitted; their assay data is
unaffected.
"""

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem

from utils.series_id_utils import parse_series_range_id

MIN_MEMBERS_FOR_CONSISTENCY = 3
MIN_MCS_HEAVY_ATOMS = 8
MIN_MCS_RATIO = 0.5

_NAME_DASH_TRANSLATION = str.maketrans({
    '–': '-', '—': '-', '‑': '-', ' ': ' ', '\u2009': ' ', '\u00a0': ' ',
})

_CJK_RE = None


def _contains_cjk(text):
    global _CJK_RE
    if _CJK_RE is None:
        import re
        _CJK_RE = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')
    return bool(_CJK_RE.search(str(text or '')))


def english_name_for_lookup(full_name, audit_path=None, series_id=''):
    """Return an English compound name for external lookup.

    Non-English names are translated by the text model — translation only,
    never structure generation. An empty result means the text does not name
    a definite compound.
    """
    name = ' '.join(str(full_name or '').translate(_NAME_DASH_TRANSLATION).split())
    if not name:
        return ''
    if not _contains_cjk(name):
        return name
    from utils.llm_utils import translate_compound_name
    translated = translate_compound_name(
        name,
        audit_path=audit_path,
        metadata={'scope': 'series_member_synthesis', 'series_id': series_id, 'path': 'name_translation'},
    )
    return ' '.join(str(translated or '').split())


def name_to_molecule(full_name, audit_path=None, series_id=''):
    """Resolve a complete chemical name to a molecule through external authorities.

    Lookup order: PubChem (authoritative record with CID), then OPSIN
    (deterministic local grammar parse for compounds not yet deposited).
    A miss on both is final — no structure is ever guessed.
    """
    from utils.pubchem_client import pubchem_lookup_name

    name = english_name_for_lookup(full_name, audit_path=audit_path, series_id=series_id)
    if not name:
        return None, 'name did not resolve to English'
    record = pubchem_lookup_name(name)
    if record is not None:
        mol = parse_full_molecule_smiles(record.get('isomeric_smiles'))
        if mol is not None:
            mol.SetProp('_LookupCID', str(record.get('cid')))
            mol.SetProp('_LookupSource', 'pubchem')
            return mol, ''
    try:
        import warnings
        from py2opsin import py2opsin as opsin_convert
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            smiles = opsin_convert(name)
    except Exception:
        return None, 'no lookup source available'
    if not smiles:
        return None, 'name not found in PubChem and not parseable by OPSIN'
    mol = parse_full_molecule_smiles(smiles)
    if mol is None:
        return None, 'OPSIN output failed chemical validation'
    mol.SetProp('_LookupSource', 'opsin')
    return mol, ''


def parse_full_molecule_smiles(smiles):
    """Parse and sanitize a complete-molecule SMILES; None when invalid."""
    text = str(smiles or '').strip()
    if not text or '*' in text:
        return None
    try:
        mol = Chem.MolFromSmiles(text)
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() <= 1:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    return mol


def parse_attachment_fragment_smiles(smiles):
    """Parse a fragment SMILES that must carry exactly one dummy atom."""
    text = str(smiles or '').strip()
    if not text or '*' not in text:
        return None
    try:
        mol = Chem.MolFromSmiles(text)
    except Exception:
        return None
    if mol is None:
        return None
    dummies = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != 1:
        return None
    if dummies[0].GetDegree() != 1:
        return None
    if mol.GetNumAtoms() - len(dummies) < 1:
        return None
    return mol


def molblock_dummy_atoms(molblock):
    """Return (index, label) for dummy atoms in a molblock molecule."""
    mol = Chem.MolFromMolBlock(str(molblock or ''), sanitize=False)
    if mol is None:
        return []
    result = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            result.append((atom.GetIdx(), atom.GetSymbol()))
    return result


MIN_MEDOID_CORE_RATIO = 0.8


def _pair_shared_core_atoms(mol_a, mol_b):
    try:
        mcs = rdFMCS.FindMCS(
            [mol_a, mol_b],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=10,
        )
    except Exception:
        return 0
    return mcs.numAtoms


def quarantine_core_outliers(candidate_mols):
    """Quarantine members that conflict with the series majority core.

    Documents occasionally carry corrupted member names (a ring heteroatom
    mistyped, a synonym pasted from a neighbour row). The medoid member is
    the one agreeing with the most others; members that share too little
    with it are excluded from emission instead of being silently corrected.
    Returns (kept dict, quarantine reasons dict).
    """
    if not candidate_mols:
        return {}, {}
    items = list(candidate_mols.items())
    if len(items) <= 2:
        return dict(items), {}
    mols = [entry[1] for _, entry in items]

    def consistent(idx_a, idx_b):
        shared = _pair_shared_core_atoms(mols[idx_a], mols[idx_b])
        smallest = min(mols[idx_a].GetNumAtoms(), mols[idx_b].GetNumAtoms())
        return smallest > 0 and shared >= MIN_MEDOID_CORE_RATIO * smallest

    agreement = [sum(1 for j in range(len(mols)) if j != i and consistent(i, j)) for i in range(len(mols))]
    medoid_idx = max(range(len(mols)), key=lambda i: agreement[i])
    kept = {}
    quarantined = {}
    for idx, (compound_id, entry) in enumerate(items):
        if idx == medoid_idx or consistent(medoid_idx, idx):
            kept[compound_id] = entry
        else:
            quarantined[compound_id] = 'structure conflicts with the series majority core'
    return kept, quarantined


def member_set_common_core(mols):
    """Find the maximum common substructure shared by all member molecules.

    Returns (mcs_mol, reason) — mcs_mol is None with a reason when the member
    set does not agree on a substantial common core.
    """
    if not mols:
        return None, 'no members'
    if len(mols) < MIN_MEMBERS_FOR_CONSISTENCY:
        return None, 'consistency check requires more members'
    try:
        mcs = rdFMCS.FindMCS(
            mols,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=15,
        )
    except Exception as exc:
        return None, f'mcs computation failed: {exc}'
    if mcs.numAtoms <= 0 or not mcs.smartsString:
        return None, 'no common substructure'
    core = Chem.MolFromSmarts(mcs.smartsString)
    if core is None:
        return None, 'common substructure is not parseable'
    if mcs.numAtoms < MIN_MCS_HEAVY_ATOMS:
        return None, f'common core too small: {mcs.numAtoms} atoms'
    sizes = sorted(mol.GetNumAtoms() for mol in mols)
    median_size = sizes[len(sizes) // 2]
    if mcs.numAtoms < MIN_MCS_RATIO * median_size:
        return None, (
            f'common core covers {mcs.numAtoms}/{median_size} atoms of the median member'
        )
    Chem.FastFindRings(core)
    if core.GetRingInfo().NumRings() == 0:
        return None, 'common core contains no ring system'
    return core, ''


def synthesize_member_from_name(full_name, smiles=None):
    """Verify a name-derived member structure; returns mol or None."""
    mol = parse_full_molecule_smiles(smiles) if smiles else None
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    try:
        Chem.SanitizeMol(mol)
        AllChem.Compute2DCoords(mol)
    except Exception:
        return None
    return mol


def substituent_consistent_with_name(substituent_text, full_name):
    """Cross-check the two textual evidence sources for one member.

    Both directions of normalized containment count; members whose table
    substituent contradicts the stated name fail the evidence check.
    """
    substituent = ' '.join(str(substituent_text or '').lower().translate(_NAME_DASH_TRANSLATION).split())
    name = ' '.join(str(full_name or '').lower().translate(_NAME_DASH_TRANSLATION).split())
    substituent = substituent.strip(' ()[]')
    name = name.strip(' ()[]')
    if not substituent or not name or len(substituent) < 3:
        return True
    if substituent in name:
        return True
    compact_substituent = substituent.replace('-', '').replace(' ', '')
    compact_name = name.replace('-', '').replace(' ', '')
    if compact_substituent and compact_substituent in compact_name:
        return True
    token_overlap = {token for token in substituent.split() if len(token) > 3}
    if token_overlap and token_overlap.issubset(set(name.split())):
        return True
    return False


def substituent_text_to_attachment_fragment(substituent_text, audit_path=None, series_id=''):
    """Convert a substituent description to a verified attachment fragment.

    Deterministic sources only: a non-English description is first translated
    by the text model (translation only), then OPSIN parses the group with a
    wildcard radical marking the attachment atom. A miss is final - fragments
    are never guessed.
    """
    name = english_name_for_lookup(substituent_text, audit_path=audit_path, series_id=series_id)
    if not name:
        return None
    try:
        import warnings
        from py2opsin import py2opsin as opsin_convert
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            smiles = opsin_convert(name, allow_radicals=True, wildcard_radicals=True)
    except Exception:
        return None
    if not smiles:
        return None
    return parse_attachment_fragment_smiles(smiles)


def _connectivity_smiles(mol):
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False)
    except Exception:
        return Chem.MolToSmiles(mol, isomericSmiles=False)


def structures_agree(mol_a, mol_b):
    '''True when two molecules have identical connectivity.'''
    if mol_a is None or mol_b is None:
        return False
    return _connectivity_smiles(mol_a) == _connectivity_smiles(mol_b)


def assemble_member_on_scaffold(scaffold_molblock, attachment_smiles, variable_position=''):
    """Attach a text-derived substituent fragment onto a scaffold dummy site.

    The scaffold's drawn pose is preserved; coordinates are generated for the
    substituent atoms only. Returns (mol, reason).
    """
    from utils.markush_assembly import _combine_at_site, _dummy_sites_by_variable

    fragment = parse_attachment_fragment_smiles(attachment_smiles)
    if fragment is None:
        return None, 'substituent fragment did not pass attachment validation'
    scaffold = Chem.MolFromMolBlock(str(scaffold_molblock or ''))
    if scaffold is None:
        return None, 'scaffold molblock is not parseable'
    try:
        AllChem.Compute2DCoords(fragment)
    except Exception:
        return None, 'substituent fragment coordinates failed'
    from utils.markush_text_substituent import normalize_variable_position

    sites_by_variable, unlabeled = _dummy_sites_by_variable(scaffold)
    all_sites = [site for sites in sites_by_variable.values() for site in sites] + unlabeled
    label = normalize_variable_position(str(variable_position or '').strip())
    if label and sites_by_variable.get(label):
        candidates = sites_by_variable[label]
    elif len(all_sites) == 1:
        candidates = all_sites
    else:
        candidates = []
    if len(candidates) != 1:
        if not all_sites:
            return None, 'scaffold has no dummy attachment site'
        return None, (
            f'scaffold has {len(all_sites)} attachment sites and no unambiguous '
            f'match for variable position {label!r}'
        )
    mol, reason, note = _combine_at_site(scaffold, candidates[0], fragment)
    if mol is None:
        return None, reason or 'assembly failed'
    return mol, note


def _source_pages_from_row(series_row):
    raw = series_row.get('source_pages')
    if raw is None:
        raw = series_row.get('PAGE_NUM', '')
    if isinstance(raw, (list, tuple)):
        candidates = [str(page) for page in raw]
    else:
        candidates = str(raw or '').replace(';', ',').split(',')
    pages = []
    for candidate in candidates:
        text = candidate.strip()
        if text.isdigit() and int(text) not in pages:
            pages.append(int(text))
    return pages


def build_member_record(
    series_row,
    member,
    mol,
    method,
    synthesis_note='',
    lookup_evidence='',
):
    """Build a structures.csv row for a synthesized series member."""
    smiles = Chem.MolToSmiles(mol)
    molblock = Chem.MolToMolBlock(mol)
    source_pages = _source_pages_from_row(series_row)
    for page in member.get('evidence_pages') or []:
        if int(page) not in source_pages:
            source_pages.append(int(page))
    return {
        'COMPOUND_ID': member['compound_id'],
        'SMILES': smiles,
        'PAGE_NUM': source_pages[0] if source_pages else series_row.get('PAGE_NUM', ''),
        'MOLBLOCK': molblock,
        'STRUCTURE_TYPE': 'series_member',
        'IS_COMPLETE_COMPOUND': True,
        'FILTERED_OUT': False,
        'IMAGE_FILE': series_row.get('IMAGE_FILE', ''),
        'SEGMENT_FILE': series_row.get('SEGMENT_FILE', ''),
        'BOX_COORDS_FILE': series_row.get('BOX_COORDS_FILE', ''),
        'PAGE_IMAGE_FILE': series_row.get('PAGE_IMAGE_FILE', ''),
        'source_pages': source_pages,
        'SERIES_COMPOUND_ID': series_row.get('COMPOUND_ID', ''),
        'SERIES_MEMBER_SUBSTITUENT': member.get('substituent_text', ''),
        'SERIES_MEMBER_NAME': member.get('full_name', ''),
        'SERIES_MEMBER_VARIABLE': member.get('variable_position', ''),
        'SERIES_SYNTHESIS_METHOD': method,
        'SERIES_SYNTHESIS_NOTE': str(synthesis_note or ''),
        'SERIES_MEMBER_LOOKUP': str(lookup_evidence or ''),
        'SERIES_EVIDENCE_PAGES': ','.join(str(p) for p in member.get('evidence_pages') or []),
        'SERIES_MEMBER_EVIDENCE': member.get('evidence_summary', ''),
    }


def collect_series_rows(structure_records):
    """Select structure records whose compound id is a series range."""
    series_rows = []
    for record in structure_records or []:
        if not isinstance(record, dict):
            continue
        compound_id = str(record.get('COMPOUND_ID') or '').strip()
        if not compound_id:
            continue
        series = parse_series_range_id(compound_id)
        if not series:
            continue
        if not (record.get('MOLBLOCK') or record.get('SMILES')):
            continue
        row = dict(record)
        row['series_members'] = series['members']
        series_rows.append(row)
    return series_rows


_TEXT_SERIES_RANGE_RE = None
_FIGURE_CONTEXT_RE = None


def discover_text_declared_series(page_contexts, min_named_members=3, max_members=64):
    """Find series declarations in document text.

    A letter-suffix range token with at least three named members declares a
    compound series. Numeric-only ranges are page/year-like ambiguity and are
    not treated as series declarations. Range tokens preceded by figure,
    table, or scheme markers denote subpanels, not compounds.
    """
    global _TEXT_SERIES_RANGE_RE, _FIGURE_CONTEXT_RE
    import re
    if _TEXT_SERIES_RANGE_RE is None:
        _TEXT_SERIES_RANGE_RE = re.compile(
            r'(?<![0-9A-Za-z])((?:[A-Za-z]?\d+|[IVXLCM]{1,5})([a-z]))\s*[-\u2013\u2014]\s*([a-z])(?![0-9A-Za-z])'
        )
        _FIGURE_CONTEXT_RE = re.compile(
            r'(?:fig|figure|table|scheme|panel)\s*\.?\s*\d*\s*[a-z]{0,3}\s*$',
            flags=re.IGNORECASE,
        )
    combined_pages = []
    series_members = {}
    for context in page_contexts or []:
        try:
            page = int(context.get('page'))
        except (TypeError, ValueError):
            continue
        text = str(context.get('markdown') or '')
        if not text.strip():
            continue
        combined_pages.append({'page': page, 'markdown': text})
        for match in _TEXT_SERIES_RANGE_RE.finditer(text):
            prefix_window = text[max(0, match.start() - 24):match.start()]
            if _FIGURE_CONTEXT_RE.search(prefix_window):
                continue
            series_id = f"{match.group(1)[:-1]}{match.group(2)}-{match.group(3)}"
            parsed = parse_series_range_id(series_id)
            if not parsed:
                continue
            members = [m for m in parsed['members'][:max_members]]
            named = [m for m in members if re.search(rf'(?<![0-9A-Za-z]){re.escape(m)}(?![0-9A-Za-z])', text)]
            if len(named) >= min_named_members:
                existing = series_members.get(series_id)
                series_members[series_id] = existing if existing is not None and len(existing) > len(named) else named
    if not series_members:
        return []
    # A declaration whose member set is contained in another declaration's
    # member set is a prose subset mention (a partial range of the same
    # series), not an independent series.
    series_ids = list(series_members.keys())
    for series_id in series_ids:
        members = set(series_members[series_id])
        for other_id, other_members in series_members.items():
            if other_id == series_id or len(members) >= len(set(other_members)):
                continue
            if members.issubset(set(other_members)):
                series_members.pop(series_id, None)
                break
    declared = []
    for series_id, members in series_members.items():
        evidence_pages = [
            entry['page'] for entry in combined_pages
            if any(re.search(rf'(?<![0-9A-Za-z]){re.escape(m)}(?![0-9A-Za-z])', entry['markdown']) for m in members)
        ]
        declared.append(
            {
                'COMPOUND_ID': series_id,
                'SMILES': '',
                'MOLBLOCK': '',
                'STRUCTURE_TYPE': 'text_declared_series',
                'PAGE_NUM': evidence_pages[0] if evidence_pages else '',
                'series_members': members,
                'declared_in_text': True,
                'source_pages': evidence_pages,
            }
        )
    return declared


def synthesize_series_members(
    structure_records,
    page_contexts,
    audit_path=None,
    max_members_per_series=64,
):
    """Synthesize verified member structures for series-range structure rows.

    Returns (member_rows, report). member_rows are structures.csv rows;
    report records per-series decisions for the output evidence file.
    """
    from utils.llm_utils import extract_series_member_assignments

    page_contexts = [item for item in (page_contexts or []) if isinstance(item, dict)]
    series_rows = collect_series_rows(structure_records)
    known_ids = {str(row.get('COMPOUND_ID') or '') for row in series_rows}
    for declared in discover_text_declared_series(page_contexts):
        if str(declared.get('COMPOUND_ID') or '') not in known_ids:
            series_rows.append(declared)
    if not series_rows or not page_contexts:
        return [], {'series': [], 'reason': 'no series rows or no page text'}

    series_inputs = [
        {
            'series_id': row.get('COMPOUND_ID', ''),
            'member_ids': row['series_members'][:max_members_per_series],
            'smiles': str(row.get('SMILES') or ''),
            'structure_type': str(row.get('STRUCTURE_TYPE') or ''),
            'declared_in_text': bool(row.get('declared_in_text')),
        }
        for row in series_rows
    ]
    assignments = extract_series_member_assignments(
        series_inputs,
        page_contexts,
        audit_path=audit_path,
        metadata={'scope': 'series_member_synthesis'},
    )
    members_by_series = {
        str(entry.get('series_id') or ''): list(entry.get('members') or [])
        for entry in (assignments.get('series') or [])
    }

    member_rows = []
    report = {'series': []}
    for series_row in series_rows:
        series_id = str(series_row.get('COMPOUND_ID') or '')
        members = members_by_series.get(series_id) or []
        series_report = {
            'series_id': series_id,
            'defined_members': len(members),
            'methods': {},
            'emitted': 0,
            'consistency': '',
        }
        if not members:
            series_report['consistency'] = 'no member definitions found in text'
            report['series'].append(series_report)
            continue

        name_members = [m for m in members if m.get('full_name')]
        candidate_mols = {}
        method_notes = {}
        lookup_evidence = {}
        for member in name_members:
            if not substituent_consistent_with_name(member.get('substituent_text'), member.get('full_name')):
                method_notes[member['compound_id']] = 'substituent text contradicts stated name'
                continue
            mol, reason = name_to_molecule(
                member['full_name'],
                audit_path=audit_path,
                series_id=series_id,
            )
            if mol is None:
                method_notes[member['compound_id']] = reason or 'name did not resolve to a molecule'
                continue
            try:
                AllChem.Compute2DCoords(mol)
            except Exception:
                continue
            source = mol.GetProp('_LookupSource') if mol.HasProp('_LookupSource') else ''
            method = 'pubchem_lookup' if source == 'pubchem' else 'opsin_parse'
            if mol.HasProp('_LookupCID'):
                lookup_evidence[member['compound_id']] = f"PubChem CID {mol.GetProp('_LookupCID')}"
            candidate_mols[member['compound_id']] = (member, mol, method)

        substituent_members = [
            m for m in members
            if m.get('substituent_text') and series_row.get('MOLBLOCK')
            and molblock_dummy_atoms(series_row['MOLBLOCK'])
        ]
        for member in substituent_members:
            compound_id = member['compound_id']
            fragment = substituent_text_to_attachment_fragment(
                member['substituent_text'],
                audit_path=audit_path,
                series_id=series_id,
            )
            if fragment is None:
                if compound_id not in candidate_mols:
                    method_notes[compound_id] = 'substituent text did not resolve to a fragment'
                continue
            assembled, note = assemble_member_on_scaffold(
                series_row['MOLBLOCK'],
                Chem.MolToSmiles(fragment),
                variable_position=member.get('variable_position', ''),
            )
            if assembled is None:
                if compound_id not in candidate_mols:
                    method_notes[compound_id] = note
                continue
            existing = candidate_mols.get(compound_id)
            if existing is not None:
                # Both evidence paths produced a structure: they must agree.
                if not structures_agree(existing[1], assembled):
                    del candidate_mols[compound_id]
                    method_notes[compound_id] = (
                        'name-derived and substituent-assembled structures disagree'
                    )
                else:
                    lookup_evidence[compound_id] = (
                        (lookup_evidence.get(compound_id, '') + '; ' if lookup_evidence.get(compound_id) else '')
                        + 'substituent assembly agrees'
                    )
                continue
            candidate_mols[compound_id] = (member, assembled, 'substituent_assembly')

        kept_mols, quarantined = quarantine_core_outliers(candidate_mols)
        for compound_id, reason in quarantined.items():
            method_notes[compound_id] = reason
        candidate_mols = kept_mols
        series_report['quarantined'] = len(quarantined)

        mols = [item[1] for item in candidate_mols.values()]
        core, core_reason = member_set_common_core(mols)
        if core is None:
            series_report['consistency'] = core_reason
            report['series'].append(series_report)
            continue
        series_report['consistency'] = 'common core verified across synthesized members'

        for compound_id, (member, mol, method) in candidate_mols.items():
            record = build_member_record(
                series_row,
                member,
                mol,
                method,
                method_notes.get(compound_id, ''),
                lookup_evidence=lookup_evidence.get(compound_id, ''),
            )
            member_rows.append(record)
            series_report['methods'][method] = series_report['methods'].get(method, 0) + 1
        series_report['emitted'] = len(candidate_mols)
        report['series'].append(series_report)
    return member_rows, report
