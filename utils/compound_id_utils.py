import ast
import json
import re
from collections import OrderedDict, defaultdict


COMPOUND_ID_PREFIXES = (
    'Example',
    'Ex.',
    'Ex',
    'No.',
    'No',
    'Compound',
    'Embodiment',
    'Intermediate',
    'Int.',
    'Preparation',
    'Formula',
    '实施例',
    '化合物',
    '编号',
)
TARGET_COMPOUND_ID_PREFIXES = (
    'Example',
    'Ex.',
    'Ex',
    'No.',
    'No',
    'Compound',
    'Formula',
    '实施例',
    '化合物',
    '编号',
)
NON_TARGET_COMPOUND_ID_PREFIXES = (
    'Embodiment',
    'Intermediate',
    'Int.',
    'Preparation',
)

_PREFIX_PATTERN = '|'.join(re.escape(prefix) for prefix in COMPOUND_ID_PREFIXES)
_COMPOUND_ID_SUFFIX_PATTERN = r'(?:[-–—][A-Za-z0-9]+)?'
_COMPOUND_ID_CORE_PATTERN = rf'(?:[A-Za-z]?\d+[A-Za-z]?{_COMPOUND_ID_SUFFIX_PATTERN}|[IVXLCM]+[A-Za-z]?{_COMPOUND_ID_SUFFIX_PATTERN})'
_KEYWORD_ID_PATTERN = re.compile(
    rf'\b(?:{_PREFIX_PATTERN})\s*({_COMPOUND_ID_CORE_PATTERN})\b',
    flags=re.IGNORECASE,
)
_SHORT_ID_PATTERN = re.compile(rf'^\(?\s*({_COMPOUND_ID_CORE_PATTERN})\s*\)?$', flags=re.IGNORECASE)
_CONCISE_IDENTIFIER_LABEL_PATTERN = re.compile(
    r'^[A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff\s.\-–—_/()]*$',
    flags=re.IGNORECASE,
)
_PAREN_PREFIX_IDENTIFIER_LABEL_PATTERN = re.compile(
    r'^\([A-Za-z0-9,+\-]+\)\s*[-–—]?\s*[A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff\s.\-–—_/()]*$',
    flags=re.IGNORECASE,
)
_RESOLUTION_STATE_CACHE = OrderedDict()
_RESOLUTION_STATE_CACHE_MAX_ENTRIES = 8
_LLM_ALIAS_CACHE = OrderedDict()
_LLM_ALIAS_CACHE_MAX_ENTRIES = 512


def _looks_like_concise_identifier_label(value):
    """Return True for a complete local/row label that should not be truncated."""
    compact_value = re.sub(r'\s+', ' ', str(value or '')).strip()
    if not compact_value or len(compact_value) > 64:
        return False
    if any(char in compact_value for char in '\n\r\t:;,{}[]'):
        return False
    if len(compact_value.split()) > 4:
        return False
    if not (
        _CONCISE_IDENTIFIER_LABEL_PATTERN.fullmatch(compact_value)
        or _PAREN_PREFIX_IDENTIFIER_LABEL_PATTERN.fullmatch(compact_value)
    ):
        return False
    if not re.search(r'[0-9IVXLCM]', compact_value, flags=re.IGNORECASE):
        return False
    prose_terms = {
        'answer',
        'returned',
        'model',
        'label',
        'identifier',
        'id is',
        'was',
    }
    lowered = compact_value.lower()
    return not any(term in lowered for term in prose_terms)


def normalize_compound_id_text(raw_value):
    if raw_value is None:
        return ''
    value = str(raw_value).strip()
    if not value:
        return ''

    extracted = _extract_embedded_compound_id_text(value)
    if extracted:
        value = extracted

    value = re.sub(r'\s+', ' ', value).strip()
    return value.strip(' ,;:[]{}')


def _extract_embedded_compound_id_text(raw_text):
    value = str(raw_text or '').strip()
    if not value:
        return ''

    if '```' in value:
        for part in value.split('```'):
            stripped_part = part.strip()
            if not stripped_part:
                continue
            if stripped_part.startswith('{') and stripped_part.endswith('}'):
                value = stripped_part
                break
            if '\n' in stripped_part:
                maybe_payload = stripped_part.split('\n', 1)[1].strip()
                if maybe_payload.startswith('{') and maybe_payload.endswith('}'):
                    value = maybe_payload
                    break

    for loader in (json.loads, ast.literal_eval):
        try:
            payload = loader(value)
        except Exception:
            continue

        extracted = _extract_compound_id_from_payload(payload)
        if extracted:
            return extracted

    compact_value = re.sub(r'\s+', ' ', value).strip()
    keyword_matches = re.findall(
        rf'\b(?:{_PREFIX_PATTERN})\s*{_COMPOUND_ID_CORE_PATTERN}\b',
        compact_value,
        flags=re.I,
    )
    if keyword_matches:
        return keyword_matches[-1].strip()

    if _looks_like_concise_identifier_label(compact_value):
        return compact_value

    short_matches = re.findall(rf'\(?\s*{_COMPOUND_ID_CORE_PATTERN}\s*\)?', compact_value, flags=re.I)
    if len(short_matches) == 1:
        candidate = short_matches[0].strip()
        alnum_len = len(re.sub(r'[^0-9A-Za-z\u4e00-\u9fff]', '', candidate))
        if alnum_len <= 6:
            return candidate

    return ''


def _extract_compound_id_from_payload(payload):
    if isinstance(payload, dict):
        for key in ('COMPOUND_ID', 'compound_id', 'VALUE', 'value', 'ID', 'id', 'answer', 'Answer'):
            if key not in payload:
                continue
            extracted = normalize_compound_id_text(payload.get(key))
            if extracted and extracted.lower() not in {'none', 'null'}:
                return extracted

        for candidate in payload.values():
            extracted = normalize_compound_id_text(candidate)
            if extracted and extracted.lower() not in {'none', 'null'}:
                return extracted

    if isinstance(payload, (list, tuple)):
        for candidate in payload:
            extracted = normalize_compound_id_text(candidate)
            if extracted and extracted.lower() not in {'none', 'null'}:
                return extracted

    if isinstance(payload, str):
        extracted = _extract_embedded_compound_id_text(payload)
        if extracted:
            return extracted

    return ''


def canonicalize_alias_token(raw_value):
    value = normalize_compound_id_text(raw_value)
    if not value:
        return ''
    return re.sub(r'[^0-9a-z\u4e00-\u9fff]+', '', value.lower())


def parse_compound_id_parts(raw_value):
    value = normalize_compound_id_text(raw_value)
    if not value:
        return None

    keyword_match = re.search(
        rf'((?:{_PREFIX_PATTERN}))\s*({_COMPOUND_ID_CORE_PATTERN})',
        value,
        flags=re.IGNORECASE,
    )
    if keyword_match:
        prefix = keyword_match.group(1)
        core = keyword_match.group(2)
        return {
            'text': value,
            'prefix': prefix,
            'core': core,
            'has_keyword': True,
        }

    short_match = _SHORT_ID_PATTERN.fullmatch(value)
    if short_match:
        return {
            'text': value,
            'prefix': '',
            'core': short_match.group(1),
            'has_keyword': False,
        }

    if _looks_like_concise_identifier_label(value):
        return None

    tail_match = re.search(rf'(\(?\s*(?:{_COMPOUND_ID_CORE_PATTERN})\s*\)?)\s*$', value, flags=re.IGNORECASE)
    if tail_match:
        candidate = normalize_compound_id_text(tail_match.group(1))
        short_tail = _SHORT_ID_PATTERN.fullmatch(candidate)
        if short_tail:
            return {
                'text': value,
                'prefix': '',
                'core': short_tail.group(1),
                'has_keyword': False,
            }
    return None


def build_compound_id_alias_map(compound_id_list):
    official_ids = []
    for compound_id in compound_id_list or []:
        normalized = normalize_compound_id_text(compound_id)
        if normalized and normalized not in official_ids:
            official_ids.append(normalized)

    parsed_by_official = {}
    core_to_officials = defaultdict(list)
    alias_candidates = defaultdict(set)

    for official_id in official_ids:
        parts = parse_compound_id_parts(official_id)
        parsed_by_official[official_id] = parts
        alias_candidates[canonicalize_alias_token(official_id)].add(official_id)
        if not parts:
            continue
        core = normalize_compound_id_text(parts['core'])
        core_to_officials[canonicalize_alias_token(core)].append(official_id)
        if parts.get('has_keyword'):
            alias_candidates[canonicalize_alias_token(f"{parts['prefix']} {core}")].add(official_id)

    for core_key, matched_officials in core_to_officials.items():
        if len(set(matched_officials)) != 1:
            continue
        official_id = matched_officials[0]
        parts = parsed_by_official.get(official_id)
        if not parts:
            continue
        core = normalize_compound_id_text(parts['core'])
        alias_variants = {
            core,
            f'({core})',
            f'No.{core}',
            f'No. {core}',
            f'Ex.{core}',
            f'Ex. {core}',
            f'编号{core}',
        }
        for prefix in TARGET_COMPOUND_ID_PREFIXES:
            alias_variants.add(f'{prefix} {core}')
        for alias in alias_variants:
            alias_candidates[canonicalize_alias_token(alias)].add(official_id)

    resolved_alias_map = {}
    ambiguous_aliases = set()
    for alias_key, matched_officials in alias_candidates.items():
        unique_matches = [item for item in official_ids if item in matched_officials]
        if len(unique_matches) == 1:
            resolved_alias_map[alias_key] = unique_matches[0]
        else:
            ambiguous_aliases.add(alias_key)

    return resolved_alias_map, ambiguous_aliases


def _normalize_official_ids(compound_id_list):
    official_ids = []
    for compound_id in compound_id_list or []:
        normalized = normalize_compound_id_text(compound_id)
        if normalized and normalized not in official_ids:
            official_ids.append(normalized)
    return official_ids


def _set_lru_cache_entry(cache, key, value, max_entries):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)


def get_compound_id_resolution_state(compound_id_list):
    official_ids = tuple(_normalize_official_ids(compound_id_list))
    cache_key = official_ids
    cached = _RESOLUTION_STATE_CACHE.get(cache_key)
    if cached is not None:
        _RESOLUTION_STATE_CACHE.move_to_end(cache_key)
        return cached

    alias_map, ambiguous_aliases = build_compound_id_alias_map(official_ids)
    preferred_map = build_preferred_compound_id_map(official_ids)
    core_keys = set()
    for official_id in official_ids:
        parts = parse_compound_id_parts(official_id)
        if not parts:
            continue
        core_key = canonicalize_alias_token(parts.get('core'))
        if core_key:
            core_keys.add(core_key)

    state = {
        'official_ids': official_ids,
        'alias_map': alias_map,
        'ambiguous_aliases': ambiguous_aliases,
        'preferred_map': preferred_map,
        'core_keys': core_keys,
    }
    _set_lru_cache_entry(_RESOLUTION_STATE_CACHE, cache_key, state, _RESOLUTION_STATE_CACHE_MAX_ENTRIES)
    return state


def build_preferred_compound_id_map(compound_id_list):
    official_ids = []
    for compound_id in compound_id_list or []:
        normalized = normalize_compound_id_text(compound_id)
        if normalized and normalized not in official_ids:
            official_ids.append(normalized)

    grouped = defaultdict(list)
    for official_id in official_ids:
        parts = parse_compound_id_parts(official_id)
        if not parts:
            continue
        core_key = canonicalize_alias_token(parts.get('core'))
        if core_key:
            grouped[core_key].append((official_id, parts))

    preferred_map = {}
    for _, entries in grouped.items():
        keyword_entries = [entry for entry in entries if entry[1].get('has_keyword')]
        bare_entries = [entry for entry in entries if not entry[1].get('has_keyword')]
        if len(keyword_entries) != 1:
            continue
        preferred_id = keyword_entries[0][0]
        for bare_id, _ in bare_entries:
            preferred_map[canonicalize_alias_token(bare_id)] = preferred_id
    return preferred_map


def resolve_compound_id_alias(raw_value, alias_map, ambiguous_aliases=None):
    alias_key = canonicalize_alias_token(raw_value)
    if not alias_key:
        return None
    if ambiguous_aliases and alias_key in ambiguous_aliases:
        return None
    return alias_map.get(alias_key)


def resolve_compound_id_with_trace(raw_value, compound_id_list, resolver_fn=None, context=''):
    normalized = normalize_compound_id_text(raw_value)
    if not normalized:
        return {
            'raw': raw_value,
            'canonical': '',
            'source': 'empty',
        }

    state = get_compound_id_resolution_state(compound_id_list)
    official_ids = state['official_ids']
    alias_map = state['alias_map']
    ambiguous_aliases = state['ambiguous_aliases']
    preferred_map = state['preferred_map']
    alias_key = canonicalize_alias_token(normalized)

    resolved = preferred_map.get(alias_key)
    if resolved:
        return {
            'raw': raw_value,
            'canonical': resolved,
            'source': 'deterministic_preferred',
        }

    resolved = resolve_compound_id_alias(normalized, alias_map, ambiguous_aliases=ambiguous_aliases)
    if resolved:
        source = 'exact_match' if normalize_compound_id_text(resolved) == normalized else 'deterministic_alias'
        return {
            'raw': raw_value,
            'canonical': resolved,
            'source': source,
        }

    candidate_parts = parse_compound_id_parts(normalized)
    if candidate_parts:
        candidate_core_key = canonicalize_alias_token(candidate_parts.get('core'))
        if candidate_core_key and candidate_core_key not in state['core_keys']:
            return {
                'raw': raw_value,
                'canonical': '',
                'source': 'unresolved',
            }

    if resolver_fn is not None:
        llm_cache_key = (alias_key, official_ids)
        cached_llm_resolution = _LLM_ALIAS_CACHE.get(llm_cache_key)
        if cached_llm_resolution is not None:
            _LLM_ALIAS_CACHE.move_to_end(llm_cache_key)
            resolved = cached_llm_resolution
        else:
            try:
                resolved = resolver_fn(normalized, official_ids, context=context)
            except Exception as exc:
                return {
                    'raw': raw_value,
                    'canonical': '',
                    'source': 'llm_error',
                    'error': str(exc),
                }
            if isinstance(resolved, str):
                resolved = normalize_compound_id_text(resolved)
            else:
                resolved = ''
            if resolved.lower() == 'none':
                resolved = ''
            _set_lru_cache_entry(_LLM_ALIAS_CACHE, llm_cache_key, resolved, _LLM_ALIAS_CACHE_MAX_ENTRIES)
        if isinstance(resolved, str):
            resolved = normalize_compound_id_text(resolved)
            if resolved and resolved in official_ids:
                return {
                    'raw': raw_value,
                    'canonical': resolved,
                    'source': 'llm_few_shot',
                }

    return {
        'raw': raw_value,
        'canonical': normalized if normalized in official_ids else '',
        'source': 'unresolved',
    }


def canonicalize_record_compound_ids(records, resolver_fn=None, context_builder=None, overwrite_compound_id=True):
    if not records:
        return records

    observed_ids = []
    for record in records:
        if not isinstance(record, dict):
            continue
        normalized = normalize_compound_id_text(record.get('COMPOUND_ID'))
        if normalized and normalized not in observed_ids:
            observed_ids.append(normalized)

    if not observed_ids:
        return records

    for record in records:
        if not isinstance(record, dict):
            continue
        raw_value = record.get('COMPOUND_ID')
        normalized = normalize_compound_id_text(raw_value)
        if not normalized:
            continue
        record.setdefault('RAW_COMPOUND_ID', normalized)
        context = ''
        if context_builder is not None:
            try:
                context = context_builder(record) or ''
            except Exception:
                context = ''
        trace = resolve_compound_id_with_trace(
            normalized,
            observed_ids,
            resolver_fn=resolver_fn,
            context=context,
        )
        canonical = trace.get('canonical') or normalized
        if overwrite_compound_id:
            record['COMPOUND_ID'] = canonical
        record['CANONICAL_COMPOUND_ID'] = canonical
        record['ALIAS_RESOLUTION_SOURCE'] = trace.get('source', '')
    return records


def remap_assay_dict_to_official_ids(assay_dict, compound_id_list, resolver_fn=None, context_by_key=None):
    if not assay_dict or not compound_id_list:
        return assay_dict or {}
    remapped = {}
    resolver_cache = {}
    for raw_key, value in (assay_dict or {}).items():
        cache_key = canonicalize_alias_token(raw_key)
        if cache_key in resolver_cache:
            resolved_key = resolver_cache[cache_key]
        else:
            context = ''
            if isinstance(context_by_key, dict):
                context = context_by_key.get(raw_key, '') or ''
            trace = resolve_compound_id_with_trace(
                raw_key,
                compound_id_list,
                resolver_fn=resolver_fn,
                context=context,
            )
            resolved_key = trace.get('canonical') or None
            if trace.get('source') == 'unresolved':
                resolved_key = None
            resolver_cache[cache_key] = resolved_key
        if resolved_key is None:
            continue
        if resolved_key not in remapped:
            remapped[resolved_key] = value
    return remapped
