"""Series-range compound identifiers, expanded only to members named in text."""

import re

from utils.compound_id_utils import parse_compound_id_parts

# Letter-suffix ranges: 6a-r, IIIa-j, A1a-d. Base: optional letter, digits,
# or roman numeral; the variable is one lowercase letter spanning start..end.
_LETTER_SUFFIX_RANGE_RE = re.compile(
    r'^(?P<base>[A-Za-z]?\d+|[IVXLCM]+)(?P<start>[a-z])\s*[-–—~]\s*(?P<end>[a-z])$'
)
_NUMERIC_RANGE_RE = re.compile(r'^(?P<start>\d+)\s*[-–—~]\s*(?P<end>\d+)$')

_MAX_LETTER_SPAN = 26
_MAX_NUMERIC_SPAN = 500

_MEMBER_BOUNDARY_TEMPLATE = r'(?<![0-9A-Za-z]){token}(?![0-9A-Za-z])'


def _enumerate_series_members(series):
    kind = series['kind']
    if kind == 'letter_suffix':
        start = ord(series['start'])
        end = ord(series['end'])
        if not (ord('a') <= start <= ord('z') and ord('a') <= end <= ord('z')):
            return []
        if end <= start or end - start >= _MAX_LETTER_SPAN:
            return []
        return [f"{series['base']}{chr(code)}" for code in range(start, end + 1)]
    if kind == 'numeric':
        start = int(series['start'])
        end = int(series['end'])
        if end <= start or end - start >= _MAX_NUMERIC_SPAN:
            return []
        return [str(value) for value in range(start, end + 1)]
    return []


def parse_series_range_id(raw_value):
    """Parse a range-form identifier into its series grammar.

    Returns a dict with the base token, member kind, and enumerated member
    identifiers, or None when the value is not a range-form identifier.
    """
    parts = parse_compound_id_parts(raw_value)
    core = str(parts.get('core') or '').strip() if parts else str(raw_value or '').strip()
    if not core:
        return None
    match = _LETTER_SUFFIX_RANGE_RE.match(core)
    if match:
        series = {
            'kind': 'letter_suffix',
            'base': match.group('base'),
            'start': match.group('start'),
            'end': match.group('end'),
        }
    else:
        match = _NUMERIC_RANGE_RE.match(core)
        if not match:
            return None
        series = {
            'kind': 'numeric',
            'base': '',
            'start': match.group('start'),
            'end': match.group('end'),
        }
    members = _enumerate_series_members(series)
    if not members:
        return None
    series['members'] = members
    return series


def discover_series_members_in_text(series_id, text):
    """Return the series members that the given text names, in series order."""
    series = parse_series_range_id(series_id)
    if not series:
        return []
    page_text = str(text or '')
    if not page_text.strip():
        return []
    found = []
    for member in series['members']:
        pattern = re.compile(_MEMBER_BOUNDARY_TEMPLATE.format(token=re.escape(member)))
        if pattern.search(page_text):
            found.append(member)
    return found


def expand_official_ids_with_series_members(official_ids, text):
    """Expand range-form official identifiers with members named in the text.

    Returns (ids with appended members, series id -> discovered members).
    """
    ids = []
    seen = set()
    expansion = {}
    for raw_id in official_ids or []:
        identifier = str(raw_id or '').strip()
        if not identifier:
            continue
        if identifier not in seen:
            seen.add(identifier)
            ids.append(identifier)
        members = discover_series_members_in_text(identifier, text)
        if not members:
            continue
        expansion[identifier] = members
        for member in members:
            if member not in seen:
                seen.add(member)
                ids.append(member)
    return ids, expansion


def split_series_member_keys(keys, official_ids):
    """Split keys into standalone keys and keys that are members of an official series.

    A member that is itself an official identifier stays standalone.
    """
    official = {str(item or '').strip() for item in official_ids or []}
    member_keys = set()
    for official_id in official:
        series = parse_series_range_id(official_id)
        if not series:
            continue
        member_keys.update(series['members'])
    member_keys -= official
    standalone = []
    members = []
    for raw_key in keys or []:
        key = str(raw_key or '').strip()
        if key in member_keys:
            members.append(key)
        else:
            standalone.append(key)
    return standalone, members

