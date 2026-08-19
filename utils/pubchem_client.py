"""PubChem compound-name lookup through the PUG REST API.

Returns the authoritative record (CID + SMILES) for an English name, or None.
"""

import json
import time
import urllib.parse
import urllib.request

try:
    import constants
except ImportError:
    constants = None

DEFAULT_BASE_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_RETRIES = 2

_lookup_cache: dict[str, dict | None] = {}


def _setting(name, default):
    if constants is None:
        return default
    return getattr(constants, name, default)


def _property_url(base_url, name):
    encoded = urllib.parse.quote(str(name or '').strip(), safe='')
    return f"{base_url.rstrip('/')}/compound/name/{encoded}/property/IsomericSMILES,CanonicalSMILES/JSON"


def _fetch_json(url, timeout_seconds):
    request = urllib.request.Request(url, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode('utf-8'))


def pubchem_lookup_name(name, timeout_seconds=None, retries=None):
    """Look up a compound by English name.

    Returns {'cid': int, 'isomeric_smiles': str, 'canonical_smiles': str} on a
    hit and None on any miss (unknown name, timeout, or service error after
    bounded retries).
    """
    key = ' '.join(str(name or '').split())
    if not key:
        return None
    if key in _lookup_cache:
        return _lookup_cache[key]
    base_url = str(_setting('PUBCHEM_REST_BASE_URL', DEFAULT_BASE_URL) or DEFAULT_BASE_URL)
    timeout = float(_setting('PUBCHEM_TIMEOUT_SECONDS', DEFAULT_TIMEOUT_SECONDS) or DEFAULT_TIMEOUT_SECONDS)
    if timeout_seconds is not None:
        timeout = float(timeout_seconds)
    attempts = max(1, int(retries if retries is not None else _setting('PUBCHEM_RETRIES', DEFAULT_RETRIES) or DEFAULT_RETRIES))
    url = _property_url(base_url, key)
    for attempt in range(1, attempts + 1):
        try:
            payload = _fetch_json(url, timeout)
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                _lookup_cache[key] = None
                return None
            if exc.code in (400, 501):
                _lookup_cache[key] = None
                return None
            if attempt >= attempts:
                _lookup_cache[key] = None
                return None
            time.sleep(1.0 * attempt)
        except Exception:
            if attempt >= attempts:
                _lookup_cache[key] = None
                return None
            time.sleep(1.0 * attempt)
    properties = ((payload or {}).get('PropertyTable') or {}).get('Properties') or []
    if not properties:
        _lookup_cache[key] = None
        return None
    record = properties[0]
    cid = record.get('CID')
    smiles = str(record.get('IsomericSMILES') or record.get('SMILES') or '').strip()
    canonical = str(record.get('CanonicalSMILES') or record.get('ConnectivitySMILES') or '').strip()
    if not cid or not smiles:
        _lookup_cache[key] = None
        return None
    result = {
        'cid': int(cid),
        'isomeric_smiles': smiles,
        'canonical_smiles': canonical or smiles,
        'queried_name': key,
    }
    _lookup_cache[key] = result
    return result
