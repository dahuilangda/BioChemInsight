"""Conservative PUG-REST access shared by all local worker processes.

Official policy: https://pubchem.ncbi.nlm.nih.gov/pcfe/docs/markdown/pug-rest.md
Dynamic feedback: https://pubchem.ncbi.nlm.nih.gov/pcfe/docs/markdown/dynamic-request-throttling.md
One cross-process gate serializes requests and shares worst-case cooldowns.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

try:
    import constants
except ImportError:
    constants = None

DEFAULT_BASE_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_RETRIES = 2
_LOG = logging.getLogger(__name__)
_PROCESS_LOCK = threading.RLock()


def _setting(name, default):
    return os.environ.get(name, getattr(constants, name, default) if constants else default)


def _property_url(base_url, name=None):
    # POST carries the name intact, including slash/plus/stereochemical notation.
    return f"{base_url.rstrip('/')}/compound/name/property/SMILES,ConnectivitySMILES,InChIKey,MolecularFormula/JSON?name_type=complete"


def retry_after_seconds(value, now=None):
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(str(value))
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0.0, date.timestamp() - (time.time() if now is None else now))
        except (ValueError, TypeError, OverflowError):
            return 0.0


def throttle_delay(header):
    """Follow the worst status, including service load, rather than the last host."""
    statuses = set(re.findall(r'\b(Green|Yellow|Red|Black)\b', str(header or ''), re.I))
    levels = {value.lower() for value in statuses}
    return 60.0 if 'black' in levels else 5.0 if 'red' in levels else 2.0 if 'yellow' in levels else 0.0


@contextmanager
def _bounded_process_gate(timeout):
    if not _PROCESS_LOCK.acquire(timeout=timeout):
        raise TimeoutError('PubChem in-process queue busy')
    try:
        yield
    finally:
        _PROCESS_LOCK.release()


class PubChemClient:
    def __init__(self, state_dir=None, *, base_url=None, opener=None, clock=None, sleep=None):
        self.state_dir = Path(state_dir or _setting('PUBCHEM_STATE_DIR', Path(__file__).resolve().parents[1] / 'data' / 'pubchem'))
        self.base_url = str(base_url or _setting('PUBCHEM_REST_BASE_URL', DEFAULT_BASE_URL))
        self.opener = opener or urllib.request.urlopen
        self.clock = clock or time.time
        self.sleep = sleep or time.sleep
        # Keep ample headroom below the documented limit; serialize runtime too.
        self.interval = max(1.0, float(_setting('PUBCHEM_MIN_INTERVAL_SECONDS', 1.0)))
        self.max_wait = min(30.0, max(0.0, float(_setting('PUBCHEM_MAX_WAIT_SECONDS', 10))))

    @contextmanager
    def _gate(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.max_wait
        with _bounded_process_gate(self.max_wait), (self.state_dir / 'requests.lock').open('a+b') as lock:
            while True:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('PubChem request queue busy')
                    self.sleep(.05)
            try:
                with sqlite3.connect(self.state_dir / 'cache.sqlite3', timeout=10) as connection:
                    connection.execute('CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, expires REAL NOT NULL, payload TEXT NOT NULL)')
                    connection.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), next_request REAL NOT NULL, cooldown REAL NOT NULL)')
                    columns = {row[1] for row in connection.execute('PRAGMA table_info(state)')}
                    for name in ('adaptive_interval', 'slow_until'):
                        if name not in columns:
                            connection.execute(f'ALTER TABLE state ADD COLUMN {name} REAL NOT NULL DEFAULT 0')
                    connection.execute('INSERT OR IGNORE INTO state (id, next_request, cooldown) VALUES (1, 0, 0)')
                    yield connection
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _feedback(self, connection, headers, status):
        now = self.clock()
        delay = max(throttle_delay(headers.get('X-Throttling-Control')), retry_after_seconds(headers.get('Retry-After'), now))
        if status in {429, 503}:
            delay = max(delay, 30.0)
        if status == 403:
            delay = max(delay, 3600.0)
        if delay:
            adaptive = 5.0 if delay >= 5 else 2.0
            connection.execute('UPDATE state SET adaptive_interval=MAX(adaptive_interval, ?), slow_until=MAX(slow_until, ?) WHERE id=1', (adaptive, now + max(60, delay)))
            # Later Green responses cannot erase worse feedback from another server.
            connection.execute('UPDATE state SET cooldown=MAX(cooldown, ?) WHERE id=1', (now + delay,))
            connection.commit()
        return delay

    def _store(self, connection, key, result, ttl):
        connection.execute('INSERT OR REPLACE INTO cache VALUES (?, ?, ?)', (key, self.clock() + ttl, json.dumps(result)))
        connection.commit()
        return result

    def lookup_details(self, name, timeout_seconds=None, retries=None):
        query = ' '.join(str(name or '').split())
        if not query or len(query) > 4096:
            return {'status': 'invalid', 'record': None}
        # Preserve case: stereochemical r/R and s/S can carry different meanings.
        key = self.base_url + '\nproperties-v2\n' + query
        timeout = min(30.0, max(1.0, float(timeout_seconds or _setting('PUBCHEM_TIMEOUT_SECONDS', DEFAULT_TIMEOUT_SECONDS))))
        attempts = min(3, max(1, int(retries if retries is not None else _setting('PUBCHEM_RETRIES', DEFAULT_RETRIES))))
        try:
            with self._gate() as connection:
                cached = connection.execute('SELECT payload FROM cache WHERE key=? AND expires>?', (key, self.clock())).fetchone()
                if cached:
                    return dict(json.loads(cached[0]), cached=True)
                for attempt in range(attempts):
                    next_at, cooldown, adaptive, slow_until = connection.execute('SELECT next_request, cooldown, adaptive_interval, slow_until FROM state WHERE id=1').fetchone()
                    wait = max(next_at, cooldown) - self.clock()
                    if wait > self.max_wait:
                        return {'status': 'cooldown', 'record': None, 'retry_after': wait}
                    if wait > 0:
                        self.sleep(wait)
                    connection.execute('UPDATE state SET next_request=? WHERE id=1', (self.clock() + max(self.interval, adaptive if slow_until > self.clock() else 0),))
                    connection.commit()
                    headers = {'Accept': 'application/json', 'Content-Type': 'application/x-www-form-urlencoded',
                               'User-Agent': str(_setting('PUBCHEM_USER_AGENT', 'BioChemInsight/0.1 (PUG-REST name resolution; shared rate limit)'))}
                    request = urllib.request.Request(_property_url(self.base_url), data=urllib.parse.urlencode({'name': query}).encode(), headers=headers, method='POST')
                    try:
                        with self.opener(request, timeout=timeout) as response:
                            self._feedback(connection, response.headers, response.status)
                            if response.status != 200:
                                return {'status': 'unavailable', 'record': None, 'http_status': response.status}
                            # HTML 503 bodies never reach the JSON parser.
                            payload = json.loads(response.read(2_000_000).decode('utf-8'))
                    except urllib.error.HTTPError as exc:
                        delay = self._feedback(connection, exc.headers or {}, exc.code)
                        if exc.code == 404:
                            return self._store(connection, key, {'status': 'not_found', 'record': None}, 6 * 3600)
                        if exc.code not in {429, 500, 502, 503, 504}:
                            return {'status': 'rejected', 'record': None, 'http_status': exc.code}
                        if delay > self.max_wait or attempt + 1 == attempts:
                            return {'status': 'cooldown' if delay else 'unavailable', 'record': None, 'http_status': exc.code, 'retry_after': delay}
                        self.sleep(max(delay, 2 ** attempt) + random.uniform(0, .5))
                        continue
                    except (OSError, ValueError, TimeoutError) as exc:
                        if attempt + 1 == attempts:
                            self._feedback(connection, {}, 503)
                            return {'status': 'unavailable', 'record': None, 'error': type(exc).__name__}
                        self.sleep(2 ** attempt + random.uniform(0, .5))
                        continue
                    properties = ((payload or {}).get('PropertyTable') or {}).get('Properties') or []
                    if not isinstance(properties, list) or not properties:
                        return {'status': 'invalid_response', 'record': None}
                    unique = {str(item.get('CID')): item for item in properties if isinstance(item, dict) and item.get('CID')}
                    if len(unique) != 1:
                        return self._store(connection, key, {'status': 'ambiguous', 'record': None, 'candidate_cids': list(unique)[:20]}, 3600)
                    record = next(iter(unique.values()))
                    smiles = str(record.get('SMILES') or record.get('IsomericSMILES') or '').strip()
                    if not smiles:
                        return {'status': 'invalid_response', 'record': None}
                    result = {'cid': int(record['CID']), 'isomeric_smiles': smiles,
                        'canonical_smiles': str(record.get('ConnectivitySMILES') or record.get('CanonicalSMILES') or smiles),
                        'inchikey': str(record.get('InChIKey') or ''), 'formula': str(record.get('MolecularFormula') or ''),
                        'queried_name': query, 'source': 'pubchem', 'retrieved_at': self.clock()}
                    return self._store(connection, key, {'status': 'found', 'record': result}, 30 * 86400)
        except (OSError, sqlite3.Error, TimeoutError) as exc:
            # Never bypass the shared limiter if the state directory is unavailable.
            _LOG.warning('PubChem client unavailable: %s', type(exc).__name__)
            return {'status': 'unavailable', 'record': None, 'error': type(exc).__name__}
        return {'status': 'unavailable', 'record': None}


def pubchem_lookup_name_details(name, timeout_seconds=None, retries=None):
    return PubChemClient().lookup_details(name, timeout_seconds, retries)


def pubchem_lookup_name(name, timeout_seconds=None, retries=None):
    """Compatibility API: only an unambiguous record is a hit.

    Transient errors are never persisted as a permanent chemical-name miss.
    """
    return pubchem_lookup_name_details(name, timeout_seconds, retries).get('record')
