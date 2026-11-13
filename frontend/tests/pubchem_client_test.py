"""All quota, cache and retry checks use a fake transport; no PubChem load test."""
import io
import json
import tempfile
import unittest
import urllib.error
from urllib.parse import parse_qs
from utils.pubchem_client import PubChemClient, throttle_delay, retry_after_seconds


class Clock:
    def __init__(self): self.value = 1000.0
    def __call__(self): return self.value
    def sleep(self, seconds): self.value += seconds


class Response:
    status = 200
    def __init__(self, payload, headers=None): self.payload = payload; self.headers = headers or {}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, size=-1): return json.dumps(self.payload).encode()


def hit(cid=701):
    return {'PropertyTable': {'Properties': [{'CID': cid, 'SMILES': 'CCO', 'ConnectivitySMILES': 'CCO'}]}}


class PubChemClientTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(); self.addCleanup(self.folder.cleanup)
        self.clock = Clock(); self.calls = []
    def client(self, responses):
        def fetch(request, timeout):
            self.calls.append((self.clock(), request))
            result = responses.pop(0)
            if isinstance(result, Exception): raise result
            return result
        return PubChemClient(self.folder.name, opener=fetch, clock=self.clock, sleep=self.clock.sleep)
    def error(self, status, headers=None):
        return urllib.error.HTTPError('https://pubchem.ncbi.nlm.nih.gov/', status, 'simulated', headers or {}, io.BytesIO(b'<html>busy</html>'))

    def test_cache_is_shared_by_new_client_instances(self):
        self.assertEqual(self.client([Response(hit())]).lookup_details('name')['status'], 'found')
        result = self.client([]).lookup_details('name')
        self.assertTrue(result['cached']); self.assertEqual(len(self.calls), 1)

    def test_no_burst_and_special_names_are_posted_intact(self):
        client = self.client([Response(hit()), Response(hit()), Response(hit())])
        for name in ['first', '(R)-name/a+b', 'third']: client.lookup_details(name)
        self.assertTrue(all(b[0] - a[0] >= 1 for a,b in zip(self.calls, self.calls[1:])))
        self.assertEqual(parse_qs(self.calls[1][1].data.decode())['name'], ['(R)-name/a+b'])
        self.assertEqual(self.calls[1][1].method, 'POST')

    def test_multi_hit_name_is_ambiguous(self):
        payload = {'PropertyTable': {'Properties': hit(1)['PropertyTable']['Properties'] + hit(2)['PropertyTable']['Properties']}}
        result = self.client([Response(payload)]).lookup_details('ambiguous')
        self.assertEqual(result['status'], 'ambiguous'); self.assertIsNone(result['record'])

    def test_throttling_stops_new_names_across_clients(self):
        first = self.client([self.error(503, {'Retry-After': '120'})]).lookup_details('first')
        self.assertEqual(first['status'], 'cooldown')
        second = self.client([]).lookup_details('second')
        self.assertEqual(second['status'], 'cooldown'); self.assertEqual(len(self.calls), 1)
        self.assertGreaterEqual(second['retry_after'], 120)

    def test_error_is_not_cached_as_a_permanent_name_miss(self):
        result = self.client([TimeoutError()]).lookup_details('first', retries=1)
        self.assertEqual(result['status'], 'unavailable')
        self.clock.value += 61
        self.assertEqual(self.client([Response(hit())]).lookup_details('first')['status'], 'found')

    def test_worst_feedback_remains_after_a_green_response(self):
        client = self.client([Response(hit(), {'X-Throttling-Control': 'Request Count status: Green (1%), Service status: Red (80%)'}), Response(hit()), Response(hit())])
        for name in ['first', 'second', 'third']: client.lookup_details(name)
        self.assertGreaterEqual(self.calls[1][0] - self.calls[0][0], 5)
        self.assertGreaterEqual(self.calls[2][0] - self.calls[1][0], 5)

    def test_404_is_negative_cached_but_forbidden_is_not_retried(self):
        first = self.client([self.error(404)]).lookup_details('absent')
        self.assertEqual(first['status'], 'not_found')
        self.assertTrue(self.client([]).lookup_details('absent')['cached'])
        self.assertEqual(self.client([self.error(403)]).lookup_details('forbidden')['status'], 'rejected')
        self.assertEqual(self.client([]).lookup_details('other')['status'], 'cooldown')
        self.assertEqual(len(self.calls), 2)

    def test_parse_feedback(self):
        self.assertEqual(throttle_delay('Service status: Black (100%)'), 60)
        self.assertEqual(retry_after_seconds('Thu, 01 Jan 1970 00:20:00 GMT', now=1000), 200)


if __name__ == '__main__': unittest.main()
