"""Real HTTP checks for compression, browser caching and streaming PDF ranges."""
import functools
import gzip
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('biochem_static', ROOT / 'docker/static-server.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    (tmp_path / 'assets').mkdir()
    original = b'export const example = "reusable static asset";\n' * 300
    asset = tmp_path / 'assets/index-a1b2c3d4.js'
    asset.write_bytes(original)
    (tmp_path / 'index.html').write_text('<main>Workbench</main>')
    subprocess.run(['node', str(ROOT / 'frontend/ui/scripts/precompress-assets.js'), str(tmp_path)], check=True, capture_output=True)
    payload = bytes(range(256)) * 32768  # 8 MiB, large enough to detect buffering.
    upstream_finished = threading.Event()

    class Upstream(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()

        def do_GET(self):
            requested = self.headers.get('Range')
            if requested:
                first, last = map(int, requested.removeprefix('bytes=').split('-'))
                content = payload[first:last + 1]
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {first}-{last}/{len(payload)}')
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()
            try:
                for offset in range(0, len(payload), 65536):
                    self.wfile.write(payload[offset:offset + 65536])
                    self.wfile.flush()
                    time.sleep(.002)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                upstream_finished.set()

        def do_POST(self):
            remaining = int(self.headers['Content-Length'])
            digest = hashlib.sha256()
            while remaining:
                block = self.rfile.read(min(65536, remaining))
                digest.update(block)
                remaining -= len(block)
            result = json.dumps({'sha256': digest.hexdigest()}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(result)))
            self.end_headers()
            self.wfile.write(result)

        def log_message(self, *args): pass

    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    monkeypatch.setattr(module, 'API_UPSTREAM_HOST', '127.0.0.1')
    monkeypatch.setattr(module, 'API_UPSTREAM_PORT', upstream.server_port)

    class QuietHandler(module.SpaRequestHandler):
        def log_message(self, *args): pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(QuietHandler, directory=str(tmp_path)))
    workers = [threading.Thread(target=service.serve_forever, daemon=True) for service in (upstream, server)]
    for worker in workers: worker.start()
    try:
        yield server.server_port, original, payload, upstream_finished
    finally:
        for service in (server, upstream): service.shutdown(); service.server_close()
        for worker in workers: worker.join()


def request(port, method, path, headers=None, body=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    result = response.status, {key.title(): value for key, value in response.getheaders()}, response.read()
    connection.close()
    return result


def test_precompressed_asset_and_conditional_cache(delivery):
    port, original, _, _ = delivery
    status, headers, body = request(port, 'GET', '/assets/index-a1b2c3d4.js', {'Accept-Encoding': 'gzip'})
    assert status == 200
    assert headers['Content-Encoding'] == 'gzip'
    assert gzip.decompress(body) == original
    assert len(body) < len(original) // 3
    assert headers['Cache-Control'].endswith('immutable')
    assert headers['Vary'] == 'Accept-Encoding'
    assert 'javascript' in headers['Content-Type']
    status, repeated, body = request(port, 'GET', '/assets/index-a1b2c3d4.js', {
        'Accept-Encoding': 'gzip', 'If-Modified-Since': headers['Last-Modified'],
    })
    assert status == 304 and body == b''
    status, plain, body = request(port, 'GET', '/assets/index-a1b2c3d4.js', {'Accept-Encoding': 'gzip;q=0,br;q=0'})
    assert status == 200 and 'Content-Encoding' not in plain and body == original
    status, head, body = request(port, 'HEAD', '/assets/index-a1b2c3d4.js', {'Accept-Encoding': 'gzip'})
    assert status == 200 and body == b'' and int(head['Content-Length']) < len(original)


def test_spa_html_revalidates_and_missing_assets_are_not_html(delivery):
    port, _, _, _ = delivery
    status, headers, body = request(port, 'GET', '/workspace')
    assert status == 200 and b'Workbench' in body
    assert headers['Cache-Control'] == 'no-cache'
    status, headers, _ = request(port, 'GET', '/assets/missing.js')
    assert status == 404 and headers['Cache-Control'] == 'no-store'


def test_proxy_preserves_range_and_head(delivery):
    port, _, payload, _ = delivery
    status, headers, content = request(port, 'GET', '/api/pdfs/one/file', {'Range': 'bytes=123-456'})
    assert status == 206 and content == payload[123:457]
    assert headers['Content-Range'] == f'bytes 123-456/{len(payload)}'
    assert int(headers['Content-Length']) == 334
    status, headers, content = request(port, 'HEAD', '/api/pdfs/one/file')
    assert status == 200 and content == b''
    assert int(headers['Content-Length']) == len(payload)


def test_proxy_sends_pdf_headers_before_buffering_the_file(delivery):
    port, _, payload, finished = delivery
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    connection.request('GET', '/api/pdfs/one/file')
    response = connection.getresponse()
    assert int(response.getheader('Content-Length')) == len(payload)
    assert not finished.is_set(), 'headers must arrive while upstream is still streaming'
    assert response.read(1) == payload[:1]
    response.close(); connection.close()
    assert finished.wait(5), 'abandoned PDF request should close the upstream connection'


def test_proxy_streams_upload_without_changing_bytes(delivery):
    port, _, payload, _ = delivery
    body = payload[:2_000_000]
    status, _, content = request(port, 'POST', '/api/pdfs', {'Content-Type': 'application/octet-stream'}, body)
    assert status == 200
    assert json.loads(content)['sha256'] == hashlib.sha256(body).hexdigest()
