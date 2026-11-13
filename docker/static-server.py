#!/usr/bin/env python
"""SPA/static server with precompressed assets and a streaming API proxy."""
import argparse
import http.client
import mimetypes
import os
import re
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

API_UPSTREAM_HOST = os.getenv('API_UPSTREAM_HOST', '127.0.0.1')
API_UPSTREAM_PORT = int(os.getenv('API_UPSTREAM_PORT', '8000'))
_HOP_HEADERS = {'connection', 'transfer-encoding', 'keep-alive', 'proxy-authenticate',
                'proxy-authorization', 'te', 'trailer', 'upgrade'}


class SpaRequestHandler(SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, '.mjs': 'text/javascript', '.wasm': 'application/wasm'}

    def send_response(self, code, message=None):
        self._response_status = code
        super().send_response(code, message)

    def _is_api_request(self):
        path = urllib.parse.urlsplit(self.path).path
        return path == '/api' or path.startswith('/api/')

    def guess_type(self, path):
        return getattr(self, '_original_content_type', None) or super().guess_type(path)

    def end_headers(self):
        if not self._is_api_request():
            self.send_header('Vary', 'Accept-Encoding')
            original = getattr(self, '_static_path', self.path)
            if getattr(self, '_response_status', 0) not in (200, 304):
                cache = 'no-store'
            elif original.endswith('.html') or '.nocache.' in original:
                cache = 'no-cache'
            elif re.search(r'-[A-Za-z0-9_-]{8,}\.(?:js|mjs|css|svg|png|woff2?)$|/[0-9A-Fa-f]{32}(?:/|\.cache\.)', original):
                cache = 'public, max-age=31536000, immutable'
            else:
                cache = 'public, max-age=3600'
            self.send_header('Cache-Control', cache)
            encoding = getattr(self, '_content_encoding', None)
            if encoding:
                self.send_header('Content-Encoding', encoding)
        super().end_headers()

    def send_head(self):
        self._original_content_type = None
        self._content_encoding = None
        parsed = urllib.parse.urlsplit(self.path)
        path = self.translate_path(self.path)
        if not os.path.exists(path):
            if parsed.path.startswith(('/assets/', '/jsme/')):
                self.send_error(404, 'Asset not found')
                return None
            self.path = '/index.html'
            parsed = urllib.parse.urlsplit(self.path)
            path = self.translate_path(self.path)
        self._static_path = parsed.path
        if not os.path.isfile(path):
            return super().send_head()

        accepted = {}
        for item in self.headers.get('Accept-Encoding', '').lower().split(','):
            parts = [part.strip() for part in item.split(';')]
            quality = 1.0
            for part in parts[1:]:
                if part.startswith('q='):
                    try:
                        quality = float(part[2:])
                    except ValueError:
                        quality = 0
            accepted[parts[0]] = quality
        options = [('br', '.br'), ('gzip', '.gz')]
        options.sort(key=lambda item: accepted.get(item[0], accepted.get('*', 0)), reverse=True)
        for encoding, suffix in options:
            if accepted.get(encoding, accepted.get('*', 0)) <= 0 or not os.path.isfile(path + suffix):
                continue
            self._original_content_type = mimetypes.guess_type(path)[0] or 'application/octet-stream'
            self._content_encoding = encoding
            original = self.path
            self.path = urllib.parse.urlunsplit(parsed._replace(path=parsed.path + suffix))
            try:
                return super().send_head()
            finally:
                self.path = original
        return super().send_head()

    def _proxy_api(self, method):
        # Keep only a chunk of the upload/download in memory. In particular,
        # send PDF headers immediately so PDF.js can cancel a full GET and
        # request the page's byte ranges without downloading the entire file.
        length = int(self.headers.get('Content-Length') or 0)
        headers = {key: value for key, value in self.headers.items()
                   if key.lower() not in _HOP_HEADERS | {'host', 'content-length'}}
        if length:
            headers['Content-Length'] = str(length)

        def upload():
            remaining = length
            while remaining:
                block = self.rfile.read(min(64 * 1024, remaining))
                if not block:
                    raise ConnectionError('Upload interrupted')
                remaining -= len(block)
                yield block

        upstream = http.client.HTTPConnection(API_UPSTREAM_HOST, API_UPSTREAM_PORT, timeout=600)
        sent_headers = False
        try:
            upstream.request(method, self.path, body=upload() if length else None, headers=headers)
            response = upstream.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in _HOP_HEADERS:
                    self.send_header(key, value)
            if response.getheader('Content-Length') is None:
                self.send_header('Connection', 'close')
                self.close_connection = True
            self.end_headers()
            sent_headers = True
            if method != 'HEAD':
                while True:
                    block = response.read(64 * 1024)
                    if not block:
                        break
                    self.wfile.write(block)
        except (BrokenPipeError, ConnectionResetError):
            # A canceled PDF range or abandoned download is not a server error.
            self.close_connection = True
        except (OSError, http.client.HTTPException):
            self.close_connection = True
            if not sent_headers:
                self.send_error(502, 'API service unavailable')
        finally:
            upstream.close()

    def do_GET(self):
        if self._is_api_request():
            return self._proxy_api('GET')
        super().do_GET()

    def do_HEAD(self):
        if self._is_api_request():
            return self._proxy_api('HEAD')
        super().do_HEAD()

    def _write_method(self, method):
        if self._is_api_request():
            return self._proxy_api(method)
        self.send_error(405)

    def do_POST(self): self._write_method('POST')
    def do_PUT(self): self._write_method('PUT')
    def do_DELETE(self): self._write_method('DELETE')
    def do_PATCH(self): self._write_method('PATCH')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=3000)
    args = parser.parse_args()
    root = str(Path(args.directory).resolve())
    handler = lambda *a, **kw: SpaRequestHandler(*a, directory=root, **kw)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f'Serving {root} at http://{args.host}:{args.port} (/api -> {API_UPSTREAM_HOST}:{API_UPSTREAM_PORT})', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
