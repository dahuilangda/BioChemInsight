#!/usr/bin/env python
import argparse
import http.client
import os
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

API_UPSTREAM_HOST = os.getenv("API_UPSTREAM_HOST", "127.0.0.1")
API_UPSTREAM_PORT = int(os.getenv("API_UPSTREAM_PORT", "8000"))


class SpaRequestHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path) or os.path.exists(path):
            return super().send_head()

        self.path = "/index.html"
        return super().send_head()

    def _is_api_request(self) -> bool:
        return urllib.parse.urlparse(self.path).path.startswith("/api/")

    def _proxy_api(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        try:
            upstream = http.client.HTTPConnection(API_UPSTREAM_HOST, API_UPSTREAM_PORT, timeout=600)
            upstream.request(
                method,
                self.path,
                body=body,
                headers={
                    key: value
                    for key, value in self.headers.items()
                    if key.lower() not in {"host", "connection", "content-length"}
                },
            )
            response = upstream.getresponse()
            payload = response.read()
        except OSError as exc:
            self.send_error(502, f"API upstream unreachable: {exc}")
            return
        self.send_response(response.status, response.reason)
        for key, value in response.getheaders():
            if key.lower() in {"connection", "transfer-encoding", "content-length"}:
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if method != "HEAD":
            self.wfile.write(payload)

    def do_GET(self):
        if self._is_api_request():
            self._proxy_api("GET")
            return
        super().do_GET()

    def do_HEAD(self):
        if self._is_api_request():
            self._proxy_api("HEAD")
            return
        super().do_HEAD()

    def do_POST(self):
        if self._is_api_request():
            self._proxy_api("POST")
            return
        self.send_error(405)

    def do_PUT(self):
        if self._is_api_request():
            self._proxy_api("PUT")
            return
        self.send_error(405)

    def do_DELETE(self):
        if self._is_api_request():
            self._proxy_api("DELETE")
            return
        self.send_error(405)

    def do_PATCH(self):
        if self._is_api_request():
            self._proxy_api("PATCH")
            return
        self.send_error(405)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()

    handler = lambda *handler_args, **handler_kwargs: SpaRequestHandler(
        *handler_args,
        directory=args.directory,
        **handler_kwargs,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {args.directory} at http://{args.host}:{args.port} (/api -> {API_UPSTREAM_HOST}:{API_UPSTREAM_PORT})")
    server.serve_forever()


if __name__ == "__main__":
    main()
