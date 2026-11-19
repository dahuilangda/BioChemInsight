from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

LOG = logging.getLogger("vision_proxy")
UPSTREAM_URL = os.environ.get(
    "VISION_UPSTREAM_URL", "http://219.146.211.42:29568/v1"
).rstrip("/")
UPSTREAM_TIMEOUT_S = float(os.environ.get("VISION_UPSTREAM_TIMEOUT_S", "300"))
RETRY_AFTER_S = max(1, math.ceil(float(os.environ.get("VISION_RETRY_AFTER_S", "15"))))
MAX_CONCURRENT_REQUESTS = max(1, int(os.environ.get("VISION_MAX_CONCURRENT_REQUESTS", "1")))

@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient() as client:
        app.state.upstream_client = client
        yield


app = FastAPI(title="vision-proxy", version="0.1.0", lifespan=lifespan)
_active_requests = 0


def _error(status, kind, message, headers=None):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": kind, "code": status}},
        headers=headers,
    )


async def _forward(request, path, timeout, body=None):
    headers = {
        name: request.headers[name]
        for name in ("authorization", "content-type")
        if name in request.headers
    }
    try:
        async with asyncio.timeout(timeout):
            upstream = await request.app.state.upstream_client.request(
                request.method, f"{UPSTREAM_URL}/{path}",
                headers=headers, content=body, timeout=timeout,
            )
    except (TimeoutError, httpx.TimeoutException):
        return _error(504, "upstream_timeout", "upstream vision model timeout")
    except httpx.RequestError as exc:
        LOG.warning("Upstream request failed: %s", type(exc).__name__)
        return _error(502, "upstream_error", "upstream vision model unavailable")

    response_headers = {
        name: upstream.headers[name]
        for name in ("content-type", "retry-after", "www-authenticate")
        if name in upstream.headers
    }
    return Response(upstream.content, status_code=upstream.status_code, headers=response_headers)


@app.get("/healthz")
async def healthz():
    return {
        "in_flight": _active_requests > 0,
        "active_requests": _active_requests,
        "capacity": MAX_CONCURRENT_REQUESTS,
        "upstream": UPSTREAM_URL,
    }


@app.get("/v1/models")
async def models(request: Request):
    return await _forward(request, "models", 30)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global _active_requests

    body = await request.body()
    if _active_requests >= MAX_CONCURRENT_REQUESTS:
        return _error(
            429, "busy", "vision model busy; retry",
            headers={"Retry-After": str(RETRY_AFTER_S)},
        )

    _active_requests += 1
    started = time.monotonic()
    try:
        response = await _forward(request, "chat/completions", UPSTREAM_TIMEOUT_S, body)
        LOG.info("Chat request: %d bytes, HTTP %d, %.1fs",
                 len(body), response.status_code, time.monotonic() - started)
        return response
    finally:
        _active_requests -= 1
