from __future__ import annotations

import logging
import os
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

LOG = logging.getLogger("vision_proxy")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(message)s")

UPSTREAM_URL = os.environ.get(
    "VISION_UPSTREAM_URL", "http://219.146.211.42:29568/v1").rstrip("/")
UPSTREAM_TIMEOUT_S = float(os.environ.get("VISION_UPSTREAM_TIMEOUT_S", "300"))
RETRY_AFTER_S = float(os.environ.get("VISION_RETRY_AFTER_S", "15"))

app = FastAPI(title="vision-proxy", version="0.1.0")

_IN_FLIGHT = False
_LAST_SERVED = 0.0


@app.get("/healthz")
async def healthz():
    return {"in_flight": _IN_FLIGHT,
            "upstream": UPSTREAM_URL}


@app.get("/v1/models")
async def models():
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{UPSTREAM_URL}/models")
        return Response(content=resp.content, status_code=resp.status_code,
                        media_type=resp.headers.get("content-type", "application/json"))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global _IN_FLIGHT, _WAITING, _LAST_SERVED

    body = await request.body()
    headers = {"Content-Type": request.headers.get(
        "content-type", "application/json")}
    api_key = request.headers.get("authorization")
    if api_key:
        headers["Authorization"] = api_key

    if _IN_FLIGHT:
        wait = max(1, int(RETRY_AFTER_S - (time.monotonic() - _LAST_SERVED)))
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "vision model busy; retry",
                               "type": "busy", "code": 429}},
            headers={"Retry-After": str(wait)},
        )

    _IN_FLIGHT = True
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_S) as client:
            resp = await client.post(f"{UPSTREAM_URL}/chat/completions",
                                     content=body, headers=headers)
        elapsed = time.monotonic() - started
        LOG.info("chat %dB -> %d in %.1fs", len(body), resp.status_code,
                 elapsed)
        _LAST_SERVED = time.monotonic()
        return Response(content=resp.content, status_code=resp.status_code,
                        media_type=resp.headers.get("content-type", "application/json"))
    except httpx.TimeoutException:
        LOG.error("upstream timeout after %.1fs", time.monotonic() - started)
        return JSONResponse(status_code=504, content={
            "error": {"message": "upstream vision model timeout",
                      "type": "upstream_timeout", "code": 504}})
    except Exception as exc:
        LOG.error("upstream error: %s", type(exc).__name__)
        return JSONResponse(status_code=502, content={
            "error": {"message": f"upstream error: {type(exc).__name__}",
                      "type": "upstream_error", "code": 502}})
    finally:
        _IN_FLIGHT = False
