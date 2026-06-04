from __future__ import annotations

import os
from typing import Callable

from fastapi import HTTPException, Request
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from src.observability import get_logger, metrics

log = get_logger("rate_limit")


def identify(request: Request) -> str:
    """Key function — prefer the authenticated user_id, fall back to IP."""
    return getattr(request.state, "user_id", None) or get_remote_address(request)

limiter = Limiter(
    key_func=identify,
    storage_uri=os.environ.get("RATE_LIMIT_STORAGE_URI", "memory://"),
    default_limits=["120/hour"],   
    headers_enabled=False,         
)

ROUTE_LIMITS: dict[str, str] = {
    "/onboarding":   "5/hour;1/minute",
    "/cv/base":      "10/hour",
    "/cv/targeted":  "20/hour;3/minute",
    "/healthz":      "1000/hour",
    "/metrics":      "1000/hour",
}


def limit_for(route: str) -> str:
    return ROUTE_LIMITS.get(route, "60/hour")

async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Custom 429 handler — increments metric and logs (no PII)."""
    route = request.scope.get("route")
    endpoint = getattr(route, "path", None) or request.url.path
    metrics.rate_limited_total.labels(endpoint=endpoint).inc()
    log.warning("rate_limit.block", endpoint=endpoint, limit=str(exc.detail))
    return JSONResponse(
        status_code=429,
        content={"detail": "rate limit exceeded", "retry_after_seconds": 60},
        headers={"Retry-After": "60"},
    )
