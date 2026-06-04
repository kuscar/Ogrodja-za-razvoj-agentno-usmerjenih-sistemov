from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.observability.logging import (
    get_logger,
    request_id_var,
    user_id_var,
)
from src.observability.metrics import (
    http_request_duration_seconds,
    http_requests_total,
    in_flight_requests,
)

log = get_logger("http")


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or request.url.path


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        req_token = request_id_var.set(request_id)
        usr_token = user_id_var.set("-")

        in_flight_requests.inc()
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            dur = time.perf_counter() - start
            in_flight_requests.dec()
            endpoint = _route_template(request)
            http_requests_total.labels(
                endpoint=endpoint, method=request.method, status=str(status)
            ).inc()
            http_request_duration_seconds.labels(
                endpoint=endpoint, method=request.method
            ).observe(dur)
            log.info(
                "http.request",
                endpoint=endpoint,
                method=request.method,
                status=status,
                duration_ms=int(dur * 1000),
            )
            request_id_var.reset(req_token)
            user_id_var.reset(usr_token)
