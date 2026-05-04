from __future__ import annotations

import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: object) -> Response:
        t0 = time.perf_counter()
        response: Response = await call_next(request)  # type: ignore[operator]
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("%s %s → %d (%.1fms)", request.method, request.url.path, response.status_code, elapsed)
        return response
