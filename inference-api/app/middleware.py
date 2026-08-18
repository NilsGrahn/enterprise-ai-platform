import json
import sys
import time
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, times the request, emits a structured log line,
    and converts unhandled exceptions into an ErrorResponse.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get('X-Request-ID') or str(uuid4())
        request.state.request_id = request_id

        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            self._log(request, 500, latency_ms, request_id, error=str(exc))

            from monitoring.prediction_logger import log_service_event
            log_service_event('inference-api', 'error', {
                'request_id': request_id,
                'path': request.url.path,
                'error': str(exc),
            })

            return JSONResponse(
                status_code=500,
                content={
                    'error': 'internal_error',
                    'detail': 'An unexpected error occurred.',
                    'request_id': request_id,
                },
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        response.headers['X-Request-ID'] = request_id
        response.headers['X-Response-Time-ms'] = str(latency_ms)
        self._log(request, response.status_code, latency_ms, request_id)
        return response

    @staticmethod
    def _log(request, status_code, latency_ms, request_id, error=None):
        entry = {
            'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'method': request.method,
            'path': request.url.path,
            'status': status_code,
            'latency_ms': latency_ms,
            'request_id': request_id,
        }
        if error:
            entry['error'] = error
        stream = sys.stderr if error else sys.stdout
        print(json.dumps(entry), file=stream)