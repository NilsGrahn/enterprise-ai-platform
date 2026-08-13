from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text

from data_platform.db import get_engine

from app.model_store import MODEL_STORE
from app.settings import get_settings
from app.routers import predict, health, metrics

settings = get_settings()


def _log_service_event(event_type: str, details: dict):
    """Best-effort write to monitoring.service_event. Never raises."""
    import json
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO monitoring.service_event (service_name, event_type, details)
                VALUES (:service, :event_type, CAST(:details AS JSONB))
            """), {
                'service': settings.app_name,
                'event_type': event_type,
                'details': json.dumps(details, default=str),
            })
    except Exception as exc:
        print(f"[main] could not log service_event '{event_type}': {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    MODEL_STORE.load(settings)
    _log_service_event('model_loaded', {
        'pipeline': settings.active_pipeline,
        'model_version': settings.model_version,
        'success': MODEL_STORE.is_ready(),
        'error': MODEL_STORE.load_error,
    })

    yield

    # --- shutdown ---
    _log_service_event('shutdown', {'uptime_seconds': MODEL_STORE.uptime_seconds()})


app = FastAPI(
    title='Enterprise AI Platform — Inference API',
    description=(
        'Serves predictions, SHAP explanations and optional LLM narratives '
        'for whichever pipeline ACTIVE_PIPELINE names.'
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(predict.router)
app.include_router(health.router)
app.include_router(metrics.router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            'error': 'validation_error',
            'detail': str(exc.errors()),
            'request_id': request.headers.get('X-Request-ID', str(uuid4())),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = request.headers.get('X-Request-ID', str(uuid4()))
    print(f"[main] unhandled exception on {request.url.path} "
          f"(request_id={request_id}): {exc}")
    return JSONResponse(
        status_code=500,
        content={
            'error': 'internal_error',
            'detail': 'An unexpected error occurred.',
            'request_id': request_id,
        },
    )