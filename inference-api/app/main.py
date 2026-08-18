from contextlib import asynccontextmanager
from uuid import uuid4

from app.middleware import RequestContextMiddleware
from app.model_store import MODEL_STORE
from app.routers import health, metrics, predict
from app.settings import get_settings
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

settings = get_settings()
from monitoring.prediction_logger import log_service_event


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    MODEL_STORE.load(settings)
    log_service_event(settings.app_name, 'model_loaded', {
        'pipeline': settings.active_pipeline,
        'model_version': settings.model_version,
        'success': MODEL_STORE.is_ready(),
        'error': MODEL_STORE.load_error,
    })

    yield

    # --- shutdown ---
    log_service_event(settings.app_name, 'shutdown', {'uptime_seconds': MODEL_STORE.uptime_seconds()})


app = FastAPI(
    title='Enterprise AI Platform — Inference API',
    description=(
        'Serves predictions, SHAP explanations and optional LLM narratives '
        'for whichever pipeline ACTIVE_PIPELINE names.'
    ),
    version=settings.app_version,
    lifespan=lifespan,
)
app.add_middleware(RequestContextMiddleware)
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