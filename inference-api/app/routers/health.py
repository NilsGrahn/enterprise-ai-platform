from app.model_store import MODEL_STORE
from app.schemas import HealthResponse
from app.settings import get_settings
from data_platform.db import get_engine
from fastapi import APIRouter, Response
from sqlalchemy import text

router = APIRouter(tags=['health'])
settings = get_settings()


def _database_reachable() -> bool:
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text('SELECT 1'))
        return True
    except Exception:  # noqa: BLE001
        return False


@router.get('/health', response_model=HealthResponse, summary='Full health picture')
def health(response: Response):
    model_loaded = MODEL_STORE.is_ready()
    database_reachable = _database_reachable()

    if model_loaded and database_reachable:
        status = 'ok'
    elif model_loaded:
        status = 'degraded'
    else:
        status = 'error'

    # 200 for ok and degraded, 503 only when the model is missing, so a
    # database blip does not take the container out of rotation.
    response.status_code = 503 if status == 'error' else 200

    llm_reachable = None
    if settings.llm_enabled and MODEL_STORE.llm_client is not None:
        # Never call the LLM API on a health check — report breaker state.
        llm_reachable = not MODEL_STORE.llm_client.breaker.is_open

    return HealthResponse(
        status=status,
        app_version=settings.app_version,
        pipeline_name=settings.active_pipeline,
        model_version=settings.model_version,
        model_loaded=model_loaded,
        database_reachable=database_reachable,
        llm_reachable=llm_reachable,
        uptime_seconds=round(MODEL_STORE.uptime_seconds(), 1),
    )


@router.get('/health/live', summary='Liveness — is the process running')
def live():
    return {'status': 'alive'}


@router.get('/health/ready', summary='Readiness — can it serve traffic')
def ready(response: Response):
    model_loaded = MODEL_STORE.is_ready()
    database_reachable = _database_reachable()
    if not (model_loaded and database_reachable):
        response.status_code = 503
        return {
            'status': 'not_ready',
            'model_loaded': model_loaded,
            'database_reachable': database_reachable,
        }
    return {'status': 'ready'}