import time

from app.model_store import MODEL_STORE
from app.schemas import MetricsResponse
from app.settings import get_settings
from data_platform.db import get_engine
from fastapi import APIRouter
from sqlalchemy import text

router = APIRouter(tags=['metrics'])
settings = get_settings()

_cache = {'value': None, 'expires_at': 0.0}


def _empty_metrics() -> MetricsResponse:
    return MetricsResponse(
        predictions_total=0,
        predictions_last_hour=0,
        errors_total=0,
        latency_p50_ms=0.0,
        latency_p95_ms=0.0,
        average_probability_last_hour=None,
        active_model_version=settings.model_version,
        drift_status='UNKNOWN',
    )


def _query_metrics() -> MetricsResponse:
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT
                count(*)                                          AS predictions_total,
                count(*) FILTER (WHERE received_at > now() - interval '1 hour')
                                                                  AS predictions_last_hour,
                count(*) FILTER (WHERE status <> 'OK')            AS errors_total,
                coalesce(percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms), 0)
                                                                  AS p50,
                coalesce(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms), 0)
                                                                  AS p95,
                avg(probability_default) FILTER
                    (WHERE received_at > now() - interval '1 hour')
                                                                  AS avg_probability
            FROM monitoring.prediction_log
        """)).mappings().fetchone()

        drift = conn.execute(text("""
            SELECT drift_status FROM monitoring.drift_report
            WHERE computed_at = (SELECT max(computed_at) FROM monitoring.drift_report)
            ORDER BY CASE drift_status
                       WHEN 'ALERT' THEN 1 WHEN 'WARN' THEN 2 ELSE 3
                     END
            LIMIT 1
        """)).fetchone()

    return MetricsResponse(
        predictions_total=int(row['predictions_total'] or 0),
        predictions_last_hour=int(row['predictions_last_hour'] or 0),
        errors_total=int(row['errors_total'] or 0),
        latency_p50_ms=float(row['p50'] or 0.0),
        latency_p95_ms=float(row['p95'] or 0.0),
        average_probability_last_hour=(
            float(row['avg_probability']) if row['avg_probability'] is not None else None
        ),
        active_model_version=(
            MODEL_STORE.metadata['model_version']
            if MODEL_STORE.metadata else settings.model_version
        ),
        drift_status=drift[0] if drift else 'UNKNOWN',
    )


@router.get('/metrics', response_model=MetricsResponse, summary='Service metrics')
def metrics():
    now = time.monotonic()
    if _cache['value'] is not None and now < _cache['expires_at']:
        return _cache['value']

    try:
        value = _query_metrics()
    except Exception as exc:  # noqa: BLE001
        print(f"[metrics] query failed: {exc}")
        value = _empty_metrics()

    _cache['value'] = value
    _cache['expires_at'] = now + settings.metrics_cache_seconds
    return value