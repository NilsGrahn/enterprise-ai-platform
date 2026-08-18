import json
import math
import sys

from sqlalchemy import text

from data_platform.db import get_engine


def _json_safe(obj):
    """Replace NaN/Infinity with None, recursively.

    Valid Python, invalid strict JSON — Postgres's JSONB parser rejects them.
    Same fix as registry.py in Phase 3.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if hasattr(obj, 'item'):          # numpy scalar -> native Python
        try:
            return _json_safe(obj.item())
        except Exception:
            return str(obj)
    return obj


def log_prediction(request_id, pipeline_name, model_version, request_payload,
                   feature_vector, probability=None, predicted_class=None,
                   risk_band=None, latency_ms=None, status='OK',
                   error_message=None):
    """Append one row to monitoring.prediction_log.

    NEVER raises. A logging failure must not turn a successful prediction into
    a 500. On failure it prints to stderr and returns False.
    """
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO monitoring.prediction_log
                    (request_id, pipeline_name, model_version,
                     request_payload, feature_vector,
                     probability_default, predicted_class, risk_band,
                     latency_ms, status, error_message)
                VALUES
                    (CAST(:request_id AS UUID), :pipeline_name, :model_version,
                     CAST(:request_payload AS JSONB), CAST(:feature_vector AS JSONB),
                     :probability, :predicted_class, :risk_band,
                     :latency_ms, :status, :error_message)
            """), {
                'request_id': str(request_id),
                'pipeline_name': pipeline_name,
                'model_version': model_version,
                'request_payload': json.dumps(_json_safe(request_payload), default=str),
                'feature_vector': json.dumps(_json_safe(feature_vector), default=str),
                'probability': None if probability is None else float(probability),
                'predicted_class': None if predicted_class is None else int(predicted_class),
                'risk_band': risk_band,
                'latency_ms': None if latency_ms is None else int(latency_ms),
                'status': status,
                'error_message': error_message,
            })
        return True

    except Exception as exc:
        print(f"[prediction_logger] failed to log {request_id}: {exc}",
              file=sys.stderr)
        return False


def log_service_event(service_name, event_type, details=None):
    """Append one row to monitoring.service_event. Never raises."""
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO monitoring.service_event
                    (service_name, event_type, details)
                VALUES (:service_name, :event_type, CAST(:details AS JSONB))
            """), {
                'service_name': service_name,
                'event_type': event_type,
                'details': json.dumps(_json_safe(details or {}), default=str),
            })
        return True
    except Exception as exc:
        print(f"[prediction_logger] failed to log event '{event_type}': {exc}",
              file=sys.stderr)
        return False