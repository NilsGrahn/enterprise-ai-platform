import json
import math

from data_platform.db import get_engine
from sqlalchemy import text


def _json_safe(obj):
    """Recursively replace NaN/Infinity floats with None, since they are
    valid Python but invalid strict JSON, which Postgres's JSONB rejects."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


class NoActiveModelError(Exception):
    """Raised when a pipeline has no active model registered."""


def register_model(pipeline_name, model_version, algorithm, hyperparameters,
                   feature_list, training_rows, metrics):
    """Insert or update this model's row in gold.dim_model.

    Returns the model_key. Does NOT activate the model — that is a separate,
    deliberate step.
    """
    engine = get_engine()

    params = {
        'pipeline_name': pipeline_name,
        'model_name': f"{pipeline_name}_{algorithm}",
        'model_version': model_version,
        'algorithm': algorithm,
        'hyperparameters': json.dumps(_json_safe(hyperparameters), default=str),
        'feature_list': json.dumps(list(feature_list)),
        'training_rows': int(training_rows),
        'metric_auc': float(metrics.get('auc', 0.0)),
        'metric_ks': float(metrics.get('ks', 0.0)),
    }

    with engine.begin() as conn:
        row = conn.execute(text("""
            INSERT INTO gold.dim_model
                (pipeline_name, model_name, model_version, algorithm,
                 hyperparameters, feature_list, training_rows, trained_at,
                 metric_auc, metric_ks, is_active)
            VALUES
                (:pipeline_name, :model_name, :model_version, :algorithm,
                 CAST(:hyperparameters AS JSONB), CAST(:feature_list AS JSONB),
                 :training_rows, now(),
                 :metric_auc, :metric_ks, FALSE)
            ON CONFLICT (pipeline_name, model_version) DO UPDATE SET
                model_name      = EXCLUDED.model_name,
                algorithm       = EXCLUDED.algorithm,
                hyperparameters = EXCLUDED.hyperparameters,
                feature_list    = EXCLUDED.feature_list,
                training_rows   = EXCLUDED.training_rows,
                trained_at      = EXCLUDED.trained_at,
                metric_auc      = EXCLUDED.metric_auc,
                metric_ks       = EXCLUDED.metric_ks
            RETURNING model_key
        """), params).fetchone()

    return int(row[0])


def activate_model(pipeline_name, model_version):
    """Make this version the single active model for its pipeline.

    Both statements run in one transaction so there is never a moment with
    two active models — which the partial unique index would reject anyway.
    """
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE gold.dim_model
            SET is_active = FALSE
            WHERE pipeline_name = :pipeline_name AND is_active
        """), {'pipeline_name': pipeline_name})

        result = conn.execute(text("""
            UPDATE gold.dim_model
            SET is_active = TRUE
            WHERE pipeline_name = :pipeline_name
              AND model_version = :model_version
            RETURNING model_key
        """), {'pipeline_name': pipeline_name, 'model_version': model_version}).fetchone()

        if result is None:
            raise ValueError(
                f"Cannot activate: no registered model for pipeline "
                f"'{pipeline_name}' version '{model_version}'"
            )

    return int(result[0])


def get_active_model(pipeline_name):
    """Return the active model row for this pipeline as a dict."""
    engine = get_engine()

    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT model_key, pipeline_name, model_name, model_version,
                   algorithm, feature_list, training_rows, trained_at,
                   metric_auc, metric_ks
            FROM gold.dim_model
            WHERE pipeline_name = :pipeline_name AND is_active
        """), {'pipeline_name': pipeline_name}).mappings().fetchone()

    if row is None:
        raise NoActiveModelError(
            f"No active model for pipeline '{pipeline_name}'. "
            f"Train one and activate it with: python -m ml_service.train "
            f"--pipeline {pipeline_name} --activate"
        )

    return dict(row)