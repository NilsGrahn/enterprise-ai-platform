"""Compute PSI drift for the active model over a recent traffic window.

CLI: python -m monitoring.run_drift_check --window-hours 24
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from data_platform.db import get_engine
from ml_service.artifacts import artifact_path
from ml_service.config import get_settings
from monitoring.drift import classify, psi_from_profile
from monitoring.prediction_logger import log_service_event
from sqlalchemy import text

MIN_ROWS = 100


def parse_args():
    settings = get_settings()
    parser = argparse.ArgumentParser(description='Run a PSI drift check.')
    parser.add_argument('--pipeline', default=settings.active_pipeline)
    parser.add_argument('--version', default=settings.model_version)
    parser.add_argument('--window-hours', type=int, default=24)
    parser.add_argument('--warn', type=float, default=None)
    parser.add_argument('--alert', type=float, default=None)
    return parser.parse_args()


def load_profile(pipeline_name, version) -> dict:
    path = artifact_path(pipeline_name, version) / 'reference_profile.json'
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No reference profile at {path}. Build one first:  "
            f"python -m monitoring.build_reference --pipeline {pipeline_name} "
            f"--version {version}"
        )
    with open(path) as f:
        return json.load(f)


def load_current_window(pipeline_name, version, window_hours) -> pd.DataFrame:
    """Expand the logged feature_vector JSONB into a DataFrame."""
    engine = get_engine()
    rows = pd.read_sql(text("""
        SELECT feature_vector
        FROM monitoring.prediction_log
        WHERE pipeline_name = :pipeline
          AND model_version = :version
          AND status = 'OK'
          AND received_at > now() - make_interval(hours => :hours)
    """), engine, params={
        'pipeline': pipeline_name,
        'version': version,
        'hours': window_hours,
    })

    if rows.empty:
        return pd.DataFrame()

    return pd.DataFrame(list(rows['feature_vector']))


def record_drift(pipeline_name, version, results, window_hours, profile):
    engine = get_engine()
    reference_window = f"{profile['split']} split, {profile['bins']} bins"
    current_window = f"last {window_hours}h of prediction_log"

    with engine.begin() as conn:
        for row in results:
            conn.execute(text("""
                INSERT INTO monitoring.drift_report
                    (pipeline_name, model_version, feature_name, psi,
                     drift_status, reference_window, current_window,
                     n_reference, n_current)
                VALUES
                    (:pipeline, :version, :feature, :psi,
                     :status, :ref_window, :cur_window,
                     :n_reference, :n_current)
            """), {
                'pipeline': pipeline_name,
                'version': version,
                'feature': row['feature'],
                'psi': row['psi'],
                'status': row['status'],
                'ref_window': reference_window,
                'cur_window': current_window,
                'n_reference': row['n_reference'],
                'n_current': row['n_current'],
            })


def main():
    args = parse_args()
    settings = get_settings()

    warn = args.warn if args.warn is not None else 0.10
    alert = args.alert if args.alert is not None else 0.25

    profile = load_profile(args.pipeline, args.version)
    current = load_current_window(args.pipeline, args.version, args.window_hours)

    if len(current) < MIN_ROWS:
        print(f"insufficient data: {len(current)} rows in the last "
              f"{args.window_hours}h, need at least {MIN_ROWS}. "
              f"Nothing written.")
        return

    print(f"comparing {len(current):,} recent rows against the reference "
          f"({profile['split']} split)\n")

    results = []
    for feature in profile['feature_names']:
        if feature not in current.columns:
            print(f"  warning: '{feature}' missing from logged vectors, skipped")
            continue

        feature_profile = profile['features'][feature]
        psi = psi_from_profile(feature_profile, current[feature])
        status = classify(psi, warn=warn, alert=alert)

        results.append({
            'feature': feature,
            'psi': psi,
            'status': status,
            'n_reference': feature_profile['n_reference'],
            'n_current': int(current[feature].notna().sum()),
        })

    results.sort(key=lambda r: r['psi'], reverse=True)

    print(f"  {'feature':<28} {'psi':>10}   status")
    print(f"  {'-' * 28} {'-' * 10}   ------")
    for row in results:
        print(f"  {row['feature']:<28} {row['psi']:>10.4f}   {row['status']}")
    print()

    record_drift(args.pipeline, args.version, results, args.window_hours, profile)

    alerts = [r for r in results if r['status'] == 'ALERT']
    warns = [r for r in results if r['status'] == 'WARN']

    if alerts:
        print(f"ALERT on {len(alerts)} feature(s): "
              f"{', '.join(r['feature'] for r in alerts)}")
        log_service_event('monitoring', 'drift_alert', {
            'pipeline': args.pipeline,
            'model_version': args.version,
            'window_hours': args.window_hours,
            'features': [{'feature': r['feature'], 'psi': round(r['psi'], 4)}
                         for r in alerts],
        })
    elif warns:
        print(f"WARN on {len(warns)} feature(s)")
    else:
        print("all features stable")

    print(f"wrote {len(results)} rows to monitoring.drift_report")


if __name__ == '__main__':
    main()