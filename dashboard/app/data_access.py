from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text

# dashboard/app/data_access.py -> app/ -> dashboard/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / '.env')

from data_platform.db import get_engine

CACHE_TTL = 300


@st.cache_data(ttl=CACHE_TTL)
def portfolio_summary() -> dict:
    """Headline counts and the observed default rate."""
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT
                count(*)                                   AS total_applicants,
                avg(is_serious_delinquency::numeric)       AS default_rate,
                count(*) FILTER (WHERE dataset_split = 'train') AS train_rows,
                count(*) FILTER (WHERE dataset_split = 'valid') AS valid_rows,
                count(*) FILTER (WHERE dataset_split = 'test')  AS test_rows
            FROM gold.fact_credit_assessment
        """)).mappings().fetchone()
    return dict(row) if row else {}


@st.cache_data(ttl=CACHE_TTL)
def segment_default_rates() -> pd.DataFrame:
    """Default rate by delinquency severity and income band.

    This is the star-schema join from Phase 2, Step 2.4 — the query that
    proved the dimensional model works.
    """
    engine = get_engine()
    return pd.read_sql(text("""
        SELECT d.severity_label,
               i.band_label,
               count(*)                                        AS n,
               round(avg(f.is_serious_delinquency)::numeric, 4) AS default_rate
        FROM gold.fact_credit_assessment f
        JOIN gold.dim_delinquency_profile d USING (delinquency_profile_key)
        JOIN gold.dim_income_band i         USING (income_band_key)
        GROUP BY 1, 2
        ORDER BY default_rate DESC
    """), engine)


@st.cache_data(ttl=CACHE_TTL)
def default_rate_by_severity() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(text("""
        SELECT d.severity_label,
               count(*)                                        AS n,
               round(avg(f.is_serious_delinquency)::numeric, 4) AS default_rate
        FROM gold.fact_credit_assessment f
        JOIN gold.dim_delinquency_profile d USING (delinquency_profile_key)
        GROUP BY 1
        ORDER BY default_rate DESC
    """), engine)


@st.cache_data(ttl=CACHE_TTL)
def default_rate_by_income_band() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(text("""
        SELECT i.band_label,
               i.sort_order,
               count(*)                                        AS n,
               round(avg(f.is_serious_delinquency)::numeric, 4) AS default_rate
        FROM gold.fact_credit_assessment f
        JOIN gold.dim_income_band i USING (income_band_key)
        GROUP BY 1, 2
        ORDER BY i.sort_order
    """), engine)


@st.cache_data(ttl=CACHE_TTL)
def active_model() -> dict:
    """The currently active model row for the currently configured pipeline."""
    import os
    pipeline_name = os.getenv('ACTIVE_PIPELINE', 'credit')

    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT pipeline_name, model_version, algorithm,
                   metric_auc, metric_ks, training_rows, trained_at
            FROM gold.dim_model
            WHERE is_active AND pipeline_name = :pipeline_name
            LIMIT 1
        """), {'pipeline_name': pipeline_name}).mappings().fetchone()
    return dict(row) if row else {}


@st.cache_data(ttl=60)
def prediction_volume() -> pd.DataFrame:
    """Predictions per hour from the monitoring log.

    Empty until Phase 7 wires log_prediction into the API.
    """
    engine = get_engine()
    return pd.read_sql(text("""
        SELECT date_trunc('hour', received_at) AS hour,
               count(*)                        AS n,
               avg(probability_default)        AS avg_probability
        FROM monitoring.prediction_log
        WHERE received_at > now() - interval '7 days'
        GROUP BY 1
        ORDER BY 1
    """), engine)


@st.cache_data(ttl=60)
def prediction_distribution() -> pd.DataFrame:
    """Raw predicted probabilities for a histogram."""
    engine = get_engine()
    return pd.read_sql(text("""
        SELECT probability_default, risk_band, received_at
        FROM monitoring.prediction_log
        WHERE probability_default IS NOT NULL
        ORDER BY received_at DESC
        LIMIT 5000
    """), engine)


@st.cache_data(ttl=60)
def drift_latest() -> pd.DataFrame:
    """Most recent drift report, worst status first.

    Empty until Phase 7 builds the drift checker.
    """
    engine = get_engine()
    return pd.read_sql(text("""
        SELECT feature_name, psi, drift_status, computed_at,
               n_reference, n_current
        FROM monitoring.drift_report
        WHERE computed_at = (SELECT max(computed_at) FROM monitoring.drift_report)
        ORDER BY psi DESC
    """), engine)


@st.cache_data(ttl=CACHE_TTL)
def random_test_applicant() -> dict:
    """One random applicant from the test split, for the 'load a random
    applicant' button.

    Selects only the ten fields the form has. total_delinquency_events is
    derived by the API, so it is deliberately not fetched.
    """
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT applicant_id,
                   revolving_utilisation, age, debt_ratio, monthly_income,
                   open_credit_lines, real_estate_loans,
                   times_30_59_days_late, times_60_89_days_late,
                   times_90_days_late, number_of_dependents,
                   is_serious_delinquency
            FROM gold.v_credit_assessment
            WHERE dataset_split = 'test' AND age IS NOT NULL
            ORDER BY random()
            LIMIT 1
        """)).mappings().fetchone()
    return dict(row) if row else {}