import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

import api_client
import data_access

st.set_page_config(
    page_title='Enterprise AI Platform',
    page_icon='🏦',
    layout='wide',
)

st.title('Enterprise AI Platform')
st.caption('A domain-agnostic machine learning platform, currently configured '
           'for consumer credit risk.')

# --------------------------------------------------------------------------
# Service health strip
# --------------------------------------------------------------------------

health, health_error = api_client.health()

st.subheader('Service health')

if health_error:
    st.error(health_error)
else:
    col1, col2, col3, col4 = st.columns(4)

    status = health['status']
    icon = {'ok': '🟢', 'degraded': '🟡', 'error': '🔴'}.get(status, '⚪')
    col1.metric('API', f"{icon} {status}")
    col2.metric('Model loaded', 'yes' if health['model_loaded'] else 'no')
    col3.metric('Database', 'reachable' if health['database_reachable'] else 'unreachable')

    llm = health.get('llm_reachable')
    col4.metric('LLM', 'disabled' if llm is None else ('reachable' if llm else 'unreachable'))

    st.caption(
        f"Serving `{health['pipeline_name']}` `{health['model_version']}` · "
        f"app v{health['app_version']} · up {health['uptime_seconds']:.0f}s"
    )

st.divider()

# --------------------------------------------------------------------------
# Active model
# --------------------------------------------------------------------------

st.subheader('Active model')

model = data_access.active_model()
if not model:
    st.warning(
        'No active model registered. Train and activate one:  '
        '`python -m ml_service.train --pipeline credit --version v1 --activate`'
    )
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Pipeline', model['pipeline_name'])
    col2.metric('Version', model['model_version'])
    col3.metric('Validation AUC', f"{float(model['metric_auc']):.4f}")
    col4.metric('KS statistic', f"{float(model['metric_ks']):.4f}")
    st.caption(
        f"{model['algorithm']} trained on {int(model['training_rows']):,} rows "
        f"at {model['trained_at']}"
    )

st.divider()

# --------------------------------------------------------------------------
# What this is
# --------------------------------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader('How it works')
    st.markdown("""
    1. **Ingestion** lands raw source rows in a bronze append-only log with a
       SHA-256 fingerprint and a full audit trail.
    2. **Silver** casts types, flags data-quality problems rather than repairing
       them, and deduplicates on the most recent record per applicant.
    3. **Gold** is a star schema — a fact table of assessments surrounded by
       date, borrower, income band, utilisation band, delinquency profile and
       model dimensions.
    4. **The ML pipeline** is an abstract base class with a fixed training
       protocol. Swapping domain means adding one class and changing one
       environment variable.
    5. **The API** serves predictions with SHAP explanations and an optional,
       schema-validated language-model narrative.
    """)

with right:
    st.subheader('Views')
    st.markdown("""
    **Analyst View** — score one applicant, see the SHAP contributions that
    drove the result, with reason codes and an optional written assessment.

    **Portfolio View** — population default rates by segment, scoring volume,
    the distribution of predicted probabilities, and input drift status.
    """)
    st.info('Use the sidebar to switch views.')

st.divider()
st.caption(
    'Decision support only. The model produces a modelled probability and the '
    'statistical associations behind it; a human officer makes the decision.'
)