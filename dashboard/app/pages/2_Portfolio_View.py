import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import plotly.express as px
import streamlit as st

import api_client
import data_access

st.set_page_config(page_title='Portfolio View', page_icon='📊', layout='wide')

st.title('Portfolio View')
st.caption('Population-level risk, scoring activity, and model health.')

# --------------------------------------------------------------------------
# KPI row
# --------------------------------------------------------------------------

summary = data_access.portfolio_summary()
model = data_access.active_model()
drift = data_access.drift_latest()
service_metrics, metrics_error = api_client.metrics()

worst_drift = 'UNKNOWN'
if not drift.empty:
    for status in ('ALERT', 'WARN', 'OK'):
        if (drift['drift_status'] == status).any():
            worst_drift = status
            break

col1, col2, col3, col4 = st.columns(4)
col1.metric('Applicants in gold', f"{summary.get('total_applicants', 0):,}")
col2.metric('Observed default rate',
            f"{float(summary.get('default_rate') or 0):.2%}")
col3.metric('Predictions served',
            f"{(service_metrics or {}).get('predictions_total', 0):,}")
col4.metric('Drift status', worst_drift)

if metrics_error:
    st.warning(f"Live service metrics unavailable — {metrics_error}")

if model:
    st.caption(
        f"Active model: `{model['pipeline_name']}` `{model['model_version']}` "
        f"({model['algorithm']}) · AUC {float(model['metric_auc']):.4f} · "
        f"KS {float(model['metric_ks']):.4f} · "
        f"trained on {int(model['training_rows']):,} rows"
    )

st.divider()

# --------------------------------------------------------------------------
# Population risk
# --------------------------------------------------------------------------

st.header('Population risk')

left, right = st.columns(2)

with left:
    st.subheader('Default rate by delinquency severity')
    severity = data_access.default_rate_by_severity()
    if severity.empty:
        st.info('No data. Run the ETL first: `python -m data_platform.run_etl`')
    else:
        severity['default_rate'] = severity['default_rate'].astype(float)
        figure = px.bar(
            severity, x='severity_label', y='default_rate',
            hover_data=['n'], labels={
                'severity_label': 'Delinquency severity',
                'default_rate': 'Observed default rate',
            },
        )
        figure.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        figure.update_yaxes(tickformat='.1%')
        st.plotly_chart(figure, use_container_width=True)

with right:
    st.subheader('Default rate by income band')
    income = data_access.default_rate_by_income_band()
    if income.empty:
        st.info('No data. Run the ETL first.')
    else:
        income['default_rate'] = income['default_rate'].astype(float)
        figure = px.bar(
            income, x='band_label', y='default_rate',
            hover_data=['n'], labels={
                'band_label': 'Monthly income band',
                'default_rate': 'Observed default rate',
            },
        )
        figure.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        figure.update_yaxes(tickformat='.1%')
        st.plotly_chart(figure, use_container_width=True)

st.subheader('Highest-risk segments')
st.caption(
    'The star-schema join from Phase 2 — delinquency profile crossed with '
    'income band. This is the query that proves the dimensional model works.'
)

segments = data_access.segment_default_rates()
if segments.empty:
    st.info('No data.')
else:
    top_n = st.slider('Segments to show', 5, 40, 10)
    display = segments.head(top_n).copy()
    display['default_rate'] = display['default_rate'].astype(float)
    display.columns = ['Delinquency severity', 'Income band',
                       'Applicants', 'Default rate']
    st.dataframe(
        display.style.format({'Default rate': '{:.2%}', 'Applicants': '{:,}'}),
        hide_index=True, use_container_width=True,
    )

st.divider()

# --------------------------------------------------------------------------
# Scoring activity
# --------------------------------------------------------------------------

st.header('Scoring activity')

volume = data_access.prediction_volume()
distribution = data_access.prediction_distribution()

if volume.empty and distribution.empty:
    st.info(
        'No predictions have been logged yet. Prediction logging is wired in '
        'during Phase 7 — until then this section stays empty even though the '
        'API is serving requests.'
    )
else:
    left, right = st.columns(2)

    with left:
        st.subheader('Predictions per hour')
        if volume.empty:
            st.info('No volume data.')
        else:
            figure = px.line(volume, x='hour', y='n', markers=True,
                             labels={'hour': 'Hour', 'n': 'Predictions'})
            figure.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(figure, use_container_width=True)

    with right:
        st.subheader('Distribution of predicted probabilities')
        if distribution.empty:
            st.info('No prediction data.')
        else:
            figure = px.histogram(
                distribution, x='probability_default', nbins=40,
                labels={'probability_default': 'Predicted probability of default'},
            )
            figure.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(figure, use_container_width=True)

if service_metrics:
    st.subheader('Service latency')
    a, b, c = st.columns(3)
    a.metric('p50 latency', f"{service_metrics['latency_p50_ms']:.0f} ms")
    b.metric('p95 latency', f"{service_metrics['latency_p95_ms']:.0f} ms")
    c.metric('Errors', f"{service_metrics['errors_total']:,}")

st.divider()

# --------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------

st.header('Input drift')

if drift.empty:
    st.info(
        'No drift reports yet. The PSI drift checker is built in Phase 7 '
        '(`python -m monitoring.run_drift_check`).'
    )
else:
    st.caption(f"Most recent check: {drift['computed_at'].iloc[0]}")
    display = drift.copy()
    display['psi'] = display['psi'].astype(float)
    st.dataframe(
        display[['feature_name', 'psi', 'drift_status', 'n_reference', 'n_current']]
        .style.format({'psi': '{:.4f}'}),
        hide_index=True, use_container_width=True,
    )
    if (drift['drift_status'] == 'ALERT').any():
        st.error('One or more features are in ALERT. Investigate before relying '
                 'on current scores.')