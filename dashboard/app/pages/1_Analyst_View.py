import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import api_client
import data_access

st.set_page_config(page_title='Analyst View', page_icon='🔍', layout='wide')

st.title('Analyst View')
st.caption('Score a single applicant and inspect what drove the result.')

# --------------------------------------------------------------------------
# Defaults and session state
# --------------------------------------------------------------------------

DEFAULTS = {
    'revolving_utilisation': 0.35,
    'age': 45,
    'debt_ratio': 0.40,
    'monthly_income': 5000.0,
    'open_credit_lines': 8,
    'real_estate_loans': 1,
    'times_30_59_days_late': 0,
    'times_60_89_days_late': 0,
    'times_90_days_late': 0,
    'number_of_dependents': 1,
}

for field, value in DEFAULTS.items():
    st.session_state.setdefault(field, value)

st.session_state.setdefault('loaded_applicant_id', None)
st.session_state.setdefault('loaded_actual_outcome', None)


def load_random_applicant():
    """Callback for the sidebar button. Writes into session state."""
    data_access.random_test_applicant.clear()      # bypass the cache
    applicant = data_access.random_test_applicant()
    if not applicant:
        st.session_state['load_error'] = 'No test-split applicants found.'
        return

    for field in DEFAULTS:
        value = applicant.get(field)
        if value is None or pd.isna(value):
            st.session_state[field] = DEFAULTS[field]
        elif isinstance(DEFAULTS[field], int):
            st.session_state[field] = int(value)
        else:
            st.session_state[field] = float(value)

    st.session_state['loaded_applicant_id'] = applicant.get('applicant_id')
    st.session_state['loaded_actual_outcome'] = applicant.get('is_serious_delinquency')


# --------------------------------------------------------------------------
# Sidebar — the applicant form
# --------------------------------------------------------------------------

with st.sidebar:
    st.header('Applicant details')

    st.button('Load a random applicant from the test split',
              on_click=load_random_applicant, use_container_width=True)

    if st.session_state.get('load_error'):
        st.warning(st.session_state.pop('load_error'))

    if st.session_state['loaded_applicant_id'] is not None:
        outcome = st.session_state['loaded_actual_outcome']
        st.info(
            f"Loaded applicant {st.session_state['loaded_applicant_id']} — "
            f"actual outcome: {'defaulted' if outcome == 1 else 'did not default'}"
        )

    st.divider()

    st.number_input('Credit utilisation ratio', min_value=0.0, max_value=50.0,
                    step=0.01, format='%.4f', key='revolving_utilisation',
                    help='Revolving balance divided by total credit limit')
    st.number_input('Age', min_value=18, max_value=110, step=1, key='age')
    st.number_input('Debt-to-income ratio', min_value=0.0, max_value=100.0,
                    step=0.01, format='%.4f', key='debt_ratio')
    st.number_input('Monthly income', min_value=0.0, step=100.0,
                    format='%.2f', key='monthly_income',
                    help='Set to 0 to simulate missing income')
    st.number_input('Open credit lines and loans', min_value=0, step=1,
                    key='open_credit_lines')
    st.number_input('Real estate loans or lines', min_value=0, step=1,
                    key='real_estate_loans')
    st.number_input('Payments 30-59 days late', min_value=0, max_value=20, step=1,
                    key='times_30_59_days_late')
    st.number_input('Payments 60-89 days late', min_value=0, max_value=20, step=1,
                    key='times_60_89_days_late')
    st.number_input('Payments 90+ days late', min_value=0, max_value=20, step=1,
                    key='times_90_days_late')
    st.number_input('Number of dependents', min_value=0, max_value=20, step=1,
                    key='number_of_dependents')

    st.divider()
    want_narrative = st.toggle(
        'Generate written assessment',
        value=False,
        help='Calls the language model. Slower, and requires LLM_API_KEY.',
    )
    submitted = st.button('Score applicant', type='primary',
                          use_container_width=True)

# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------

if not submitted:
    st.info('Set the applicant details in the sidebar, then press **Score applicant**.')
    st.stop()

features = {field: st.session_state[field] for field in DEFAULTS}

# A zero income is treated as "not provided", which is what the model's
# income_missing feature is designed to capture.
if features['monthly_income'] == 0:
    features['monthly_income'] = None

with st.spinner('Scoring…'):
    result, error = api_client.predict(
        features,
        include_explanation=True,
        include_narrative=want_narrative,
        applicant_id=st.session_state['loaded_applicant_id'],
    )

if error:
    st.error(error)
    st.stop()

# ---- headline metrics ----

probability = result['probability_default']
band = result['risk_band']
BAND_COLOURS = {'low': '#2e7d32', 'medium': '#f9a825',
                'high': '#ef6c00', 'very_high': '#c62828'}

col1, col2, col3, col4 = st.columns(4)
col1.metric('Probability of default', f"{probability:.1%}")
col2.metric('Risk band', band.replace('_', ' ').title())
col3.metric('Decision at threshold',
            'Refer / decline' if result['predicted_class'] == 1 else 'Within appetite')
col4.metric('Threshold used', f"{result['threshold_used']:.2f}")

st.markdown(
    f"<div style='height:8px;background:{BAND_COLOURS.get(band, '#666')};"
    f"border-radius:4px;margin:0.5rem 0 1.5rem 0;'></div>",
    unsafe_allow_html=True,
)

if st.session_state['loaded_actual_outcome'] is not None:
    actual = st.session_state['loaded_actual_outcome']
    st.caption(
        f"This applicant's recorded outcome was "
        f"**{'default' if actual == 1 else 'no default'}**. "
        f"Shown for inspection only — it was not an input to the model."
    )

# ---- contribution chart ----

contributions = result.get('explanation') or []

if contributions:
    st.subheader('What drove this score')

    chart_data = sorted(contributions, key=lambda c: abs(c['contribution']))
    labels = [c['factor'] for c in chart_data]
    values = [c['contribution'] for c in chart_data]
    colours = ['#c62828' if v > 0 else '#2e7d32' for v in values]
    hover = [
        f"{c['factor']}<br>Applicant value: {c['value']}<br>"
        f"Contribution: {c['contribution']:+.4f}"
        for c in chart_data
    ]

    figure = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker_color=colours, hovertext=hover, hoverinfo='text',
    ))
    figure.update_layout(
        height=max(280, 45 * len(labels)),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title='Contribution to modelled risk (log-odds)',
        yaxis_title=None,
        showlegend=False,
    )
    figure.add_vline(x=0, line_width=1, line_color='#999')
    st.plotly_chart(figure, use_container_width=True)

    st.caption(
        'Red increases modelled risk, green reduces it. Values are SHAP '
        'contributions in log-odds, relative to the average applicant. '
        'These are statistical associations, not causes.'
    )

    table = pd.DataFrame([{
        'Rank': c['rank'],
        'Factor': c['factor'],
        'Applicant value': c['value'],
        'Contribution': round(c['contribution'], 4),
        'Direction': c['direction'].replace('_', ' '),
        'Reason code': c.get('reason_code') or '—',
    } for c in sorted(contributions, key=lambda c: c['rank'])])
    st.dataframe(table, hide_index=True, use_container_width=True)

# ---- narrative ----

narrative = result.get('narrative')

if narrative:
    st.subheader('Written assessment')

    status = narrative['llm_status']
    if status == 'ok':
        st.success('Generated by the language model and validated against the '
                   'model output.')
    elif status == 'fallback':
        st.warning('The language model was unavailable or its output failed '
                   'validation. This is a deterministic template built from the '
                   'same figures.')
    else:
        st.info('Narrative generation is disabled. This is a deterministic '
                'template built from the same figures.')

    st.write(narrative['summary'])
    st.caption(f"Confidence: {narrative['confidence']}")

    with st.expander(f"Key risk factors ({len(narrative['key_risk_factors'])})",
                     expanded=True):
        for factor in narrative['key_risk_factors']:
            code = factor.get('reason_code') or '—'
            st.markdown(f"**{factor['factor']}**  `{code}`")
            st.write(factor['observation'])

    if narrative['mitigating_factors']:
        with st.expander(f"Mitigating factors ({len(narrative['mitigating_factors'])})"):
            for factor in narrative['mitigating_factors']:
                st.markdown(f"**{factor['factor']}**")
                st.write(factor['observation'])

    with st.expander('Data quality notes'):
        for note in narrative['data_quality_notes']:
            st.write(f"- {note}")

    with st.expander('Recommended checks'):
        for check in narrative['recommended_checks']:
            st.write(f"- {check}")

# ---- provenance ----

st.divider()
st.caption(
    f"Model `{result['pipeline_name']}` version `{result['model_version']}` · "
    f"request `{result['request_id']}` · {result['latency_ms']} ms · "
    f"scored at {result['predicted_at']}"
)