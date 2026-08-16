import os

import requests

BASE_URL = os.getenv('INFERENCE_API_URL', 'http://localhost:8000')
TIMEOUT_SECONDS = 30

# Business field name -> the alias the API expects.
# Recall Phase 5: ApplicantFeatures accepts the original dataset headers.
FIELD_ALIASES = {
    'revolving_utilisation': 'RevolvingUtilizationOfUnsecuredLines',
    'age': 'age',
    'debt_ratio': 'DebtRatio',
    'monthly_income': 'MonthlyIncome',
    'open_credit_lines': 'NumberOfOpenCreditLinesAndLoans',
    'real_estate_loans': 'NumberRealEstateLoansOrLines',
    'times_30_59_days_late': 'NumberOfTime30-59DaysPastDueNotWorse',
    'times_60_89_days_late': 'NumberOfTime60-89DaysPastDueNotWorse',
    'times_90_days_late': 'NumberOfTimes90DaysLate',
    'number_of_dependents': 'NumberOfDependents',
}


def _friendly_error(exc) -> str:
    """Turn an exception into something a non-developer can act on."""
    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            f"Could not reach the inference API at {BASE_URL}. "
            f"Is it running? Start it with:  cd inference-api && "
            f"uvicorn app.main:app --port 8000"
        )
    if isinstance(exc, requests.exceptions.Timeout):
        return f"The inference API did not respond within {TIMEOUT_SECONDS} seconds."
    return f"Unexpected error contacting the API: {exc}"


def to_api_features(values: dict) -> dict:
    """Convert business field names to the aliases the API expects."""
    return {
        alias: values.get(field)
        for field, alias in FIELD_ALIASES.items()
    }


def predict(features: dict, include_explanation=True, include_narrative=False,
            applicant_id=None):
    """POST /predict. Returns (response_dict, error_message)."""
    payload = {
        'applicant_id': applicant_id,
        'features': to_api_features(features),
        'include_explanation': include_explanation,
        'include_narrative': include_narrative,
    }

    try:
        response = requests.post(
            f'{BASE_URL}/predict', json=payload, timeout=TIMEOUT_SECONDS
        )
    except Exception as exc:
        return None, _friendly_error(exc)

    if response.status_code == 200:
        return response.json(), None

    if response.status_code == 422:
        try:
            detail = response.json().get('detail', response.text)
        except ValueError:
            detail = response.text
        return None, f"The applicant details were rejected as invalid: {detail}"

    if response.status_code == 503:
        return None, (
            "The inference API is running but has no model loaded. "
            "Train and activate one:  python -m ml_service.train --activate"
        )

    return None, f"API returned HTTP {response.status_code}: {response.text[:300]}"


def health():
    """GET /health. Returns (response_dict, error_message)."""
    try:
        response = requests.get(f'{BASE_URL}/health', timeout=5)
        return response.json(), None
    except Exception as exc:
        return None, _friendly_error(exc)


def metrics():
    """GET /metrics. Returns (response_dict, error_message)."""
    try:
        response = requests.get(f'{BASE_URL}/metrics', timeout=10)
        if response.status_code != 200:
            return None, f"API returned HTTP {response.status_code}"
        return response.json(), None
    except Exception as exc:
        return None, _friendly_error(exc)