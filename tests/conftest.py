"""Shared fixtures.

No test in this suite touches a live database or a real LLM. Anything that
would normally do so is either synthetic or mocked.
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_raw_df():
    """20 rows in bronze shape (all strings), covering every dq edge case.

    Rows 0-14 are ordinary. Rows 15-19 each trigger one data quality rule.
    """
    rows = []

    rng = np.random.default_rng(42)
    for i in range(15):
        rows.append({
            'bronze_id': i + 1,
            'raw_row_id': str(i + 1),
            'serious_dlqin_2yrs': str(int(rng.random() < 0.2)),
            'revolving_utilization_unsecured_lines': f"{rng.random():.4f}",
            'age': str(int(rng.integers(25, 70))),
            'times_30_59_days_past_due': str(int(rng.integers(0, 3))),
            'debt_ratio': f"{rng.random():.4f}",
            'monthly_income': f"{rng.integers(2000, 9000)}",
            'open_credit_lines_and_loans': str(int(rng.integers(2, 15))),
            'times_90_days_late': str(int(rng.integers(0, 2))),
            'real_estate_loans_or_lines': str(int(rng.integers(0, 3))),
            'times_60_89_days_past_due': str(int(rng.integers(0, 2))),
            'number_of_dependents': str(int(rng.integers(0, 4))),
        })

    def base(bronze_id, raw_row_id):
        return {
            'bronze_id': bronze_id,
            'raw_row_id': str(raw_row_id),
            'serious_dlqin_2yrs': '0',
            'revolving_utilization_unsecured_lines': '0.3',
            'age': '40',
            'times_30_59_days_past_due': '0',
            'debt_ratio': '0.4',
            'monthly_income': '5000',
            'open_credit_lines_and_loans': '6',
            'times_90_days_late': '0',
            'real_estate_loans_or_lines': '1',
            'times_60_89_days_past_due': '0',
            'number_of_dependents': '1',
        }

    # 15: missing income
    row = base(16, 16); row['monthly_income'] = ''
    rows.append(row)

    # 16: missing dependents
    row = base(17, 17); row['number_of_dependents'] = ''
    rows.append(row)

    # 17: impossible age
    row = base(18, 18); row['age'] = '0'
    rows.append(row)

    # 18: absurd utilisation
    row = base(19, 19); row['revolving_utilization_unsecured_lines'] = '50000'
    rows.append(row)

    # 19: sentinel code in a late-payment count
    row = base(20, 20); row['times_90_days_late'] = '98'
    rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def sample_gold_df():
    """200 rows in gold shape (already typed), suitable for training."""
    rng = np.random.default_rng(7)
    n = 200

    utilisation = rng.random(n)
    late_30 = rng.integers(0, 4, n)
    late_60 = rng.integers(0, 3, n)
    late_90 = rng.integers(0, 3, n)
    income = rng.integers(1500, 12000, n).astype(float)

    # A deliberate, learnable signal so the model has something to fit.
    risk = 0.35 * utilisation + 0.12 * late_90 + 0.08 * late_60
    target = (risk + rng.normal(0, 0.15, n) > 0.45).astype(int)

    df = pd.DataFrame({
        'applicant_id': np.arange(1, n + 1),
        'snapshot_date_key': 20260101,
        'dataset_split': ['train'] * 140 + ['valid'] * 30 + ['test'] * 30,
        'is_serious_delinquency': target,
        'revolving_utilisation': utilisation,
        'debt_ratio': rng.random(n),
        'monthly_income': income,
        'open_credit_lines': rng.integers(1, 20, n),
        'real_estate_loans': rng.integers(0, 4, n),
        'times_30_59_days_late': late_30,
        'times_60_89_days_late': late_60,
        'times_90_days_late': late_90,
        'total_delinquency_events': late_30 + late_60 + late_90,
        'age': rng.integers(21, 75, n),
        'number_of_dependents': rng.integers(0, 5, n),
    })

    # Some missing income, so the imputation path is exercised.
    df.loc[df.sample(20, random_state=3).index, 'monthly_income'] = np.nan
    return df


@pytest.fixture
def single_applicant_df(sample_gold_df):
    """One row, for prediction and explanation tests."""
    return sample_gold_df.head(1).copy()


# ---------------------------------------------------------------------------
# A trained pipeline, without a database or a saved artifact
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def trained_pipeline():
    """A CreditPipeline fitted on synthetic data.

    session scope: trained once for the whole run. Bypasses load_data() so no
    database is needed, and does not touch the real artifact.
    """
    import numpy as np
    import pandas as pd
    from ml_service.pipelines import get_pipeline

    rng = np.random.default_rng(7)
    n = 200
    utilisation = rng.random(n)
    late_30 = rng.integers(0, 4, n)
    late_60 = rng.integers(0, 3, n)
    late_90 = rng.integers(0, 3, n)
    risk = 0.35 * utilisation + 0.12 * late_90 + 0.08 * late_60
    target = (risk + rng.normal(0, 0.15, n) > 0.45).astype(int)

    df = pd.DataFrame({
        'is_serious_delinquency': target,
        'revolving_utilisation': utilisation,
        'debt_ratio': rng.random(n),
        'monthly_income': rng.integers(1500, 12000, n).astype(float),
        'open_credit_lines': rng.integers(1, 20, n),
        'real_estate_loans': rng.integers(0, 4, n),
        'times_30_59_days_late': late_30,
        'times_60_89_days_late': late_60,
        'times_90_days_late': late_90,
        'total_delinquency_events': late_30 + late_60 + late_90,
        'age': rng.integers(21, 75, n),
        'number_of_dependents': rng.integers(0, 5, n),
    })
    df.loc[df.sample(20, random_state=3).index, 'monthly_income'] = np.nan

    train = df.iloc[:150]
    valid = df.iloc[150:]

    pipeline = get_pipeline('credit')

    train_clean = pipeline.clean(train)
    valid_clean = pipeline.clean(valid)

    y_train = train_clean['is_serious_delinquency']
    y_valid = valid_clean['is_serious_delinquency']

    X_train, preprocessing = pipeline.feature_engineering(train_clean, fit=True)
    pipeline.preprocessing = preprocessing
    X_valid, _ = pipeline.feature_engineering(valid_clean, fit=False)

    result = pipeline.train(X_train, y_train, X_valid, y_valid)
    pipeline.model = result.model

    return pipeline


@pytest.fixture(scope='session')
def fake_metadata(trained_pipeline):
    """Metadata in the shape load_artifact() would return."""
    return {
        'pipeline_name': 'credit',
        'model_version': 'test',
        'trained_at': '2026-01-01T00:00:00+00:00',
        'algorithm': 'XGBClassifier',
        'feature_names': trained_pipeline.preprocessing['feature_names'],
        'preprocessing': trained_pipeline.preprocessing,
        'metrics': {'auc': 0.9, 'ks': 0.6},
        'training_rows': 150,
    }


@pytest.fixture(scope='session')
def explainer(trained_pipeline, fake_metadata):
    from explain_service.explainer import ShapTreeExplainer
    return ShapTreeExplainer(trained_pipeline, fake_metadata)


# ---------------------------------------------------------------------------
# LLM stand-ins
# ---------------------------------------------------------------------------

@pytest.fixture
def null_llm():
    from llm_service.client import NullLLMClient
    return NullLLMClient()


@pytest.fixture
def sample_explanation():
    """A hand-built ExplanationResult — no model needed."""
    from explain_service.schema import ExplanationResult, FeatureContribution

    contributions = [
        FeatureContribution('revolving_utilisation', 'Credit utilisation ratio',
                            0.95, 0.62, 'increases_risk', 1),
        FeatureContribution('times_90_days_late', 'Payments 90+ days late',
                            2.0, 0.41, 'increases_risk', 2),
        FeatureContribution('debt_ratio', 'Debt-to-income ratio',
                            0.89, 0.22, 'increases_risk', 3),
        FeatureContribution('monthly_income', 'Monthly income',
                            2500.0, -0.13, 'decreases_risk', 4),
    ]
    return ExplanationResult(
        request_id='test-request',
        probability=0.4213,
        base_value=-2.1,
        contributions=contributions,
        model_version='test',
        pipeline_name='credit',
        all_contributions=contributions,
    )


# ---------------------------------------------------------------------------
# API test client
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client(trained_pipeline, fake_metadata, explainer, null_llm, monkeypatch):
    """A TestClient with MODEL_STORE populated by the synthetic pipeline.

    No uvicorn, no network, no artifact on disk, no database.
    """
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        'monitoring.prediction_logger.log_prediction',
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        'monitoring.prediction_logger.log_service_event',
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        'app.routers.predict.log_prediction',
        lambda *args, **kwargs: True,
    )

    from app import main as app_main

    with TestClient(app_main.app) as client:
        # Patch AFTER lifespan has already loaded the real model,
        # so this overrides it rather than being overwritten by it.
        monkeypatch.setattr(app_main.MODEL_STORE, 'pipeline', trained_pipeline)
        monkeypatch.setattr(app_main.MODEL_STORE, 'metadata', fake_metadata)
        monkeypatch.setattr(app_main.MODEL_STORE, 'explainer', explainer)
        monkeypatch.setattr(app_main.MODEL_STORE, 'llm_client', null_llm)
        monkeypatch.setattr(app_main.MODEL_STORE, 'load_error', None)
        monkeypatch.setattr(app_main.MODEL_STORE, 'model_key', 1)
        yield client


@pytest.fixture
def api_client_no_model(monkeypatch):
    from app import main as app_main
    from fastapi.testclient import TestClient

    with TestClient(app_main.app) as client:
        # Patch AFTER lifespan has already loaded the real model,
        # so this overrides it rather than being overwritten by it.
        monkeypatch.setattr(app_main.MODEL_STORE, 'pipeline', None)
        monkeypatch.setattr(app_main.MODEL_STORE, 'metadata', None)
        monkeypatch.setattr(app_main.MODEL_STORE, 'load_error', 'simulated load failure')
        yield client


@pytest.fixture
def valid_request_body():
    return {
        'applicant_id': 1,
        'include_explanation': True,
        'include_narrative': False,
        'features': {
            'RevolvingUtilizationOfUnsecuredLines': 0.5,
            'age': 40,
            'DebtRatio': 0.4,
            'MonthlyIncome': 5000,
            'NumberOfOpenCreditLinesAndLoans': 6,
            'NumberRealEstateLoansOrLines': 1,
            'NumberOfTime30-59DaysPastDueNotWorse': 0,
            'NumberOfTime60-89DaysPastDueNotWorse': 0,
            'NumberOfTimes90DaysLate': 0,
            'NumberOfDependents': 1,
        },
    }