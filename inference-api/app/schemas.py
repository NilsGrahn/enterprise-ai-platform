from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class ApplicantFeatures(BaseModel):
    """One applicant.

    Field names are business names; aliases accept the original dataset
    headers so a raw row from cs-training.csv can be posted unchanged.
    """

    revolving_utilisation: float = Field(
        ..., ge=0, alias='RevolvingUtilizationOfUnsecuredLines',
        description='Balance on revolving lines divided by total credit limit',
    )
    age: int = Field(
        ..., ge=18, le=110, alias='age',
        description='Applicant age in years',
    )
    debt_ratio: float = Field(
        ..., ge=0, alias='DebtRatio',
        description='Monthly debt payments divided by monthly gross income',
    )
    monthly_income: float | None = Field(
        None, ge=0, alias='MonthlyIncome',
        description='Null is accepted; the pipeline imputes it and flags the imputation',
    )
    open_credit_lines: int = Field(
        ..., ge=0, alias='NumberOfOpenCreditLinesAndLoans',
        description='Number of open loans and lines of credit',
    )
    real_estate_loans: int = Field(
        ..., ge=0, alias='NumberRealEstateLoansOrLines',
        description='Number of mortgage and real estate loans',
    )
    times_30_59_days_late: int = Field(
        ..., ge=0, alias='NumberOfTime30-59DaysPastDueNotWorse',
    )
    times_60_89_days_late: int = Field(
        ..., ge=0, alias='NumberOfTime60-89DaysPastDueNotWorse',
    )
    times_90_days_late: int = Field(
        ..., ge=0, alias='NumberOfTimes90DaysLate',
    )
    number_of_dependents: int | None = Field(
        None, ge=0, alias='NumberOfDependents',
    )

    model_config = {
        'populate_by_name': True,
        'json_schema_extra': {
            'examples': [{
                'RevolvingUtilizationOfUnsecuredLines': 0.95,
                'age': 32,
                'DebtRatio': 0.9,
                'MonthlyIncome': 2500,
                'NumberOfOpenCreditLinesAndLoans': 12,
                'NumberRealEstateLoansOrLines': 0,
                'NumberOfTime30-59DaysPastDueNotWorse': 3,
                'NumberOfTime60-89DaysPastDueNotWorse': 1,
                'NumberOfTimes90DaysLate': 2,
                'NumberOfDependents': 3,
            }],
        },
    }

    @field_validator('times_30_59_days_late', 'times_60_89_days_late',
                     'times_90_days_late')
    @classmethod
    def reject_sentinel(cls, v):
        if v in (96, 98):
            raise ValueError(
                '96 and 98 are source sentinel codes meaning "unknown", not '
                'counts. Send null or omit the field instead.'
            )
        return v


def to_pipeline_frame(features: 'ApplicantFeatures') -> pd.DataFrame:
    """Convert a validated request into the frame the pipeline expects.

    Derives total_delinquency_events using the same formula as the Phase 2
    ETL (load_fact_assessment.py), because the pipeline requires that column
    but a caller has no reason to supply it.
    """
    row = features.model_dump()

    row['total_delinquency_events'] = (
        (row['times_30_59_days_late'] or 0)
        + (row['times_60_89_days_late'] or 0)
        + (row['times_90_days_late'] or 0)
    )

    return pd.DataFrame([row])


class PredictRequest(BaseModel):
    applicant_id: int | None = None
    features: ApplicantFeatures
    include_explanation: bool = True
    include_narrative: bool = False      # costs an LLM call, so opt in


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, max_length=500)


class ContributionOut(BaseModel):
    factor: str
    value: float
    contribution: float
    direction: Literal['increases_risk', 'decreases_risk']
    reason_code: str | None = None
    rank: int


class NarrativeOut(BaseModel):
    summary: str
    key_risk_factors: list[dict]
    mitigating_factors: list[dict]
    data_quality_notes: list[str]
    recommended_checks: list[str]
    confidence: Literal['high', 'medium', 'low']
    llm_status: Literal['ok', 'fallback', 'disabled']


class PredictResponse(BaseModel):
    request_id: str
    applicant_id: int | None = None
    probability_default: float = Field(..., ge=0, le=1)
    predicted_class: Literal[0, 1]
    risk_band: Literal['low', 'medium', 'high', 'very_high']
    threshold_used: float
    model_version: str
    pipeline_name: str
    explanation: list[ContributionOut] | None = None
    narrative: NarrativeOut | None = None
    latency_ms: int
    predicted_at: datetime


class HealthResponse(BaseModel):
    status: Literal['ok', 'degraded', 'error']
    app_version: str
    pipeline_name: str
    model_version: str
    model_loaded: bool
    database_reachable: bool
    llm_reachable: bool | None = None
    uptime_seconds: float


class MetricsResponse(BaseModel):
    predictions_total: int
    predictions_last_hour: int
    errors_total: int
    latency_p50_ms: float
    latency_p95_ms: float
    average_probability_last_hour: float | None = None
    active_model_version: str
    drift_status: Literal['OK', 'WARN', 'ALERT', 'UNKNOWN']


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str