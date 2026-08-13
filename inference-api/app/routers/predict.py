from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from llm_service.report import generate_credit_report

from app.model_store import MODEL_STORE
from app.schemas import (
    BatchPredictRequest,
    ContributionOut,
    ErrorResponse,
    NarrativeOut,
    PredictRequest,
    PredictResponse,
    to_pipeline_frame,
)
from app.settings import get_settings

router = APIRouter(tags=['prediction'])
settings = get_settings()


def band_for(probability: float) -> str:
    """Map a probability to a risk band. Boundaries are a policy choice."""
    if probability < 0.05:
        return 'low'
    if probability < 0.15:
        return 'medium'
    if probability < 0.35:
        return 'high'
    return 'very_high'


def _dq_flags(features) -> dict:
    """Data quality flags for the LLM payload."""
    return {
        'monthly_income_imputed': features.monthly_income is None,
        'dependents_missing': features.number_of_dependents is None,
    }


def _score_one(request: PredictRequest, request_id: str) -> PredictResponse:
    started = perf_counter()

    df = to_pipeline_frame(request.features)

    explanation = None
    if request.include_explanation or request.include_narrative:
        explanation = MODEL_STORE.explainer.explain(df, request_id=request_id)
        probability = explanation.probability
    else:
        probability = float(MODEL_STORE.pipeline.predict(df)[0])

    threshold = settings.decision_threshold
    predicted_class = int(probability >= threshold)
    risk_band = band_for(probability)

    narrative = None
    if request.include_narrative:
        report = generate_credit_report(
            explanation,
            risk_band,
            threshold,
            _dq_flags(request.features),
            MODEL_STORE.llm_client,
        )
        narrative = NarrativeOut(**report)

    contributions = None
    if request.include_explanation and explanation is not None:
        from llm_service.reason_codes import REASON_CODES
        contributions = [
            ContributionOut(
                factor=c.display_name,
                value=round(float(c.value), 4),
                contribution=round(float(c.shap_value), 4),
                direction=c.direction,
                reason_code=(REASON_CODES[c.feature][0]
                             if c.direction == 'increases_risk'
                             and c.feature in REASON_CODES else None),
                rank=c.rank,
            )
            for c in explanation.contributions
        ]

    latency_ms = int((perf_counter() - started) * 1000)

    return PredictResponse(
        request_id=request_id,
        applicant_id=request.applicant_id,
        probability_default=probability,
        predicted_class=predicted_class,
        risk_band=risk_band,
        threshold_used=threshold,
        model_version=MODEL_STORE.metadata['model_version'],
        pipeline_name=MODEL_STORE.metadata['pipeline_name'],
        explanation=contributions,
        narrative=narrative,
        latency_ms=latency_ms,
        predicted_at=datetime.now(timezone.utc),
    )


@router.post(
    '/predict',
    response_model=PredictResponse,
    status_code=200,
    responses={422: {'model': ErrorResponse}, 503: {'model': ErrorResponse}},
    summary='Score one applicant',
)
def predict(request: PredictRequest):
    if not MODEL_STORE.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {MODEL_STORE.load_error or 'unknown reason'}",
        )

    request_id = str(uuid4())
    response = _score_one(request, request_id)

    # Phase 7 wires prediction logging in here. It must never raise into the
    # response — see monitoring/src/monitoring/prediction_logger.py.

    return response


@router.post(
    '/predict/batch',
    response_model=list[PredictResponse],
    responses={422: {'model': ErrorResponse}, 503: {'model': ErrorResponse}},
    summary='Score up to 500 applicants',
)
def predict_batch(request: BatchPredictRequest):
    if not MODEL_STORE.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {MODEL_STORE.load_error or 'unknown reason'}",
        )

    results = []
    for item in request.items:
        # Explanations are expensive; default them off for batch work unless
        # the caller explicitly asked per item.
        results.append(_score_one(item, str(uuid4())))
    return results