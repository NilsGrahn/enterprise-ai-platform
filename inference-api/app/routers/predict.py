from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

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
from fastapi import APIRouter, HTTPException
from llm_service.report import generate_credit_report
from monitoring.prediction_logger import log_prediction

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


def _score_one(request: PredictRequest, request_id: str):
    """Returns (response, feature_frame). The frame is logged by the caller."""
    started = perf_counter()

    df = to_pipeline_frame(request.features)

    # What the model actually consumes — logged so drift is measured on the
    # engineered features, not the raw request.
    feature_frame = MODEL_STORE.pipeline.build_feature_frame(df)

    explanation = None
    if request.include_explanation or request.include_narrative:
        explanation = MODEL_STORE.explainer.explain(df, request_id=request_id)
        probability = explanation.probability
    else:
        probability = float(
            MODEL_STORE.pipeline.model.predict_proba(feature_frame)[:, 1][0]
        )

    threshold = settings.decision_threshold
    predicted_class = int(probability >= threshold)
    risk_band = band_for(probability)

    narrative = None
    if request.include_narrative:
        report = generate_credit_report(
            explanation, risk_band, threshold,
            _dq_flags(request.features), MODEL_STORE.llm_client,
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

    response = PredictResponse(
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

    return response, feature_frame


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

    try:
        response, feature_frame = _score_one(request, request_id)
    except Exception as exc:
        log_prediction(
            request_id=request_id,
            pipeline_name=MODEL_STORE.metadata['pipeline_name'],
            model_version=MODEL_STORE.metadata['model_version'],
            request_payload=request.features.model_dump(),
            feature_vector={},
            status='ERROR',
            error_message=str(exc),
        )
        raise


    log_prediction(
        request_id=request_id,
        pipeline_name=response.pipeline_name,
        model_version=response.model_version,
        request_payload=request.features.model_dump(),
        feature_vector=feature_frame.iloc[0].to_dict(),
        probability=response.probability_default,
        predicted_class=response.predicted_class,
        risk_band=response.risk_band,
        latency_ms=response.latency_ms,
        status='OK',
    )

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
        request_id = str(uuid4())
        response, feature_frame = _score_one(item, request_id)
        log_prediction(
            request_id=request_id,
            pipeline_name=response.pipeline_name,
            model_version=response.model_version,
            request_payload=item.features.model_dump(),
            feature_vector=feature_frame.iloc[0].to_dict(),
            probability=response.probability_default,
            predicted_class=response.predicted_class,
            risk_band=response.risk_band,
            latency_ms=response.latency_ms,
            status='OK',
        )
        results.append(response)
    return results