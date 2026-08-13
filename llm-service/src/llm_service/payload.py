from llm_service.reason_codes import code_for

POPULATION_BASELINE_DEFAULT_RATE = 0.0668


def build_model_output(explanation, risk_band, threshold, dq_flags,
                       algorithm='xgboost',
                       population_baseline=POPULATION_BASELINE_DEFAULT_RATE) -> dict:
    """Build the ONLY structure the LLM is permitted to see.

    No raw dataframe, no PII, no free text from the request.
    """
    contributions = []
    for c in explanation.contributions:
        entry = {
            'factor': c.display_name,
            'value': round(float(c.value), 4),
            'contribution': round(float(c.shap_value), 4),
            'direction': c.direction,
            'rank': c.rank,
        }
        if c.direction == 'increases_risk':
            code, _ = code_for(c.feature)
            entry['reason_code'] = code
        contributions.append(entry)

    return {
        'request_id': explanation.request_id,
        'model': {
            'pipeline': explanation.pipeline_name,
            'version': explanation.model_version,
            'algorithm': algorithm,
        },
        'prediction': {
            'probability_of_default': round(float(explanation.probability), 4),
            'risk_band': risk_band,
            'threshold': round(float(threshold), 4),
        },
        'population_baseline': {
            'average_probability_of_default': round(float(population_baseline), 4),
        },
        'contributions': contributions,
        'data_quality': {k: bool(v) for k, v in dq_flags.items()},
    }