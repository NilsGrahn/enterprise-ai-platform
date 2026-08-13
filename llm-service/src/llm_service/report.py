import json
import re

from pydantic import BaseModel, Field, ValidationError
from typing import Literal

from llm_service.client import LLMUnavailableError
from llm_service.payload import build_model_output
from llm_service.prompts import SYSTEM_PROMPT, USER_TEMPLATE, RETRY_SUFFIX


class ReportValidationError(Exception):
    """Raised when LLM output fails schema, reason-code or number validation."""


class KeyRiskFactor(BaseModel):
    factor: str
    observation: str
    reason_code: str


class MitigatingFactor(BaseModel):
    factor: str
    observation: str


class CreditReport(BaseModel):
    summary: str
    key_risk_factors: list[KeyRiskFactor] = Field(..., min_length=1, max_length=5)
    mitigating_factors: list[MitigatingFactor] = Field(default_factory=list)
    data_quality_notes: list[str] = Field(default_factory=list)
    recommended_checks: list[str] = Field(default_factory=list)
    confidence: Literal['high', 'medium', 'low']


# Matches integers and decimals, including those inside percentages.
NUMBER_PATTERN = re.compile(r'\d+(?:\.\d+)?')

# Numbers a narrative may legitimately use without them appearing in the payload:
# small ordinals and counts used in prose ("the first factor", "3 accounts").
ALLOWED_INCIDENTAL = {'0', '1', '2', '3', '4', '5'}


def _strip_fences(raw: str) -> str:
    """Remove ```json ... ``` fences the model sometimes adds despite instructions."""
    text = raw.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    return text.strip()


def _payload_numbers(payload: dict) -> set:
    """Every numeric value in the payload, as strings, for membership checks."""
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            found.add(str(node))
            found.add(str(abs(node)))
            # percentage form: 0.4213 -> 42.13, 42.1, 42
            as_pct = abs(node) * 100
            found.update({str(round(as_pct, 2)), str(round(as_pct, 1)),
                          str(int(as_pct)), str(round(abs(node), 2)),
                          str(round(abs(node), 1))})
        elif isinstance(node, str):
            for match in NUMBER_PATTERN.findall(node):
                found.add(match)

    walk(payload)
    return {n.rstrip('0').rstrip('.') if '.' in n else n for n in found} | found


def _validate_reason_codes(report: CreditReport, payload: dict):
    supplied = {
        c['reason_code'] for c in payload['contributions'] if 'reason_code' in c
    }
    used = {f.reason_code for f in report.key_risk_factors}
    fabricated = used - supplied
    if fabricated:
        raise ReportValidationError(
            f"Report used reason code(s) not supplied in the payload: "
            f"{sorted(fabricated)}. Supplied: {sorted(supplied)}."
        )


def _validate_numbers(report: CreditReport, payload: dict):
    allowed = _payload_numbers(payload) | ALLOWED_INCIDENTAL

    text_parts = [report.summary]
    text_parts += [f.observation for f in report.key_risk_factors]
    text_parts += [f.observation for f in report.mitigating_factors]
    text_parts += report.data_quality_notes
    text_parts += report.recommended_checks

    for text in text_parts:
        for raw_number in NUMBER_PATTERN.findall(text):
            normalised = raw_number.rstrip('0').rstrip('.') if '.' in raw_number else raw_number
            if raw_number not in allowed and normalised not in allowed:
                raise ReportValidationError(
                    f"Report contains the number '{raw_number}', which does not "
                    f"appear in the model output payload. Text: \"{text[:120]}\""
                )


def _template_report(payload: dict, status: str) -> dict:
    """Deterministic fallback. No LLM involved."""
    prediction = payload['prediction']
    risk_factors = [c for c in payload['contributions']
                    if c['direction'] == 'increases_risk']
    mitigating = [c for c in payload['contributions']
                  if c['direction'] == 'decreases_risk']
    imputed = [k for k, v in payload['data_quality'].items() if v]

    return {
        'summary': (
            f"Modelled probability of default is "
            f"{prediction['probability_of_default']}, placing this application in the "
            f"{prediction['risk_band']} risk band against a decision threshold of "
            f"{prediction['threshold']}. The portfolio average is "
            f"{payload['population_baseline']['average_probability_of_default']}. "
            f"This note was generated without narrative language support."
        ),
        'key_risk_factors': [
            {
                'factor': c['factor'],
                'observation': (
                    f"Recorded value {c['value']}; contribution to modelled risk "
                    f"{c['contribution']}."
                ),
                'reason_code': c.get('reason_code', ''),
            }
            for c in risk_factors[:5]
        ],
        'mitigating_factors': [
            {
                'factor': c['factor'],
                'observation': (
                    f"Recorded value {c['value']}; contribution to modelled risk "
                    f"{c['contribution']}."
                ),
            }
            for c in mitigating
        ],
        'data_quality_notes': (
            [f"Field flagged as imputed or missing: {k}." for k in imputed]
            or ["No data quality flags were raised for this application."]
        ),
        'recommended_checks': [
            "Verify reported income against documentary evidence.",
            "Confirm the delinquency history against the bureau record.",
            "Review current outstanding balances across all open accounts.",
        ],
        'confidence': 'low',
        'llm_status': status,
    }


def generate_credit_report(explanation, risk_band, threshold, dq_flags, client) -> dict:
    """Produce a validated narrative report. Always returns a usable dict.

    'llm_status' is one of:
      'ok'       - the model produced output that passed every check
      'fallback' - the model failed or produced invalid output; template used
      'disabled' - the LLM is switched off; template used
    """
    payload = build_model_output(explanation, risk_band, threshold, dq_flags)

    if not client.enabled:
        return _template_report(payload, 'disabled')

    user_prompt = USER_TEMPLATE.format(
        model_output_json=json.dumps(payload, indent=2)
    )

    last_error = None
    for attempt in range(2):                       # initial call, then one retry
        prompt = user_prompt
        if attempt == 1:
            prompt = user_prompt + RETRY_SUFFIX.format(error=last_error)

        try:
            raw = client.generate(SYSTEM_PROMPT, prompt)
        except LLMUnavailableError as exc:
            last_error = str(exc)
            break                                  # transport failure: do not retry here

        try:
            parsed = json.loads(_strip_fences(raw))
            report = CreditReport(**parsed)
            _validate_reason_codes(report, payload)
            _validate_numbers(report, payload)
        except (json.JSONDecodeError, ValidationError, ReportValidationError) as exc:
            last_error = str(exc)
            continue

        result = report.model_dump()
        result['llm_status'] = 'ok'
        return result

    print(f"[llm] falling back to template report: {last_error}")
    return _template_report(payload, 'fallback')