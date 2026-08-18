"""The Phase 4 LLM guardrails.

No test here makes a network call. The client is either the null client or a
mock returning a canned string.
"""

import json
from unittest.mock import MagicMock

import pytest
from llm_service.client import LLMUnavailableError
from llm_service.payload import build_model_output
from llm_service.report import (
    CreditReport,
    ReportValidationError,
    _validate_numbers,
    _validate_reason_codes,
    generate_credit_report,
)


@pytest.fixture
def payload(sample_explanation):
    return build_model_output(sample_explanation, 'very_high', 0.5,
                              {'monthly_income_imputed': False,
                               'dependents_missing': True})


def good_report_dict(payload):
    """A response that should pass every check."""
    return {
        'summary': (
            f"Modelled probability of default is "
            f"{payload['prediction']['probability_of_default']}, against a "
            f"threshold of {payload['prediction']['threshold']}."
        ),
        'key_risk_factors': [
            {'factor': 'Credit utilisation ratio',
             'observation': 'Utilisation is elevated.',
             'reason_code': 'R01'},
            {'factor': 'Payments 90+ days late',
             'observation': 'A history of serious delinquency is present.',
             'reason_code': 'R02'},
            {'factor': 'Debt-to-income ratio',
             'observation': 'Existing debt is high relative to income.',
             'reason_code': 'R03'},
        ],
        'mitigating_factors': [
            {'factor': 'Monthly income', 'observation': 'Income is recorded.'},
        ],
        'data_quality_notes': ['Dependents were not supplied.'],
        'recommended_checks': ['Verify income against documentation.'],
        'confidence': 'medium',
    }


class TestPayload:

    def test_contains_only_the_expected_keys(self, payload):
        assert set(payload) == {
            'request_id', 'model', 'prediction', 'population_baseline',
            'contributions', 'data_quality',
        }

    def test_no_personal_data_reaches_the_llm(self, payload):
        text = json.dumps(payload).lower()
        for forbidden in ('applicant_id', 'name', 'address', 'ssn'):
            assert forbidden not in text

    def test_reason_codes_only_on_risk_increasing_factors(self, payload):
        for c in payload['contributions']:
            if c['direction'] == 'increases_risk':
                assert 'reason_code' in c
            else:
                assert 'reason_code' not in c


class TestDisabledLLM:

    def test_returns_a_template_with_disabled_status(self, sample_explanation,
                                                     null_llm):
        report = generate_credit_report(sample_explanation, 'very_high', 0.5,
                                        {'monthly_income_imputed': True}, null_llm)
        assert report['llm_status'] == 'disabled'
        assert report['confidence'] == 'low'

    def test_template_validates_against_the_schema(self, sample_explanation,
                                                   null_llm):
        report = generate_credit_report(sample_explanation, 'very_high', 0.5,
                                        {}, null_llm)
        CreditReport(**{k: v for k, v in report.items() if k != 'llm_status'})

    def test_template_mentions_imputed_fields(self, sample_explanation, null_llm):
        report = generate_credit_report(sample_explanation, 'high', 0.5,
                                        {'monthly_income_imputed': True}, null_llm)
        notes = ' '.join(report['data_quality_notes']).lower()
        assert 'monthly_income_imputed' in notes or 'income' in notes


class TestReasonCodeValidation:

    def test_fabricated_code_is_rejected(self, payload):
        bad = good_report_dict(payload)
        bad['key_risk_factors'][0]['reason_code'] = 'R99'
        with pytest.raises(ReportValidationError) as exc:
            _validate_reason_codes(CreditReport(**bad), payload)
        assert 'R99' in str(exc.value)

    def test_supplied_codes_are_accepted(self, payload):
        _validate_reason_codes(CreditReport(**good_report_dict(payload)), payload)


class TestNumberValidation:

    def test_fabricated_statistic_is_rejected(self, payload):
        bad = good_report_dict(payload)
        bad['summary'] = 'Default probability is 87.5 percent, far above average.'
        with pytest.raises(ReportValidationError):
            _validate_numbers(CreditReport(**bad), payload)

    def test_numbers_from_the_payload_are_accepted(self, payload):
        _validate_numbers(CreditReport(**good_report_dict(payload)), payload)

    def test_percentage_form_is_accepted(self, payload):
        report = good_report_dict(payload)
        as_pct = round(payload['prediction']['probability_of_default'] * 100, 2)
        report['summary'] = f"The modelled probability is {as_pct} percent."
        _validate_numbers(CreditReport(**report), payload)

    def test_fabricated_number_in_an_observation_is_rejected(self, payload):
        bad = good_report_dict(payload)
        bad['key_risk_factors'][0]['observation'] = 'Utilisation is 73.4 percent.'
        with pytest.raises(ReportValidationError):
            _validate_numbers(CreditReport(**bad), payload)


class TestMockedLLM:
    """A mock client — never a real API call."""

    @staticmethod
    def client_returning(text):
        client = MagicMock()
        client.enabled = True
        client.generate.return_value = text
        return client

    def test_valid_response_gives_ok_status(self, sample_explanation, payload):
        client = self.client_returning(json.dumps(good_report_dict(payload)))
        report = generate_credit_report(sample_explanation, 'very_high', 0.5,
                                        {'monthly_income_imputed': False,
                                         'dependents_missing': True}, client)
        assert report['llm_status'] == 'ok'
        assert client.generate.call_count == 1

    def test_code_fences_are_stripped(self, sample_explanation, payload):
        fenced = '```json\n' + json.dumps(good_report_dict(payload)) + '\n```'
        client = self.client_returning(fenced)
        report = generate_credit_report(sample_explanation, 'very_high', 0.5,
                                        {'monthly_income_imputed': False,
                                         'dependents_missing': True}, client)
        assert report['llm_status'] == 'ok'

    def test_malformed_json_retries_once_then_falls_back(self, sample_explanation):
        client = self.client_returning('this is not json at all')
        report = generate_credit_report(sample_explanation, 'very_high', 0.5,
                                        {}, client)
        assert report['llm_status'] == 'fallback'
        assert client.generate.call_count == 2, "one retry, then give up"

    def test_fabricated_code_falls_back(self, sample_explanation, payload):
        bad = good_report_dict(payload)
        bad['key_risk_factors'][0]['reason_code'] = 'R99'
        client = self.client_returning(json.dumps(bad))
        report = generate_credit_report(sample_explanation, 'very_high', 0.5,
                                        {'monthly_income_imputed': False,
                                         'dependents_missing': True}, client)
        assert report['llm_status'] == 'fallback'

    def test_transport_failure_falls_back_without_retrying(self, sample_explanation):
        client = MagicMock()
        client.enabled = True
        client.generate.side_effect = LLMUnavailableError('connection refused')
        report = generate_credit_report(sample_explanation, 'very_high', 0.5,
                                        {}, client)
        assert report['llm_status'] == 'fallback'
        assert client.generate.call_count == 1, \
            "the client already retried internally"


class TestCircuitBreaker:

    def test_opens_after_repeated_failures(self):
        from llm_service.client import CircuitBreaker
        breaker = CircuitBreaker(threshold=3)
        assert not breaker.is_open
        for _ in range(3):
            breaker.record_failure()
        assert breaker.is_open

    def test_success_resets_the_count(self):
        from llm_service.client import CircuitBreaker
        breaker = CircuitBreaker(threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        assert not breaker.is_open