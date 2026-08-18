"""API endpoints. No uvicorn, no network, no database."""

import pytest


class TestHealth:

    def test_health_returns_the_expected_shape(self, api_client):
        response = api_client.get('/health')
        assert response.status_code == 200
        body = response.json()
        for key in ('status', 'app_version', 'pipeline_name', 'model_version',
                    'model_loaded', 'database_reachable', 'uptime_seconds'):
            assert key in body

    def test_model_loaded_is_true(self, api_client):
        assert api_client.get('/health').json()['model_loaded'] is True

    def test_liveness_checks_nothing_else(self, api_client):
        response = api_client.get('/health/live')
        assert response.status_code == 200
        assert response.json()['status'] == 'alive'

    def test_liveness_works_without_a_model(self, api_client_no_model):
        """Liveness must not depend on the model — otherwise a load failure
        would trigger endless restarts."""
        assert api_client_no_model.get('/health/live').status_code == 200

    def test_health_reports_error_without_a_model(self, api_client_no_model):
        response = api_client_no_model.get('/health')
        assert response.status_code == 503
        assert response.json()['model_loaded'] is False


class TestPredict:

    def test_valid_request_succeeds(self, api_client, valid_request_body):
        response = api_client.post('/predict', json=valid_request_body)
        assert response.status_code == 200
        body = response.json()
        assert 0 <= body['probability_default'] <= 1
        assert body['predicted_class'] in (0, 1)
        assert body['risk_band'] in ('low', 'medium', 'high', 'very_high')

    def test_response_carries_provenance(self, api_client, valid_request_body):
        body = api_client.post('/predict', json=valid_request_body).json()
        assert body['request_id']
        assert body['model_version']
        assert body['pipeline_name']
        assert body['latency_ms'] >= 0

    def test_explanation_is_returned_when_requested(self, api_client,
                                                    valid_request_body):
        body = api_client.post('/predict', json=valid_request_body).json()
        assert body['explanation'] is not None
        assert len(body['explanation']) > 0
        first = body['explanation'][0]
        assert first['rank'] == 1
        assert first['direction'] in ('increases_risk', 'decreases_risk')

    def test_explanation_is_omitted_when_not_requested(self, api_client,
                                                       valid_request_body):
        body = dict(valid_request_body, include_explanation=False)
        assert api_client.post('/predict', json=body).json()['explanation'] is None

    def test_narrative_is_off_by_default(self, api_client, valid_request_body):
        body = dict(valid_request_body)
        body.pop('include_narrative')
        assert api_client.post('/predict', json=body).json()['narrative'] is None

    def test_narrative_is_returned_when_requested(self, api_client,
                                                  valid_request_body):
        body = dict(valid_request_body, include_narrative=True)
        narrative = api_client.post('/predict', json=body).json()['narrative']
        assert narrative is not None
        assert narrative['llm_status'] == 'disabled'   # null client in tests

    def test_missing_income_is_accepted(self, api_client, valid_request_body):
        body = dict(valid_request_body)
        body['features'] = dict(body['features'], MonthlyIncome=None)
        assert api_client.post('/predict', json=body).status_code == 200

    def test_service_is_unavailable_without_a_model(self, api_client_no_model,
                                                    valid_request_body):
        assert api_client_no_model.post('/predict',
                                        json=valid_request_body).status_code == 503


class TestValidation:

    @pytest.mark.parametrize('field,value', [
        ('age', 12),
        ('age', 200),
        ('RevolvingUtilizationOfUnsecuredLines', -0.5),
        ('MonthlyIncome', -1000),
        ('NumberOfOpenCreditLinesAndLoans', -1),
    ])
    def test_out_of_range_values_are_rejected(self, api_client, valid_request_body,
                                              field, value):
        body = dict(valid_request_body)
        body['features'] = dict(body['features'], **{field: value})
        assert api_client.post('/predict', json=body).status_code == 422

    @pytest.mark.parametrize('field', [
        'NumberOfTimes90DaysLate',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfTime30-59DaysPastDueNotWorse',
    ])
    @pytest.mark.parametrize('sentinel', [96, 98])
    def test_sentinel_codes_are_rejected(self, api_client, valid_request_body,
                                         field, sentinel):
        body = dict(valid_request_body)
        body['features'] = dict(body['features'], **{field: sentinel})
        assert api_client.post('/predict', json=body).status_code == 422

    def test_missing_required_field_is_rejected(self, api_client,
                                                valid_request_body):
        body = dict(valid_request_body)
        features = dict(body['features'])
        del features['age']
        body['features'] = features
        assert api_client.post('/predict', json=body).status_code == 422

    def test_business_field_names_also_work(self, api_client):
        """populate_by_name means both spellings are accepted."""
        body = {
            'include_explanation': False,
            'features': {
                'revolving_utilisation': 0.5, 'age': 40, 'debt_ratio': 0.4,
                'monthly_income': 5000, 'open_credit_lines': 6,
                'real_estate_loans': 1, 'times_30_59_days_late': 0,
                'times_60_89_days_late': 0, 'times_90_days_late': 0,
                'number_of_dependents': 1,
            },
        }
        assert api_client.post('/predict', json=body).status_code == 200


class TestBatch:

    def test_batch_returns_one_result_per_item(self, api_client,
                                               valid_request_body):
        body = {'items': [valid_request_body] * 3}
        response = api_client.post('/predict/batch', json=body)
        assert response.status_code == 200
        assert len(response.json()) == 3

    def test_oversized_batch_is_rejected(self, api_client, valid_request_body):
        body = {'items': [valid_request_body] * 501}
        assert api_client.post('/predict/batch', json=body).status_code == 422

    def test_empty_batch_is_rejected(self, api_client):
        assert api_client.post('/predict/batch', json={'items': []}).status_code == 422


class TestMiddleware:

    def test_request_id_header_is_returned(self, api_client):
        assert 'x-request-id' in api_client.get('/health').headers

    def test_supplied_request_id_is_reused(self, api_client):
        response = api_client.get('/health',
                                  headers={'X-Request-ID': 'my-trace-id'})
        assert response.headers['x-request-id'] == 'my-trace-id'

    def test_response_time_header_is_present(self, api_client):
        assert 'x-response-time-ms' in api_client.get('/health').headers


class TestDocs:

    def test_openapi_schema_is_generated(self, api_client):
        response = api_client.get('/openapi.json')
        assert response.status_code == 200
        schema = response.json()
        assert '/predict' in schema['paths']
        assert '/health' in schema['paths']