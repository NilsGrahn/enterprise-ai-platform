"""SHAP explanation structure and correctness."""

import numpy as np
import pytest
from explain_service.explainer import DISPLAY_NAMES, MissingDisplayNameError, ShapTreeExplainer


class TestDisplayNames:

    def test_every_engineered_feature_has_a_label(self, trained_pipeline):
        missing = [f for f in trained_pipeline.preprocessing['feature_names']
                   if f not in DISPLAY_NAMES]
        assert missing == [], f"no display name for: {missing}"

    def test_missing_label_raises_at_construction(self, trained_pipeline,
                                                  fake_metadata):
        incomplete = {k: v for k, v in DISPLAY_NAMES.items()
                      if k != 'monthly_income'}
        with pytest.raises(MissingDisplayNameError):
            ShapTreeExplainer(trained_pipeline, fake_metadata,
                              display_names=incomplete)


class TestExplanationStructure:

    def test_returns_the_requested_number_of_contributions(self, explainer,
                                                           single_applicant_df):
        result = explainer.explain(single_applicant_df, request_id='t', top_n=5)
        assert len(result.contributions) == 5

    def test_contributions_are_ranked_by_absolute_value(self, explainer,
                                                        single_applicant_df):
        result = explainer.explain(single_applicant_df, request_id='t')
        magnitudes = [abs(c.shap_value) for c in result.contributions]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_ranks_are_sequential_from_one(self, explainer, single_applicant_df):
        result = explainer.explain(single_applicant_df, request_id='t')
        assert [c.rank for c in result.contributions] == [1, 2, 3, 4, 5]

    def test_direction_matches_the_sign(self, explainer, single_applicant_df):
        result = explainer.explain(single_applicant_df, request_id='t')
        for c in result.all_contributions:
            expected = 'increases_risk' if c.shap_value > 0 else 'decreases_risk'
            assert c.direction == expected

    def test_all_contributions_covers_every_feature(self, explainer,
                                                    single_applicant_df,
                                                    trained_pipeline):
        result = explainer.explain(single_applicant_df, request_id='t')
        assert len(result.all_contributions) == \
            len(trained_pipeline.preprocessing['feature_names'])

    def test_rejects_multiple_rows(self, explainer, sample_gold_df):
        with pytest.raises(ValueError):
            explainer.explain(sample_gold_df.head(3), request_id='t')


class TestAdditivity:
    """base_value + sum(shap) must reconstruct the model's output."""

    def test_contributions_sum_to_the_prediction(self, explainer,
                                                 single_applicant_df):
        result = explainer.explain(single_applicant_df, request_id='t')
        total = result.base_value + sum(c.shap_value for c in result.all_contributions)
        reconstructed = 1 / (1 + np.exp(-total))
        assert abs(reconstructed - result.probability) < 1e-4, \
            "if this fails, the wrong class index was taken and every sign is inverted"

    def test_probability_matches_the_pipeline(self, explainer, trained_pipeline,
                                              single_applicant_df):
        result = explainer.explain(single_applicant_df, request_id='t')
        direct = trained_pipeline.predict(single_applicant_df)[0]
        assert abs(result.probability - direct) < 1e-9


class TestReasonCodes:

    def test_every_feature_has_a_reason_code(self, trained_pipeline):
        from llm_service.reason_codes import REASON_CODES
        missing = [f for f in trained_pipeline.preprocessing['feature_names']
                   if f not in REASON_CODES]
        assert missing == [], f"no reason code for: {missing}"

    def test_only_risk_increasing_factors_get_codes(self, sample_explanation):
        from llm_service.reason_codes import to_reason_codes
        codes = to_reason_codes(sample_explanation)
        assert len(codes) == 3
        assert [c['rank'] for c in codes] == [1, 2, 3]