"""CreditPipeline — cleaning, feature engineering, prediction."""

import numpy as np
import pandas as pd
import pytest
from ml_service.pipelines import get_pipeline


class TestClean:

    def test_drops_rows_with_a_null_target(self, sample_gold_df):
        pipeline = get_pipeline('credit')
        df = sample_gold_df.copy()
        df.loc[0:4, 'is_serious_delinquency'] = np.nan
        assert len(pipeline.clean(df)) == len(df) - 5

    def test_tolerates_a_missing_target_column(self, sample_gold_df):
        """At serve time the target does not exist at all."""
        pipeline = get_pipeline('credit')
        df = sample_gold_df.drop(columns=['is_serious_delinquency'])
        assert len(pipeline.clean(df)) == len(df)

    def test_clips_age_to_a_plausible_range(self, sample_gold_df):
        pipeline = get_pipeline('credit')
        df = sample_gold_df.copy()
        df.loc[0, 'age'] = 5
        df.loc[1, 'age'] = 200
        out = pipeline.clean(df)
        assert out.loc[0, 'age'] == 18
        assert out.loc[1, 'age'] == 100

    def test_clips_negative_counts_to_zero(self, sample_gold_df):
        pipeline = get_pipeline('credit')
        df = sample_gold_df.copy()
        df.loc[0, 'times_90_days_late'] = -3
        assert pipeline.clean(df).loc[0, 'times_90_days_late'] == 0

    def test_does_not_impute(self, sample_gold_df):
        """Imputation is a modelling decision — it belongs in feature_engineering."""
        pipeline = get_pipeline('credit')
        df = sample_gold_df.copy()
        df.loc[0, 'monthly_income'] = np.nan
        assert pd.isna(pipeline.clean(df).loc[0, 'monthly_income'])

    def test_does_not_mutate_the_input(self, sample_gold_df):
        pipeline = get_pipeline('credit')
        before = sample_gold_df.copy()
        pipeline.clean(sample_gold_df)
        pd.testing.assert_frame_equal(sample_gold_df, before)


class TestFeatureEngineering:

    def test_fit_learns_and_returns_parameters(self, sample_gold_df):
        pipeline = get_pipeline('credit')
        _, p = pipeline.feature_engineering(pipeline.clean(sample_gold_df), fit=True)
        for key in ('median_income', 'utilisation_cap', 'debt_ratio_cap',
                    'feature_names'):
            assert key in p

    def test_no_fit_without_stored_preprocessing_raises(self, sample_gold_df):
        pipeline = get_pipeline('credit')
        with pytest.raises(ValueError):
            pipeline.feature_engineering(pipeline.clean(sample_gold_df), fit=False)

    def test_imputation_uses_the_stored_median_not_a_new_one(self, sample_gold_df):
        """The core train/serve skew guarantee."""
        pipeline = get_pipeline('credit')
        cleaned = pipeline.clean(sample_gold_df)
        _, p = pipeline.feature_engineering(cleaned, fit=True)
        pipeline.preprocessing = p
        stored_median = p['median_income']

        # A frame whose own median is wildly different.
        other = cleaned.head(10).copy()
        other['monthly_income'] = 50000.0
        other.loc[other.index[0], 'monthly_income'] = np.nan

        X, _ = pipeline.feature_engineering(other, fit=False)
        assert X['monthly_income'].iloc[0] == stored_median, \
            "serving must reuse the training median, never recompute one"

    def test_missingness_flag_is_set_before_imputation(self, sample_gold_df):
        pipeline = get_pipeline('credit')
        cleaned = pipeline.clean(sample_gold_df)
        X, _ = pipeline.feature_engineering(cleaned, fit=True)
        expected = int(cleaned['monthly_income'].isna().sum())
        assert X['income_missing'].sum() == expected
        assert expected > 0, "the fixture should contain some missing income"

    def test_output_has_no_missing_values(self, sample_gold_df):
        pipeline = get_pipeline('credit')
        X, _ = pipeline.feature_engineering(pipeline.clean(sample_gold_df), fit=True)
        assert X.isna().sum().sum() == 0

    def test_derived_features_are_present(self, sample_gold_df):
        pipeline = get_pipeline('credit')
        X, _ = pipeline.feature_engineering(pipeline.clean(sample_gold_df), fit=True)
        for feature in ('monthly_debt_payment', 'income_per_dependent',
                        'delinquency_ratio', 'has_any_delinquency',
                        'utilisation_bucket'):
            assert feature in X.columns

    def test_target_does_not_leak_into_features(self, sample_gold_df):
        pipeline = get_pipeline('credit')
        X, _ = pipeline.feature_engineering(pipeline.clean(sample_gold_df), fit=True)
        assert 'is_serious_delinquency' not in X.columns
        assert 'applicant_id' not in X.columns
        assert 'dataset_split' not in X.columns

    def test_column_order_is_stable_across_calls(self, sample_gold_df):
        pipeline = get_pipeline('credit')
        cleaned = pipeline.clean(sample_gold_df)
        X1, p = pipeline.feature_engineering(cleaned, fit=True)
        pipeline.preprocessing = p
        X2, _ = pipeline.feature_engineering(cleaned.head(5), fit=False)
        assert list(X1.columns) == list(X2.columns)


class TestPredict:

    def test_returns_probabilities_in_range(self, trained_pipeline, sample_gold_df):
        probabilities = trained_pipeline.predict(sample_gold_df.head(20))
        assert len(probabilities) == 20
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

    def test_is_deterministic(self, trained_pipeline, single_applicant_df):
        first = trained_pipeline.predict(single_applicant_df)
        second = trained_pipeline.predict(single_applicant_df)
        assert np.allclose(first, second)

    def test_without_a_model_raises(self, sample_gold_df):
        pipeline = get_pipeline('credit')
        with pytest.raises(ValueError):
            pipeline.predict(sample_gold_df)

    def test_higher_risk_profile_scores_higher(self, trained_pipeline,
                                               single_applicant_df):
        low = single_applicant_df.copy()
        low['revolving_utilisation'] = 0.05
        low['times_90_days_late'] = 0
        low['times_60_89_days_late'] = 0
        low['total_delinquency_events'] = 0

        high = single_applicant_df.copy()
        high['revolving_utilisation'] = 0.99
        high['times_90_days_late'] = 3
        high['times_60_89_days_late'] = 2
        high['total_delinquency_events'] = 6

        assert trained_pipeline.predict(high)[0] > trained_pipeline.predict(low)[0]