"""The Phase 2 bronze -> silver transformation rules."""

import hashlib

import pandas as pd
from data_platform.transforms.bronze_to_silver import (
    apply_dq_rules,
    coerce_numeric,
    dedupe,
)


def prepared(df):
    """Run the two steps that always precede the rules."""
    return apply_dq_rules(coerce_numeric(df.copy()))


class TestCoercion:

    def test_strings_become_numbers(self, sample_raw_df):
        out = coerce_numeric(sample_raw_df.copy())
        assert pd.api.types.is_numeric_dtype(out['monthly_income'])
        assert pd.api.types.is_numeric_dtype(out['age'])

    def test_unconvertible_values_become_nan(self, sample_raw_df):
        df = sample_raw_df.copy()
        df.loc[0, 'monthly_income'] = 'not a number'
        out = coerce_numeric(df)
        assert pd.isna(out.loc[0, 'monthly_income'])


class TestDataQualityRules:

    def test_missing_income_is_flagged(self, sample_raw_df):
        out = prepared(sample_raw_df)
        assert out.loc[15, 'dq_income_missing']
        assert not out.loc[0, 'dq_income_missing']

    def test_missing_dependents_is_flagged(self, sample_raw_df):
        out = prepared(sample_raw_df)
        assert out.loc[16, 'dq_dependents_missing']

    def test_invalid_age_is_flagged_and_nulled(self, sample_raw_df):
        out = prepared(sample_raw_df)
        assert out.loc[17, 'dq_age_invalid']
        assert pd.isna(out.loc[17, 'age']), "an invalid age must be nulled, not kept"

    def test_utilisation_outlier_is_flagged_but_not_nulled(self, sample_raw_df):
        out = prepared(sample_raw_df)
        assert out.loc[18, 'dq_utilisation_outlier']
        assert not pd.isna(out.loc[18, 'revolving_utilization_unsecured_lines']), \
            "the outlier rule flags but does not null — capping happens in the pipeline"

    def test_sentinel_is_flagged_and_nulled(self, sample_raw_df):
        out = prepared(sample_raw_df)
        assert out.loc[19, 'dq_delinquency_sentinel']
        assert pd.isna(out.loc[19, 'times_90_days_late'])

    def test_sentinel_only_nulls_the_offending_column(self, sample_raw_df):
        out = prepared(sample_raw_df)
        assert not pd.isna(out.loc[19, 'times_30_59_days_past_due']), \
            "other late-payment columns in the same row stay intact"

    def test_valid_rows_are_not_flagged(self, sample_raw_df):
        out = prepared(sample_raw_df)
        flags = ['dq_income_missing', 'dq_dependents_missing', 'dq_age_invalid',
                 'dq_utilisation_outlier', 'dq_delinquency_sentinel',
                 'dq_row_quarantined']
        assert not out.loc[0, flags].any()

    def test_invalid_target_is_quarantined(self, sample_raw_df):
        df = sample_raw_df.copy()
        df.loc[0, 'serious_dlqin_2yrs'] = ''
        out = prepared(df)
        assert out.loc[0, 'dq_row_quarantined']

    def test_rules_never_drop_rows(self, sample_raw_df):
        out = prepared(sample_raw_df)
        assert len(out) == len(sample_raw_df), \
            "silver flags problems, it does not silently drop rows"


class TestDeduplication:

    def test_highest_bronze_id_wins(self, sample_raw_df):
        df = sample_raw_df.copy()
        duplicate = df.iloc[0].copy()
        duplicate['bronze_id'] = 999
        duplicate['monthly_income'] = '9999'
        df = pd.concat([df, duplicate.to_frame().T], ignore_index=True)

        out = dedupe(coerce_numeric(df))

        matching = out[out['applicant_id'] == 1]
        assert len(matching) == 1
        assert matching.iloc[0]['monthly_income'] == 9999, \
            "the most recently ingested row must win"

    def test_no_duplicate_applicants_remain(self, sample_raw_df):
        df = pd.concat([sample_raw_df, sample_raw_df], ignore_index=True)
        df['bronze_id'] = range(1, len(df) + 1)
        out = dedupe(coerce_numeric(df))
        assert out['applicant_id'].duplicated().sum() == 0


class TestHashSplit:

    @staticmethod
    def split_for(applicant_id):
        h = int(hashlib.md5(f"{applicant_id}".encode()).hexdigest()[:8], 16) % 100
        return 'train' if h < 70 else ('valid' if h < 85 else 'test')

    def test_split_is_stable(self):
        for applicant_id in [1, 42, 4521, 150000]:
            assert self.split_for(applicant_id) == self.split_for(applicant_id)

    def test_split_proportions_are_roughly_right(self):
        splits = [self.split_for(i) for i in range(1, 10001)]
        train = splits.count('train') / len(splits)
        valid = splits.count('valid') / len(splits)
        test = splits.count('test') / len(splits)
        assert 0.67 < train < 0.73
        assert 0.13 < valid < 0.17
        assert 0.13 < test < 0.17

    def test_all_splits_are_valid_labels(self):
        assert {self.split_for(i) for i in range(1, 1001)} == {'train', 'valid', 'test'}