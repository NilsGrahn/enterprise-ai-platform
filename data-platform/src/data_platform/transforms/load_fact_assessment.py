import hashlib
import pandas as pd
import numpy as np
from data_platform.db import get_engine
from sqlalchemy import text


def read_silver():
    engine = get_engine()
    query = text("""
        SELECT applicant_id, snapshot_date, revolving_utilisation, debt_ratio,
               monthly_income, open_credit_lines, real_estate_loans,
               times_30_59_days_late, times_60_89_days_late, times_90_days_late,
               is_serious_delinquency, dq_delinquency_sentinel
        FROM silver.loan_applications
        WHERE NOT dq_row_quarantined
    """)
    return pd.read_sql(query, engine)


def load_dimension_lookups():
    engine = get_engine()
    income_bands = pd.read_sql(text(
        "SELECT income_band_key, band_label, lower_bound, upper_bound, is_unknown FROM gold.dim_income_band"
    ), engine)
    util_bands = pd.read_sql(text(
        "SELECT utilisation_band_key, band_label, lower_bound, upper_bound, is_anomalous FROM gold.dim_utilisation_band"
    ), engine)
    delinq_profiles = pd.read_sql(text(
        "SELECT delinquency_profile_key, has_30_59_late, has_60_89_late, has_90_plus_late, has_sentinel_code FROM gold.dim_delinquency_profile"
    ), engine)
    borrowers = pd.read_sql(text(
        "SELECT borrower_key, applicant_id FROM gold.dim_borrower WHERE is_current"
    ), engine)
    return income_bands, util_bands, delinq_profiles, borrowers


def bucket_value(value, bands_df):
    if pd.isna(value):
        unknown_row = bands_df[bands_df['is_unknown'] == True]
        return int(unknown_row.iloc[0].iloc[0])

    for _, band in bands_df.iterrows():
        lower = band['lower_bound']
        upper = band['upper_bound']
        if pd.isna(lower) and value < upper:
            return int(band.iloc[0])
        if pd.isna(upper) and value >= lower:
            return int(band.iloc[0])
        if not pd.isna(lower) and not pd.isna(upper) and lower <= value < upper:
            return int(band.iloc[0])

    return int(bands_df.iloc[-1].iloc[0])


def assign_dimension_keys(df, income_bands, util_bands, delinq_profiles, borrowers):
    df['snapshot_date_key'] = pd.to_datetime(df['snapshot_date']).dt.strftime('%Y%m%d').astype(int)

    df = df.merge(borrowers, on='applicant_id', how='left')

    df['income_band_key'] = df['monthly_income'].apply(lambda v: bucket_value(v, income_bands))
    df['utilisation_band_key'] = df['revolving_utilisation'].apply(lambda v: bucket_value(v, util_bands))

    df['has_30_59'] = df['times_30_59_days_late'].fillna(0) > 0
    df['has_60_89'] = df['times_60_89_days_late'].fillna(0) > 0
    df['has_90_plus'] = df['times_90_days_late'].fillna(0) > 0
    df['has_sentinel'] = df['dq_delinquency_sentinel']

    df = df.merge(
        delinq_profiles,
        left_on=['has_30_59', 'has_60_89', 'has_90_plus', 'has_sentinel'],
        right_on=['has_30_59_late', 'has_60_89_late', 'has_90_plus_late', 'has_sentinel_code'],
        how='left'
    )

    return df


def compute_derived_fields(df):
    df['total_delinquency_events'] = (
        df['times_30_59_days_late'].fillna(0) +
        df['times_60_89_days_late'].fillna(0) +
        df['times_90_days_late'].fillna(0)
    ).astype(int)

    def split_for(applicant_id):
        h = int(hashlib.md5(f"{applicant_id}".encode()).hexdigest()[:8], 16) % 100
        if h < 70:
            return 'train'
        elif h < 85:
            return 'valid'
        else:
            return 'test'

    df['dataset_split'] = df['applicant_id'].apply(split_for)
    df['monthly_income_imputed'] = False

    return df


def assert_no_null_keys(df):
    key_cols = ['snapshot_date_key', 'borrower_key', 'income_band_key',
                'utilisation_band_key', 'delinquency_profile_key']
    for col in key_cols:
        n_null = df[col].isna().sum()
        if n_null > 0:
            raise ValueError(f"{n_null} rows have NULL {col} — dimension seeding is incomplete")


def write_to_gold(df):
    engine = get_engine()

    fact_cols = [
        'snapshot_date_key', 'borrower_key', 'income_band_key', 'utilisation_band_key',
        'delinquency_profile_key', 'applicant_id', 'revolving_utilisation', 'debt_ratio',
        'monthly_income', 'open_credit_lines', 'real_estate_loans',
        'times_30_59_days_late', 'times_60_89_days_late', 'times_90_days_late',
        'total_delinquency_events', 'is_serious_delinquency', 'monthly_income_imputed',
        'dataset_split'
    ]

    staging = df[fact_cols].copy()

    with engine.begin() as conn:
        staging.to_sql('gold_fact_staging', conn, if_exists='replace', index=False)

        conn.execute(text(f"""
            INSERT INTO gold.fact_credit_assessment ({','.join(fact_cols)})
            SELECT {','.join(fact_cols)} FROM gold_fact_staging
            ON CONFLICT (applicant_id, snapshot_date_key) DO UPDATE SET
                {','.join(f"{c} = EXCLUDED.{c}" for c in fact_cols if c not in ('applicant_id', 'snapshot_date_key'))}
        """))

        conn.execute(text("DROP TABLE gold_fact_staging"))

    return len(staging)


def main():
    df = read_silver()
    income_bands, util_bands, delinq_profiles, borrowers = load_dimension_lookups()
    df = assign_dimension_keys(df, income_bands, util_bands, delinq_profiles, borrowers)
    df = compute_derived_fields(df)
    assert_no_null_keys(df)
    n = write_to_gold(df)
    print(f"loaded {n} rows into gold.fact_credit_assessment")


if __name__ == '__main__':
    main()