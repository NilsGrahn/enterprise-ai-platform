import sys

import numpy as np
import pandas as pd
from data_platform.db import get_engine
from sqlalchemy import text


def get_target_ingestion_id():
    if '--ingestion-id' in sys.argv:
        idx = sys.argv.index('--ingestion-id')
        return sys.argv[idx + 1]

    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(text("""
            SELECT ingestion_id FROM bronze.ingestion_runs
            WHERE status = 'SUCCESS'
            ORDER BY completed_at DESC
            LIMIT 1
        """)).fetchone()
    return str(result[0])


def read_bronze(ingestion_id):
    engine = get_engine()
    query = text("SELECT * FROM bronze.loan_applications WHERE ingestion_id = :iid")
    return pd.read_sql(query, engine, params={'iid': ingestion_id})


def coerce_numeric(df):
    numeric_cols = [
        'raw_row_id', 'serious_dlqin_2yrs', 'revolving_utilization_unsecured_lines',
        'age', 'times_30_59_days_past_due', 'debt_ratio', 'monthly_income',
        'open_credit_lines_and_loans', 'times_90_days_late',
        'real_estate_loans_or_lines', 'times_60_89_days_past_due',
        'number_of_dependents'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def apply_dq_rules(df):
    df['dq_income_missing'] = df['monthly_income'].isna()

    df['dq_dependents_missing'] = df['number_of_dependents'].isna()

    df['dq_age_invalid'] = (df['age'] <= 0) | (df['age'] > 110)
    df.loc[df['dq_age_invalid'], 'age'] = np.nan

    df['dq_utilisation_outlier'] = df['revolving_utilization_unsecured_lines'] > 1.0

    sentinel_cols = ['times_30_59_days_past_due', 'times_60_89_days_past_due', 'times_90_days_late']
    df['dq_delinquency_sentinel'] = False
    for col in sentinel_cols:
        is_sentinel = df[col].isin([96, 98])
        df['dq_delinquency_sentinel'] = df['dq_delinquency_sentinel'] | is_sentinel
        df.loc[is_sentinel, col] = np.nan

    df['dq_row_quarantined'] = ~df['serious_dlqin_2yrs'].isin([0, 1])

    return df



def dedupe(df):
    df['applicant_id'] = df['raw_row_id'].astype('Int64')
    df = df.sort_values('bronze_id', ascending=False)
    df = df.drop_duplicates(subset='applicant_id', keep='first')
    return df


def get_snapshot_date(ingestion_id):
    if '--snapshot-date' in sys.argv:
        idx = sys.argv.index('--snapshot-date')
        return sys.argv[idx + 1]

    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(text("""
            SELECT completed_at::date FROM bronze.ingestion_runs
            WHERE ingestion_id = :iid
        """), {'iid': ingestion_id}).fetchone()
    return str(result[0])



def write_to_silver(df):
    engine = get_engine()

    silver_cols = [
        'applicant_id', 'ingestion_id', 'snapshot_date',
        'is_serious_delinquency', 'revolving_utilisation', 'age', 'debt_ratio',
        'monthly_income', 'open_credit_lines', 'real_estate_loans',
        'times_30_59_days_late', 'times_60_89_days_late', 'times_90_days_late',
        'number_of_dependents',
        'dq_income_missing', 'dq_dependents_missing', 'dq_age_invalid',
        'dq_utilisation_outlier', 'dq_delinquency_sentinel', 'dq_row_quarantined'
    ]

    staging = df.rename(columns={
        'serious_dlqin_2yrs': 'is_serious_delinquency',
        'revolving_utilization_unsecured_lines': 'revolving_utilisation',
        'times_30_59_days_past_due': 'times_30_59_days_late',
        'open_credit_lines_and_loans': 'open_credit_lines',
        'real_estate_loans_or_lines': 'real_estate_loans',
        'times_60_89_days_past_due': 'times_60_89_days_late',
    })[silver_cols]

    with engine.begin() as conn:
        staging.to_sql('silver_staging', conn, if_exists='replace', index=False)

        select_cols = ','.join(
            f"{c}::uuid" if c == 'ingestion_id'
            else f"{c}::date" if c == 'snapshot_date'
            else c
            for c in silver_cols
        )

        conn.execute(text(f"""
            INSERT INTO silver.loan_applications ({','.join(silver_cols)})
            SELECT {select_cols} FROM silver_staging
            ON CONFLICT (applicant_id) DO UPDATE SET
                {','.join(f"{c} = EXCLUDED.{c}" for c in silver_cols if c != 'applicant_id')}
        """))

        conn.execute(text("DROP TABLE silver_staging"))

    return len(staging)



def log_dq_results(df, ingestion_id):
    engine = get_engine()
    rules = {
        'income_missing': df['dq_income_missing'].sum(),
        'dependents_missing': df['dq_dependents_missing'].sum(),
        'age_invalid': df['dq_age_invalid'].sum(),
        'utilisation_outlier': df['dq_utilisation_outlier'].sum(),
        'delinquency_sentinel': df['dq_delinquency_sentinel'].sum(),
        'row_quarantined': df['dq_row_quarantined'].sum(),
    }
    with engine.begin() as conn:
        for rule_name, rows_affected in rules.items():
            conn.execute(text("""
                INSERT INTO silver.data_quality_log
                    (ingestion_id, rule_name, severity, rows_affected)
                VALUES (:iid, :rule, 'WARN', :count)
            """), {'iid': ingestion_id, 'rule': rule_name, 'count': int(rows_affected)})


def main():
    ingestion_id = get_target_ingestion_id()
    snapshot_date = get_snapshot_date(ingestion_id)

    df = read_bronze(ingestion_id)
    df = coerce_numeric(df)
    df = apply_dq_rules(df)
    df = dedupe(df)
    df['snapshot_date'] = snapshot_date

    n = write_to_silver(df)
    log_dq_results(df, ingestion_id)

    print(f"transformed {n} rows into silver, ingestion_id={ingestion_id}")


if __name__ == '__main__':
    main()