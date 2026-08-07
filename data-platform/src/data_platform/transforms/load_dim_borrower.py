import pandas as pd
import numpy as np
from data_platform.db import get_engine
from sqlalchemy import text


def read_silver_borrowers():
    engine = get_engine()
    query = text("""
        SELECT applicant_id, age, number_of_dependents, snapshot_date
        FROM silver.loan_applications
        WHERE NOT dq_row_quarantined
    """)
    return pd.read_sql(query, engine)


def derive_bands(df):
    age_bins = [0, 25, 35, 45, 55, 65, 200]
    age_labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
    df['age_band'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    df['age_band'] = df['age_band'].astype('object')
    df.loc[df['age'].isna(), 'age_band'] = 'unknown'

    def dependents_band(n):
        if pd.isna(n):
            return 'unknown'
        elif n == 0:
            return '0'
        elif n == 1:
            return '1'
        elif n == 2:
            return '2'
        else:
            return '3+'

    df['dependents_band'] = df['number_of_dependents'].apply(dependents_band)

    return df


def scd2_merge(df):
    engine = get_engine()

    with engine.begin() as conn:
        current = pd.read_sql(text("""
            SELECT borrower_key, applicant_id, age, age_band,
                   number_of_dependents, dependents_band
            FROM gold.dim_borrower
            WHERE is_current
        """), conn)
        merged = df.merge(
            current, on='applicant_id', how='left', suffixes=('', '_old')
        )

        new_applicants = merged[merged['borrower_key'].isna()]

        existing = merged[merged['borrower_key'].notna()]
        changed_mask = (
            (existing['age_band'] != existing['age_band_old']) |
            (existing['dependents_band'] != existing['dependents_band_old'])
        )
        changed = existing[changed_mask]

        for _, row in new_applicants.iterrows():
            conn.execute(text("""
                INSERT INTO gold.dim_borrower
                    (applicant_id, age, age_band, number_of_dependents,
                     dependents_band, effective_from, is_current)
                VALUES (:applicant_id, :age, :age_band, :dependents,
                        :dependents_band, :snapshot_date, TRUE)
            """), {
                'applicant_id': int(row['applicant_id']),
                'age': None if pd.isna(row['age']) else int(row['age']),
                'age_band': row['age_band'],
                'dependents': None if pd.isna(row['number_of_dependents']) else int(row['number_of_dependents']),
                'dependents_band': row['dependents_band'],
                'snapshot_date': row['snapshot_date']
            })

        for _, row in changed.iterrows():
            conn.execute(text("""
                UPDATE gold.dim_borrower
                SET is_current = FALSE, effective_to = :snapshot_date::date - 1
                WHERE borrower_key = :borrower_key
            """), {
                'snapshot_date': row['snapshot_date'],
                'borrower_key': int(row['borrower_key'])
            })
            conn.execute(text("""
                INSERT INTO gold.dim_borrower
                    (applicant_id, age, age_band, number_of_dependents,
                     dependents_band, effective_from, is_current)
                VALUES (:applicant_id, :age, :age_band, :dependents,
                        :dependents_band, :snapshot_date, TRUE)
            """), {
                'applicant_id': int(row['applicant_id']),
                'age': None if pd.isna(row['age']) else int(row['age']),
                'age_band': row['age_band'],
                'dependents': None if pd.isna(row['number_of_dependents']) else int(row['number_of_dependents']),
                'dependents_band': row['dependents_band'],
                'snapshot_date': row['snapshot_date']
            })

    return len(new_applicants), len(changed)


def main():
    df = read_silver_borrowers()
    df = derive_bands(df)
    n_new, n_changed = scd2_merge(df)
    print(f"dim_borrower: {n_new} new, {n_changed} changed")


if __name__ == '__main__':
    main()