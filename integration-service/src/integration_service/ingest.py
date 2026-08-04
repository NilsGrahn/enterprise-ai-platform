from uuid import uuid4
import pandas as pd
from data_platform.db import get_engine
from integration_service.sources import CsvFileSource
from sqlalchemy import text
import sys

COLUMN_MAP = {
    'Unnamed: 0': 'raw_row_id',
    'SeriousDlqin2yrs': 'serious_dlqin_2yrs',
    'RevolvingUtilizationOfUnsecuredLines': 'revolving_utilization_unsecured_lines',
    'age': 'age',
    'NumberOfTime30-59DaysPastDueNotWorse': 'times_30_59_days_past_due',
    'DebtRatio': 'debt_ratio',
    'MonthlyIncome': 'monthly_income',
    'NumberOfOpenCreditLinesAndLoans': 'open_credit_lines_and_loans',
    'NumberOfTimes90DaysLate': 'times_90_days_late',
    'NumberRealEstateLoansOrLines': 'real_estate_loans_or_lines',
    'NumberOfTime60-89DaysPastDueNotWorse': 'times_60_89_days_past_due',
    'NumberOfDependents': 'number_of_dependents',
}

def ingest(path):
    source = CsvFileSource(path)
    desc = source.descriptor()
    ingestion_id = uuid4()
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO bronze.ingestion_runs
                (ingestion_id, source_file, file_sha256, status)
            VALUES (:ingestion_id, :source_file, :sha256, 'RUNNING')
        """), {
            'ingestion_id': str(ingestion_id),
            'source_file': desc['source_file'],
            'sha256': desc['sha256']
        })

    try:
        row_offset = 0
        for chunk in source.read():
            chunk = chunk.rename(columns=COLUMN_MAP)
            chunk['ingestion_id'] = str(ingestion_id)
            chunk['source_file'] = desc['source_file']
            chunk['source_row_number'] = range(row_offset, row_offset + len(chunk))

            chunk.to_sql(
                'loan_applications',
                engine,
                schema='bronze',
                if_exists='append',
                index=False,
                method='multi'
            )

            row_offset += len(chunk)

        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE bronze.ingestion_runs
                SET status = 'SUCCESS', row_count = :row_count, completed_at = now()
                WHERE ingestion_id = :ingestion_id
            """), {'row_count': row_offset, 'ingestion_id': str(ingestion_id)})

        print(f"ingested {row_offset} rows, ingestion_id={ingestion_id}")

    except Exception as e:
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE bronze.ingestion_runs
                SET status = 'FAILED', error_message = :error
                WHERE ingestion_id = :ingestion_id
            """), {'error': str(e), 'ingestion_id': str(ingestion_id)})
        raise


if __name__ == '__main__':
    path = sys.argv[2]  # sys.argv[0]=script, [1]='--path', [2]=actual path
    ingest(path)