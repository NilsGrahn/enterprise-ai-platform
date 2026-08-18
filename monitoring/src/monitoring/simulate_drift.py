"""Generate synthetic scoring traffic, optionally with a deliberate shift.

CLI: python -m monitoring.simulate_drift --n 500 --shift none|moderate|severe

Samples applicants from the test split and POSTs them to the API. With a shift,
utilisation is inflated and income reduced before sending, so the drift checker
has something real to detect.

This is a demonstration tool for the portfolio, not production code.
"""

import argparse
import os
import sys

import pandas as pd
import requests
from data_platform.db import get_engine

API_URL = os.getenv('INFERENCE_API_URL', 'http://localhost:8000')

SHIFTS = {
    'none':     {'utilisation_multiplier': 1.0, 'income_multiplier': 1.0},
    'moderate': {'utilisation_multiplier': 1.5, 'income_multiplier': 0.8},
    'severe':   {'utilisation_multiplier': 3.0, 'income_multiplier': 0.5},
}

ALIASES = {
    'revolving_utilisation': 'RevolvingUtilizationOfUnsecuredLines',
    'age': 'age',
    'debt_ratio': 'DebtRatio',
    'monthly_income': 'MonthlyIncome',
    'open_credit_lines': 'NumberOfOpenCreditLinesAndLoans',
    'real_estate_loans': 'NumberRealEstateLoansOrLines',
    'times_30_59_days_late': 'NumberOfTime30-59DaysPastDueNotWorse',
    'times_60_89_days_late': 'NumberOfTime60-89DaysPastDueNotWorse',
    'times_90_days_late': 'NumberOfTimes90DaysLate',
    'number_of_dependents': 'NumberOfDependents',
}

INTEGER_FIELDS = {
    'age', 'open_credit_lines', 'real_estate_loans',
    'times_30_59_days_late', 'times_60_89_days_late',
    'times_90_days_late', 'number_of_dependents',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Generate scoring traffic.')
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--shift', default='none', choices=list(SHIFTS))
    parser.add_argument('--explain', action='store_true',
                        help='Request SHAP explanations (slower)')
    return parser.parse_args()


def sample_applicants(n) -> pd.DataFrame:
    return pd.read_sql(f"""
        SELECT revolving_utilisation, age, debt_ratio, monthly_income,
               open_credit_lines, real_estate_loans,
               times_30_59_days_late, times_60_89_days_late,
               times_90_days_late, number_of_dependents
        FROM gold.v_credit_assessment
        WHERE dataset_split = 'test' AND age IS NOT NULL
        ORDER BY random()
        LIMIT {int(n)}
    """, get_engine())


def to_payload(row, shift) -> dict:
    features = {}
    for column, alias in ALIASES.items():
        value = row[column]

        if pd.isna(value):
            features[alias] = None
            continue

        value = float(value)
        if column == 'revolving_utilisation':
            value *= shift['utilisation_multiplier']
        elif column == 'monthly_income':
            value *= shift['income_multiplier']

        features[alias] = int(round(value)) if column in INTEGER_FIELDS else value

    return features


def main():
    args = parse_args()
    shift = SHIFTS[args.shift]

    print(f"sampling {args.n} applicants from the test split")
    applicants = sample_applicants(args.n)
    print(f"shift '{args.shift}': utilisation x{shift['utilisation_multiplier']}, "
          f"income x{shift['income_multiplier']}")
    print(f"posting to {API_URL}/predict\n")

    sent = 0
    failed = 0
    for index, row in applicants.iterrows():
        payload = {
            'features': to_payload(row, shift),
            'include_explanation': args.explain,
            'include_narrative': False,
        }
        try:
            response = requests.post(f'{API_URL}/predict', json=payload, timeout=30)
            if response.status_code == 200:
                sent += 1
            else:
                failed += 1
                if failed <= 3:
                    print(f"  HTTP {response.status_code}: {response.text[:200]}")
        except Exception as exc:
            failed += 1
            if failed <= 3:
                print(f"  request failed: {exc}")

        if sent and sent % 100 == 0:
            print(f"  {sent} sent…")

    print(f"\nsent {sent}, failed {failed}")

    if sent == 0:
        print("Nothing was scored. Is the API running?", file=sys.stderr)
        sys.exit(1)

    print("\nNow run:  python -m monitoring.run_drift_check --window-hours 1")


if __name__ == '__main__':
    main()