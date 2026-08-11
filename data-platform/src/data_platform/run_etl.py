import sys
import time
from data_platform import migrate, seed_dimensions
from data_platform.transforms import bronze_to_silver, load_dim_borrower, load_fact_assessment
from integration_service import ingest


def run_stage(name, func):
    print(f"--- starting {name} ---")
    start = time.time()
    func()
    elapsed = time.time() - start
    print(f"--- finished {name} in {elapsed:.1f}s ---")


def main():
    full = '--full' in sys.argv

    if full:
        run_stage('ingest', lambda: ingest.ingest('data/raw/cs-training.csv'))

    run_stage('migrate', migrate.main)
    run_stage('seed_dimensions', seed_dimensions.main)
    run_stage('bronze_to_silver', bronze_to_silver.main)
    run_stage('load_dim_borrower', load_dim_borrower.main)
    run_stage('load_fact_assessment', load_fact_assessment.main)

    print("ETL complete")


if __name__ == '__main__':
    main()