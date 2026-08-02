from pathlib import Path
from data_platform.db import run_sql_file
sql_files = sorted(Path("data-platform/sql").glob("*.sql"))

for schemas in sql_files:
    run_sql_file(schemas)