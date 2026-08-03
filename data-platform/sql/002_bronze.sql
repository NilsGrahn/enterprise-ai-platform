CREATE TABLE IF NOT EXISTS bronze.loan_applications (
    bronze_id              BIGSERIAL PRIMARY KEY,
    ingestion_id           UUID        NOT NULL,
    source_file            TEXT        NOT NULL,
    source_row_number      INTEGER     NOT NULL,
    ingested_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- every source column as TEXT, no casting, no cleaning
    raw_row_id                             TEXT,   -- the "Unnamed: 0" column
    serious_dlqin_2yrs                     TEXT,
    revolving_utilization_unsecured_lines  TEXT,
    age                                    TEXT,
    times_30_59_days_past_due              TEXT,
    debt_ratio                             TEXT,
    monthly_income                         TEXT,
    open_credit_lines_and_loans            TEXT,
    times_90_days_late                     TEXT,
    real_estate_loans_or_lines             TEXT,
    times_60_89_days_past_due              TEXT,
    number_of_dependents                   TEXT
);

CREATE TABLE IF NOT EXISTS bronze.ingestion_runs (
    ingestion_id   UUID PRIMARY KEY,
    source_file    TEXT NOT NULL,
    file_sha256    TEXT NOT NULL,
    row_count      INTEGER,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at   TIMESTAMPTZ,
    status         TEXT NOT NULL DEFAULT 'RUNNING'
        CHECK (status IN ('RUNNING','SUCCESS','FAILED')),
    error_message  TEXT
);

CREATE INDEX IF NOT EXISTS ix_bronze_loan_ingestion
    ON bronze.loan_applications (ingestion_id);