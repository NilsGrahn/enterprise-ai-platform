CREATE TABLE IF NOT EXISTS silver.loan_applications (
    applicant_id                 INTEGER PRIMARY KEY,
    ingestion_id                 UUID NOT NULL,
    snapshot_date                DATE NOT NULL,

    is_serious_delinquency       SMALLINT CHECK (is_serious_delinquency IN (0,1)),
    revolving_utilisation        NUMERIC(14,6),
    age                          SMALLINT,
    debt_ratio                   NUMERIC(16,6),
    monthly_income               NUMERIC(14,2),
    open_credit_lines            SMALLINT,
    real_estate_loans            SMALLINT,
    times_30_59_days_late        SMALLINT,
    times_60_89_days_late        SMALLINT,
    times_90_days_late           SMALLINT,
    number_of_dependents         SMALLINT,

    -- data quality flags: never silently repair, always record
    dq_income_missing            BOOLEAN NOT NULL DEFAULT FALSE,
    dq_dependents_missing        BOOLEAN NOT NULL DEFAULT FALSE,
    dq_age_invalid               BOOLEAN NOT NULL DEFAULT FALSE,
    dq_utilisation_outlier       BOOLEAN NOT NULL DEFAULT FALSE,
    dq_delinquency_sentinel      BOOLEAN NOT NULL DEFAULT FALSE,
    dq_row_quarantined           BOOLEAN NOT NULL DEFAULT FALSE,

    processed_at                 TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS silver.data_quality_log (
    dq_id          BIGSERIAL PRIMARY KEY,
    ingestion_id   UUID NOT NULL,
    rule_name      TEXT NOT NULL,
    severity       TEXT NOT NULL CHECK (severity IN ('INFO','WARN','ERROR')),
    rows_affected  INTEGER NOT NULL,
    checked_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);