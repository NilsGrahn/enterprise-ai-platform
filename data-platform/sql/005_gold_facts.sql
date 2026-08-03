-- GRAIN: one row per applicant per snapshot_date.
CREATE TABLE IF NOT EXISTS gold.fact_credit_assessment (
    assessment_key           BIGSERIAL PRIMARY KEY,

    -- foreign keys to dimensions
    snapshot_date_key        INTEGER NOT NULL REFERENCES gold.dim_date(date_key),
    borrower_key             BIGINT  NOT NULL REFERENCES gold.dim_borrower(borrower_key),
    income_band_key          INTEGER NOT NULL REFERENCES gold.dim_income_band(income_band_key),
    utilisation_band_key     INTEGER NOT NULL REFERENCES gold.dim_utilisation_band(utilisation_band_key),
    delinquency_profile_key  INTEGER NOT NULL REFERENCES gold.dim_delinquency_profile(delinquency_profile_key),

    -- degenerate dimension (natural key, no dimension table of its own)
    applicant_id             INTEGER NOT NULL,

    -- additive / semi-additive measures
    revolving_utilisation    NUMERIC(14,6),
    debt_ratio               NUMERIC(16,6),
    monthly_income           NUMERIC(14,2),
    open_credit_lines        SMALLINT,
    real_estate_loans        SMALLINT,
    times_30_59_days_late    SMALLINT,
    times_60_89_days_late    SMALLINT,
    times_90_days_late       SMALLINT,
    total_delinquency_events SMALLINT,

    -- label + governance
    is_serious_delinquency   SMALLINT CHECK (is_serious_delinquency IN (0,1)),
    monthly_income_imputed   BOOLEAN NOT NULL DEFAULT FALSE,
    dataset_split            TEXT NOT NULL CHECK (dataset_split IN ('train','valid','test')),
    loaded_at                TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (applicant_id, snapshot_date_key)
);
CREATE INDEX IF NOT EXISTS ix_fact_assessment_split
    ON gold.fact_credit_assessment (dataset_split);
CREATE INDEX IF NOT EXISTS ix_fact_assessment_date
    ON gold.fact_credit_assessment (snapshot_date_key);