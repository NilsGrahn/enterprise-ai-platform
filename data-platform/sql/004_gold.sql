-- 1. DATE ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gold.dim_date (
    date_key      INTEGER PRIMARY KEY,          -- YYYYMMDD
    full_date     DATE NOT NULL UNIQUE,
    year          SMALLINT NOT NULL,
    quarter       SMALLINT NOT NULL,
    month         SMALLINT NOT NULL,
    month_name    TEXT NOT NULL,
    day_of_month  SMALLINT NOT NULL,
    day_of_week   SMALLINT NOT NULL,
    is_month_end  BOOLEAN NOT NULL
);

-- 2. BORROWER (SCD type 2) ----------------------------------------------
CREATE TABLE IF NOT EXISTS gold.dim_borrower (
    borrower_key        BIGSERIAL PRIMARY KEY,
    applicant_id        INTEGER NOT NULL,       -- natural key
    age                 SMALLINT,
    age_band            TEXT,                   -- '<25','25-34','35-44','45-54','55-64','65+','unknown'
    number_of_dependents SMALLINT,
    dependents_band     TEXT,                   -- '0','1','2','3+','unknown'
    effective_from      DATE NOT NULL,
    effective_to        DATE NOT NULL DEFAULT DATE '9999-12-31',
    is_current          BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_dim_borrower_current
    ON gold.dim_borrower (applicant_id) WHERE is_current;

-- 3. INCOME BAND --------------------------------------------------------
CREATE TABLE IF NOT EXISTS gold.dim_income_band (
    income_band_key SERIAL PRIMARY KEY,
    band_label      TEXT NOT NULL UNIQUE,
    lower_bound     NUMERIC(14,2),
    upper_bound     NUMERIC(14,2),
    is_unknown      BOOLEAN NOT NULL DEFAULT FALSE,
    sort_order      SMALLINT NOT NULL
);

-- 4. UTILISATION BAND ---------------------------------------------------
CREATE TABLE IF NOT EXISTS gold.dim_utilisation_band (
    utilisation_band_key SERIAL PRIMARY KEY,
    band_label           TEXT NOT NULL UNIQUE,
    lower_bound          NUMERIC(10,4),
    upper_bound          NUMERIC(10,4),
    is_anomalous         BOOLEAN NOT NULL DEFAULT FALSE,
    sort_order           SMALLINT NOT NULL
);

-- 5. DELINQUENCY PROFILE (junk dimension) -------------------------------
CREATE TABLE IF NOT EXISTS gold.dim_delinquency_profile (
    delinquency_profile_key SERIAL PRIMARY KEY,
    has_30_59_late          BOOLEAN NOT NULL,
    has_60_89_late          BOOLEAN NOT NULL,
    has_90_plus_late        BOOLEAN NOT NULL,
    has_sentinel_code       BOOLEAN NOT NULL,
    severity_label          TEXT NOT NULL,      -- 'none','mild','moderate','severe','unreliable'
    UNIQUE (has_30_59_late, has_60_89_late, has_90_plus_late, has_sentinel_code)
);

-- 6. MODEL --------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gold.dim_model (
    model_key        SERIAL PRIMARY KEY,
    pipeline_name    TEXT NOT NULL,             -- 'credit', later 'fraud', etc.
    model_name       TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    algorithm        TEXT NOT NULL,
    hyperparameters  JSONB,
    feature_list     JSONB,
    training_rows    INTEGER,
    trained_at       TIMESTAMPTZ NOT NULL,
    metric_auc       NUMERIC(6,4),
    metric_ks        NUMERIC(6,4),
    is_active        BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (pipeline_name, model_version)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_dim_model_active
    ON gold.dim_model (pipeline_name) WHERE is_active;