-- append-only, written synchronously by the API; JSONB so schema changes never break the API
CREATE TABLE IF NOT EXISTS monitoring.prediction_log (
    log_id              BIGSERIAL PRIMARY KEY,
    request_id          UUID NOT NULL,
    received_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    pipeline_name       TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    request_payload     JSONB NOT NULL,
    feature_vector      JSONB NOT NULL,
    probability_default NUMERIC(9,6),
    predicted_class     SMALLINT,
    risk_band           TEXT,
    latency_ms          INTEGER,
    status              TEXT NOT NULL DEFAULT 'OK',
    error_message       TEXT
);
CREATE INDEX IF NOT EXISTS ix_pred_log_received ON monitoring.prediction_log (received_at);

CREATE TABLE IF NOT EXISTS monitoring.drift_report (
    drift_id        BIGSERIAL PRIMARY KEY,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    pipeline_name   TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    feature_name    TEXT NOT NULL,
    psi             NUMERIC(10,6) NOT NULL,
    drift_status    TEXT NOT NULL CHECK (drift_status IN ('OK','WARN','ALERT')),
    reference_window TEXT NOT NULL,
    current_window   TEXT NOT NULL,
    n_reference     INTEGER,
    n_current       INTEGER
);

CREATE TABLE IF NOT EXISTS monitoring.service_event (
    event_id     BIGSERIAL PRIMARY KEY,
    occurred_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    service_name TEXT NOT NULL,
    event_type   TEXT NOT NULL,      -- 'startup','model_loaded','error','drift_alert'
    details      JSONB
);