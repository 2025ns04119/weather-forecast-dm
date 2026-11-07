-- ============================================================
-- Weather Forecast DM project - Neon (cloud PostgreSQL) schema
-- Medallion architecture: raw (bronze) -> clean (silver) -> features (gold)
-- Run once in Neon SQL Editor
-- ============================================================

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS clean;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS quarantine;
CREATE SCHEMA IF NOT EXISTS ml;

-- ---------- BRONZE: raw API payload, stored as received ----------
CREATE TABLE IF NOT EXISTS raw.weather_api (
    id          BIGSERIAL PRIMARY KEY,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    location    TEXT        NOT NULL,
    source      TEXT        NOT NULL DEFAULT 'open-meteo',
    endpoint    TEXT        NOT NULL,          -- forecast / archive
    payload     JSONB       NOT NULL
);

-- ---------- SILVER: validated, cleaned hourly observations ----------
CREATE TABLE IF NOT EXISTS clean.weather_hourly (
    location              TEXT        NOT NULL,
    ts                    TIMESTAMPTZ NOT NULL,
    temperature_2m        REAL,
    relative_humidity_2m  REAL,
    dew_point_2m          REAL,
    pressure_msl          REAL,
    cloud_cover           REAL,
    wind_speed_10m        REAL,
    wind_direction_10m    REAL,
    precipitation         REAL,
    raw_id                BIGINT REFERENCES raw.weather_api(id),  -- lineage
    ingested_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (location, ts)
);

-- ---------- QUARANTINE: rows that failed validation ----------
CREATE TABLE IF NOT EXISTS quarantine.weather_rejects (
    id           BIGSERIAL PRIMARY KEY,
    location     TEXT,
    ts           TIMESTAMPTZ,
    reason       TEXT        NOT NULL,
    record       JSONB       NOT NULL,
    raw_id       BIGINT,
    rejected_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------- GOLD: ML features ----------
CREATE TABLE IF NOT EXISTS features.weather_features (
    location          TEXT        NOT NULL,
    ts                TIMESTAMPTZ NOT NULL,
    hour              SMALLINT,
    month             SMALLINT,
    day_of_week       SMALLINT,
    temp_lag_1h       REAL,
    temp_lag_3h       REAL,
    temp_lag_24h      REAL,
    temp_roll_mean_6h REAL,
    temp_roll_mean_24h REAL,
    humidity          REAL,
    pressure          REAL,
    wind_speed        REAL,
    cloud_cover       REAL,
    target_temp_24h   REAL,                    -- label: temperature 24h ahead
    feature_version   TEXT        NOT NULL DEFAULT 'v1',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (location, ts, feature_version)
);

-- ---------- ML: predictions, drift, pipeline run log ----------
CREATE TABLE IF NOT EXISTS ml.predictions (
    id             BIGSERIAL PRIMARY KEY,
    location       TEXT        NOT NULL,
    made_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    target_ts      TIMESTAMPTZ NOT NULL,
    predicted_temp REAL        NOT NULL,
    model_name     TEXT        NOT NULL,
    model_version  TEXT        NOT NULL
);

CREATE TABLE IF NOT EXISTS ml.drift_reports (
    id              BIGSERIAL PRIMARY KEY,
    run_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    drift_detected  BOOLEAN     NOT NULL,
    drift_share     REAL,
    details         JSONB
);

CREATE TABLE IF NOT EXISTS ml.pipeline_runs (            -- metadata / lineage log
    run_id      BIGSERIAL PRIMARY KEY,
    pipeline    TEXT        NOT NULL,   -- ingest / validate / features / train / predict
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status      TEXT,                   -- success / failed
    rows_in     INT,
    rows_out    INT,
    notes       TEXT
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_raw_fetched   ON raw.weather_api (fetched_at);
CREATE INDEX IF NOT EXISTS idx_pred_target   ON ml.predictions (location, target_ts);
