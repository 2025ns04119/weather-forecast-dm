-- Runs automatically the first time the database container starts

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Medallion layers
CREATE SCHEMA IF NOT EXISTS raw;        -- bronze: API data as received
CREATE SCHEMA IF NOT EXISTS clean;      -- silver: validated + cleaned
CREATE SCHEMA IF NOT EXISTS features;   -- gold: ML features
CREATE SCHEMA IF NOT EXISTS quarantine; -- rows that failed validation
CREATE SCHEMA IF NOT EXISTS ml;         -- predictions + monitoring

-- Bronze table: raw API payload
CREATE TABLE IF NOT EXISTS raw.weather_api (
    id           BIGSERIAL,
    fetched_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    location     TEXT        NOT NULL,
    payload      JSONB       NOT NULL,
    PRIMARY KEY (id, fetched_at)
);
SELECT create_hypertable('raw.weather_api', 'fetched_at', if_not_exists => TRUE);

-- Governance: role-based access
CREATE ROLE etl_user     LOGIN PASSWORD 'etl_pass';
CREATE ROLE ml_user      LOGIN PASSWORD 'ml_pass';
CREATE ROLE analyst_user LOGIN PASSWORD 'analyst_pass';

GRANT USAGE, CREATE ON SCHEMA raw, clean, quarantine TO etl_user;
GRANT USAGE ON SCHEMA clean TO ml_user;
GRANT USAGE, CREATE ON SCHEMA features, ml TO ml_user;
GRANT USAGE ON SCHEMA clean, features, ml TO analyst_user;

GRANT ALL ON ALL TABLES IN SCHEMA raw TO etl_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA raw TO etl_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA clean GRANT SELECT ON TABLES TO ml_user, analyst_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA features, ml GRANT SELECT ON TABLES TO analyst_user;
