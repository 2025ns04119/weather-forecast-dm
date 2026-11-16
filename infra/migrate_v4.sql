-- ============================================================
-- Migration v4: second data source (city catalogue CSV) + data-lake lineage
-- Safe to re-run.   python infra/run_sql.py infra/migrate_v4.sql
-- ============================================================

-- BRONZE: every CSV row stored as received; file_md5 identifies the file version
CREATE TABLE IF NOT EXISTS raw.city_catalogue (
    id           BIGSERIAL PRIMARY KEY,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_file  TEXT        NOT NULL,
    file_md5     TEXT        NOT NULL,
    lake_path    TEXT,                       -- copy of the file in the data lake
    row_num      INT         NOT NULL,
    record       JSONB       NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_citycat_md5 ON raw.city_catalogue (file_md5);

-- QUARANTINE: CSV rows that failed validation
CREATE TABLE IF NOT EXISTS quarantine.location_rejects (
    id           BIGSERIAL PRIMARY KEY,
    city         TEXT,
    reason       TEXT        NOT NULL,
    record       JSONB       NOT NULL,
    file_md5     TEXT,
    rejected_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- SILVER: location dimension enriched from the catalogue
ALTER TABLE clean.locations
    ADD COLUMN IF NOT EXISTS country      TEXT,
    ADD COLUMN IF NOT EXISTS continent    TEXT,
    ADD COLUMN IF NOT EXISTS timezone     TEXT,
    ADD COLUMN IF NOT EXISTS climate_zone TEXT,
    ADD COLUMN IF NOT EXISTS source       TEXT,
    ADD COLUMN IF NOT EXISTS updated_at   TIMESTAMPTZ;

-- Lineage: link each bronze API record to its file in the data lake
ALTER TABLE raw.weather_api ADD COLUMN IF NOT EXISTS lake_path TEXT;
