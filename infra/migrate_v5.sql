-- ============================================================
-- Migration v5: feature store registry table (safe to re-run)
--   python infra/run_sql.py infra/migrate_v5.sql
-- Filled from pipeline/feature_store/registry.yaml by:  python -m pipeline.feature_store.store --sync
-- ============================================================
CREATE TABLE IF NOT EXISTS features.feature_registry (
    feature_name     TEXT        NOT NULL,
    feature_version  TEXT        NOT NULL,
    dtype            TEXT,
    source           TEXT,
    transformation   TEXT,
    description      TEXT,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (feature_name, feature_version)
);
