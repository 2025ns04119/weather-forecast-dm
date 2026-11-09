-- ============================================================
-- Migration v2: multi-location support (safe to re-run)
-- ============================================================

-- Location dimension table (reference data for every place we store)
CREATE TABLE IF NOT EXISTS clean.locations (
    location    TEXT PRIMARY KEY,
    latitude    REAL NOT NULL,
    longitude   REAL NOT NULL,
    elevation   REAL,
    role        TEXT NOT NULL DEFAULT 'training',   -- monitored / training
    added_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO clean.locations (location, latitude, longitude, role)
VALUES ('Bengaluru', 12.97, 77.59, 'monitored')
ON CONFLICT (location) DO UPDATE SET role = 'monitored';

-- Feature version v2 adds latitude (location-agnostic global model)
ALTER TABLE features.weather_features ADD COLUMN IF NOT EXISTS latitude REAL;

-- Faster "latest forecast" lookups for the dashboard
CREATE INDEX IF NOT EXISTS idx_pred_made ON ml.predictions (location, made_at);
CREATE INDEX IF NOT EXISTS idx_feat_loc_ver ON features.weather_features (location, feature_version, ts);
