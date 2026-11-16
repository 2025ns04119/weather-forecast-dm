"""
Simple custom feature store.

    Registry  : feature_store/registry.yaml  (metadata, versioned with Git)
                -> synced to the table features.feature_registry
    Offline   : features.weather_features    (gold layer, one row per location + hour + version)
    Retrieval : FeatureStore.get_training_features(version)        -> training (many rows)
                FeatureStore.get_online_features(location, version) -> inference (latest rows)
Both retrieval methods use the same registry, so training and serving always get the
same feature names in the same order for a given version.

Usage (from the project root):
    python -m feature_store.store --check      # registry matches the code + data freshness
    python -m feature_store.store --sync       # copy registry.yaml into features.feature_registry
    python -m feature_store.store --describe   # print the feature list
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy import text

from common.logger import get_logger

log = get_logger("feature_store")
REGISTRY_PATH = Path(__file__).with_name("registry.yaml")


def load_registry() -> dict:
    with REGISTRY_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def feature_names(version: str | None = None, registry: dict | None = None) -> list[str]:
    """Feature names of a version, in registry order."""
    reg = registry or load_registry()
    version = version or reg["store"]["current_version"]
    names = [f["name"] for f in reg["features"] if version in f["versions"]]
    if not names:
        raise ValueError(f"Unknown feature version '{version}'")
    return names


class FeatureStore:
    def __init__(self, engine=None):
        if engine is None:
            from common.db import get_engine
            engine = get_engine()
        self.engine = engine
        self.registry = load_registry()
        self.table = self.registry["store"]["offline_table"]
        self.current_version = self.registry["store"]["current_version"]

    # ------------------------------------------------------------ retrieval
    def get_training_features(self, version: str | None = None, locations: list[str] | None = None,
                              start=None, end=None) -> pd.DataFrame:
        """Historical feature rows for training: [location, ts, <features of the version>]."""
        version = version or self.current_version
        cols = feature_names(version, self.registry)
        sql = f"SELECT location, ts, {', '.join(cols)} FROM {self.table} WHERE feature_version = :v"
        params = {"v": version}
        if locations:
            sql += " AND location = ANY(:locs)"
            params["locs"] = list(locations)
        if start is not None:
            sql += " AND ts >= :start"
            params["start"] = start
        if end is not None:
            sql += " AND ts < :end"
            params["end"] = end
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql + " ORDER BY location, ts"), conn, params=params)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        log.info("training retrieval: version=%s rows=%d locations=%d features=%d",
                 version, len(df), df["location"].nunique(), len(cols))
        return df

    def get_online_features(self, location: str, version: str | None = None,
                            n_hours: int = 24) -> pd.DataFrame:
        """Latest n_hours feature rows of one location for inference: [ts, <features>]."""
        version = version or self.current_version
        cols = feature_names(version, self.registry)
        with self.engine.connect() as conn:
            df = pd.read_sql(text(f"""
                SELECT ts, {', '.join(cols)} FROM {self.table}
                WHERE location = :loc AND feature_version = :v
                ORDER BY ts DESC LIMIT :n"""), conn,
                params={"loc": location, "v": version, "n": n_hours})
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.sort_values("ts").reset_index(drop=True)
        if not df.empty:
            age = (pd.Timestamp.now(tz="UTC") - df["ts"].max()).total_seconds() / 3600
            log.info("online retrieval: %s version=%s rows=%d latest=%s (%.1f h old)",
                     location, version, len(df), df["ts"].max(), age)
        return df

    # ------------------------------------------------------------ governance
    def check(self) -> dict:
        """Registry vs code, and freshness / volume of the offline store."""
        from features.build_features import FEATURE_COLS, FEATURE_VERSION
        reg_cols = feature_names(FEATURE_VERSION, self.registry)
        if reg_cols != FEATURE_COLS:
            raise RuntimeError(f"Registry and code disagree for {FEATURE_VERSION}: "
                               f"registry={reg_cols} code={FEATURE_COLS}")
        with self.engine.connect() as conn:
            stats = pd.read_sql(text(f"""
                SELECT feature_version, COUNT(*) AS rows, COUNT(DISTINCT location) AS locations,
                       MAX(ts) AS latest
                FROM {self.table} GROUP BY feature_version ORDER BY feature_version"""), conn)
        log.info("registry matches code: %d features in %s", len(reg_cols), FEATURE_VERSION)
        for r in stats.itertuples(index=False):
            log.info("  version %s: %d rows, %d locations, latest %s",
                     r.feature_version, r.rows, r.locations, r.latest)
        return {"features": len(reg_cols), "versions": stats.to_dict("records")}

    def sync_registry(self) -> int:
        """Copy registry.yaml into features.feature_registry (one row per feature + version)."""
        rows = [{"name": f["name"], "version": v, "dtype": f["dtype"], "source": f["source"],
                 "transformation": f["transformation"], "description": f["description"]}
                for f in self.registry["features"] for v in f["versions"]]
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO features.feature_registry
                    (feature_name, feature_version, dtype, source, transformation, description, updated_at)
                VALUES (:name, :version, :dtype, :source, :transformation, :description, now())
                ON CONFLICT (feature_name, feature_version) DO UPDATE SET
                    dtype = EXCLUDED.dtype, source = EXCLUDED.source,
                    transformation = EXCLUDED.transformation,
                    description = EXCLUDED.description, updated_at = now()"""), rows)
        log.info("registry synced: %d feature/version rows -> features.feature_registry", len(rows))
        return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()

    if args.describe or not (args.check or args.sync):
        reg = load_registry()
        table = pd.DataFrame([{"feature": f["name"], "versions": ",".join(f["versions"]),
                               "transformation": f["transformation"]} for f in reg["features"]])
        print(table.to_string(index=False))
        return

    from common.db import finish_run, get_engine, start_run
    engine = get_engine()
    run_id = start_run(engine, "feature_store")
    try:
        fs = FeatureStore(engine)
        n = fs.sync_registry() if args.sync else 0
        info = fs.check() if args.check else {}
        finish_run(engine, run_id, "success", rows_in=n, rows_out=info.get("features"),
                   notes=json.dumps(info, default=str)[:500])
    except Exception as e:
        finish_run(engine, run_id, "failed", notes=str(e)[:500])
        log.exception("feature store step FAILED: %s", e)
        raise


if __name__ == "__main__":
    main()
