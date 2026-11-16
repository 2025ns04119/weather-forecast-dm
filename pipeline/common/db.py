"""Shared database helpers: Neon connection + pipeline run log (metadata/lineage)."""
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def get_engine():
    url = os.getenv("NEON_DB_URL")
    if not url:
        raise RuntimeError("NEON_DB_URL is not set (check .env or GitHub Secrets)")
    # SQLAlchemy 2.1 defaults to the psycopg (v3) driver; force psycopg2
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    return create_engine(url, pool_pre_ping=True)


def start_run(engine, pipeline: str) -> int:
    """Insert a row in ml.pipeline_runs and return its run_id."""
    with engine.begin() as conn:
        return conn.execute(
            text("INSERT INTO ml.pipeline_runs (pipeline, status) "
                 "VALUES (:p, 'running') RETURNING run_id"),
            {"p": pipeline},
        ).scalar()


def finish_run(engine, run_id: int, status: str,
               rows_in: int = None, rows_out: int = None, notes: str = None):
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE ml.pipeline_runs SET finished_at = now(), status = :s, "
                 "rows_in = :ri, rows_out = :ro, notes = :n WHERE run_id = :id"),
            {"s": status, "ri": rows_in, "ro": rows_out, "n": notes, "id": run_id},
        )
