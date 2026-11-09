"""Run a SQL file against Neon.   Usage:  python infra/run_sql.py infra/migrate_v2.sql"""
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv()
path = Path(sys.argv[1] if len(sys.argv) > 1 else "infra/migrate_v2.sql")
conn = psycopg2.connect(os.getenv("NEON_DB_URL"))
conn.autocommit = True
conn.cursor().execute(path.read_text(encoding="utf-8"))
conn.close()
print(f"Applied {path}")
