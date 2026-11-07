import os
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

load_dotenv()
sql_file = Path(__file__).parent / "neon_init.sql"

conn = psycopg2.connect(os.getenv("NEON_DB_URL"))
conn.autocommit = True
cur = conn.cursor()
cur.execute(sql_file.read_text(encoding="utf-8"))
print("Schema applied.\n")

cur.execute("""
    SELECT table_schema || '.' || table_name
    FROM information_schema.tables
    WHERE table_schema IN ('raw','clean','features','quarantine','ml')
    ORDER BY 1
""")
for r in cur.fetchall():
    print("  -", r[0])
conn.close()