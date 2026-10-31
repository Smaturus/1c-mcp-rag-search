import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = Path(__file__).resolve().parents[2]
DB = PROJECT / "data" / "chroma" / "chroma.sqlite3"

db = sqlite3.connect(DB)
db.row_factory = sqlite3.Row

for table in ("embeddings", "embedding_metadata", "embedding_fulltext_search_content",
              "embeddings_queue", "segments"):
    print(f"--- {table} ---")
    try:
        cols = [r[1] for r in db.execute(f"PRAGMA table_info({table})")]
        print("cols:", cols)
        for row in db.execute(f"SELECT * FROM {table} LIMIT 2"):
            vals = {}
            for k in row.keys():
                v = row[k]
                if isinstance(v, bytes) and len(v or b"") > 40:
                    v = f"<bytes {len(v)}>"
                vals[k] = v
            print(vals)
    except Exception as exc:
        print("ERR:", exc)

print("--- metadata keys distribution ---")
for row in db.execute(
    "SELECT key, COUNT(*) FROM embedding_metadata GROUP BY key ORDER BY 2 DESC"
):
    print(tuple(row))

print("--- collections map ---")
for row in db.execute("SELECT id, name FROM collections"):
    print(tuple(row))

print("--- embeddings per segment ---")
for row in db.execute("SELECT segment_id, COUNT(*) FROM embeddings GROUP BY segment_id"):
    print(tuple(row))

db.close()
