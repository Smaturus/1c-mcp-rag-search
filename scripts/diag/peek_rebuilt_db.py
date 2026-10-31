import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = Path(__file__).resolve().parents[2]
DB = PROJECT / "data" / "chroma_rebuilt" / "chroma.sqlite3"

db = sqlite3.connect(DB)
print("segments:")
for r in db.execute("SELECT id, type, scope, collection FROM segments"):
    print(tuple(r))
print("segment_metadata:")
for r in db.execute("SELECT * FROM segment_metadata"):
    print(tuple(r))
print("collections:")
for r in db.execute("SELECT id, name FROM collections"):
    print(tuple(r))
print("collection_metadata:")
for r in db.execute("SELECT * FROM collection_metadata"):
    print(tuple(r))
print("embeddings_queue rows:", db.execute("SELECT COUNT(*) FROM embeddings_queue").fetchone()[0])
print("embeddings rows:", db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
db.close()
