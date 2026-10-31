import pickle
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = Path(__file__).resolve().parents[2]
BACKUP = PROJECT / "data" / "chroma_backup_segments"

BASES = [
    ("metadata vec segment (1ad4885e)", BACKUP / "1ad4885e-9691-43c0-9ecf-f30b7c63108b" / "index_metadata.pickle"),
    ("code vec segment (ec5cc1f7)", BACKUP / "ec5cc1f7-b259-4dee-a4c3-f23f8c2089c5" / "index_metadata.pickle"),
]

for label, path in BASES:
    print(f"=== {label} ===")
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print("pickle.load OK, type:", type(obj))
        if isinstance(obj, dict):
            for k, v in obj.items():
                desc = f"{type(v).__name__}"
                if hasattr(v, "__len__"):
                    desc += f" len={len(v)}"
                print(f"  {k}: {desc}")
                if isinstance(v, dict) and len(str(v)) < 600:
                    print(f"    {v}")
        elif isinstance(obj, (list, tuple)):
            print("len:", len(obj))
            print("first elem type:", type(obj[0]) if obj else None)
    except Exception as exc:
        print(f"pickle.load FAIL: {type(exc).__name__}: {exc}")

print()
print("=== sqlite space usage (top 12) ===")
db = sqlite3.connect(PROJECT / "data" / "chroma" / "chroma.sqlite3")
try:
    for row in db.execute(
        "SELECT name, ROUND(SUM(pgsize)/1048576.0,1) AS mb FROM dbstat "
        "GROUP BY name ORDER BY mb DESC LIMIT 12"
    ):
        print(tuple(row))
except Exception as exc:
    print("dbstat FAIL:", exc)

print()
print("=== WAL queue vectors ===")
for row in db.execute(
    "SELECT topic, COUNT(*), SUM(LENGTH(vector))/1048576.0 FROM embeddings_queue GROUP BY topic"
):
    print(tuple(row))
db.close()
