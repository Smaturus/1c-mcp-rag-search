"""Read-only peek into local Chroma chroma.sqlite3 (tables, collections, sample rows)."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "chroma" / "chroma.sqlite3"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--table", default=None, help="Dump sample rows from this table")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    db = args.db
    if not db.exists():
        print(f"Not found: {db}")
        return 1

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cur = con.cursor()
    print(f"DB: {db}")
    print(f"size_MB: {db.stat().st_size / 1e6:.1f}")

    print("\n--- tables (row counts) ---")
    tables = [
        r[0]
        for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
        )
    ]
    for name in tables:
        try:
            count = cur.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        except Exception as exc:  # noqa: BLE001
            count = f"err:{exc}"
        print(f"  {name}: {count}")

    print("\n--- collections ---")
    try:
        for row in cur.execute("SELECT id, name, dimension FROM collections"):
            print(" ", row)
    except Exception as exc:  # noqa: BLE001
        print(" ", exc)

    if args.table:
        print(f"\n--- sample from {args.table} (limit {args.limit}) ---")
        cols = [r[1] for r in cur.execute(f'PRAGMA table_info("{args.table}")')]
        print("  cols:", cols)
        rows = cur.execute(
            f'SELECT * FROM "{args.table}" LIMIT ?', (args.limit,)
        ).fetchall()
        for row in rows:
            # Truncate long blobs/strings for readability
            shown = []
            for cell in row:
                if isinstance(cell, (bytes, memoryview)):
                    shown.append(f"<blob {len(cell)} bytes>")
                elif isinstance(cell, str) and len(cell) > 120:
                    shown.append(cell[:120] + "…")
                else:
                    shown.append(cell)
            print(" ", shown)

    print(
        "\nTip: vectors themselves live mostly in segment folders "
        "(*.bin), not as readable floats in sqlite."
    )
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
