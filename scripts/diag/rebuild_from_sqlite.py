"""Rebuild chroma collections by re-embedding documents from the old sqlite.

The 1.5.9 compactor bug (#7090) destroyed the vectors (WAL purged, vector
files never written); documents/metadata in sqlite are intact. This script
re-embeds them with the same model/prefix as the original indexer and fills
a NEW database (data/chroma_rebuilt) with hnsw:sync_threshold=1000000 so the
compactor never runs again.

Resume-safe: journal data/reindex_done.txt (one chroma id per line, appended
after each successful upsert); upsert makes reruns idempotent.

Usage:
  python scripts\\diag\\rebuild_from_sqlite.py            # full run
  python scripts\\diag\\rebuild_from_sqlite.py --reset    # drop chroma_rebuilt first
  python scripts\\diag\\rebuild_from_sqlite.py --max-batches 1  # smoke
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = Path(__file__).resolve().parents[2]
OLD_DB = PROJECT / "data" / "chroma" / "chroma.sqlite3"
NEW_DIR = PROJECT / "data" / "chroma_rebuilt"
JOURNAL = PROJECT / "data" / "reindex_done.txt"
STATUS = PROJECT / "data" / "reindex_status.txt"

READ_BATCH = 256
EMBED_BATCH = 32
STATUS_EVERY = 5

MODEL = "intfloat/multilingual-e5-small"

SQL = """
SELECT c.name AS coll, e.embedding_id AS cid, e.id AS num,
       md.string_value AS doc,
       src.string_value AS source,
       kind.string_value AS kind,
       COALESCE(idx.int_value, 0) AS chunk_index
FROM embeddings e
JOIN segments s ON s.id = e.segment_id
JOIN collections c ON c.id = s.collection
LEFT JOIN embedding_metadata md  ON md.id  = e.id AND md.key  = 'chroma:document'
LEFT JOIN embedding_metadata src ON src.id = e.id AND src.key = 'source'
LEFT JOIN embedding_metadata kind ON kind.id = e.id AND kind.key = 'kind'
LEFT JOIN embedding_metadata idx ON idx.id = e.id AND idx.key = 'chunk_index'
ORDER BY e.id
"""


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)


def write_status(finished: bool, done: int, total: int, rate: float, note: str) -> None:
    eta_h = (total - done) / rate / 60.0 if rate > 0 else float("inf")
    STATUS.write_text(
        "\n".join(
            [
                f"time={time.strftime('%Y-%m-%d %H:%M:%S')}",
                "phase=chroma_rebuild",
                f"finished={'True' if finished else 'False'}",
                f"summary_ru=Пересборка индекса: {done}/{total} чанков, {rate:.0f} чанков/мин, ETA {eta_h:.1f} ч",
                f"progress: {done}/{total} ({done / max(total, 1) * 100:.1f}%), rate={rate:.0f}/min, eta={eta_h:.2f}h",
                f"note: {note}",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="drop chroma_rebuilt and journal")
    ap.add_argument("--max-batches", type=int, default=0, help="stop after N batches (smoke)")
    args = ap.parse_args()

    if args.reset:
        import shutil

        shutil.rmtree(NEW_DIR, ignore_errors=True)
        JOURNAL.unlink(missing_ok=True)
        log("reset: removed chroma_rebuilt and journal")

    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    done_ids: set[str] = set()
    if JOURNAL.exists():
        done_ids = {
            line.strip()
            for line in JOURNAL.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        log(f"resume: {len(done_ids)} ids already in journal")

    new_client = chromadb.PersistentClient(
        path=str(NEW_DIR), settings=Settings(anonymized_telemetry=False)
    )
    cols = {}
    for name in ("code", "metadata"):
        cols[name] = new_client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine", "hnsw:sync_threshold": 1000000},
        )
        log(f"collection {name}: metadata={cols[name].metadata}")

    log(f"loading model {MODEL} ...")
    model = SentenceTransformer(MODEL)
    log("model loaded")

    db = sqlite3.connect(f"file:{OLD_DB.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    cur = db.execute(SQL)

    journal_f = JOURNAL.open("a", encoding="utf-8")
    total_expected = 124253
    t_start = time.time()
    embedded = 0
    skipped = 0
    batch_no = 0

    while True:
        rows = cur.fetchmany(READ_BATCH)
        if not rows:
            break
        batch_no += 1
        pending = []
        for r in rows:
            if r["cid"] in done_ids:
                skipped += 1
                continue
            if not r["doc"]:
                log(f"WARN: empty document for id={r['cid']} ({r['coll']}), skip")
                continue
            pending.append(r)
        if pending:
            texts = [r["doc"] for r in pending]
            vectors = model.encode(
                [f"passage: {t}" for t in texts],
                batch_size=EMBED_BATCH,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            cols[pending[0]["coll"]].upsert(
                ids=[r["cid"] for r in pending],
                embeddings=vectors,
                documents=texts,
                metadatas=[
                    {
                        "source": r["source"] or "",
                        "kind": r["kind"] or "",
                        "chunk_index": int(r["chunk_index"] or 0),
                    }
                    for r in pending
                ],
            )
            journal_f.write("\n".join(r["cid"] for r in pending) + "\n")
            journal_f.flush()
            done_ids.update(r["cid"] for r in pending)
            embedded += len(pending)
        if batch_no % STATUS_EVERY == 0 or args.max_batches:
            elapsed = time.time() - t_start
            rate = embedded / elapsed * 60.0 if elapsed > 0 else 0.0
            write_status(False, embedded + skipped, total_expected, rate, "running")
            log(
                f"batch {batch_no}: embedded={embedded} skipped={skipped} "
                f"rate={rate:.0f}/min eta={(total_expected - embedded - skipped) / max(rate, 1):.1f}min"
            )
        if args.max_batches and batch_no >= args.max_batches:
            log(f"smoke stop after {batch_no} batches")
            break

    db.close()
    journal_f.close()
    counts = {name: cols[name].count() for name in cols}
    elapsed = time.time() - t_start
    log(f"DONE in {elapsed / 60:.1f} min, counts={counts}")
    write_status(True, embedded + skipped, total_expected, embedded / elapsed * 60.0 if elapsed else 0, f"counts={counts}")


if __name__ == "__main__":
    main()
