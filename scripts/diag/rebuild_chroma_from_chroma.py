"""Rebuild chroma index from sqlite into a fresh dir, reusing stored embeddings."""

import shutil
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from chromadb.config import Settings

OLD = r"D:\MCP_1С_homework\data\chroma"
NEW = r"D:\MCP_1С_homework\data\chroma_rebuilt"
BATCH = 2000

shutil.rmtree(NEW, ignore_errors=True)

old_client = chromadb.PersistentClient(path=OLD, settings=Settings(anonymized_telemetry=False))
new_client = chromadb.PersistentClient(path=NEW, settings=Settings(anonymized_telemetry=False))

for name in ("code", "metadata"):
    src = old_client.get_collection(name)
    dst = new_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "hnsw:sync_threshold": 1000000},
    )
    total = src.count()
    t0 = time.time()
    offset = 0
    copied = 0
    while offset < total:
        page = src.get(
            limit=BATCH,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = page["ids"]
        if not ids:
            break
        dst.add(
            ids=ids,
            embeddings=page["embeddings"],
            documents=page["documents"],
            metadatas=page["metadatas"],
        )
        copied += len(ids)
        offset += BATCH
    print(f"{name}: copied {copied}/{total} in {time.time()-t0:.0f}s")
    assert copied == total, f"copy mismatch: {copied} != {total}"

print("REBUILD DONE")
