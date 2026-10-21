import sys

sys.stdout.reconfigure(encoding="utf-8")

import chromadb

from mcp_1c_rag.store import get_code_collection, get_metadata_collection

print("chromadb version:", chromadb.__version__)

for name, getter in [("metadata", get_metadata_collection), ("code", get_code_collection)]:
    try:
        col = getter()
        n = col.count()
        print(f"collection {name}: OK, count={n}")
    except Exception as exc:
        print(f"collection {name}: FAIL: {type(exc).__name__}: {str(exc)[:300]}")
