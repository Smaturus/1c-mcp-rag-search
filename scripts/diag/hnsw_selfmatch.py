import sys

sys.stdout.reconfigure(encoding="utf-8")

from mcp_1c_rag.embeddings import embed_query
from mcp_1c_rag.store import get_code_collection, query_collection

col = get_code_collection()
print("count:", col.count())

# Точная фраза из реального чанка Documents/СчетФактураВыданный/Ext/ObjectModule.bsl
probe = "Отражение начисления НДС с авансов для последующего отражения в книге продаж"
hits = query_collection(col, embed_query(probe), 5)
for i, h in enumerate(hits, 1):
    m = h.get("metadata") or {}
    print(f"[{i}] {m.get('source')} chunk={m.get('chunk_index')} dist={h['distance']:.4f}")

# Прямая проверка содержимого sqlite: есть ли вообще чанки СчетФактураВыданный в коллекции
got = col.get(where={"source": "Documents/СчетФактураВыданный/Ext/ObjectModule.bsl"}, limit=3,
              include=["documents"])
print("chunks of СчетФактураВыданный in sqlite via get():", len(got.get("ids", [])))
