import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.environ["MCP_1C_HYBRID"] = "1"

from mcp_1c_rag.search import codesearch

print("=== HYBRID codesearch('начисление НДС с аванса', 5) ===")
r = codesearch("начисление НДС с аванса", 5)
# печатаем только заголовки hits, чтобы увидеть релевантность источников
for block in r.split("\n\n---\n\n"):
    print(block.splitlines()[0])

print()
print("=== HYBRID codesearch('СчетФактураВыданный', 3) [контроль] ===")
for block in codesearch("СчетФактураВыданный", 3).split("\n\n---\n\n"):
    print(block.splitlines()[0])
