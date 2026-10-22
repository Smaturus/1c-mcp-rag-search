"""Evaluate ``meta_upp11`` (XML metadata) retrieval with structural ground truth.

The gold answers here are structural, not subjective: for an object we know its
exact name; for an attribute we know the owning object and the field name.
Queries are generated from the parsed metadata itself, so the test measures
coverage of the index, not human judgement.

Usage:
    $env:MCP_1C_CHROMA_DIR = "data\\chroma_xml"
    python scripts/eval_metadata_xml.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.bm25 import Bm25Index  # noqa: E402
from mcp_1c_rag.embeddings import embed_query  # noqa: E402
from mcp_1c_rag.store import get_collection  # noqa: E402

DATA = ROOT / "data" / "metadata_xml"
REPORT = ROOT / "data" / "eval" / "metadata_xml_report.md"
COLLECTION = "meta_upp11"


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _query_cases(objects: list[dict], limit: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    cases: list[dict] = []
    for record in objects:
        if record["kind"] in {"Document", "Catalog", "Report", "DataProcessor", "BusinessProcess"}:
            cases.append(
                {
                    "kind": "object",
                    "query": f"какие реквизиты у {record['kind_ru'].lower()} {record['name']}",
                    "expect_name": record["name"],
                    "expect_level": "object",
                }
            )
        elif record["kind"] in {"AccumulationRegister", "InformationRegister", "AccountingRegister"}:
            cases.append(
                {
                    "kind": "register",
                    "query": f"список измерений регистра {record['name']}",
                    "expect_name": record["name"],
                    "expect_level": "object",
                }
            )
        elif record["kind"] == "Enum" and record["enum_values"]:
            cases.append(
                {
                    "kind": "enum",
                    "query": f"какие значения у перечисления {record['name']}",
                    "expect_name": record["name"],
                    "expect_level": "object",
                }
            )
        for attribute in record["attributes"][:2]:
            cases.append(
                {
                    "kind": "attribute",
                    "query": f"где хранится реквизит {attribute['name']}",
                    "expect_name": record["name"],
                    "expect_level": "реквизит",
                    "expect_field": attribute["name"],
                }
            )
    rng.shuffle(cases)
    return cases[:limit]


def _vector_search(query: str, collection, top_k: int) -> list[dict]:
    raw = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        include=["metadatas", "documents"],
    )
    metas = (raw.get("metadatas") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    return [
        {"metadata": meta or {}, "text": doc or ""}
        for meta, doc in zip(metas, docs, strict=False)
    ]


def _matches(items: list[dict], case: dict, top_k: int) -> bool:
    """Проверить, что в top-k есть нужный объект/реквизит (структурная истина)."""
    for item in items[:top_k]:
        meta = item["metadata"]
        if meta.get("name") != case["expect_name"]:
            continue
        if case["expect_level"] == "object":
            return meta.get("level") == "object"
        return meta.get("field") == case.get("expect_field")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate metadata XML retrieval")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    objects = _read(DATA / "objects.jsonl")
    chunks = [c for c in _read(DATA / "chunks.jsonl") if c["metadata"].get("kind_tag") not in {
        "CommonPicture", "Style", "StyleItem", "Interface", "Language", "WSReference", "XDTOPackage", "Role"}]
    bm25 = Bm25Index.build(((c["id"], c["text"]) for c in chunks))
    collection = get_collection(COLLECTION)

    cases = _query_cases(objects, args.limit)
    if not cases:
        raise SystemExit("нет кейсов — проверьте objects.jsonl")
    hits_vec = hits_bm25 = 0
    by_kind: dict[str, list[int]] = {}
    failures: list[str] = []
    for case in cases:
        vec_items = _vector_search(case["query"], collection, args.top_k)
        bm25_items = [
            {"metadata": next(c["metadata"] for c in chunks if c["id"] == doc_id)}
            for doc_id, _ in bm25.search(case["query"], args.top_k)
        ]
        v = _matches(vec_items, case, args.top_k)
        b = _matches(bm25_items, case, args.top_k)
        hits_vec += v
        hits_bm25 += b
        bucket = by_kind.setdefault(case["kind"], [0, 0, 0])
        bucket[0] += 1
        bucket[1] += int(v)
        bucket[2] += int(b)
        if not v:
            failures.append(f"{case['kind']}: {case['query']}")

    lines = [
        "# Mетаданные из XML: качество поиска",
        "",
        f"- кейсов: {len(cases)} (структурная истина из самой выгрузки)",
        f"- top-k: {args.top_k}",
        "",
        "| Тип кейса | Кейсов | vector Hit@k | BM25 Hit@k |",
        "|---|---|---|---|",
    ]
    for kind, (total, vec, b) in sorted(by_kind.items()):
        lines.append(f"| {kind} | {total} | {vec / total:.1%} | {b / total:.1%} |")
    lines += [
        "",
        f"**Итого vector:** {hits_vec / len(cases):.1%} ({hits_vec}/{len(cases)})",
        f"**Итого BM25:** {hits_bm25 / len(cases):.1%} ({hits_bm25}/{len(cases)})",
    ]
    if failures:
        lines += ["", "## Примеры промахов vector", ""] + [f"- {item}" for item in failures[:10]]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
