"""Гейты кандидата §6.2.4 (без пересчёта длины токенов, без BM25).

Ненулевой код, если не выполнено любое из условий плана.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_COLLECTIONS = ("code_proc", "config_xml", "metadata_xml")
EXPECTED_DIM = 768
EXPECTED_META_COUNT = 25_026


def _load_dry_n_chunks(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "n_chunks" not in data:
        raise SystemExit(f"в сухом прогоне нет n_chunks: {path}")
    return int(data["n_chunks"])


def _load_code_relevant(gold_paths: list[Path]) -> set[str]:
    sources: set[str] = set()
    for path in gold_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            scope = row.get("scope")
            if scope is not None and scope != "code":
                continue
            for src in row.get("relevant_sources", []):
                norm = src.replace("\\", "/")
                if norm.lower().endswith(".bsl"):
                    sources.add(norm)
    return sources


def _nonempty_xml_sources(root: Path) -> set[str]:
    out: set[str] = set()
    for path in root.rglob("*.xml"):
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            text = raw.decode("utf-8-sig", errors="replace")
        else:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("cp1251", errors="replace")
        if text.strip():
            out.add(str(path.relative_to(root)).replace("\\", "/"))
    return out


def _fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)
    print(f"GATE_FAIL: {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify candidate Chroma gates §6.2.4")
    ap.add_argument("--chroma-dir", type=Path, required=True)
    ap.add_argument("--bsl-dry", type=Path, required=True)
    ap.add_argument("--xml-dry", type=Path, required=True)
    ap.add_argument("--xml-root", type=Path, required=True)
    ap.add_argument(
        "--gold",
        action="append",
        type=Path,
        required=True,
        help="повторяемый gold JSONL (три файла §5)",
    )
    args = ap.parse_args()

    if len(args.gold) != 3:
        print(f"нужны ровно три --gold, получено {len(args.gold)}", file=sys.stderr)
        return 1

    bsl_n = _load_dry_n_chunks(args.bsl_dry)
    xml_n = _load_dry_n_chunks(args.xml_dry)
    gold_sources = _load_code_relevant(args.gold)
    nonempty_xml = _nonempty_xml_sources(args.xml_root)

    os.environ["MCP_1C_CHROMA_DIR"] = str(args.chroma_dir)
    sys.path.insert(0, str(ROOT / "src"))

    from mcp_1c_rag.store import get_client  # noqa: E402

    client = get_client()
    errors: list[str] = []

    names = sorted(c.name for c in client.list_collections())
    if tuple(names) != EXPECTED_COLLECTIONS:
        _fail(
            f"коллекции {names} != {list(EXPECTED_COLLECTIONS)}",
            errors,
        )

    all_sources: set[str] = set()

    for name in EXPECTED_COLLECTIONS:
        if name not in names:
            _fail(f"нет коллекции {name}", errors)
            continue
        coll = client.get_collection(name)
        count = coll.count()
        if count <= 0:
            _fail(f"{name}: пустая коллекция", errors)
            continue

        # Один вектор — размерность; полный перебор документов батчами.
        peek = coll.get(limit=1, include=["embeddings", "documents", "metadatas"])
        emb = (peek.get("embeddings") or [None])[0]
        if emb is None:
            _fail(f"{name}: нет embedding для dim", errors)
            continue
        dim = len(emb)
        if dim != EXPECTED_DIM:
            _fail(f"{name}: dim={dim} != {EXPECTED_DIM}", errors)

        ids: list[str] = []
        docs: list[str] = []
        sources: list[str] = []
        offset = 0
        batch = 5000
        while offset < count:
            batch_data = coll.get(
                limit=batch,
                offset=offset,
                include=["documents", "metadatas"],
            )
            batch_ids = batch_data.get("ids") or []
            if not batch_ids:
                break
            batch_docs = batch_data.get("documents") or []
            batch_metas = batch_data.get("metadatas") or []
            for cid, doc, meta in zip(batch_ids, batch_docs, batch_metas, strict=False):
                ids.append(cid)
                docs.append(doc or "")
                src = str((meta or {}).get("source", "")).replace("\\", "/")
                sources.append(src)
                if src:
                    all_sources.add(src)
            offset += len(batch_ids)

        if any(not d.strip() for d in docs):
            _fail(f"{name}: есть пустые тексты", errors)

        if name == "code_proc":
            if len(ids) != bsl_n:
                _fail(f"code_proc count={len(ids)} != bsl dry n_chunks={bsl_n}", errors)
            if any(i.startswith("xml:") for i in ids):
                _fail("code_proc: есть id с префиксом xml:", errors)
        elif name == "config_xml":
            if len(ids) != xml_n:
                _fail(f"config_xml count={len(ids)} != xml dry n_chunks={xml_n}", errors)
            if not all(i.startswith("xml:") for i in ids):
                _fail("config_xml: не все id начинаются с xml:", errors)
            src_set = {s for s in sources if s}
            if src_set != nonempty_xml:
                missing = sorted(nonempty_xml - src_set)[:10]
                extra = sorted(src_set - nonempty_xml)[:10]
                _fail(
                    f"config_xml sources != nonempty xml "
                    f"(missing={len(nonempty_xml - src_set)} extra={len(src_set - nonempty_xml)} "
                    f"missing_ex={missing} extra_ex={extra})",
                    errors,
                )
            if any(not s.lower().endswith(".xml") for s in src_set):
                _fail("config_xml: source не оканчивается на .xml", errors)
        elif name == "metadata_xml":
            if len(ids) != EXPECTED_META_COUNT:
                _fail(
                    f"metadata_xml count={len(ids)} != {EXPECTED_META_COUNT}",
                    errors,
                )
            if not all(i.startswith("metaobj:") for i in ids):
                _fail("metadata_xml: не все id начинаются с metaobj:", errors)

    missing_gold = sorted(s for s in gold_sources if s not in all_sources)
    if missing_gold:
        _fail(
            f"code relevant_sources отсутствуют в source: {len(missing_gold)} "
            f"ex={missing_gold[:20]}",
            errors,
        )

    report = {
        "chroma_dir": str(args.chroma_dir),
        "collections": names,
        "bsl_dry_n_chunks": bsl_n,
        "xml_dry_n_chunks": xml_n,
        "n_gold_code_sources": len(gold_sources),
        "n_nonempty_xml": len(nonempty_xml),
        "errors": errors,
        "ok": not errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if errors:
        print("GATES_FAIL", flush=True)
        return 1
    print("GATES_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
