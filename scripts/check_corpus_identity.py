"""Content-hash gate: доказать, что корпус и BM25 соответствуют chroma_xml.

Без прохождения этого гейта полная сборка e5-large запрещена: если тексты
чанков или множества id расходятся, сравнение моделей превращается в
«смену модели + смену корпуса», и старый BM25 переиспользовать нельзя.

Проверяет (всё — только чтение производственных артефактов):

1. На 10 случайных ``bsl_proc:`` id в ``chroma_xml`` есть ключ
   ``chroma:document`` с непустым текстом (иначе гейт в принципе невозможен).
2. Множество ``bsl_proc:`` id из ``data/bm25_all/meta.json`` (ключ ``doc_ids``)
   равно множеству ``bsl_proc:`` id в ``chroma_xml`` и равно множеству id,
   восстановленных заново из ``1C_Corpus``.
3. SHA-256 текста каждого чанка (старый из ``chroma_xml`` vs новый из
   повторной нарезки ``chunk_bsl_file(max_chars=1000, overlap=150)`` на
   ``sorted(rglob("*.bsl"))``) совпадает для каждого id.
4. Записывает манифест: SHA-256 отсортированного списка ``путь + sha256 файла``,
   число чанков, SHA-256 множества текстов чанков.

Exit code: 0 = PASS, != 0 = STOP (любое расхождение).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.chunking_procedures import chunk_bsl_file  # noqa: E402


def chunk_id(source: str, index: int) -> str:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    return f"bsl_proc:{digest}:{index}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_lines(lines: list[str]) -> str:
    """SHA-256 детерминированной склейки отсортированных строк."""
    return hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Content-hash gate корпуса и BM25")
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--chroma-db", type=Path, required=True)
    ap.add_argument("--bm25-meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="путь к manifest.json")
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--sample-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    started = time.time()
    errors: list[str] = []

    # ---- 0. bm25 doc_ids (bsl_proc only) ---------------------------------
    meta = json.loads(args.bm25_meta.read_text(encoding="utf-8"))
    bm25_ids = [x for x in meta["doc_ids"] if x.startswith("bsl_proc:")]
    bm25_id_set = set(bm25_ids)
    n_bm25 = len(bm25_id_set)
    print(f"[bm25] bsl_proc ids = {n_bm25}", flush=True)

    # ---- 1. chroma: bsl_proc id -> (text, source) ------------------------
    db = sqlite3.connect(f"file:{args.chroma_db.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        doc_rows = db.execute(
            """
            SELECT e.embedding_id AS cid, md.string_value AS doc
            FROM embeddings e
            JOIN embedding_metadata md ON md.id = e.id AND md.key = 'chroma:document'
            WHERE e.embedding_id LIKE 'bsl_proc:%'
            """
        )
        old_text: dict[str, str] = {}
        chroma_ids: set[str] = set()
        for row in doc_rows:
            cid = row["cid"]
            chroma_ids.add(cid)
            old_text[cid] = row["doc"] or ""
    finally:
        db.close()

    print(f"[chroma] bsl_proc ids = {len(chroma_ids)}", flush=True)

    # ---- 2. 10 случайных id: chroma:document непустой --------------------
    rng = random.Random(args.seed)
    sample = rng.sample(sorted(bm25_id_set), min(args.sample_n, len(bm25_id_set)))
    empty_in_sample = [cid for cid in sample if not (old_text.get(cid) or "").strip()]
    print(f"[sample] {len(sample)} случайных id, пустых текстов: {len(empty_in_sample)}",
          flush=True)
    if empty_in_sample:
        errors.append(
            f"chroma:document пуст для {len(empty_in_sample)} из {len(sample)} "
            f"случайных id: {empty_in_sample[:5]}"
        )

    # ---- 3. повторная нарезка корпуса ------------------------------------
    files = sorted(args.corpus_root.rglob("*.bsl"))
    print(f"[corpus] файлов .bsl = {len(files)}", flush=True)

    new_text_hashes: dict[str, str] = {}
    new_ids: set[str] = set()
    file_manifest: list[str] = []
    chunk_hashes: list[str] = []

    for path in files:
        rel = str(path.relative_to(args.corpus_root)).replace("\\", "/")
        file_manifest.append(f"{rel}\t{sha256_bytes(path.read_bytes())}")
        try:
            chunks = chunk_bsl_file(
                path, args.corpus_root, max_chars=args.max_chars, overlap=args.overlap
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ошибка нарезки {rel}: {exc}")
            continue
        for c in chunks:
            cid = chunk_id(rel, c.chunk_index)
            new_ids.add(cid)
            new_text_hashes[cid] = sha256_text(c.text)
            chunk_hashes.append(sha256_text(c.text))

    n_new = len(new_ids)
    print(f"[rechunk] id = {n_new}", flush=True)

    # ---- 4. сравнение ----------------------------------------------------
    id_sets_equal = (bm25_id_set == chroma_ids == new_ids)
    if bm25_id_set != chroma_ids:
        errors.append(
            f"множества id расходятся: bm25={len(bm25_id_set)} vs "
            f"chroma={len(chroma_ids)} (симм. разность "
            f"{len(bm25_id_set ^ chroma_ids)})"
        )
    if bm25_id_set != new_ids:
        errors.append(
            f"множества id расходятся: bm25={len(bm25_id_set)} vs "
            f"rechunk={len(new_ids)} (симм. разность "
            f"{len(bm25_id_set ^ new_ids)})"
        )

    missing_old = bm25_id_set - set(old_text.keys())
    if missing_old:
        errors.append(f"нет chroma:document для {len(missing_old)} id")

    text_mismatch: list[str] = []
    for cid in sorted(bm25_id_set):
        old_h = sha256_text(old_text.get(cid, ""))
        new_h = new_text_hashes.get(cid)
        if new_h is None or old_h != new_h:
            text_mismatch.append(cid)
            if len(text_mismatch) > 20:
                break
    if text_mismatch:
        errors.append(
            f"расхождение SHA-256 текста у {len(text_mismatch)} id "
            f"(первые: {text_mismatch[:5]})"
        )

    # ---- 5. манифест -----------------------------------------------------
    manifest = {
        "file_list_sha256": sha256_lines(file_manifest),
        "n_files": len(files),
        "n_chunks": n_new,
        "chunk_text_set_sha256": sha256_lines(chunk_hashes),
        "n_bm25_bsl_proc": n_bm25,
        "n_chroma_bsl_proc": len(chroma_ids),
        "id_sets_equal": id_sets_equal,
        "text_hashes_equal": not text_mismatch,
        "sample_empty_count": len(empty_in_sample),
        "errors": errors,
        "elapsed_sec": round(time.time() - started, 1),
        "params": {"max_chars": args.max_chars, "overlap": args.overlap,
                   "seed": args.seed, "sample_n": args.sample_n},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    passed = not errors
    print("\n=== CONTENT-HASH GATE ===", flush=True)
    print(f"file_list_sha256   = {manifest['file_list_sha256']}", flush=True)
    print(f"chunk_text_set_sha = {manifest['chunk_text_set_sha256']}", flush=True)
    print(f"n_files            = {manifest['n_files']}", flush=True)
    print(f"n_chunks           = {manifest['n_chunks']}", flush=True)
    print(f"id_sets_equal      = {id_sets_equal}", flush=True)
    print(f"text_hashes_equal  = {not text_mismatch}", flush=True)
    print(f"elapsed_sec        = {manifest['elapsed_sec']}", flush=True)
    if errors:
        print("\nFAILED:", flush=True)
        for e in errors:
            print("  -", e, flush=True)
    print(f"\nRESULT: {'PASS' if passed else 'STOP'}", flush=True)
    print(f"manifest: {args.out}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
