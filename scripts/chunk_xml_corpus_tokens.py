"""Сухой прогон токенного XML-чанкера по 1C_Corpus БЕЗ эмбеддингов.

Не режим сборки кода. Токенизатор — из ``--embedding-model`` через
``get_tokenizer(model_name)``. Результат — JSON в stdout и опционально в файл.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.chunking_tokens import get_tokenizer  # noqa: E402
from mcp_1c_rag.chunking_xml_tokens import chunk_xml_file_tokens  # noqa: E402

DEFAULT_CORPUS = ROOT / "1C-Corpus" / "1C_Corpus"


def pct(s: list[int], q: float) -> int:
    if not s:
        return 0
    i = min(len(s) - 1, max(0, int(q * len(s))))
    return s[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--overlap-tokens", type=int, default=64)
    ap.add_argument("--min-tokens", type=int, default=32)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--embedding-model",
        required=True,
        help="имя токенизатора для сухого прогона XML",
    )
    args = ap.parse_args()

    files = sorted(args.root.rglob("*.xml"))
    if not files:
        raise SystemExit(f"нет .xml в {args.root}")

    tok = get_tokenizer(args.embedding_model)
    print(f"tokenizer={args.embedding_model}", flush=True)
    started = time.time()

    token_lengths: list[int] = []
    n_over_500 = 0
    n_over_512 = 0
    n_below_min = 0
    n_empty_chunks = 0
    n_chunks = 0
    files_with_chunks = 0
    files_without_chunks = 0
    nonempty_files = 0
    sources: list[str] = []
    over_500_examples: list[str] = []

    for path in files:
        try:
            chunks = chunk_xml_file_tokens(
                path,
                args.root,
                model_name=args.embedding_model,
                max_tokens=args.max_tokens,
                overlap_tokens=args.overlap_tokens,
                min_tokens=args.min_tokens,
                max_chars=args.max_chars,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {path}: {exc}", file=sys.stderr)
            continue

        raw = path.read_bytes()
        try:
            text = (
                raw.decode("utf-8-sig")
                if raw.startswith(b"\xef\xbb\xbf")
                else raw.decode("utf-8")
            )
        except UnicodeDecodeError:
            text = raw.decode("cp1251", errors="replace")

        source = str(path.relative_to(args.root)).replace("\\", "/")
        if text.strip():
            nonempty_files += 1
            if chunks:
                files_with_chunks += 1
                sources.append(source)
            else:
                files_without_chunks += 1

        for c in chunks:
            n_chunks += 1
            if not c.text.strip():
                n_empty_chunks += 1
            t = len(
                tok.encode(
                    f"passage: {c.text}", add_special_tokens=True, truncation=False
                )
            )
            token_lengths.append(t)
            if t < args.min_tokens:
                n_below_min += 1
            if t > 500:
                n_over_500 += 1
                if len(over_500_examples) < 10:
                    over_500_examples.append(f"{c.source_path}#{c.chunk_index} tok={t}")
            if t > 512:
                n_over_512 += 1

    s = sorted(token_lengths)
    result = {
        "corpus": str(args.root),
        "n_files": len(files),
        "nonempty_files": nonempty_files,
        "files_with_chunks": files_with_chunks,
        "files_without_chunks": files_without_chunks,
        "n_chunks": n_chunks,
        "n_empty_chunks": n_empty_chunks,
        "n_over_500": n_over_500,
        "n_over_512": n_over_512,
        "n_below_min_tokens": n_below_min,
        "token_stats": {
            "n": len(token_lengths),
            "min": s[0] if s else None,
            "p50": pct(s, 0.50),
            "p95": pct(s, 0.95),
            "p99": pct(s, 0.99),
            "max": s[-1] if s else None,
            "mean": round(statistics.fmean(token_lengths), 2) if token_lengths else None,
        },
        "n_sources": len(sources),
        "sources": sources,
        "over_500_examples": over_500_examples,
        "elapsed_sec": round(time.time() - started, 1),
        "params": {
            "max_tokens": args.max_tokens,
            "overlap_tokens": args.overlap_tokens,
            "min_tokens": args.min_tokens,
            "max_chars": args.max_chars,
            "embedding_model": args.embedding_model,
        },
    }

    out = json.dumps(result, ensure_ascii=False, indent=2)
    print(out, flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
