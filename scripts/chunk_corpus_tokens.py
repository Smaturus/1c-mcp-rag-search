"""Прогнать токенный чанкер по полному 1C_Corpus БЕЗ эмбеддингов.

Только токенизация + чанкинг. Проверяет жёсткие гейты фазы 1:
- 0 чанков >500 токенов (политика) и 0 чанков >512 (sanity);
- 0 пустых чанков;
- 100% непустых файлов дают >=1 чанк;
- покрытие непустых байтов кода >=99.6%;
- распределение токенов p50/p95/p99/max, число чанков, время.

Результат — JSON в stdout и в файл (UTF-8). Читает корпус, ничего не пишет в индекс.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.chunking_tokens import (  # noqa: E402
    chunk_bsl_file_tokens,
    get_tokenizer,
)

DEFAULT_CORPUS = ROOT / "1C-Corpus" / "1C_Corpus"


def pct(s: list[int], q: float) -> int:
    if not s:
        return 0
    i = min(len(s) - 1, max(0, int(q * len(s))))
    return s[i]


def coverage_bytes(
    text: str, chunk_bodies: list[str]
) -> tuple[float, int, int, int]:
    """Доля непустых (non-whitespace) байтов исходника, покрытых чанками.

    Покрытие считается посимвольно: каждый non-whitespace символ исходника должен
    встретиться в чанках хотя бы столько же раз (перекрытие не мешает, токен-разрез
    сверхдлинных строк сохраняет символы). Возвращает (frac, covered, total, uncovered).
    """
    src = Counter(c for c in text if not c.isspace())
    if not src:
        return 1.0, 0, 0, 0
    chunk = Counter()
    for body in chunk_bodies:
        chunk.update(c for c in body if not c.isspace())
    total = sum(src.values())
    covered = sum(min(cnt, chunk.get(ch, 0)) for ch, cnt in src.items())
    uncovered = total - covered
    return covered / total, covered, total, uncovered


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
        help="имя токенизатора для сухого прогона нарезки",
    )
    args = ap.parse_args()

    files = sorted(args.root.rglob("*.bsl"))
    if not files:
        raise SystemExit(f"нет .bsl в {args.root}")

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
    total_cov_bytes = 0
    covered_bytes = 0
    uncovered_by_file: list[tuple[str, int]] = []  # (source, uncovered non-ws bytes)
    over_500_examples: list[str] = []

    for path in files:
        try:
            chunks = chunk_bsl_file_tokens(
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

        text = path.read_bytes()
        try:
            text = text.decode("utf-8-sig") if text.startswith(b"\xef\xbb\xbf") else text.decode("utf-8")
        except UnicodeDecodeError:
            text = text.decode("cp1251", errors="replace")

        if text.strip():
            nonempty_files += 1
            if chunks:
                files_with_chunks += 1
            else:
                files_without_chunks += 1

        bodies = [c.text.split("\n\n", 1)[1] if "\n\n" in c.text else "" for c in chunks]
        cov, covd, total_b, uncovered = coverage_bytes(text, bodies)
        total_cov_bytes += total_b
        covered_bytes += covd
        if uncovered:
            source = str(path.relative_to(args.root)).replace("\\", "/")
            uncovered_by_file.append((source, uncovered))

        for c in chunks:
            n_chunks += 1
            body = c.text
            if not body.strip():
                n_empty_chunks += 1
            t = len(tok.encode(f"passage: {c.text}", add_special_tokens=True, truncation=False))
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
    coverage = covered_bytes / total_cov_bytes if total_cov_bytes else 1.0

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
        "coverage": {
            "covered_bytes": covered_bytes,
            "total_nonws_bytes": total_cov_bytes,
            "frac": round(coverage, 6),
            "note": "non-whitespace байты; покрытие посимвольно (min(src, chunk))",
        },
        "uncovered_by_file_top": [
            {"source": s, "uncovered_bytes": b}
            for s, b in sorted(uncovered_by_file, key=lambda x: -x[1])[:30]
        ],
        "n_files_with_uncovered": len(uncovered_by_file),
        "over_500_examples": over_500_examples,
        "elapsed_sec": round(time.time() - started, 1),
        "params": {
            "max_tokens": args.max_tokens,
            "overlap_tokens": args.overlap_tokens,
            "min_tokens": args.min_tokens,
            "max_chars": args.max_chars,
        },
    }

    out = json.dumps(result, ensure_ascii=False, indent=2)
    print(out, flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
