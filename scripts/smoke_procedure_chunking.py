"""Смоук нарезки по процедурам на реальных .bsl: размеры чанков и покрытие кода.

Считает:
- сколько процедур/чанков получается и сколько из них длиннее лимита e5;
- распределение токенов (p50/p90/p99/max);
- покрытие исходного кода: доля символов файла, попавших в чанки (без служебных
  префиксов), — проверка, что нарезка не «теряет» код.

Запуск:
    python scripts/smoke_procedure_chunking.py --files 1000
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.chunking import chunk_text, read_text_file  # noqa: E402
from mcp_1c_rag.chunking_procedures import chunk_bsl_text  # noqa: E402
from mcp_1c_rag.config import (  # noqa: E402  # noqa: E402
    BSL_CHUNK_CHARS,
    BSL_CHUNK_OVERLAP,
    DATA_DIR,
    EMBEDDING_MODEL,
    STUDY_ROOT,
)

MAX_TOKENS = 512
REPORT = DATA_DIR / "eval" / "procedure_chunking_smoke.md"


def main() -> int:
    ap = argparse.ArgumentParser(description="procedure chunking smoke")
    ap.add_argument("--root", type=Path, default=STUDY_ROOT)
    ap.add_argument("--files", type=int, default=1000)
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=150)
    args = ap.parse_args()

    files = sorted(args.root.rglob("*.bsl"))[: args.files]
    if not files:
        raise SystemExit(f"нет .bsl в {args.root}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

    started = time.time()
    token_lengths: list[int] = []
    proc_chunks = 0
    window_chunks = 0
    files_without_procs = 0
    covered = 0
    total_chars = 0
    for path in files:
        text = read_text_file(path)
        if not text.strip():
            continue
        total_chars += len(text)
        rel = str(path.relative_to(args.root)).replace("\\", "/")
        chunks = chunk_bsl_text(
            text, rel, max_chars=args.max_chars, overlap=args.overlap
        )
        if all(chunk.kind == "module" for chunk in chunks):
            files_without_procs += 1
        body_chars = 0
        for chunk in chunks:
            body = chunk.text.split("\n\n", 1)[-1]
            body_chars += len(body)
            token_lengths.append(len(tokenizer.encode(chunk.text, add_special_tokens=False)))
        covered += min(body_chars, len(text))
        proc_chunks += sum(1 for chunk in chunks if chunk.kind == "procedure")
        window_chunks += sum(1 for chunk in chunks if chunk.kind != "procedure")

    token_lengths.sort()
    over = sum(1 for value in token_lengths if value > MAX_TOKENS)
    old_chunks = 0
    for path in files:
        text = read_text_file(path)
        if not text.strip():
            continue
        old_chunks += len(
            chunk_text(
                text,
                str(path.relative_to(args.root)).replace("\\", "/"),
                BSL_CHUNK_CHARS,
                BSL_CHUNK_OVERLAP,
            )
        )

    def pct(share: float) -> int:
        index = min(len(token_lengths) - 1, int(share * len(token_lengths)))
        return token_lengths[index]

    lines = [
        "# Смоук: нарезка BSL по процедурам",
        "",
        f"- файлов: {len(files)}, без процедур (fallback на окно): {files_without_procs}",
        f"- параметры: max_chars={args.max_chars}, overlap={args.overlap}",
        f"- чанков процедур: {proc_chunks}, служебных/оконных: {window_chunks}",
        f"- было бы при символьном окне ({BSL_CHUNK_CHARS}/{BSL_CHUNK_OVERLAP}): {old_chunks}",
        f"- покрытие исходного кода: **{covered / total_chars:.1%}**",
        "",
        "| Метрика токенов | Значение |",
        "|---|---|",
        f"| p50 | {pct(0.50)} |",
        f"| p90 | {pct(0.90)} |",
        f"| p95 | {pct(0.95)} |",
        f"| p99 | {pct(0.99)} |",
        f"| max | {token_lengths[-1]} |",
        "",
        f"- чанков > лимита {MAX_TOKENS}: **{over}** ({over / len(token_lengths):.1%})",
        f"- среднее: {statistics.fmean(token_lengths):.1f} токенов",
        f"- время: {time.time() - started:.0f} c",
        "",
        "Гейт: нарезка принимается, если покрытие кода ≥ 95 % и доля чанков "
        "сверх лимита ≤ 1 %.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
