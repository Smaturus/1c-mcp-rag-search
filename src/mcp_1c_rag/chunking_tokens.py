"""Token-aware BSL-модуль чанкинг: процедуры/функции, бюджет по токенам XLM-R.

Новый чанкер для плана token-aware baseline (PLAN_TOKEN_AWARE_CHUNKING.md).
Старый ``chunk_bsl_file`` (символьное окно ``max_chars``) НЕ меняется.

Идея:
- границы ``Процедура``/``Функция`` … ``КонецПроцедуры``/``КонецФункции``;
- бюджет чанка считается ровно так, как текст увидит ``embed_passages``:
  ``len(tokenizer("passage: " + prefix + "\n\n" + body,
  add_special_tokens=True, truncation=False))``, где prefix =
  ``// file: <source> · proc: <name>``;
- политика: каждый чанк <= ``max_tokens`` (500); sanity против 512 — отдельно
  в пост-проверке;
- длинная процедура режется по строкам с токенным перекрытием; строка,
  не влезающая целиком, режется по токенам;
- хвост короче ``min_tokens`` приклеивается к предыдущему чанку, если сумма
  <= ``max_tokens``; иначе остаётся отдельным непустым чанком;
- ``chunk_index`` детерминирован; ID как раньше: ``bsl_proc:<sha1(source)[:16]>:<chunk_index>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from mcp_1c_rag.chunking import read_text_file
from mcp_1c_rag.chunking_procedures import END, PROC_PREFIX, START

__all__ = [
    "TokenChunk",
    "chunk_bsl_file_tokens",
    "chunk_bsl_text_tokens",
    "get_tokenizer",
    "model_input_tokens",
]

PASSAGE_PREFIX = "passage: "
DEFAULT_MAX_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 64
DEFAULT_MIN_TOKENS = 32
DEFAULT_MAX_CHARS = 2000


@dataclass(frozen=True)
class TokenChunk:
    """Чанк (процедура/часть процедуры/module-шапка) с токенным бюджетом."""

    text: str
    source_path: str
    chunk_index: int
    procedure_name: str
    kind: str = "procedure"
    part: int = 1
    parts: int = 1


@lru_cache(maxsize=8)
def get_tokenizer(model_name: str):
    """XLM-R токенизатор по имени модели (локальный кэш HF, без сети)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, local_files_only=True)


def model_input_tokens(prefix: str, body: str, model_name: str) -> int:
    """Точное число токенов модели для чанка ``prefix + "\n\n" + body``.

    Соответствует ``embed_passages``: ``"passage: " + текст_чанка`` и
    ``add_special_tokens=True`` (``<s>``/``</s>``).
    """
    tok = get_tokenizer(model_name)
    return len(
        tok.encode(
            f"{PASSAGE_PREFIX}{prefix}\n\n{body}",
            add_special_tokens=True,
            truncation=False,
        )
    )


def _body_fits(
    prefix: str, body: str, max_tokens: int, max_chars: int, model_name: str
) -> bool:
    if len(body) > max_chars:
        return False
    return model_input_tokens(prefix, body, model_name) <= max_tokens


def _tail_overlap(buf: list[tuple[str, int]], overlap_tokens: int) -> list[tuple[str, int]]:
    """Перенести хвостовые (строка, токены) с суммой <= overlap_tokens в следующий чанк."""
    if not buf:
        return []
    carried: list[tuple[str, int]] = []
    total = 0
    for line, lt in reversed(buf):
        if carried and total + lt > overlap_tokens:
            break
        carried.append((line, lt))
        total += lt
        if total >= overlap_tokens:
            break
    carried.reverse()
    return carried


def _split_long_line(
    prefix: str,
    line: str,
    max_tokens: int,
    overlap_tokens: int,
    max_chars: int,
    model_name: str,
) -> list[str]:
    """Разрезать одну строку, не влезающую в бюджет, по токенам с перекрытием."""
    tok = get_tokenizer(model_name)
    base = len(
        tok.encode(
            f"{PASSAGE_PREFIX}{prefix}\n\n", add_special_tokens=True, truncation=False
        )
    )
    budget = max_tokens - base
    if budget <= 0:
        raise ValueError(f"prefix слишком длинный для бюджета: {max_tokens} токенов")
    # Строка может быть длиннее max_chars — режем по токенам, не по символам.
    ids = tok.encode(line, add_special_tokens=False)
    pieces: list[str] = []
    start = 0
    n = len(ids)
    while start < n:
        end = min(start + budget, n)
        piece = tok.decode(
            ids[start:end], skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        # Проверка и сужение, чтобы гарантировать <= max_tokens.
        while end > start + 1 and not _body_fits(
            prefix, piece, max_tokens, max_chars, model_name
        ):
            end -= 1
            piece = tok.decode(
                ids[start:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
        if not piece and end > start:
            end = start + 1
            piece = tok.decode(
                ids[start:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap_tokens, start + 1)
    return pieces


def _split_exact_lines(
    prefix: str,
    body: str,
    max_tokens: int,
    overlap_tokens: int,
    max_chars: int,
    model_name: str,
) -> list[str]:
    """Точная (медленная) дорезка по строкам без потерь — только для редких промахов оценки."""
    lines = body.split("\n")
    parts: list[str] = []
    buf: list[str] = []
    for line in lines:
        cand = "\n".join([*buf, line])
        if _body_fits(prefix, cand, max_tokens, max_chars, model_name):
            buf.append(line)
            continue
        if buf:
            parts.append("\n".join(buf))
        if _body_fits(prefix, line, max_tokens, max_chars, model_name):
            buf = [line]
        else:
            parts.extend(
                _split_long_line(
                    prefix, line, max_tokens, overlap_tokens, max_chars, model_name
                )
            )
            buf = []
    if buf:
        parts.append("\n".join(buf))
    return parts


def _finalize_piece(
    buf: list[tuple[str, int]],
    prefix: str,
    max_tokens: int,
    max_chars: int,
    overlap_tokens: int,
    model_name: str,
) -> list[str]:
    """Собрать кусок; при промахе оценки дорезать точно (без потерь)."""
    text = "\n".join(line for line, _ in buf)
    if _body_fits(prefix, text, max_tokens, max_chars, model_name):
        return [text]
    return _split_exact_lines(
        prefix, text, max_tokens, overlap_tokens, max_chars, model_name
    )


def _split_body_tokens(
    prefix: str,
    body: str,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
    max_chars: int,
    model_name: str,
) -> list[str]:
    """Разбить тело (процедуры/шапки) на чанки с токенным бюджетом и перекрытием.

    Производительность: строки батчируются одной токенизацией, а в цикле
    используется O(1)-оценка сверху по токенам; точная проверка бюджета — только
    при фиксации куска (``_finalize_piece``) и при одиночной сверхдлинной строке.
    """
    body = body.strip()
    if not body:
        return []
    if _body_fits(prefix, body, max_tokens, max_chars, model_name):
        return [body]

    tok = get_tokenizer(model_name)
    lines = body.split("\n")
    counts = [len(ids) for ids in tok(lines, add_special_tokens=False)["input_ids"]]
    base = len(
        tok.encode(
            f"{PASSAGE_PREFIX}{prefix}\n\n", add_special_tokens=True, truncation=False
        )
    )

    pieces: list[str] = []
    buf: list[tuple[str, int]] = []
    buf_sum = 0
    buf_chars = 0
    buf_n = 0

    for line, lc in zip(lines, counts, strict=False):
        cl = len(line)
        # Оценка сверху: base + сумма токенов строк + число переносов строк.
        est_tokens = base + buf_sum + lc + buf_n
        est_chars = buf_chars + cl + buf_n
        if est_tokens <= max_tokens and est_chars <= max_chars:
            buf.append((line, lc))
            buf_sum += lc
            buf_chars += cl
            buf_n += 1
            continue

        # Бюджет исчерпан: фиксируем накопленное и начинаем следующий чанк
        # с хвостового перекрытия уже выданного чанка.
        if buf:
            pieces.extend(
                _finalize_piece(
                    buf, prefix, max_tokens, max_chars, overlap_tokens, model_name
                )
            )
            buf = _tail_overlap(buf, overlap_tokens)
            buf_sum = sum(c for _, c in buf)
            buf_chars = sum(len(line_text) for line_text, _ in buf)
            buf_n = len(buf)
            if base + buf_sum + lc + buf_n <= max_tokens and buf_chars + cl + buf_n <= max_chars:
                buf.append((line, lc))
                buf_sum += lc
                buf_chars += cl
                buf_n += 1
                continue
            # Перекрытие не помещается со строкой: эти строки уже в предыдущем
            # чанке, поэтому их можно отбросить, а строку положить с нуля.
            buf, buf_sum, buf_chars, buf_n = [], 0, 0, 0

        if base + lc <= max_tokens and cl <= max_chars:
            buf = [(line, lc)]
            buf_sum = lc
            buf_chars = cl
            buf_n = 1
        else:
            # Одна строка не влезает целиком — режем по токенам.
            pieces.extend(
                _split_long_line(
                    prefix, line, max_tokens, overlap_tokens, max_chars, model_name
                )
            )

    if buf:
        pieces.extend(
            _finalize_piece(
                buf, prefix, max_tokens, max_chars, overlap_tokens, model_name
            )
        )

    return _merge_tail(pieces, prefix, max_tokens, min_tokens, model_name)


def _merge_tail(
    pieces: list[str],
    prefix: str,
    max_tokens: int,
    min_tokens: int,
    model_name: str,
) -> list[str]:
    """Приклеить короткий хвост к предыдущему чанку, если помещается в бюджет."""
    if len(pieces) < 2:
        return [p for p in pieces if p.strip()]
    tail = pieces[-1]
    tail_tokens = model_input_tokens(prefix, tail, model_name) if tail.strip() else 0
    if tail_tokens < min_tokens:
        prev = pieces[-2]
        if model_input_tokens(prefix, "\n".join([prev, tail]), model_name) <= max_tokens:
            merged = "\n".join([prev, tail])
            return [*pieces[:-2], merged]
    return [p for p in pieces if p.strip()]


def _proc_prefix(source: str, name: str) -> str:
    return PROC_PREFIX.format(source=source, name=name)


def chunk_bsl_text_tokens(
    text: str,
    source: str,
    *,
    model_name: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[TokenChunk]:
    """Нарезать BSL-текст по процедурам с токенным бюджетом (новый чанкер)."""
    if not model_name:
        raise ValueError("model_name is required")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be >= 0 and < max_tokens")
    if min_tokens <= 0 or min_tokens >= max_tokens:
        raise ValueError("min_tokens must be > 0 and < max_tokens")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    lines = text.replace("\r\n", "\n").split("\n")
    chunks: list[TokenChunk] = []
    index = 0

    header: list[str] = []
    current: list[str] = []
    current_name: str | None = None
    pending_directives: list[str] = []

    def flush_procedure() -> None:
        nonlocal index, current, current_name
        if current_name is None or not "".join(current).strip():
            current, current_name = [], None
            return
        body = "\n".join(current).strip()
        pieces = _split_body_tokens(
            _proc_prefix(source, current_name),
            body,
            max_tokens,
            overlap_tokens,
            min_tokens,
            max_chars,
            model_name,
        )
        total = len(pieces)
        for part, piece in enumerate(pieces, start=1):
            chunks.append(
                TokenChunk(
                    text=f"{_proc_prefix(source, current_name)}\n\n{piece}",
                    source_path=source,
                    chunk_index=index,
                    procedure_name=current_name,
                    kind="procedure",
                    part=part,
                    parts=total,
                )
            )
            index += 1
        current, current_name = [], None

    def flush_header(header_text: str) -> None:
        nonlocal index
        header_text = header_text.strip()
        if not header_text:
            return
        pieces = _split_body_tokens(
            _proc_prefix(source, "module"),
            header_text,
            max_tokens,
            overlap_tokens,
            min_tokens,
            max_chars,
            model_name,
        )
        for piece in pieces:
            chunks.append(
                TokenChunk(
                    text=f"{_proc_prefix(source, 'module')}\n\n{piece}",
                    source_path=source,
                    chunk_index=index,
                    procedure_name="module",
                    kind="module",
                )
            )
            index += 1

    for line in lines:
        start_match = START.match(line)
        if start_match and current_name is None:
            flush_header("\n".join(header))
            header = []
            current_name = start_match.group(1)
            current = [*pending_directives, line]
            pending_directives = []
            continue
        if END.match(line) and current_name is not None:
            current.append(line)
            flush_procedure()
            continue
        if current_name is not None:
            current.append(line)
        elif line.lstrip().startswith("&"):
            pending_directives.append(line)
        else:
            pending_directives = []
            header.append(line)

    flush_procedure()
    if pending_directives:
        # «Висячие» директивы компиляции (&НаКлиенте/&НаСервере) без следующей
        # процедуры — не теряем, прикладываем к module-шапке.
        header.extend(pending_directives)
        pending_directives = []
    flush_header("\n".join(header))

    if not chunks and text.strip():
        # Нет процедур: весь файл как module-шапка.
        flush_header(text)

    return chunks


def chunk_bsl_file_tokens(
    path: Path,
    root: Path,
    *,
    model_name: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[TokenChunk]:
    """Прочитать .bsl и вернуть токенные чанки с относительным путём."""
    rel = str(path.relative_to(root)).replace("\\", "/")
    return chunk_bsl_text_tokens(
        read_text_file(path),
        rel,
        model_name=model_name,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
        max_chars=max_chars,
    )
