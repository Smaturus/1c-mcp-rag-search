"""Unit-тесты токенного чанкера (без индексации/эмбеддингов)."""

from __future__ import annotations

import hashlib

import pytest

from mcp_1c_rag.chunking_tokens import (
    DEFAULT_MAX_TOKENS,
    chunk_bsl_file_tokens,
    chunk_bsl_text_tokens,
    get_tokenizer,
    model_input_tokens,
)

MODEL = "intfloat/multilingual-e5-base"
SRC = "CommonModules/Тест/Ext/ObjectModule.bsl"


def chunk_id(source: str, index: int) -> str:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    return f"bsl_proc:{digest}:{index}"


def chunk_tokens(chunk) -> int:
    """Точное число токенов, которое увидит модель для чанка (как embed_passages)."""
    tok = get_tokenizer(MODEL)
    return len(tok.encode(f"passage: {chunk.text}", add_special_tokens=True, truncation=False))


def chunk_body(chunk) -> str:
    """Тело чанка (текст после первого разделителя '\n\n')."""
    return chunk.text.split("\n\n", 1)[1] if "\n\n" in chunk.text else ""


def _proc(name: str, body: str) -> str:
    return f"Процедура {name}()\n{body}\nКонецПроцедуры\n"


def test_long_procedure_all_within_budget():
    body = "\n".join(f'\tСтрока{i} = "значение {i}";' for i in range(2000))
    chunks = chunk_bsl_text_tokens(_proc("Длинная", body), SRC, model_name=MODEL)
    assert len(chunks) >= 2
    for c in chunks:
        assert chunk_tokens(c) <= DEFAULT_MAX_TOKENS, f"чанк {c.chunk_index} превысил бюджет"


def test_overlap_present_but_not_full_duplicate():
    body = "\n".join(f"\tСтрока{i} = {i};" for i in range(3000))
    chunks = [
        c
        for c in chunk_bsl_text_tokens(_proc("Д", body), SRC, model_name=MODEL)
        if c.kind == "procedure"
    ]
    assert len(chunks) >= 2
    parts = {c.part: c for c in chunks}
    # Перекрытие: соседние части разделяют хотя бы одну строку, но не весь чанк.
    for part in sorted(parts):
        if part + 1 not in parts:
            continue
        a_lines = set(chunk_body(parts[part]).split("\n"))
        b_lines = set(chunk_body(parts[part + 1]).split("\n"))
        assert a_lines & b_lines, f"части {part}/{part+1} не перекрываются"
        assert a_lines != b_lines, f"части {part}/{part+1} полностью совпадают"


def test_procedure_boundaries_not_lost():
    text = _proc("Первая", "x = 1;") + _proc("Вторая", "y = 2;")
    chunks = chunk_bsl_text_tokens(text, SRC, model_name=MODEL)
    names = {c.procedure_name for c in chunks}
    assert {"Первая", "Вторая"} <= names
    # Каждая процедура присутствует и не слилась в один module-чанк.
    assert any(c.kind == "procedure" for c in chunks)


def test_empty_and_bom_only(tmp_path):
    assert chunk_bsl_text_tokens("", SRC, model_name=MODEL) == []
    assert chunk_bsl_text_tokens("   \n\t\n  ", SRC, model_name=MODEL) == []
    # BOM-only файл
    bom = tmp_path / "bom.bsl"
    bom.write_bytes(b"\xef\xbb\xbf")
    assert chunk_bsl_file_tokens(bom, tmp_path, model_name=MODEL) == []
    # Пустой файл
    empty = tmp_path / "empty.bsl"
    empty.write_text("", encoding="utf-8")
    assert chunk_bsl_file_tokens(empty, tmp_path, model_name=MODEL) == []


def test_determinism_and_unique_ids():
    text = _proc("А", "a = 1;") + _proc("Б", "b = 2;") + _proc("В", "c = 3;")
    c1 = chunk_bsl_text_tokens(text, SRC, model_name=MODEL)
    c2 = chunk_bsl_text_tokens(text, SRC, model_name=MODEL)
    ids1 = [chunk_id(SRC, c.chunk_index) for c in c1]
    ids2 = [chunk_id(SRC, c.chunk_index) for c in c2]
    assert ids1 == ids2
    assert len(ids1) == len(set(ids1))
    assert [c.source_path for c in c1] == [SRC] * len(c1)


def test_prefix_accounted_in_budget():
    body = "x = 1;\n" * 400
    short_prefix = "// file: A · proc: p"
    long_prefix = "// file: " + ("ОченьДлинныйПутьИсточника/") * 40 + " · proc: p"
    assert model_input_tokens(long_prefix, body, MODEL) > model_input_tokens(
        short_prefix, body, MODEL
    )
    long_src = ("ОченьДлинныйПутьИсточника/") * 40 + "Module.bsl"
    chunks = chunk_bsl_text_tokens(_proc("П", body), long_src, model_name=MODEL)
    assert chunks
    for c in chunks:
        assert chunk_tokens(c) <= DEFAULT_MAX_TOKENS


def test_single_line_over_budget_splits():
    # Одна строка существенно длиннее бюджета: тысячи токенов.
    line = " ".join(["токен"] * 3000)
    text = _proc("Однострочная", line)
    chunks = chunk_bsl_text_tokens(text, SRC, model_name=MODEL)
    assert chunks
    for c in chunks:
        assert chunk_tokens(c) <= DEFAULT_MAX_TOKENS
        assert chunk_tokens(c) > 0


def test_module_header_chunk_for_no_procedure_file():
    text = "Перем Глобальная;\nГлобальная = 1;\n"
    chunks = chunk_bsl_text_tokens(text, SRC, model_name=MODEL)
    assert len(chunks) == 1
    assert chunks[0].kind == "module"
    assert chunks[0].procedure_name == "module"


def test_tail_merge_keeps_nonempty():
    # Короткая процедура после длинной не должна теряться.
    text = _proc("Д", "\n".join(f"x{i} = {i};" for i in range(1500))) + _proc("К", "y = 1;")
    chunks = chunk_bsl_text_tokens(text, SRC, model_name=MODEL)
    names = [c.procedure_name for c in chunks]
    assert "К" in names
    for c in chunks:
        assert c.text.strip()
        assert chunk_tokens(c) <= DEFAULT_MAX_TOKENS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
