"""Тесты нарезки BSL по процедурам."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from mcp_1c_rag.chunking_procedures import chunk_bsl_text  # noqa: E402

SIMPLE = """\
// Модуль объекта
Процедура ПередЗаписью(Отказ, РежимЗаписи)
    // проверка заполнения
    Если Не ЗначениеЗаполнено(Ссылка) Тогда
        Отказ = Истина;
    КонецЕсли;
КонецПроцедуры

Функция ПолучитьСумму() Экспорт
    Возврат 1;
КонецФункции
"""


def test_two_procedures_produce_two_chunks() -> None:
    chunks = chunk_bsl_text(SIMPLE, "Documents/Тест/Ext/ObjectModule.bsl")
    procedures = [c for c in chunks if c.kind == "procedure"]
    assert [c.procedure_name for c in procedures] == ["ПередЗаписью", "ПолучитьСумму"]
    assert all("Documents/Тест/Ext/ObjectModule.bsl" in c.text for c in procedures)


def test_header_becomes_module_chunk() -> None:
    chunks = chunk_bsl_text(SIMPLE, "src.bsl")
    module_chunks = [c for c in chunks if c.kind == "module"]
    assert len(module_chunks) == 1
    assert "Модуль объекта" in module_chunks[0].text
    assert module_chunks[0].part == 1


def test_compilation_directive_attached_to_procedure() -> None:
    text = """\
&НаКлиенте
Процедура ПриОткрытии(Отказ)
    Сообщить("ok");
КонецПроцедуры
"""
    chunks = chunk_bsl_text(text, "Forms/Форма/Ext/Form/Module.bsl")
    assert len(chunks) == 1
    assert "&НаКлиенте" in chunks[0].text
    assert chunks[0].procedure_name == "ПриОткрытии"


def test_long_procedure_is_split_with_parts() -> None:
    body = "\n".join(f"    А = {index};" for index in range(400))
    text = f"Процедура Большая()\n{body}\nКонецПроцедуры\n"
    chunks = chunk_bsl_text(text, "big.bsl", max_chars=500, overlap=50)
    assert len(chunks) > 1
    assert all(c.procedure_name == "Большая" for c in chunks)
    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))
    assert chunks[0].parts == len(chunks)


def test_module_without_procedures_falls_back_to_windows() -> None:
    text = "// только комментарии\n" * 100
    chunks = chunk_bsl_text(text, "no_procs.bsl", max_chars=300, overlap=30)
    assert len(chunks) >= 1
    assert all(c.kind == "module" for c in chunks)


def test_empty_text_returns_nothing() -> None:
    assert chunk_bsl_text("   \n\n ", "empty.bsl") == []


def test_chunk_indexes_are_sequential() -> None:
    chunks = chunk_bsl_text(SIMPLE, "x.bsl")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError):
        chunk_bsl_text("Процедура А()\nКонецПроцедуры", "x.bsl", max_chars=0)
    with pytest.raises(ValueError):
        chunk_bsl_text("Процедура А()\nКонецПроцедуры", "x.bsl", max_chars=100, overlap=100)
