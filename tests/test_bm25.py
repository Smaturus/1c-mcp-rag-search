"""Тесты BM25: токенизация, глобальный индекс, поиск, сохранение/загрузка."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_homework.bm25 import Bm25Index, BM25Okapi, docs_from_sqlite, tokenize  # noqa: E402


def test_tokenize_keeps_abbreviations_and_codes() -> None:
    """Аббревиатуры и дефисные коды остаются целыми токенами."""
    tokens = tokenize("НДС с аванса, НФ-001 и ОСВ; ПередЗаписью")
    assert tokens == ["ндс", "с", "аванса", "нф-001", "и", "осв", "передзаписью"]


def test_tokenize_ignores_punctuation_only() -> None:
    """Только пунктуация — пустой список токенов."""
    assert tokenize("  ,.;:!?  ") == []


def test_bm25_okapi_scores_relevant_doc_higher() -> None:
    """Классический BM25 ранжирует документ с термином выше остальных."""
    corpus = [tokenize(t) for t in ("НДС с аванса", "печать этикеток", "регистр сведений")]
    bm25 = BM25Okapi(corpus)
    scores = bm25.scores("НДС")
    assert scores[0] > scores[1] and scores[0] > scores[2]


@pytest.fixture()
def small_index(tmp_path: Path) -> Bm25Index:
    """Небольшой индекс из 4 документов для проверок поиска."""
    docs = [
        ("bsl:1", "Процедура ПередЗаписью начисляет НДС с аванса"),
        ("bsl:2", "Печать этикеток и штрихкодов на принтере"),
        ("meta:3", "Справочник Номенклатура: печать этикеток"),
        ("bsl:4", "Регистр сведений цены номенклатуры"),
    ]
    return Bm25Index.build(docs)


def test_index_search_finds_lexical_match(small_index: Bm25Index) -> None:
    """Лексический поиск находит документ с точным термином."""
    hits = small_index.search("ПередЗаписью", top_k=2)
    assert hits
    assert hits[0].doc_id == "bsl:1"


def test_index_search_scores_are_positive(small_index: Bm25Index) -> None:
    """Все возвращённые скоры положительны, порядок убывающий."""
    hits = small_index.search("НДС аванса", top_k=3)
    assert hits
    assert all(h.score > 0 for h in hits)
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_index_save_load_roundtrip(small_index: Bm25Index, tmp_path: Path) -> None:
    """Индекс сохраняется и загружается без потери результатов."""
    small_index.save(tmp_path)
    loaded = Bm25Index.load(tmp_path)
    assert loaded.stats()["n_docs"] == small_index.stats()["n_docs"]
    assert [h.doc_id for h in loaded.search("печать этикеток", 3)] == [
        h.doc_id for h in small_index.search("печать этикеток", 3)
    ]


def test_index_empty_query_returns_nothing(small_index: Bm25Index) -> None:
    """Пустой запрос не находит документов."""
    assert small_index.search("", top_k=5) == []


def test_docs_from_sqlite_reads_documents(tmp_path: Path) -> None:
    """Чтение пар (id, document) из sqlite Chroma-совместимой схемы."""
    import sqlite3

    db = sqlite3.connect(tmp_path / "test.sqlite3")
    db.executescript(
        """
        CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT);
        CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT);
        INSERT INTO embeddings (id, segment_id, embedding_id) VALUES (1, 's', 'bsl:1');
        INSERT INTO embedding_metadata (id, key, string_value) VALUES (1, 'chroma:document', 'текст');
        """
    )
    db.commit()
    db.close()
    pairs = list(docs_from_sqlite(tmp_path / "test.sqlite3"))
    assert pairs == [("bsl:1", "текст")]
