"""Сгенерировать gold-set для eval: 10 курируемых запросов + 20 из реальных файлов.

Релевантность задаётся на уровне **файлов-источников** (``relevant_sources``),
поэтому набор не зависит от того, какой чанк крупного модуля вернул поиск.

Формат ``data/eval/gold_set.jsonl`` — по строке на запрос::

    {"id": "exact-peredzapisyu", "type": "exact",
     "query": "ПередЗаписью",
     "relevant_sources": ["Documents/....bsl"],
     "source": "Documents/....bsl", "note": "..."}

Курируемая часть — 10 доменных запросов трёх типов (точное имя / смысл /
ловушка) с заранее известными файлами. Автоматическая часть — 20 запросов,
выведенных из реальных модулей конфигурации (релевантен один файл).

Пример:
    python scripts/make_goldset.py
    python scripts/make_goldset.py --auto 20 --out data/eval/gold_set.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_homework.config import CHROMA_DIR, DATA_DIR  # noqa: E402

_PROC = re.compile(r"(?i)\b(Процедура|Функция)\s+([A-Za-zА-Яа-яЁё_][A-Za-zА-Яа-яЁё0-9_]*)")

# Курируемые запросы: файлы-эталоны заданы вручную по разбору выгрузки configuration.
CURATED: list[dict] = [
    {
        "id": "exact-peredzapisyu",
        "type": "exact",
        "query": "ПередЗаписью",
        "sources": [
            "Documents/СчетФактураВыданный/Ext/ObjectModule.bsl",
            "Documents/СчетФактураПолученный/Ext/ObjectModule.bsl",
        ],
        "note": "Точное имя обработчика: ожидаем сильный BM25",
    },
    {
        "id": "exact-nomer-dokumenta",
        "type": "exact",
        "query": "НомерДокумента",
        "sources": [
            "Documents/СчетФактураПолученный/Ext/ObjectModule.bsl",
            "Documents/ВводНачальныхОстатковНДС/Ext/ObjectModule.bsl",
        ],
        "note": "Точный идентификатор 1С",
    },
    {
        "id": "semantic-nds-avans",
        "type": "semantic",
        "query": "начисление НДС с авансов полученных",
        "sources": [
            "Documents/СчетФактураВыданный/Ext/ObjectModule.bsl",
            "Documents/ВводНачальныхОстатковНДС/Ext/ObjectModule.bsl",
            "Documents/ФормированиеЗаписейКнигиПродаж/Ext/ObjectModule.bsl",
        ],
        "note": "Смысловой: НДС с авансов, ожидаем сильный cosine",
    },
    {
        "id": "semantic-zapolnenie-tch",
        "type": "semantic",
        "query": "заполнение таблицы товаров документа программно",
        "sources": [
            "Documents/ФормированиеЗаписейКнигиПокупок/Ext/ObjectModule.bsl",
            "Documents/СчетФактураПолученный/Ext/ObjectModule.bsl",
        ],
        "note": "Смысловой: заполнение ТЧ",
    },
    {
        "id": "semantic-pechat-etiketok",
        "type": "semantic",
        "query": "печать этикеток и штрихкодов на принтере",
        "sources": [
            "DataProcessors/ПечатьЭтикетокСтикеровЗаказКМ/Forms/Форма/Ext/Form/Module.bsl",
        ],
        "note": "Смысловой: печать этикеток",
    },
    {
        "id": "semantic-provedenie-nds",
        "type": "semantic",
        "query": "проведение документа по регистрам накопления НДС",
        "sources": [
            "Documents/ВводНачальныхОстатковНДС/Ext/ObjectModule.bsl",
            "Documents/СчетФактураВыданный/Ext/ObjectModule.bsl",
        ],
        "note": "Смысловой: проведение по регистрам",
    },
    {
        "id": "semantic-nds-pokupka",
        "type": "semantic",
        "query": "НДС по приобретенным ценностям вычет",
        "sources": [
            "Documents/СчетФактураПолученный/Ext/ObjectModule.bsl",
            "Documents/ФормированиеЗаписейКнигиПокупок/Ext/ObjectModule.bsl",
        ],
        "note": "Смысловой: НДС к вычету",
    },
    {
        "id": "trap-nds-kniga-prodazh",
        "type": "trap",
        "query": "НДС счет-фактура выданный книга продаж",
        "sources": [
            "Documents/СчетФактураВыданный/Ext/ObjectModule.bsl",
            "Documents/ФормированиеЗаписейКнигиПродаж/Ext/ObjectModule.bsl",
        ],
        "note": "Ловушка: имя объекта + контекст",
    },
    {
        "id": "trap-identifikatory",
        "type": "trap",
        "query": "идентификаторы объектов метаданных конфигурации",
        "sources": [
            "Catalogs/ИдентификаторыОбъектовМетаданных/Ext/ManagerModule.bsl",
        ],
        "note": "Ловушка: предметная область без точного имени",
    },
    {
        "id": "trap-osnovnye-sredstva",
        "type": "trap",
        "query": "основные средства поступление и печать",
        "sources": [
            "Catalogs/ОсновныеСредства/Ext/ObjectModule.bsl",
        ],
        "note": "Ловушка: два сигнала — объект и действие",
    },
]

_DOMAIN_WORDS = {
    "Documents": "документ",
    "Catalogs": "справочник",
    "DataProcessors": "обработка",
    "Reports": "отчет",
    "CommonModules": "общий модуль",
}


def _connect(sqlite_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def _existing_sources(db: sqlite3.Connection) -> set[str]:
    rows = db.execute(
        "SELECT DISTINCT string_value FROM embedding_metadata WHERE key = 'source'"
    )
    return {r[0] for r in rows if r[0]}


def _group_sources(db: sqlite3.Connection) -> dict[str, list[tuple[str, str]]]:
    """source -> [(chroma_id, document)] по всем чанкам коллекции code."""
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in db.execute(
        """
        SELECT e.embedding_id AS cid, s.string_value AS source, d.string_value AS doc
        FROM embeddings e
        JOIN embedding_metadata d ON d.id = e.id AND d.key = 'chroma:document'
        LEFT JOIN embedding_metadata s ON s.id = e.id AND s.key = 'source'
        WHERE e.embedding_id LIKE 'bsl:%'
        ORDER BY e.id
        """
    ):
        grouped[row["source"] or ""].append((row["cid"], row["doc"] or ""))
    return grouped


def _humanize(source: str) -> str:
    """``Documents/СчетФактураВыданный/...`` -> ``документ счет фактура выданный``."""
    parts = source.split("/")
    obj = parts[1] if len(parts) > 1 else parts[0]
    kind = _DOMAIN_WORDS.get(parts[0], "объект")
    words = re.findall(r"[A-ZА-Я][a-zа-я0-9]+", obj)
    return f"{kind} {' '.join(w.lower() for w in words)}".strip()


def build_curated(db: sqlite3.Connection) -> list[dict]:
    """Курируемые запросы; файлы-эталоны проверяются по фактическому индексу."""
    existing = _existing_sources(db)
    rows: list[dict] = []
    for item in CURATED:
        present = [src for src in item["sources"] if src in existing]
        if not present:
            print(f"пропуск {item['id']}: ни один файл-эталон не найден в индексе")
            continue
        rows.append(
            {
                "id": item["id"],
                "type": item["type"],
                "query": item["query"],
                "relevant_sources": present,
                "source": present[0],
                "note": item["note"],
            }
        )
    return rows


def build_auto(db: sqlite3.Connection, n: int) -> list[dict]:
    """N синтетических запросов из реальных модулей (релевантен один файл)."""
    grouped = {
        src: chunks
        for src, chunks in _group_sources(db).items()
        if src.startswith(("Documents/", "Catalogs/", "DataProcessors/")) and chunks
    }
    sources = sorted(grouped)
    if not sources or n <= 0:
        return []
    step = max(1, len(sources) // n)
    picked = sources[::step][:n]
    rows: list[dict] = []
    for index, source in enumerate(picked, start=1):
        proc = next(
            (m.group(2) for _, doc in grouped[source] if (m := _PROC.search(doc))),
            None,
        )
        human = _humanize(source)
        if proc and index % 3 == 1:
            qtype, query = "exact", proc
            note = "Авто: имя процедуры из модуля"
        elif index % 3 == 2:
            qtype, query = "semantic", f"как реализовано {human}"
            note = "Авто: смысловой запрос по назначению файла"
        else:
            qtype = "trap"
            query = f"{human} {proc or ''}".strip()
            note = "Авто: название объекта плюс имя процедуры"
        rows.append(
            {
                "id": f"auto-{index:02d}",
                "type": qtype,
                "query": query,
                "relevant_sources": [source],
                "source": source,
                "note": note,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Сгенерировать gold-set (file-level релевантность)")
    ap.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "eval" / "gold_set.jsonl")
    ap.add_argument("--auto", type=int, default=20)
    args = ap.parse_args()

    sqlite_path = args.chroma_dir / "chroma.sqlite3"
    if not sqlite_path.exists():
        raise SystemExit(f"не найдена база: {sqlite_path}")
    db = _connect(sqlite_path)
    rows = build_curated(db)
    auto_rows = build_auto(db, args.auto)
    rows.extend(auto_rows)
    db.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["type"]] += 1
    print(
        f"записано {len(rows)} запросов в {args.out} "
        f"(curated={len(rows) - len(auto_rows)}, auto={len(auto_rows)}, types={dict(counts)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
