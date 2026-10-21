"""Gold-set v2: запросы в естественных формулировках (цель — 70 штук).

Отличие от v1: курируемая часть выросла до 30 доменных вопросов, а авточасть
строится по реальным объектам конфигурации с естественными шаблонами
(«как проводится документ …» вместо «как реализовано документ …»).

Релевантность — по файлам-источникам. Курируемые записи задают фрагменты имён
объектов, которые генератор разрешает в реальные пути через sqlite: если файл
не найден, запись отбрасывается с предупреждением.

Ограничение: авточасть — шаблонная (не написана человеком). Это зафиксировано
в отчёте; следующий шаг — ручная разметка.
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

from mcp_1c_rag.config import CHROMA_DIR, DATA_DIR  # noqa: E402

ACRONYMS = {"НДС", "УСН", "КМ", "ОС", "ТЧ", "КИЗ", "СМР", "БУ", "МСФО", "XML"}
_WORD = re.compile(r"[A-ZА-Я]+(?![a-zа-я])|[A-ZА-Я][a-zа-я0-9]+")

# 30 курируемых запросов: фрагмент → реальный путь разрешается по индексу.
CURATED: list[dict] = [
    {"id": "c01", "type": "exact", "query": "ПередЗаписью",
     "frags": ["СчетФактураВыданный", "СчетФактураПолученный"],
     "note": "точное имя обработчика"},
    {"id": "c02", "type": "exact", "query": "ПечатьЭтикеток",
     "frags": ["ПечатьЭтикетокСтикеров"],
     "note": "точное имя процедуры печати"},
    {"id": "c03", "type": "exact", "query": "ОбменДаннымиXML",
     "frags": ["ОбменДаннымиXML"],
     "note": "точное имя подсистемы обмена"},
    {"id": "c04", "type": "exact", "query": "ИнвентаризацияТоваровНаСкладе",
     "frags": ["ИнвентаризацияТоваровНаСкладе"],
     "note": "точное имя документа"},
    {"id": "c05", "type": "semantic", "query": "как начислить НДС с авансов полученных",
     "frags": ["СчетФактураВыданный", "ВводНачальныхОстатковНДС"]},
    {"id": "c06", "type": "semantic", "query": "где рассчитывается НДС к вычету с авансов",
     "frags": ["ФормированиеЗаписейКнигиПокупок", "ВводНачальныхОстатковНДС"]},
    {"id": "c07", "type": "semantic", "query": "печать этикеток и штрихкодов на принтере",
     "frags": ["ПечатьЭтикетокСтикеров"]},
    {"id": "c08", "type": "semantic", "query": "как заполняется книга продаж при реализации",
     "frags": ["ФормированиеЗаписейКнигиПродаж"]},
    {"id": "c09", "type": "semantic", "query": "распределение косвенных расходов по НДС",
     "frags": ["РаспределениеНДСКосвенныхРасходов"]},
    {"id": "c10", "type": "semantic", "query": "проведение документов по регистрам НДС",
     "frags": ["ПроведениеДокументовПоРегистрамНДС"]},
    {"id": "c11", "type": "semantic", "query": "инвентаризация товаров на складе",
     "frags": ["ИнвентаризацияТоваровНаСкладе"]},
    {"id": "c12", "type": "semantic", "query": "обмен данными с сайтом",
     "frags": ["ОбменССайтом"]},
    {"id": "c13", "type": "semantic", "query": "выгрузка данных в XML для обмена между базами",
     "frags": ["УниверсальныйОбменДаннымиXML", "ОбменДаннымиXML"]},
    {"id": "c14", "type": "semantic", "query": "как рассчитать зарплату к выплате",
     "frags": ["ЗарплатаКВыплатеОрганизаций"]},
    {"id": "c15", "type": "semantic", "query": "отражение зарплаты в управленческом учёте",
     "frags": ["ОтражениеЗарплатыВУпрУчете"]},
    {"id": "c16", "type": "semantic", "query": "как заполнить спецификацию номенклатуры",
     "frags": ["СпецификацииНоменклатуры"]},
    {"id": "c17", "type": "semantic", "query": "подбор номенклатуры в документ",
     "frags": ["ПодборНоменклатуры"]},
    {"id": "c18", "type": "semantic", "query": "анализ цен номенклатуры",
     "frags": ["АнализЦен"]},
    {"id": "c19", "type": "semantic", "query": "стоимостная оценка склада в ценах номенклатуры",
     "frags": ["СтоимостнаяОценкаСкладаВЦенахНоменклатуры"]},
    {"id": "c20", "type": "semantic", "query": "обмен с клиент-банком",
     "frags": ["КлиентБанк"]},
    {"id": "c21", "type": "semantic", "query": "расходный кассовый ордер",
     "frags": ["РасходныйКассовыйОрдер"]},
    {"id": "c22", "type": "semantic", "query": "как формируется кассовая книга",
     "frags": ["КассоваяКнига"]},
    {"id": "c23", "type": "semantic", "query": "отчёт по бюджету",
     "frags": ["ОтчетПоБюджету"]},
    {"id": "c24", "type": "semantic", "query": "обороты по статьям бюджетов",
     "frags": ["ОборотыПоСтатьямБюджетов"]},
    {"id": "c25", "type": "semantic", "query": "контролируемые значения по бюджетам",
     "frags": ["КонтролируемыеЗначенияПоБюджетам"]},
    {"id": "c26", "type": "semantic", "query": "регламентные операции по налогу на прибыль",
     "frags": ["РегламентныеОперацииНалоговогоУчетаПоНалогуНаПрибыль"]},
    {"id": "c27", "type": "semantic", "query": "расчёты по налогу на прибыль",
     "frags": ["РасчетыПоНалогуНаПрибыль"]},
    {"id": "c28", "type": "trap", "query": "НДС счёт-фактура выданный книга продаж",
     "frags": ["СчетФактураВыданный", "ФормированиеЗаписейКнигиПродаж"]},
    {"id": "c29", "type": "trap", "query": "взаиморасчёты с контрагентами по договорам",
     "frags": ["ДоговорыКонтрагентов"]},
    {"id": "c30", "type": "trap", "query": "диспетчеризация производства продукции",
     "frags": ["ДиспетчеризацияПроизводстваПродукции"]},
]

AUTO_TEMPLATES = {
    "Documents": ["как проводится документ {name}", "как заполняется документ {name}",
                  "логика документа {name}"],
    "Catalogs": ["как получить данные из справочника {name}",
                 "где хранятся настройки справочника {name}"],
    "DataProcessors": ["как работает обработка {name}", "что делает обработка {name}"],
    "Reports": ["как формируется отчёт {name}", "какие показатели в отчёте {name}"],
}


def _connect(sqlite_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def _sources(db: sqlite3.Connection) -> list[tuple[str, int]]:
    rows = db.execute(
        """
        SELECT s.string_value AS source, count(*) AS n
        FROM embeddings e JOIN embedding_metadata s ON s.id = e.id AND s.key = 'source'
        WHERE s.string_value LIKE 'Documents/%' OR s.string_value LIKE 'Catalogs/%'
           OR s.string_value LIKE 'DataProcessors/%' OR s.string_value LIKE 'Reports/%'
        GROUP BY s.string_value
        """
    ).fetchall()
    return [(r["source"], r["n"]) for r in rows]


def _humanize(name: str) -> str:
    words = _WORD.findall(name)
    out = []
    for word in words:
        upper = word.upper()
        out.append(upper if upper in ACRONYMS else word.lower())
    return " ".join(out)


def _resolve(db: sqlite3.Connection, frags: list[str]) -> list[str]:
    resolved: list[str] = []
    for frag in frags:
        row = db.execute(
            "SELECT DISTINCT string_value FROM embedding_metadata "
            "WHERE key='source' AND string_value LIKE ? ORDER BY length(string_value) LIMIT 1",
            (f"%{frag}%",),
        ).fetchone()
        if row:
            resolved.append(row[0])
    return resolved


def build_curated(db: sqlite3.Connection) -> list[dict]:
    rows: list[dict] = []
    for item in CURATED:
        sources = _resolve(db, item["frags"])
        if not sources:
            print(f"skip {item['id']}: sources not found")
            continue
        rows.append(
            {
                "id": item["id"],
                "type": item["type"],
                "query": item["query"],
                "relevant_sources": sources,
                "source": sources[0],
                "note": item.get("note", "curated v2"),
            }
        )
    return rows


def build_auto(db: sqlite3.Connection, n: int, exclude: set[str]) -> list[dict]:
    files = [(s, c) for s, c in _sources(db) if s not in exclude and c >= 3]
    files.sort()
    if not files or n <= 0:
        return []
    step = max(1, len(files) // n)
    picked = files[::step][:n]
    rows: list[dict] = []
    for index, (source, _chunks) in enumerate(picked, start=1):
        parts = source.split("/")
        prefix, obj = parts[0], (parts[1] if len(parts) > 1 else "объект")
        templates = AUTO_TEMPLATES.get(prefix)
        if not templates:
            continue
        template = templates[(index - 1) % len(templates)]
        rows.append(
            {
                "id": f"v2-auto-{index:02d}",
                "type": "semantic",
                "query": template.format(name=_humanize(obj)),
                "relevant_sources": [source],
                "source": source,
                "note": "template-derived from real object name",
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Gold-set v2 (natural queries)")
    ap.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "eval" / "gold_set_v2.jsonl")
    ap.add_argument("--auto", type=int, default=40)
    args = ap.parse_args()

    sqlite_path = args.chroma_dir / "chroma.sqlite3"
    db = _connect(sqlite_path)
    curated = build_curated(db)
    exclude = {row["source"] for row in curated}
    auto = build_auto(db, args.auto, exclude)
    rows = curated + auto
    db.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    types: dict[str, int] = defaultdict(int)
    for row in rows:
        types[row["type"]] += 1
    print(f"written {len(rows)} queries -> {args.out} "
          f"(curated={len(curated)}, auto={len(auto)}, types={dict(types)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
