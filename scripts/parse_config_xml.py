"""Parse a 1C XML configuration dump into structured metadata and search chunks.

Input: directory produced by "Выгрузить конфигурацию в файлы" (XML dump):
    <Root>/Documents/<Name>.xml, <Root>/Catalogs/<Name>.xml, ...

Output (all local, the dump is confidential — never commit):
    data/metadata_xml/objects.jsonl   — one record per metadata object
    data/metadata_xml/chunks.jsonl    — searchable chunks (object/attribute/form/...)
    data/metadata_xml/stats.json      — coverage counters
    data/lexicon/names.json       — identifier -> object/kind for the router

Usage:
    python scripts/parse_config_xml.py --root "<path to configuration dump>"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as etree  # stdlib: no extra dependency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "data" / "metadata_xml"
LEXICON = ROOT / "data" / "lexicon" / "names.json"

OBJECT_KINDS = {
    "Document": "Документ",
    "Catalog": "Справочник",
    "DataProcessor": "Обработка",
    "Report": "Отчёт",
    "CommonModule": "ОбщийМодуль",
    "AccumulationRegister": "РегистрНакопления",
    "InformationRegister": "РегистрСведений",
    "AccountingRegister": "РегистрБухгалтерии",
    "ChartOfAccounts": "ПланСчетов",
    "ChartOfCharacteristicTypes": "ПланВидовХарактеристик",
    "ChartOfCalculationTypes": "ПланВидовРасчета",
    "Enum": "Перечисление",
    "Constant": "Константа",
    "CommonForm": "ОбщаяФорма",
    "BusinessProcess": "БизнесПроцесс",
    "DocumentJournal": "ЖурналДокументов",
    "CommonCommand": "ОбщаяКоманда",
    "DefinedType": "ОпределяемыйТип",
    "DocumentNumerator": "НумераторДокументов",
    "CommandGroup": "ГруппаКоманд",
    "CommonTemplate": "ОбщийМакет",
    "CommonPicture": "ОбщаяКартинка",
    "SessionParameter": "ПараметрСеанса",
    "FilterCriterion": "КритерийОтбора",
    "Sequence": "Последовательность",
    "WebService": "WebСервис",
    "HTTPService": "HTTPСервис",
    "XDTOPackage": "ПакетXDTO",
}


def _local(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _find(element: Any, name: str) -> Any:
    if element is None:
        return None
    for child in element:
        if _local(child.tag) == name:
            return child
    return None


def _findall(element: Any, name: str) -> list[Any]:
    if element is None:
        return []
    return [child for child in element if _local(child.tag) == name]


def _text(element: Any) -> str:
    return (element.text or "").strip() if element is not None else ""


def _synonym(element: Any) -> str:
    synonym = _find(element, "Synonym")
    if synonym is None:
        return ""
    for item in _findall(synonym, "item"):
        lang = _text(_find(item, "lang"))
        if lang == "ru":
            return _text(_find(item, "content"))
    return ""


def _types(properties: Any) -> list[str]:
    type_element = _find(properties, "Type")
    if type_element is None:
        return []
    found: list[str] = []
    for element in type_element.iter():
        if _local(element.tag) == "Type":
            value = (element.text or "").strip()
            if value:
                found.append(value.replace("cfg:", ""))
    return sorted(set(found))


def _field(entry: Any) -> dict[str, Any]:
    props = _find(entry, "Properties")
    name = _text(_find(props, "Name")) or _text(_find(props, "Name"))
    return {
        "name": name,
        "synonym": _synonym(props) if props is not None else "",
        "comment": _text(_find(props, "Comment")) if props is not None else "",
        "types": _types(props) if props is not None else [],
    }


def parse_object(xml_path: Path) -> dict[str, Any] | None:
    try:
        tree = etree.parse(str(xml_path))
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {xml_path.name}: {exc}")
        return None
    root = tree.getroot()
    children = [_local(child.tag) for child in root]
    kind_tag = next((tag for tag in children if tag != "InternalInfo"), None)
    if not kind_tag:
        return None
    obj = next(child for child in root if _local(child.tag) == kind_tag)
    props = _find(obj, "Properties")
    if props is None:
        return None
    name = _text(_find(props, "Name"))
    if not name:
        return None
    child_objects = _find(obj, "ChildObjects")

    attributes: list[dict[str, Any]] = []
    tabular: list[dict[str, Any]] = []
    forms: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    dimensions: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    enum_values: list[str] = []
    if child_objects is not None:
        for entry in child_objects:
            tag = _local(entry.tag)
            props = _find(entry, "Properties")
            if tag == "Attribute":
                attributes.append(_field(entry))
            elif tag == "TabularSection":
                section = _field(entry)
                section["attributes"] = [_field(item) for item in _findall(entry, "Attribute")]
                tabular.append(section)
            elif tag == "Form":
                forms.append(
                    {
                        "name": _text(_find(props, "Name")),
                        "type": _text(_find(props, "FormType")) or "Managed",
                    }
                )
            elif tag == "Command":
                commands.append(_field(entry))
            elif tag == "Dimension":
                dimensions.append(_field(entry))
            elif tag == "Resource":
                resources.append(_field(entry))
            elif tag == "EnumValue":
                value_name = _text(_find(props, "Name"))
                if value_name:
                    enum_values.append(value_name)

    return {
        "kind": kind_tag,
        "kind_ru": OBJECT_KINDS.get(kind_tag, kind_tag),
        "name": name,
        "synonym": _synonym(props),
        "comment": _text(_find(props, "Comment")),
        "attributes": attributes,
        "tabular_sections": tabular,
        "forms": forms,
        "commands": commands,
        "dimensions": dimensions,
        "resources": resources,
        "enum_values": enum_values,
        "source_file": str(xml_path.name),
    }


def _type_hint(entry: dict[str, Any]) -> str:
    if entry["types"]:
        return ", ".join(entry["types"][:4])
    return "—"


def build_chunks(record: dict[str, Any]) -> list[dict[str, Any]]:
    head = f"{record['kind_ru']} · {record['name']}"
    if record["synonym"]:
        head += f" ({record['synonym']})"
    summary = [
        head,
        f"Комментарий: {record['comment']}" if record["comment"] else "",
        "Реквизиты: " + ", ".join(a["name"] for a in record["attributes"]) if record["attributes"] else "",
        "ТЧ: " + ", ".join(t["name"] for t in record["tabular_sections"]) if record["tabular_sections"] else "",
        "Измерения: " + ", ".join(d["name"] for d in record["dimensions"]) if record["dimensions"] else "",
        "Ресурсы: " + ", ".join(r["name"] for r in record["resources"]) if record["resources"] else "",
        "Формы: " + ", ".join(f["name"] for f in record["forms"]) if record["forms"] else "",
        "Команды: " + ", ".join(c["name"] for c in record["commands"]) if record["commands"] else "",
        "Значения: " + ", ".join(record["enum_values"]) if record["enum_values"] else "",
    ]
    text = "\n".join(line for line in summary if line)
    chunks = [
        {
            "id": f"metaobj:{record['kind']}:{record['name']}",
            "text": text,
            "metadata": {
                "kind": record["kind_ru"],
                "name": record["name"],
                "synonym": record["synonym"],
                "kind_tag": record["kind"],
                "level": "object",
                "source": f"{record['kind']}/{record['name']}.xml",
            },
        }
    ]

    def add(level: str, name: str, synonym: str, types: str) -> None:
        if not name:
            return
        line = f"{record['kind_ru']} {record['name']} · {level} {name}"
        if synonym:
            line += f" ({synonym})"
        if types and types != "—":
            line += f" · тип: {types}"
        chunks.append(
            {
                "id": f"metaobj:{record['kind']}:{record['name']}:{level}:{name}",
                "text": line,
                "metadata": {
                    "kind": record["kind_ru"],
                    "kind_tag": record["kind"],
                    "name": record["name"],
                    "level": level,
                    "field": name,
                    "source": f"{record['kind']}/{record['name']}.xml",
                },
            }
        )

    for attribute in record["attributes"]:
        add("реквизит", attribute["name"], attribute["synonym"], _type_hint(attribute))
    for section in record["tabular_sections"]:
        add("ТЧ", section["name"], section["synonym"], "—")
        for attribute in section.get("attributes", []):
            add(f"ТЧ {section['name']}", attribute["name"], attribute["synonym"], _type_hint(attribute))
    for dimension in record["dimensions"]:
        add("измерение", dimension["name"], dimension["synonym"], _type_hint(dimension))
    for resource in record["resources"]:
        add("ресурс", resource["name"], resource["synonym"], _type_hint(resource))
    for form in record["forms"]:
        add("форма", form["name"], "", form["type"])
    for command in record["commands"]:
        add("команда", command["name"], command["synonym"], "—")
    for value in record["enum_values"]:
        add("значение", value, "", "—")
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse 1C XML configuration dump")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="ограничить число объектов (смоук)")
    args = ap.parse_args()
    if not args.root.exists():
        raise SystemExit(f"нет каталога: {args.root}")

    xml_files = sorted(
        path
        for path in args.root.glob("*/*.xml")
        if path.name not in {"Configuration.xml", "ConfigDumpInfo.xml"}
    )
    if args.limit:
        xml_files = xml_files[: args.limit]
    print(f"object xml files: {len(xml_files)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LEXICON.parent.mkdir(parents=True, exist_ok=True)
    kinds: Counter[str] = Counter()
    lexicon: dict[str, dict[str, str]] = {}
    objects_written = 0
    chunks_written = 0

    with (OUT_DIR / "objects.jsonl").open("w", encoding="utf-8") as obj_fh, (
        OUT_DIR / "chunks.jsonl"
    ).open("w", encoding="utf-8") as chunk_fh:
        for path in xml_files:
            record = parse_object(path)
            if not record:
                continue
            kinds[record["kind"]] += 1
            objects_written += 1
            obj_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            for chunk in build_chunks(record):
                chunk_fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                chunks_written += 1
            for field in (
                record["attributes"]
                + [a for section in record["tabular_sections"] for a in section.get("attributes", [])]
                + record["dimensions"]
                + record["resources"]
                + [{"name": form["name"]} for form in record["forms"]]
                + record["commands"]
            ):
                field_name = field["name"]
                if field_name and re.fullmatch(r"[A-Za-zА-Яа-я_][\w]*", field_name):
                    lexicon.setdefault(
                        field_name,
                        {"kind": record["kind_ru"], "object": record["name"]},
                    )
            lexicon.setdefault(
                record["name"], {"kind": record["kind_ru"], "object": record["name"]}
            )

    stats = {
        "objects": objects_written,
        "chunks": chunks_written,
        "by_kind": dict(sorted(kinds.items(), key=lambda item: -item[1])),
        "lexicon_size": len(lexicon),
    }
    (OUT_DIR / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LEXICON.write_text(json.dumps(lexicon, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("out:", OUT_DIR, "| lexicon:", LEXICON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
