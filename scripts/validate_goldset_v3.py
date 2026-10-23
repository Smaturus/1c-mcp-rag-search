"""Validate the v3 hold-out gold-set and the lexical anchor file.

Checks:
- valid JSONL, sequential unique ids, required fields and allowed values;
- every relevant_sources path exists in the served corpus (code + metadata);
- scope/answerable/primary_source/finetune_ready invariants;
- anchor targets exist and match the linked golden row's primary source.

Usage:
    python scripts/validate_goldset_v3.py
    python scripts/validate_goldset_v3.py --golden data/eval/gold_set_v3_holdout.jsonl
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_GOLDEN = DATA / "eval" / "gold_set_v3_holdout.jsonl"
DEFAULT_ANCHORS = DATA / "eval" / "anchors_v3.jsonl"
DEFAULT_SQLITE = DATA / "chroma_xml" / "chroma.sqlite3"

ROUTES = {"lexical", "vector", "hybrid"}
TYPES = {"exact", "semantic", "trap"}
CONFIDENCE = {"high", "medium", "low"}
SCOPES = {"code", "metadata", "none"}
GOLDEN_FIELDS = {
    "id", "category", "query", "relevant_sources", "scope", "answerable",
    "confidence", "expected_route", "type", "primary_source", "finetune_ready",
    "notes",
}
ANCHOR_FIELDS = {"id", "phrase", "target_source", "top_k", "golden_id", "note"}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def load_sources(sqlite_path: Path) -> set[str]:
    conn = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
    try:
        return {
            row[0]
            for row in conn.execute(
                "select distinct string_value from embedding_metadata "
                "where key='source' and string_value is not null"
            )
        }
    finally:
        conn.close()


def validate_golden(rows: list[dict], sources: set[str], errors: list[str]) -> None:
    if not rows:
        errors.append("golden set is empty")
        return
    ids = [r.get("id") for r in rows]
    if ids != list(range(1, len(rows) + 1)):
        errors.append(f"ids must be sequential 1..{len(rows)}, got {ids[:5]}...")
    if len(set(ids)) != len(ids):
        errors.append("duplicate ids")
    for row in rows:
        rid = row.get("id", "?")
        missing = GOLDEN_FIELDS - set(row)
        if missing:
            errors.append(f"#{rid}: missing fields {sorted(missing)}")
            continue
        if row["expected_route"] not in ROUTES:
            errors.append(f"#{rid}: bad route {row['expected_route']!r}")
        if row["type"] not in TYPES:
            errors.append(f"#{rid}: bad type {row['type']!r}")
        if row["confidence"] not in CONFIDENCE:
            errors.append(f"#{rid}: bad confidence {row['confidence']!r}")
        if row["scope"] not in SCOPES:
            errors.append(f"#{rid}: bad scope {row['scope']!r}")
        if not isinstance(row["relevant_sources"], list):
            errors.append(f"#{rid}: relevant_sources must be a list")
            continue
        missing_sources = [s for s in row["relevant_sources"] if s not in sources]
        if missing_sources:
            errors.append(f"#{rid}: sources missing from corpus: {missing_sources}")
        if row["answerable"] and not row["relevant_sources"]:
            errors.append(f"#{rid}: answerable=true with empty relevant_sources")
        if not row["answerable"] and row["relevant_sources"]:
            errors.append(f"#{rid}: answerable=false with non-empty relevant_sources")
        if row["answerable"]:
            expected_scope = "code" if all(s.endswith(".bsl") for s in row["relevant_sources"]) else "metadata"
            if row["scope"] != expected_scope:
                errors.append(f"#{rid}: scope={row['scope']} but sources imply {expected_scope}")
            if row["primary_source"] not in row["relevant_sources"]:
                errors.append(f"#{rid}: primary_source not in relevant_sources")
        else:
            if row["scope"] != "none" or row["primary_source"] is not None:
                errors.append(f"#{rid}: non-answerable row must have scope=none and primary_source=null")
        if row["finetune_ready"] and (len(row["relevant_sources"]) != 1 or row["confidence"] == "low"):
            errors.append(f"#{rid}: finetune_ready requires one source and confidence != low")
        if not str(row["notes"]).strip():
            errors.append(f"#{rid}: empty notes")


def validate_anchors(anchors: list[dict], golden_by_id: dict[int, dict], sources: set[str], errors: list[str]) -> None:
    if not 5 <= len(anchors) <= 10:
        errors.append(f"anchors must contain 5..10 rows, got {len(anchors)}")
    seen = set()
    for anchor in anchors:
        aid = anchor.get("id", "?")
        missing = ANCHOR_FIELDS - set(anchor)
        if missing:
            errors.append(f"{aid}: missing fields {sorted(missing)}")
            continue
        if aid in seen:
            errors.append(f"{aid}: duplicate anchor id")
        seen.add(aid)
        if anchor["target_source"] not in sources:
            errors.append(f"{aid}: target source missing from corpus")
        if not isinstance(anchor["top_k"], int) or anchor["top_k"] < 1:
            errors.append(f"{aid}: top_k must be a positive int")
        golden = golden_by_id.get(anchor["golden_id"])
        if golden is None:
            errors.append(f"{aid}: golden_id {anchor['golden_id']} not found")
        elif golden["primary_source"] != anchor["target_source"]:
            errors.append(f"{aid}: target_source differs from golden primary_source")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate v3 gold-set and anchors")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE)
    args = parser.parse_args()

    errors: list[str] = []
    golden = load_jsonl(args.golden)
    anchors = load_jsonl(args.anchors)
    sources = load_sources(args.sqlite)

    validate_golden(golden, sources, errors)
    golden_by_id = {r["id"]: r for r in golden if "id" in r}
    validate_anchors(anchors, golden_by_id, sources, errors)

    print(f"golden rows: {len(golden)} | anchors: {len(anchors)} | corpus sources: {len(sources)}")
    if golden:
        print("routes:", dict(Counter(r.get("expected_route") for r in golden)))
        print("types:", dict(Counter(r.get("type") for r in golden)))
        print("scope:", dict(Counter(r.get("scope") for r in golden)))
        print("answerable:", dict(Counter(r.get("answerable") for r in golden)))
        print("confidence:", dict(Counter(r.get("confidence") for r in golden)))
        print("finetune_ready:", [r["id"] for r in golden if r.get("finetune_ready")])
    if errors:
        print(f"\nFAILED with {len(errors)} error(s):")
        for error in errors:
            print("  -", error)
        return 1
    print("\nOK: gold-set and anchors are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
