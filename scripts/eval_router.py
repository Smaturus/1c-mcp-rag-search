"""Оценка роутера: точность на routing-кейсах и end-to-end nDCG@10 на gold-set v2.

Почему два разных набора:

- **routing-кейсы** (`data/eval/routing_cases.jsonl`) — вручную размеченные
  фразы по контракту из промпта ассистента (exact → lexical, смысл → vector,
  смешанное/требования → hybrid). Только они измеряют точность роутера.
- **gold-set v2** — измеряет end-to-end качество: сравнение «routed» и
  фиксированного `hybrid`. Здесь нельзя ожидать маршрут по типу запроса:
  semantic-запросы в v2 часто содержат точное имя объекта и по контракту
  обязаны уходить в hybrid.

Запуск:
    python scripts/eval_router.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.config import DATA_DIR  # noqa: E402
from mcp_1c_rag.eval_metrics import ndcg_at_k, recall_at_k  # noqa: E402
from mcp_1c_rag.router import Route, route_query  # noqa: E402
from mcp_1c_rag.search_modes import search  # noqa: E402

CASES = DATA_DIR / "eval" / "routing_cases.jsonl"
GOLDEN = DATA_DIR / "eval" / "gold_set_v2.jsonl"
REPORT_MD = DATA_DIR / "eval" / "router_report.md"
REPORT_JSON = DATA_DIR / "eval" / "router_report.json"
K = 10

MODE_BY_ROUTE = {Route.LEXICAL: "lexical", Route.VECTOR: "vector", Route.HYBRID: "hybrid"}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"нет файла: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _routing_accuracy(cases: list[dict]) -> tuple[float, dict, dict]:
    correct = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    failures: list[dict] = []
    for case in cases:
        decision = route_query(case["query"])
        expected = case["expected"]
        confusion[expected][decision.route.value] += 1
        if decision.route.value == expected:
            correct += 1
        else:
            failures.append(
                {
                    "query": case["query"],
                    "expected": expected,
                    "actual": decision.route.value,
                    "reason": decision.reason,
                }
            )
    return correct / len(cases), {k: dict(v) for k, v in confusion.items()}, {"failures": failures}


def _e2e(mode_resolver, rows: list[dict]) -> dict[str, float]:
    ndcgs: list[float] = []
    recalls: list[float] = []
    for row in rows:
        candidates = search(row["query"], top_k=K, mode=mode_resolver(row), collection="code")
        sources = [c.source for c in candidates]
        relevant = row["relevant_sources"]
        ndcgs.append(ndcg_at_k(sources, relevant, k=K))
        recalls.append(recall_at_k(sources, relevant, k=K))
    return {"ndcg@10": sum(ndcgs) / len(ndcgs), "recall@10": sum(recalls) / len(recalls)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate query router")
    ap.add_argument("--routing-only", action="store_true",
                    help="только точность маршрута, без end-to-end прогона поиска")
    args = ap.parse_args()

    cases = _read_jsonl(CASES)
    rows = _read_jsonl(GOLDEN)

    accuracy, confusion, details = _routing_accuracy(cases)
    wrote_e2e = not args.routing_only
    routed = _e2e(lambda row: MODE_BY_ROUTE[route_query(row["query"]).route], rows) if wrote_e2e else {}
    fixed = _e2e(lambda _row: "hybrid", rows) if wrote_e2e else {}

    lines = [
        "# Оценка роутера запросов",
        "",
        f"- routing-кейсов: {len(cases)} (ручная разметка по контракту)",
        f"- **точность маршрута: {accuracy:.1%}** "
        f"({round(accuracy * len(cases))}/{len(cases)})",
        "",
        "## Матрица маршрутов (expected → actual)",
        "",
        "| Ожидался | lexical | vector | hybrid |",
        "|---|---|---|---|",
    ]
    for expected in ("lexical", "vector", "hybrid"):
        row = confusion.get(expected, {})
        lines.append(
            f"| `{expected}` | {row.get('lexical', 0)} | {row.get('vector', 0)} | {row.get('hybrid', 0)} |"
        )
    if details["failures"]:
        lines += ["", "## Ошибки роутинга", "", "| Запрос | Ждали | Получили | Причина |", "|---|---|---|---|"]
        for item in details["failures"]:
            lines.append(
                f"| {item['query']} | {item['expected']} | {item['actual']} | {item['reason']} |"
            )
    lines += [
        "",
        "## End-to-end качество на gold-set v2 (коллекция `code`, k=10)",
        "",
    ]
    if not wrote_e2e:
        lines += ["_Не считалось: запуск с `--routing-only`._"]
    else:
        lines += [
            "| Стратегия | nDCG@10 | Recall@10 |",
            "|---|---|---|",
            f"| fixed `hybrid` (безопасный дефолт) | {fixed['ndcg@10']:.3f} | {fixed['recall@10']:.3f} |",
            f"| **routed** | {routed['ndcg@10']:.3f} | {routed['recall@10']:.3f} |",
            "",
            "Контракт маршрутизации не должен ухудшать качество относительно дефолта; "
            "если ухудшает — дефолт остаётся `hybrid`, а правила фиксируются как эвристика.",
        ]

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    REPORT_JSON.write_text(
        json.dumps(
            {
                "routing_cases": len(cases),
                "accuracy": accuracy,
                "confusion": confusion,
                "failures": details["failures"],
                "fixed_hybrid": fixed,
                "routed": routed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
