"""Quality gate: цикломатическая сложность и индекс поддерживаемости.

Падает, если:
- средняя сложность по `src/` выше ``MAX_AVERAGE``;
- любой блок вне baseline сложнее ``MAX_BLOCK``;
- известный блок из baseline стал сложнее зафиксированного значения (ratchet);
- индекс поддерживаемости модуля ниже ``MIN_MI``.

Baseline хранит известные «долги» (см. «Инженерные принципы» в README):
его можно только сокращать — после рефакторинга запись удаляется.

Запуск:
    python scripts/check_complexity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radon.complexity import cc_rank, cc_visit  # noqa: E402
from radon.metrics import mi_visit  # noqa: E402

SRC = ROOT / "src" / "mcp_1c_rag"

MAX_AVERAGE = 6.0
MAX_BLOCK = 20
MIN_MI = 20.0  # radon grade A начинается с 20

# Известные сложные функции: имя -> зафиксированная сложность.
# Записи удаляются после рефакторинга; расти значениям нельзя.
BASELINE: dict[str, int] = {
    "eval_retrieval.evaluate_retrieval": 33,
    "indexer.index_metadata": 30,
}


def _qualname(module: str, block) -> str:
    """Имя вида ``module.function`` (класс добавляется через точку)."""
    name = getattr(block, "fullname", block.name)
    return f"{module}.{name}"


def main() -> int:
    files = sorted(SRC.rglob("*.py"))
    if not files:
        print(f"no files under {SRC}")
        return 1

    complexities: list[int] = []
    violations: list[str] = []
    top: list[tuple[int, str, str]] = []
    mi_results: list[tuple[str, float]] = []

    for path in files:
        source = path.read_text(encoding="utf-8")
        mi_results.append((str(path.relative_to(ROOT)), mi_visit(source, multi=True)))
        for block in cc_visit(source):
            complexity = int(block.complexity)
            complexities.append(complexity)
            module = path.stem
            name = _qualname(module, block)
            top.append((complexity, cc_rank(complexity), name))
            recorded = BASELINE.get(name)
            if recorded is None:
                if complexity > MAX_BLOCK:
                    violations.append(f"{name}: CC={complexity} > {MAX_BLOCK} (вне baseline)")
            elif complexity > recorded:
                violations.append(f"{name}: CC вырос {recorded} → {complexity}")

    average = sum(complexities) / len(complexities)
    if average > MAX_AVERAGE:
        violations.append(f"средняя сложность {average:.2f} > {MAX_AVERAGE}")

    for module, mi in mi_results:
        if mi < MIN_MI:
            violations.append(f"{module}: MI={mi:.1f} < {MIN_MI}")

    top_sorted = sorted(top, reverse=True)[:5]
    print(f"блоков: {len(complexities)}, средняя сложность: {average:.2f} (порог {MAX_AVERAGE})")
    print(f"минимум MI: {min(mi for _, mi in mi_results):.1f} (порог {MIN_MI})")
    print("самые сложные:")
    for complexity, rank, name in top_sorted:
        marker = " " if complexity <= MAX_BLOCK else "!"
        print(f"  {marker} {complexity:3d} {rank}  {name}")

    if violations:
        print("\nНАРУШЕНИЯ:")
        for item in violations:
            print("  -", item)
        return 1
    print("\nOK: complexity gate пройден")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
