#!/usr/bin/env python3
"""Curator fix plan for 1c-mcp-rag-search.

Read-only auditor: checks the current checkout and prints prioritized fixes.
It never modifies files, never calls the network and never needs credentials.

Usage:
    python scripts/curator_fix_plan.py
    python scripts/curator_fix_plan.py --json   # machine-readable output

Run it on an up-to-date main checkout (git pull / fresh clone); on an old
checkout some files are absent, and the script will say so instead of guessing.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Finding:
    priority: str
    title: str
    evidence: str
    fix: str


def read(path: str) -> str | None:
    file = ROOT / path
    if not file.exists():
        return None
    return file.read_text(encoding="utf-8", errors="replace")


def git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=15, check=False
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def check_reproduction(findings: list[Finding]) -> None:
    evaluation = read("docs/EVALUATION.md")
    if evaluation is None:
        findings.append(Finding(
            "P1", "docs/EVALUATION.md отсутствует в checkout",
            "файл не найден — похоже, checkout старше main",
            "git pull (или свежий clone), затем повторить проверку",
        ))
        return
    has_stability = "eval_stability.py" in evaluation
    tracked = git("ls-files", "--", "data/eval/code_proc", "data/eval/v3_code") or ""
    tracked_runs = [line for line in tracked.splitlines() if Path(line).name.startswith("run_")]
    if has_stability and not tracked_runs:
        findings.append(Finding(
            "P1", "Команды воспроизведения в EVALUATION не работают на свежем clone",
            "docs/EVALUATION.md вызывает eval_stability/eval_paired, но run_*.json "
            "не входят в git (в локальном checkout они могут лежать как untracked)",
            "Сгенерировать прогоны и передать их явно: "
            "4× eval_pipeline --out-dir data/eval/_runs/run_N; "
            "скопировать report_pipeline.json в run_N.json; "
            "eval_stability --out-dir data/eval/_runs; "
            "eval_paired --runs-glob \"data/eval/_runs/run_*.json\". "
            "Либо заменить в EVALUATION команды на eval_paired --report ...",
        ))
    paired = read("scripts/eval_paired.py")
    if paired and "DEFAULT_RUNS_GLOB" in paired:
        findings.append(Finding(
            "P2", "eval_paired по умолчанию ищет удалённые run_hybrid_*.json",
            "DEFAULT_RUNS_GLOB = data/eval/code_proc/run_hybrid_*.json",
            "В документации указывать --runs-glob/--report; при необходимости "
            "изменить DEFAULT_RUNS_GLOB или вернуть run-файлы",
        ))


def check_metadata_provenance(findings: list[Finding]) -> None:
    files = [
        "README.md",
        "docs/EVALUATION.md",
        "docs/adr/0008-xml-metadata-source.md",
        "data/eval/metadata_ab_report.md",
    ]
    hits = []
    for path in files:
        text = read(path)
        if text and re.search(r"99[.,]7\s*%", text):
            hits.append(path)
    if not hits:
        return
    caveat = any("ADR 0010" in (read(p) or "") for p in hits)
    if not caveat:
        findings.append(Finding(
            "P2", "Metadata A/B 99,7 % подан как текущий, но снят до ADR 0010",
            "число встречается в: " + ", ".join(hits),
            "Пометить замер как pre-ADR-0010 или перезапустить "
            "python scripts/eval_metadata_xml.py; вывод XML>text, вероятно, сохранится",
        ))


def check_roadmap(findings: list[Finding]) -> None:
    readme = read("README.md")
    if readme is None:
        return
    match = re.search(r"^- \[ \] RRF.*$", readme, flags=re.MULTILINE)
    history = read("docs/HISTORY.md") or ""
    if match and "Веса RRF" in history:
        findings.append(Finding(
            "P2", "README Roadmap противоречит HISTORY",
            match.group(0),
            "Заменить на '- [x] RRF с весами … — проверено, прироста нет "
            "(docs/HISTORY.md)' или убрать пункт",
        ))


def check_v3_table(findings: list[Finding]) -> None:
    evaluation = read("docs/EVALUATION.md")
    if evaluation is None:
        return
    if re.search(r"\| `pearson` \| — \| — \| 0\.229", evaluation):
        findings.append(Finding(
            "P3", "В таблице v3 у pearson/hybrid не заполнены Hit@1, MRR, Recall",
            "в docs/EVALUATION.md стоят прочерки при наличии значений в report_stability",
            "Заполнить из data/eval/v3_code/report_stability.md: "
            "pearson 0.113/0.228/0.229/0.342; hybrid 0.094/0.215/0.217/0.339",
        ))


def check_hygiene(findings: list[Finding]) -> None:
    logs = [p for p in ("data/meta_build2.err", "data/proc_build.err") if (ROOT / p).exists()]
    if logs:
        findings.append(Finding(
            "P3", "В репозитории лежат служебные build-логи",
            ", ".join(logs),
            "git rm --cached <files>; добавить data/*_build*.err в .gitignore",
        ))
    presentation = ROOT / "docs/presentation/en"
    if (presentation / "1c-mcp-rag-search-slides-en.pdf").exists() and not (
        presentation / "build_deck_en.py"
    ).exists():
        findings.append(Finding(
            "P3", "Исходники презентации удалены, остались только PDF/PPTX",
            "docs/presentation/en/build_deck_en.py отсутствует",
            "Для сдачи это допустимо; если колоду нужно будет править — хранить "
            "генератор вне публичного репо или восстановить его",
        ))


def check_release(findings: list[Finding]) -> None:
    tags = git("tag", "--points-at", "HEAD")
    if tags is None:
        return  # git unavailable in this shell — not a repo problem
    if "v0.5.0" not in tags.split() and "v0.5.1" not in tags.split():
        head = git("rev-parse", "--short", "HEAD") or "?"
        findings.append(Finding(
            "P2", "HEAD не совпадает с релизным тегом",
            f"HEAD {head}; tags at HEAD: {tags or '(none)'}",
            "Либо выпустить v0.5.1 после docs/code cleanup, либо явно указать "
            "куратору, что актуален main, а не release tarball",
        ))
    if (ROOT / "pyproject.toml").exists():
        version = re.search(r'^version\s*=\s*"([^"]+)"', read("pyproject.toml") or "", re.M)
        if version and tags and version.group(1) not in tags:
            findings.append(Finding(
                "P3", "Версия pyproject не имеет тега на HEAD",
                f"pyproject {version.group(1)}; tags at HEAD: {tags}",
                "Проверить release-процесс; для куратора важнее актуальный main",
            ))


def check_manual(findings: list[Finding]) -> None:
    findings.append(Finding(
        "P3", "Открыт dependabot PR #1",
        "ci: bump actions/setup-python 7.0.0 → 7.1.0; mergeable/clean",
        "Смержить или закрыть до ревью, чтобы не оставлять висящий PR",
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only curator fix plan")
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args()

    findings: list[Finding] = []
    check_reproduction(findings)
    check_metadata_provenance(findings)
    check_roadmap(findings)
    check_v3_table(findings)
    check_hygiene(findings)
    check_release(findings)
    check_manual(findings)

    if args.json:
        print(json.dumps([asdict(f) for f in findings], ensure_ascii=False, indent=2))
        return 0

    print(f"Curator fix plan for {ROOT}")
    print(f"Findings: {len(findings)}\n")
    for priority in ("P1", "P2", "P3"):
        group = [f for f in findings if f.priority == priority]
        if not group:
            continue
        print(f"== {priority} ({len(group)}) ==")
        for index, finding in enumerate(group, 1):
            print(f"{priority}.{index} {finding.title}")
            print(f"  evidence: {finding.evidence}")
            print(f"  fix:      {finding.fix}\n")
    if not findings:
        print("No findings. Checkout looks ready for curator review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
