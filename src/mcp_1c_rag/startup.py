"""Проверка индекса и env при старте MCP (без загрузки embedding-модели)."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

IssueLevel = Literal["error", "warning"]

_VALID_MODES = frozenset(
    {"lexical", "vector", "hybrid", "hybrid_rerank", "pearson"}
)
_SKIP_ENV = "MCP_1C_SKIP_STARTUP_CHECK"
_N_DOCS_RE = re.compile(r'"n_docs"\s*:\s*(\d+)')


@dataclass(frozen=True)
class ValidationIssue:
    """Одна проблема проверки runtime."""

    level: IssueLevel
    code: str
    message: str
    path: str | None = None
    actual: str | None = None
    hint: str | None = None


@dataclass
class ValidationReport:
    """Результат ``validate_runtime``."""

    issues: list[ValidationIssue] = field(default_factory=list)
    skipped: bool = False

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]


class RuntimeValidationError(Exception):
    """Фатальные ошибки startup-проверки (errors обязательны)."""

    def __init__(
        self,
        errors: list[ValidationIssue],
        warnings: list[ValidationIssue] | None = None,
    ) -> None:
        if not errors:
            raise ValueError("RuntimeValidationError требует хотя бы одну error")
        self.errors = list(errors)
        self.warnings = list(warnings or [])
        super().__init__(format_validation_error(self))


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _read_bm25_n_docs(meta_path: Path) -> int | None:
    """Прочитать n_docs из meta.json (голова файла или полный JSON)."""
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            head = handle.read(8192)
        match = _N_DOCS_RE.search(head)
        if match:
            return int(match.group(1))
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        value = data.get("n_docs")
        return int(value) if value is not None else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _issue(
    level: IssueLevel,
    code: str,
    message: str,
    *,
    path: Path | str | None = None,
    actual: Any = None,
    hint: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        level=level,
        code=code,
        message=message,
        path=str(path) if path is not None else None,
        actual=None if actual is None else str(actual),
        hint=hint,
    )


def _default_client(chroma_dir: Path) -> Any:
    """Открыть PersistentClient без mkdir и без get_or_create."""
    import chromadb

    return chromadb.PersistentClient(path=str(chroma_dir))


def _check_search_mode(mode: str) -> list[ValidationIssue]:
    """Валидировать MCP_1C_SEARCH_MODE."""
    if mode in _VALID_MODES:
        return []
    return [
        _issue(
            "error",
            "bad_search_mode",
            "неизвестный MCP_1C_SEARCH_MODE",
            actual=mode,
            hint=(
                "задайте MCP_1C_SEARCH_MODE=hybrid "
                "(допустимо: lexical|vector|hybrid|hybrid_rerank|pearson)"
            ),
        )
    ]


def _check_chroma_path(chroma_path: Path) -> list[ValidationIssue]:
    """Проверить каталог Chroma и наличие SQLite-файла."""
    if not chroma_path.exists():
        return [
            _issue(
                "error",
                "chroma_missing",
                "каталог Chroma не существует",
                path=chroma_path,
                actual="отсутствует",
                hint=(
                    "задайте MCP_1C_CHROMA_DIR=data/chroma_xml "
                    "или соберите индекс (build_procedure_index / build_metadata_index)"
                ),
            )
        ]
    if not chroma_path.is_dir():
        return [
            _issue(
                "error",
                "chroma_not_dir",
                "MCP_1C_CHROMA_DIR указывает не на каталог",
                path=chroma_path,
                actual="не каталог",
                hint="задайте путь к каталогу Chroma, например data/chroma_xml",
            )
        ]

    issues: list[ValidationIssue] = []
    if not any(chroma_path.iterdir()):
        issues.append(
            _issue(
                "error",
                "chroma_empty",
                "каталог Chroma пуст",
                path=chroma_path,
                actual="0 файлов",
                hint=(
                    "задайте MCP_1C_CHROMA_DIR=data/chroma_xml "
                    "или пересоберите индекс"
                ),
            )
        )
    sqlite_path = chroma_path / "chroma.sqlite3"
    if not sqlite_path.is_file():
        issues.append(
            _issue(
                "error",
                "chroma_sqlite_missing",
                "в каталоге Chroma нет chroma.sqlite3",
                path=sqlite_path,
                actual="файл не найден",
                hint=(
                    "проверьте MCP_1C_CHROMA_DIR "
                    "(ожидается data/chroma_xml с chroma.sqlite3)"
                ),
            )
        )
    return issues


def _check_collections(
    client: Any,
    code_name: str,
    meta_name: str,
    chroma_path: Path,
) -> list[ValidationIssue]:
    """Проверить code/metadata-коллекции через уже открытый клиент."""
    issues: list[ValidationIssue] = []
    try:
        code_coll = client.get_collection(code_name)
        code_count = int(code_coll.count())
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue(
                "error",
                "code_collection_missing",
                f"коллекция кода {code_name!r} недоступна",
                path=chroma_path,
                actual=f"{type(exc).__name__}: {exc}",
                hint=(
                    f"соберите индекс: python scripts/build_procedure_index.py "
                    f"(коллекция {code_name}) или "
                    f"MCP_1C_CODE_COLLECTION={code_name}"
                ),
            )
        )
    else:
        if code_count <= 0:
            issues.append(
                _issue(
                    "error",
                    "code_collection_empty",
                    f"коллекция кода {code_name!r} пуста",
                    path=chroma_path,
                    actual=f"count={code_count}",
                    hint=(
                        "пересоберите индекс: "
                        "python scripts/build_procedure_index.py"
                    ),
                )
            )

    try:
        meta_coll = client.get_collection(meta_name)
        meta_count = int(meta_coll.count())
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue(
                "warning",
                "metadata_collection_missing",
                f"коллекция метаданных {meta_name!r} недоступна",
                path=chroma_path,
                actual=f"{type(exc).__name__}: {exc}",
                hint=(
                    "metadata-инструменты будут пустыми; "
                    "соберите: python scripts/build_metadata_index.py "
                    f"или задайте MCP_1C_METADATA_COLLECTION={meta_name}"
                ),
            )
        )
    else:
        if meta_count <= 0:
            issues.append(
                _issue(
                    "warning",
                    "metadata_collection_empty",
                    f"коллекция метаданных {meta_name!r} пуста",
                    path=chroma_path,
                    actual=f"count={meta_count}",
                    hint=(
                        "metadata-инструменты будут пустыми; "
                        "python scripts/build_metadata_index.py"
                    ),
                )
            )
    return issues


def _check_chroma(
    chroma_path: Path,
    code_name: str,
    meta_name: str,
    factory: Callable[[Path], Any],
) -> list[ValidationIssue]:
    """Проверить Chroma и коллекции; не вызывать get_or_create."""
    issues = _check_chroma_path(chroma_path)
    if any(issue.code in {"chroma_missing", "chroma_not_dir"} for issue in issues):
        return issues
    sqlite_path = chroma_path / "chroma.sqlite3"
    if not sqlite_path.is_file() or not any(chroma_path.iterdir()):
        return issues

    try:
        client = factory(chroma_path)
    except Exception as exc:  # noqa: BLE001 — диагностика старта
        issues.append(
            _issue(
                "error",
                "chroma_open_failed",
                "не удалось открыть Chroma PersistentClient",
                path=chroma_path,
                actual=f"{type(exc).__name__}: {exc}",
                hint="проверьте MCP_1C_CHROMA_DIR и права доступа",
            )
        )
        return issues
    issues.extend(_check_collections(client, code_name, meta_name, chroma_path))
    return issues


def _check_bm25(bm25_path: Path) -> list[ValidationIssue]:
    """Проверить наличие и непустоту BM25-индекса."""
    issues: list[ValidationIssue] = []
    index_npz = bm25_path / "index.npz"
    meta_json = bm25_path / "meta.json"
    if not index_npz.is_file():
        issues.append(
            _issue(
                "error",
                "bm25_index_missing",
                "BM25 index.npz не найден",
                path=index_npz,
                actual="файл не найден",
                hint=(
                    "задайте MCP_1C_BM25_DIR=data/bm25_all или соберите: "
                    "python scripts/build_bm25.py --chroma-dir data/chroma_xml "
                    "--out data/bm25_all"
                ),
            )
        )
    if not meta_json.is_file():
        issues.append(
            _issue(
                "error",
                "bm25_meta_missing",
                "BM25 meta.json не найден",
                path=meta_json,
                actual="файл не найден",
                hint=(
                    "задайте MCP_1C_BM25_DIR=data/bm25_all или пересоберите BM25 "
                    "(scripts/build_bm25.py)"
                ),
            )
        )
        return issues

    n_docs = _read_bm25_n_docs(meta_json)
    if n_docs is None:
        issues.append(
            _issue(
                "error",
                "bm25_meta_invalid",
                "BM25 meta.json не читается или без n_docs",
                path=meta_json,
                actual="parse_error",
                hint="пересоберите BM25: python scripts/build_bm25.py",
            )
        )
    elif n_docs <= 0:
        issues.append(
            _issue(
                "error",
                "bm25_empty",
                "BM25 индекс пуст (n_docs=0)",
                path=meta_json,
                actual=f"n_docs={n_docs}",
                hint=(
                    "пересоберите BM25 после индексации Chroma: "
                    "python scripts/build_bm25.py"
                ),
            )
        )
    return issues


def validate_runtime(
    *,
    chroma_dir: Path | None = None,
    bm25_dir: Path | None = None,
    code_collection: str | None = None,
    metadata_collection: str | None = None,
    search_default_mode: str | None = None,
    skip: bool | None = None,
    client_factory: Callable[[Path], Any] | None = None,
) -> ValidationReport:
    """Проверить пути индекса, BM25, коллекции и дефолтный режим.

    Не загружает embedding-модель и не вызывает ``get_or_create_collection``.
    При ``MCP_1C_SKIP_STARTUP_CHECK=1`` возвращает ``skipped=True`` без I/O.
    """
    if skip is None:
        skip = _env_truthy(_SKIP_ENV)
    if skip:
        return ValidationReport(skipped=True)

    from mcp_1c_rag.config import (
        BM25_DIR,
        CHROMA_DIR,
        COLLECTION_CODE,
        COLLECTION_METADATA,
        SEARCH_DEFAULT_MODE,
    )

    chroma_path = Path(chroma_dir) if chroma_dir is not None else Path(CHROMA_DIR)
    bm25_path = Path(bm25_dir) if bm25_dir is not None else Path(BM25_DIR)
    code_name = code_collection if code_collection is not None else COLLECTION_CODE
    meta_name = (
        metadata_collection
        if metadata_collection is not None
        else COLLECTION_METADATA
    )
    mode = (
        search_default_mode
        if search_default_mode is not None
        else SEARCH_DEFAULT_MODE
    ).strip().lower()
    factory = client_factory or _default_client

    issues: list[ValidationIssue] = []
    issues.extend(_check_search_mode(mode))
    issues.extend(_check_chroma(chroma_path, code_name, meta_name, factory))
    issues.extend(_check_bm25(bm25_path))
    return ValidationReport(issues=issues, skipped=False)


def format_validation_error(
    exc: RuntimeValidationError | ValidationReport,
) -> str:
    """Многострочное actionable-сообщение для stderr."""
    if isinstance(exc, ValidationReport):
        errors = exc.errors
        warnings = exc.warnings
    else:
        errors = exc.errors
        warnings = exc.warnings

    lines = ["Проверка индекса при старте MCP не пройдена."]
    for issue in errors:
        lines.extend(_format_issue(issue))
    if warnings:
        lines.append("Предупреждения:")
        for issue in warnings:
            lines.extend(_format_issue(issue))
    lines.append(
        f"Чтобы пропустить проверку (не рекомендуется): {_SKIP_ENV}=1"
    )
    return "\n".join(lines)


def format_validation_warnings(report: ValidationReport) -> str:
    """Текст предупреждений для stderr (сервер всё же стартует)."""
    lines = ["Предупреждения проверки индекса при старте MCP:"]
    for issue in report.warnings:
        lines.extend(_format_issue(issue))
    return "\n".join(lines)


def _format_issue(issue: ValidationIssue) -> list[str]:
    prefix = "error" if issue.level == "error" else "warning"
    lines = [f"- [{prefix}/{issue.code}] {issue.message}"]
    if issue.path:
        lines.append(f"  путь: {issue.path}")
    if issue.actual is not None:
        lines.append(f"  факт: {issue.actual}")
    if issue.hint:
        lines.append(f"  исправление: {issue.hint}")
    return lines


def run_startup_checks(*, file: Any = None) -> ValidationReport:
    """Выполнить проверку: errors → RuntimeValidationError; warnings → stderr."""
    out = sys.stderr if file is None else file
    report = validate_runtime()
    if report.skipped:
        return report
    if report.errors:
        raise RuntimeValidationError(report.errors, report.warnings)
    if report.warnings:
        print(format_validation_warnings(report), file=out)
    return report


__all__ = [
    "RuntimeValidationError",
    "ValidationIssue",
    "ValidationReport",
    "format_validation_error",
    "format_validation_warnings",
    "run_startup_checks",
    "validate_runtime",
]
