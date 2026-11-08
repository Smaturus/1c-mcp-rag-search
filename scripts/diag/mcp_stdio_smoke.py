"""Smoke-проверка MCP-сервера по stdio: 4 tools, запросы к коду и XML-метаданным.

Поднимает ``python -m mcp_1c_rag`` через клиент MCP SDK и проверяет:
- tools/list отдаёт ровно search, evaluate_retrieval, route_query, index_stats,
  все с ``readOnlyHint=true``;
- search по BSL (``code_proc``), XML-метаданным (``metadata_xml``) и ``all``;
- legacy-синонимы ``code``/``metadata`` дают ту же выдачу, что канонические имена;
- ``config_xml`` (пока не в обслуживаемом индексе) отклоняется схемой;
- route_query, evaluate_retrieval по XML-метаданным, index_stats;
- чтение ресурсов bsl:// и metadata:// по URI из выдачи search;
- сервер не обращается к huggingface.co (``HF_HUB_OFFLINE`` по умолчанию).

Запуск (из корня репозитория, нужен собранный индекс):
    python scripts/diag/mcp_stdio_smoke.py

По умолчанию MCP_1C_CHROMA_DIR=data/chroma_xml, MCP_1C_BM25_DIR=data/bm25_all;
заданные в окружении MCP_1C_* / HF_* значения пробрасываются серверу.
Код выхода: 0 — все проверки прошли, 1 — есть провалы. stderr сервера —
во временном файле ``mcp_stdio_stderr.log``.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters, stdio_client

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = Path(__file__).resolve().parents[2]
ERRLOG = Path(tempfile.gettempdir()) / "mcp_stdio_stderr.log"
EXPECTED_TOOLS = ["evaluate_retrieval", "index_stats", "route_query", "search"]
PREVIEW_CHARS = 240


@dataclass(frozen=True)
class ToolCheck:
    """Один вызов tool и ожидание: успех с непустым текстом или ошибка схемы."""

    label: str
    tool: str
    arguments: dict[str, Any]
    expect_error: bool = False
    same_as: str | None = None


CHECKS: tuple[ToolCheck, ...] = (
    ToolCheck(
        "search: код (code_proc, hybrid)",
        "search",
        {"query": "НДС с аванса", "top_k": 3, "collection": "code_proc"},
    ),
    ToolCheck(
        "search: legacy code → code_proc",
        "search",
        {"query": "НДС с аванса", "top_k": 3, "collection": "code"},
        same_as="search: код (code_proc, hybrid)",
    ),
    ToolCheck(
        "search: XML-метаданные (metadata_xml, lexical)",
        "search",
        {
            "query": "РеализацияТоваровУслуг",
            "top_k": 3,
            "mode": "lexical",
            "collection": "metadata_xml",
        },
    ),
    ToolCheck(
        "search: legacy metadata → metadata_xml",
        "search",
        {
            "query": "РеализацияТоваровУслуг",
            "top_k": 3,
            "mode": "lexical",
            "collection": "metadata",
        },
        same_as="search: XML-метаданные (metadata_xml, lexical)",
    ),
    ToolCheck(
        "search: код + XML (all, hybrid)",
        "search",
        {"query": "проведение реализации товаров", "top_k": 5, "collection": "all"},
    ),
    ToolCheck(
        "search: config_xml отклоняется схемой",
        "search",
        {"query": "x", "collection": "config_xml"},
        expect_error=True,
    ),
    ToolCheck(
        "route_query",
        "route_query",
        {"query": "Какие проводки у документа РеализацияТоваровУслуг?"},
    ),
    ToolCheck(
        "evaluate_retrieval: XML-метаданные",
        "evaluate_retrieval",
        {
            "query": "РеализацияТоваровУслуг",
            "k": 5,
            "collection": "metadata_xml",
            "null_pairs": 0,
        },
    ),
    ToolCheck("index_stats", "index_stats", {}),
)


@dataclass
class Report:
    """Итоги проверок: строки PASS/FAIL и тексты ответов для сравнения."""

    failures: int = 0
    texts: dict[str, str] = field(default_factory=dict)

    def record(self, ok: bool, label: str, detail: str = "") -> None:
        """Напечатать результат одной проверки и учесть провал."""
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failures += 1


def _server_env() -> dict[str, str]:
    """Окружение сервера: MCP_1C_* / HF_* из текущего процесса + дефолты индекса."""
    env = {k: v for k, v in os.environ.items() if k.startswith(("MCP_1C_", "HF_"))}
    env.setdefault("MCP_1C_CHROMA_DIR", str(PROJECT / "data" / "chroma_xml"))
    env.setdefault("MCP_1C_BM25_DIR", str(PROJECT / "data" / "bm25_all"))
    return env


def _text_of(result: Any) -> str:
    """Склеить текстовые блоки ответа tools/call."""
    return "\n".join(
        block.text for block in result.content if getattr(block, "type", "") == "text"
    )


def _preview(text: str) -> str:
    """Короткий однострочный фрагмент ответа для печати."""
    return text[:PREVIEW_CHARS].replace("\n", " | ")


async def _check_tool_list(session: ClientSession, report: Report) -> None:
    """tools/list: ровно 4 tools, все помечены только для чтения."""
    tools = (await session.list_tools()).tools
    names = sorted(tool.name for tool in tools)
    report.record(names == EXPECTED_TOOLS, "tools/list: 4 tools", str(names))
    not_read_only = [
        tool.name
        for tool in tools
        if tool.annotations is None or tool.annotations.read_only_hint is not True
    ]
    report.record(not not_read_only, "tools/list: readOnlyHint=true", str(not_read_only or "все"))


async def _run_check(session: ClientSession, check: ToolCheck, report: Report) -> None:
    """Вызвать tool и сверить результат с ожиданием проверки."""
    result = await session.call_tool(check.tool, check.arguments)
    text = _text_of(result)
    report.texts[check.label] = text
    if check.expect_error:
        report.record(bool(result.is_error), check.label, _preview(text))
        return
    ok = not result.is_error and bool(text.strip())
    if ok and check.same_as is not None:
        ok = text == report.texts.get(check.same_as)
        report.record(ok, check.label, f"совпадает с «{check.same_as}»: {ok}")
        return
    report.record(ok, check.label, _preview(text))


async def _check_uris(session: ClientSession, report: Report) -> None:
    """Прочитать bsl:// и metadata:// из выдачи search: код и XML-метаданные."""
    import re

    wanted = {
        "bsl": "search: код (code_proc, hybrid)",
        "metadata": "search: XML-метаданные (metadata_xml, lexical)",
    }
    for scheme, label in wanted.items():
        match = re.search(rf"uri=({scheme}://\S+)", report.texts.get(label, ""))
        if match is None:
            report.record(False, f"uri {scheme}:// в выдаче", "нет")
            continue
        uri = match.group(1)
        try:
            result = await session.read_resource(uri)
        except Exception as exc:  # noqa: BLE001 — смоук фиксирует любой отказ чтения
            report.record(False, f"resources/read {uri}", type(exc).__name__)
            continue
        text = getattr(result.contents[0], "text", "")
        report.record(bool(text.strip()), f"resources/read {scheme}://", f"{len(text)} символов")


def _check_offline(report: Report) -> None:
    """В stderr сервера не должно быть обращений к huggingface.co."""
    log = ERRLOG.read_text(encoding="utf-8", errors="replace")
    hits = log.count("huggingface.co")
    report.record(hits == 0, "нет обращений к huggingface.co", f"упоминаний в stderr: {hits}")


async def main() -> int:
    """Прогнать все проверки; вернуть код выхода процесса."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_1c_rag"],
        env=_server_env(),
        cwd=str(PROJECT),
    )
    report = Report()
    with ERRLOG.open("w", encoding="utf-8") as errlog:
        async with (
            stdio_client(params, errlog=errlog) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            await _check_tool_list(session, report)
            for check in CHECKS:
                await _run_check(session, check, report)
            await _check_uris(session, report)
    _check_offline(report)
    print(f"\nПровалов: {report.failures}. stderr сервера: {ERRLOG}")
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
