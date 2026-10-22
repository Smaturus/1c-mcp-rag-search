"""Маршрутизация запроса к стратегии поиска.

Правила перенесены из рабочего промпта ассистента 1С и превращены в проверяемую
функцию: выбор инструмента не должен зависеть от того, как промпт «поняла»
модель. Решение объяснимо: маршрут, причина, уверенность.

Три маршрута соответствуют трём инструментам MCP-сервера:

- ``lexical`` → ``search_1c_metadata``: точные имена, проводки, движения,
  связи между документами («grep-подобный» BM25-поиск);
- ``vector`` → ``search_knowledge_base``: «как сделать», «почему не работает»,
  инструкции, опыт;
- ``hybrid`` → ``search_hybrid``: смешанные формулировки, требования, Confluence.

При сомнении выбирается ``hybrid`` — как в исходном промпте.

Проверка: 24 базовых кейса + 15 кейсов заказчика (``tests/Router_tets_cases.txt``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

__all__ = ["Route", "RouteDecision", "route_query"]


class Route(StrEnum):
    """Доступные стратегии поиска."""

    LEXICAL = "lexical"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class RouteDecision:
    """Результат маршрутизации: что выбрали и почему."""

    route: Route
    reason: str
    confidence: float
    tool: str


_TOOL_BY_ROUTE = {
    Route.LEXICAL: "search_1c_metadata",
    Route.VECTOR: "search_knowledge_base",
    Route.HYBRID: "search_hybrid",
}

# Точные имена объектов 1С и обработчиков: РеализацияТоваровУслуг, ПередЗаписью,
# ИспользоватьОбменВС. Внутренняя заглавная буква отсекает обычные слова с
# заглавной буквы в начале предложения (Почему, Инструкция).
_IDENTIFIER = re.compile(
    r"(?<![A-Za-zА-Яа-я0-9])[A-ZА-Я][a-zа-я0-9]+[A-ZА-Я][A-Za-zА-Яа-я0-9]*"
)
_CODE = re.compile(r"\b[A-ZА-Я]{2,}-\d{2,}\b")

# Лексические признаки: точные термины системы (существительные и устойчивые
# фразы). Словоформы покрываются префиксами и явными окончаниями.
_LEXICAL = re.compile(
    r"\b(?:проведени|движени|проводк|остатк|"
    r"регистр(?:а|ы|ов|ам|ах|е|ом|у|ой)?\b|хранится|"
    r"функциональн\w*\s+опци|табличн\w*\s+част|обработчик|"
    r"на\s+основании|связанн|кто\s+пишет|перед\s+записью|при\s+записи|"
    r"где\s+посмотреть)",
    re.IGNORECASE,
)

# Смысловые признаки: как сделать, почему, диагностика, инструкции.
_SEMANTIC = re.compile(
    r"\b(?:почему|зачем|инструкци|что\s+делать|что\s+такое|объясни|"
    r"лучшие\s+практики|опыт|форум|не\s+работает|не\s+проводил|не\s+проводит|"
    r"зависает|слетает|не\s+сходится|не\s+вызывается|обновлени|"
    r"как\s+(?:сделать|настро|правильно|исправить|проверить|работ|получить)|"
    r"что\s+проверить|как\s+исправить)",
    re.IGNORECASE,
)

# Признаки требований/доработок: смешанный или «Confluence»-контекст.
_HYBRID = re.compile(
    r"(?:требовани|confluence|конфлюенс|доработк|согласова|по\s+тз\b|регламент|"
    r"стандарт|добавить\s+реквизит|изменить\s+реквизит|задним\s+числом)",
    re.IGNORECASE,
)


def route_query(query: str) -> RouteDecision:
    """Выбрать маршрут поиска по правилам из промпта ассистента.

    Порядок решений:
    1. Требования/доработки/Confluence → hybrid.
    2. Точное имя + смысл → hybrid (нужны оба сигнала).
    3. Точное имя без смысла → lexical.
    4. Лексика + смысл → hybrid; только лексика → lexical.
    5. Только смысл → vector; признаков нет → hybrid (безопасный дефолт).

    Args:
        query: Вопрос или фраза пользователя.

    Returns:
        RouteDecision с маршрутом, причиной, уверенностью и именем tool'а.
    """
    text = " ".join(query.lower().split())
    identifiers = _IDENTIFIER.findall(query)
    codes = _CODE.findall(query)
    has_exact = bool(identifiers or codes)

    lexical_hits = len(_LEXICAL.findall(text))
    semantic_hits = len(_SEMANTIC.findall(text))
    hybrid_hits = len(_HYBRID.findall(text))

    if hybrid_hits:
        return RouteDecision(
            Route.HYBRID,
            f"признаки требований/доработок ({hybrid_hits}); нужен гибридный поиск",
            0.85,
            _TOOL_BY_ROUTE[Route.HYBRID],
        )
    if has_exact and semantic_hits:
        names = ", ".join((identifiers + codes)[:2])
        return RouteDecision(
            Route.HYBRID,
            f"смешанный запрос: точное имя ({names}) + смысловой контекст",
            0.75,
            _TOOL_BY_ROUTE[Route.HYBRID],
        )
    if has_exact:
        names = ", ".join((identifiers + codes)[:3])
        return RouteDecision(
            Route.LEXICAL,
            f"точные имена/коды: {names}",
            round(min(0.95, 0.65 + 0.05 * len(identifiers + codes)), 2),
            _TOOL_BY_ROUTE[Route.LEXICAL],
        )
    if lexical_hits and semantic_hits:
        return RouteDecision(
            Route.HYBRID,
            f"смешанный запрос: термины ({lexical_hits}) и смысл ({semantic_hits})",
            0.7,
            _TOOL_BY_ROUTE[Route.HYBRID],
        )
    if lexical_hits:
        return RouteDecision(
            Route.LEXICAL,
            f"точные термины системы ({lexical_hits}) без смыслового вопроса",
            round(min(0.9, 0.6 + 0.05 * lexical_hits), 2),
            _TOOL_BY_ROUTE[Route.LEXICAL],
        )
    if semantic_hits:
        return RouteDecision(
            Route.VECTOR,
            f"смысловой/диагностический вопрос ({semantic_hits}); точных имён нет",
            round(min(0.9, 0.6 + 0.05 * semantic_hits), 2),
            _TOOL_BY_ROUTE[Route.VECTOR],
        )
    return RouteDecision(
        Route.HYBRID,
        "недостаточно признаков; по правилу «сомневаешься — hybrid»",
        0.4,
        _TOOL_BY_ROUTE[Route.HYBRID],
    )
