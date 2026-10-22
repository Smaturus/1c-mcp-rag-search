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
# ClearBase. Требуем внутреннюю заглавную букву (CamelCase), чтобы обычные слова
# с заглавной буквы в начале предложения не считались идентификаторами.
_IDENTIFIER = re.compile(
    r"(?<![A-Za-zА-Яа-я0-9])[A-ZА-Я][a-zа-я0-9]+[A-ZА-Я][A-Za-zА-Яа-я0-9]*"
)
_CODE = re.compile(r"\b[A-ZА-Я]{2,}-\d{2,}\b")

_LEXICAL_PHRASES = (
    "на основании",
    "движени",
    "проводк",
    "кто пишет",
    "пишет в регистр",
    "связанные документы",
    "табличная часть",
    "обработчик",
    "перед записью",
    "при записи",
    "регистр сведений",
    "регистр накопления",
    "движения документа",
)

_SEMANTIC_PHRASES = (
    "как сделать",
    "как настроить",
    "как работает",
    "как правильно",
    "почему не работает",
    "почему не проводится",
    "почему не получается",
    "инструкция",
    "форум",
    "опыт",
    "что делать",
    "что такое",
    "объясни",
    "лучшие практики",
)

_HYBRID_MARKERS = (
    "требовани",
    "confluence",
    "конфлюенс",
    "доработк",
    "согласова",
    "по тз",
    "регламент",
    "стандарт",
)


def _count_phrases(text: str, phrases: tuple[str, ...]) -> int:
    return sum(1 for phrase in phrases if phrase in text)


def route_query(query: str) -> RouteDecision:
    """Выбрать маршрут поиска по правилам из промпта ассистента.

    Args:
        query: Вопрос или фраза пользователя.

    Returns:
        RouteDecision с маршрутом, причиной, уверенностью и именем tool'а.
    """
    text = " ".join(query.lower().split())
    identifiers = _IDENTIFIER.findall(query)
    codes = _CODE.findall(query)
    lexical = _count_phrases(text, _LEXICAL_PHRASES) + len(identifiers) + len(codes)
    semantic = _count_phrases(text, _SEMANTIC_PHRASES)
    hybrid = _count_phrases(text, _HYBRID_MARKERS)

    if hybrid:
        return RouteDecision(
            Route.HYBRID,
            f"признаки требований/регламентов ({hybrid}); нужен гибридный поиск",
            0.85,
            _TOOL_BY_ROUTE[Route.HYBRID],
        )
    if lexical and semantic:
        return RouteDecision(
            Route.HYBRID,
            f"смешанный запрос: точные признаки ({lexical}) и смысловые ({semantic})",
            0.7,
            _TOOL_BY_ROUTE[Route.HYBRID],
        )
    if lexical:
        return RouteDecision(
            Route.LEXICAL,
            f"точные имена/связи: {', '.join((identifiers + codes)[:3]) or 'термины'}",
            round(min(0.95, 0.6 + 0.1 * lexical), 2),
            _TOOL_BY_ROUTE[Route.LEXICAL],
        )
    if semantic:
        return RouteDecision(
            Route.VECTOR,
            f"смысловой вопрос ({semantic}); лексика не помогает",
            round(min(0.95, 0.6 + 0.1 * semantic), 2),
            _TOOL_BY_ROUTE[Route.VECTOR],
        )
    return RouteDecision(
        Route.HYBRID,
        "недостаточно признаков; по правилу «сомневаешься — hybrid»",
        0.4,
        _TOOL_BY_ROUTE[Route.HYBRID],
    )
