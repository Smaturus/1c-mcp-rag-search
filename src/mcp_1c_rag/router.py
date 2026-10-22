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

Валидация: 24 базовых кейса + 15 кейсов заказчика + 17 кейсов Part-2 +
15 регресс-кейсов N1–N15 (``data/eval/routing_cases.jsonl``). Blind-набор
Part-3 (20 кейсов, ``routing_cases_part3_blind.jsonl``) прогнан однократно
до правок: 15/20 = 75 %; после правок он используется только как dev-набор.
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
# ИспользоватьОбменВС. Требуем внутреннюю заглавную букву (CamelCase), чтобы
# обычные слова с заглавной буквы в начале предложения не считались именами.
_IDENTIFIER = re.compile(
    r"(?<![A-Za-zА-Яа-я0-9])[A-ZА-Я][a-zа-я0-9]+[A-ZА-Я][A-Za-zА-Яа-я0-9]*"
)
_CODE = re.compile(r"\b[A-ZА-Я]{2,}-\d{2,}\b")

# Одиночное имя реквизита/опции после слова-классификатора: «реквизит Инициатор».
_REQUISITE_NAME = re.compile(
    r"\b(?:реквизит|опция|поле|свойство|атрибут)\s+[А-ЯЁ][а-яё]{3,}\b",
    re.IGNORECASE,
)

# Лексические признаки: точные термины системы.
_LEXICAL = re.compile(
    r"\b(?:проведени|движени|проводк|остатк|"
    r"регистр(?:а|ы|ов|ам|ах|е|ом|у|ой)?\b|хранится|"
    r"функциональн\w*\s+опци|табличн\w*\s+част|обработчик|"
    r"на\s+основании|связанн|кто\s+пишет|перед\s+записью|при\s+записи|"
    r"где\s+посмотреть)",
    re.IGNORECASE,
)

# Сильные смысловые признаки: как сделать/почему/инструкция/объяснение.
_STRONG_SEMANTIC = re.compile(
    r"\b(?:почему|зачем|инструкци|что\s+делать|что\s+такое|объясни|"
    r"лучшие\s+практики|опыт|форум|не\s+работает|не\s+проводил|не\s+проводит|"
    r"слетает|обновлени|разницу|"
    r"как\s+(?:сделать|настро|правильно|исправить|проверить|работ|получить|лечить)|"
    r"что\s+проверить|как\s+исправить|что\s+это\s+значит)",
    re.IGNORECASE,
)

# Диагностика: симптом/ошибка без явного «почему» — повод для hybrid при наличии
# точного имени или как фолбэк, когда нет сильной семантики.
_DIAGNOSTIC = re.compile(
    r"(?:\bвылезает\b|\bвылетает\b|\bпадает\b|\bзависает\b|"
    r"\bне\s+срабатывает\b|\bне\s+создаются\b|\bне\s+создаётся\b|"
    r"\bне\s+сходятся\b|\bне\s+уменьшаются\b|\bне\s+вызывается\b|"
    r"\bразберись\b|\bпричина\b|"
    r"\bошибка\b[^.!?]*\bмодул\w*|\bмодул\w*[^.!?]*\bне\s+вызывается\b)",
    re.IGNORECASE,
)

# Отрицание отсутствия сущности: «документ без движений/проводок».
_NEGATION = re.compile(
    r"\bбез\s+(?:движени|провод|проведени|регистр|остатк)\w*",
    re.IGNORECASE,
)

# Явный отказ от сущности: «не трогай X, объясни…» — имя не должно перевешивать.
_IGNORE_ENTITY = re.compile(r"\b(?:не\s+трогай|не\s+надо|игнорируй)\b", re.IGNORECASE)

# Признаки требований/доработок/Confluence и механизма «задним числом».
_HYBRID = re.compile(
    r"(?:требовани|confluence|конфлюенс|доработк|согласова|по\s+тз\b|регламент|"
    r"стандарт|добавить\s+реквизит|изменить\s+реквизит|"
    r"(?:не\s+проводил|запрет\w*\s+изменени\w*)[^.!?]*задним\s+числом|"
    r"задним\s+числом[^.!?]*(?:не\s+проводил|запрет\w*\s+изменени\w*))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Features:
    """Признаки запроса — решение принимается по ним, а не по сырому тексту."""

    has_exact: bool
    exact_names: tuple[str, ...]
    has_requisite_name: bool
    lexical_hits: int
    strong_semantic: bool
    diagnostic: bool
    negation: bool
    ignores_entity: bool
    hybrid_marker: bool


def _extract_features(query: str) -> _Features:
    text = " ".join(query.lower().split())
    identifiers = _IDENTIFIER.findall(query)
    codes = _CODE.findall(query)
    ignores_entity = bool(_IGNORE_ENTITY.search(text))
    # «Не трогай X, объясни…» — имя упомянуто в отрицании и не является целью поиска.
    exact_names = tuple(identifiers + codes) if not ignores_entity else ()
    return _Features(
        has_exact=bool(exact_names),
        exact_names=exact_names,
        has_requisite_name=bool(_REQUISITE_NAME.search(query)),
        lexical_hits=len(_LEXICAL.findall(text)),
        strong_semantic=bool(_STRONG_SEMANTIC.search(text)),
        diagnostic=bool(_DIAGNOSTIC.search(text)),
        negation=bool(_NEGATION.search(text)),
        ignores_entity=ignores_entity,
        hybrid_marker=bool(_HYBRID.search(text)),
    )


def _names(features: _Features) -> str:
    return ", ".join(features.exact_names[:3])


def _decide(features: _Features) -> RouteDecision:
    """Каскад решений; порядок и пороги обоснованы в ADR 0006."""
    if features.hybrid_marker:
        return RouteDecision(
            Route.HYBRID,
            "признаки требований/доработок или механизма; нужен гибридный поиск",
            0.85,
            _TOOL_BY_ROUTE[Route.HYBRID],
        )
    if features.negation:
        return RouteDecision(
            Route.HYBRID,
            "вопрос об отсутствии сущности («без движений/проводок») — нужны оба сигнала",
            0.75,
            _TOOL_BY_ROUTE[Route.HYBRID],
        )
    if features.has_exact and (features.diagnostic or features.strong_semantic):
        return RouteDecision(
            Route.HYBRID,
            f"точное имя ({_names(features)}) + смысловая/диагностическая часть",
            0.75,
            _TOOL_BY_ROUTE[Route.HYBRID],
        )
    if features.has_exact:
        return RouteDecision(
            Route.LEXICAL,
            f"точные имена/коды: {_names(features)}",
            round(min(0.95, 0.65 + 0.05 * len(features.exact_names)), 2),
            _TOOL_BY_ROUTE[Route.LEXICAL],
        )
    if features.has_requisite_name:
        return RouteDecision(
            Route.HYBRID,
            "название реквизита/опции без строгого CamelCase — нужны BM25 и вектор",
            0.65,
            _TOOL_BY_ROUTE[Route.HYBRID],
        )
    if features.diagnostic and not features.strong_semantic:
        return RouteDecision(
            Route.HYBRID,
            "диагностическая формулировка без сильного смыслового маркера",
            0.65,
            _TOOL_BY_ROUTE[Route.HYBRID],
        )
    if features.strong_semantic:
        if features.lexical_hits >= 2:
            return RouteDecision(
                Route.HYBRID,
                f"смешанный запрос: термины ({features.lexical_hits}) и смысл",
                0.7,
                _TOOL_BY_ROUTE[Route.HYBRID],
            )
        return RouteDecision(
            Route.VECTOR,
            "смысловой/диагностический вопрос без точных имён",
            0.8,
            _TOOL_BY_ROUTE[Route.VECTOR],
        )
    if features.lexical_hits:
        return RouteDecision(
            Route.LEXICAL,
            f"точные термины системы ({features.lexical_hits}) без смыслового вопроса",
            round(min(0.9, 0.6 + 0.05 * features.lexical_hits), 2),
            _TOOL_BY_ROUTE[Route.LEXICAL],
        )
    return RouteDecision(
        Route.HYBRID,
        "недостаточно признаков; по правилу «сомневаешься — hybrid»",
        0.4,
        _TOOL_BY_ROUTE[Route.HYBRID],
    )


def route_query(query: str) -> RouteDecision:
    """Выбрать маршрут поиска по правилам из промпта ассистента.

    Args:
        query: Вопрос или фраза пользователя.

    Returns:
        RouteDecision с маршрутом, причиной, уверенностью и именем tool'а.
    """
    return _decide(_extract_features(query))
