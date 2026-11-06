"""Тесты маршрутизатора: правила из промпта ассистента 1С."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from mcp_1c_rag.router import Route, RouteDecision, route_query  # noqa: E402

LEXICAL_CASES = [
    "Какие проводки у документа РеализацияТоваровУслуг?",
    "Кто пишет в регистр накопления НДС?",
    "Движения документа ПоступлениеТоваровУслуг",
    "На основании чего создаётся СчетФактураВыданный?",
    "Связанные документы с НФ-001",
    "Где обработчик ПередЗаписью у СчетФактураПолученный?",
    "Табличная часть Товары у РеализацияТоваровУслуг",
]

VECTOR_CASES = [
    "Почему не работает печать этикеток?",
    "Как настроить обмен с сайтом",
    "Что делать, если документ не проводится",
    "Как правильно вести учёт по УСН",
    "Инструкция для нового сотрудника",
    "Объясни, как устроен учёт НДС",
    "Опыт автоматизации розницы",
]

HYBRID_CASES = [
    "Требования к доработке документа Реализация по НДС",
    "Confluence: как правильно настроить обмен и движения",
    "Согласованный регламент проводок по НДС",
    "По ТЗ доработать ПечатьЭтикеток на складе",
]

DEFAULT_CASES = [
    "склад",
    "НДС",
    "",
]

_CALL_BY_ROUTE = {
    Route.LEXICAL: ("lexical", "metadata_xml"),
    Route.VECTOR: ("vector", "all"),
    Route.HYBRID: ("hybrid", "all"),
}


def _assert_search_call(decision: RouteDecision) -> None:
    assert decision.tool == "search"
    mode, collection = _CALL_BY_ROUTE[decision.route]
    assert decision.mode == mode
    assert decision.collection == collection


@pytest.mark.parametrize("query", LEXICAL_CASES)
def test_lexical_routes(query: str) -> None:
    decision = route_query(query)
    assert decision.route is Route.LEXICAL, (query, decision)
    _assert_search_call(decision)


@pytest.mark.parametrize("query", VECTOR_CASES)
def test_semantic_routes_to_vector(query: str) -> None:
    decision = route_query(query)
    assert decision.route is Route.VECTOR, (query, decision)
    _assert_search_call(decision)


@pytest.mark.parametrize("query", HYBRID_CASES)
def test_mixed_and_requirements_route_to_hybrid(query: str) -> None:
    decision = route_query(query)
    assert decision.route is Route.HYBRID, (query, decision)
    _assert_search_call(decision)


@pytest.mark.parametrize("query", DEFAULT_CASES)
def test_unknown_defaults_to_hybrid(query: str) -> None:
    decision = route_query(query)
    assert decision.route is Route.HYBRID, (query, decision)
    assert "сомневаешься" in decision.reason
    _assert_search_call(decision)


def test_decision_explains_itself() -> None:
    decision = route_query("Какие проводки у документа РеализацияТоваровУслуг?")
    assert decision.tool == "search"
    assert decision.mode == "lexical"
    assert decision.collection == "metadata_xml"
    assert "РеализацияТоваровУслуг" in decision.reason
    assert 0.0 < decision.confidence <= 1.0


def test_english_code_identifier_is_lexical() -> None:
    decision = route_query("где обрабатывается ClearBase в драйвере оборудования")
    assert decision.route is Route.LEXICAL
    _assert_search_call(decision)


def test_semantic_with_exact_name_is_hybrid() -> None:
    decision = route_query("Как сделать, чтобы документ РеализацияТоваровУслуг не падал?")
    assert decision.route is Route.HYBRID
    _assert_search_call(decision)


CASES_FILE = ROOT / "data" / "eval" / "routing_cases.jsonl"


def test_all_labelled_routing_cases() -> None:
    """Контракт целиком: базовые + заказчик + Part-2 + N1–N15 (см. routing_cases.jsonl)."""
    import json

    if not CASES_FILE.exists():
        pytest.skip("routing_cases.jsonl отсутствует: приватный набор не публикуется")

    rows = [
        json.loads(line)
        for line in CASES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) >= 65
    mismatches = []
    for row in rows:
        decision = route_query(row["query"])
        allowed = set(row["expected_any"]) if "expected_any" in row else {row["expected"]}
        if decision.route.value not in allowed:
            mismatches.append((row["query"], "|".join(sorted(allowed)), decision.route.value))
        else:
            _assert_search_call(decision)
    assert not mismatches, mismatches
