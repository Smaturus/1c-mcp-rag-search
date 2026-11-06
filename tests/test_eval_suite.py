"""Тесты eval_suite без ML: expected/regression/partial."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_suite  # noqa: E402
from mcp_1c_rag.models import Candidate  # noqa: E402


def _cand(source: str) -> Candidate:
    return Candidate(id=source, text="t", metadata={"source": source}, score=1.0)


def test_compare_ndcg_pass_and_fail() -> None:
    ok, diff = eval_suite.compare_ndcg(0.45, 0.44, 0.02)
    assert ok and diff == pytest.approx(0.01)
    bad, diff2 = eval_suite.compare_ndcg(0.40, 0.44, 0.02)
    assert not bad and diff2 == pytest.approx(-0.04)


def test_run_set_with_fake_searcher() -> None:
    rows = [
        {
            "id": "1",
            "query": "q1",
            "relevant_sources": ["Doc/A.bsl"],
        },
        {
            "id": "2",
            "query": "q2",
            "relevant_sources": ["Doc/B.bsl"],
        },
    ]

    def searcher(query: str, *, top_k: int, mode: str, collection: str) -> list[Candidate]:
        if query == "q1":
            return [_cand("Doc/A.bsl")]
        return [_cand("other.bsl")]

    out = eval_suite.run_set(rows, searcher=searcher, k=10)
    assert out["n_queries"] == 2
    assert out["metrics"]["hit@1"] == pytest.approx(0.5)
    assert out["metrics"]["ndcg@10"] > 0


def test_main_regression_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = tmp_path / "gold_set_v2.jsonl"
    gold.write_text(
        json.dumps(
            {
                "id": "1",
                "query": "q",
                "relevant_sources": ["hit.bsl"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(eval_suite.GOLD_FILES, "v2", gold)

    def searcher(query: str, *, top_k: int, mode: str, collection: str) -> list[Candidate]:
        return [_cand("miss.bsl")]

    monkeypatch.setattr(eval_suite, "default_searcher", searcher)
    monkeypatch.setattr(
        eval_suite,
        "load_manifest",
        lambda _p: type("M", (), {})(),
    )
    monkeypatch.setattr(eval_suite, "validate_manifest", lambda *_a, **_k: [])

    expected = tmp_path / "expected.json"
    expected.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sets": {"v2": {"ndcg@10": 0.9, "delta": 0.01}},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    (tmp_path / "man.json").write_text("{}", encoding="utf-8")
    code = eval_suite.main(
        [
            "--manifest",
            str(tmp_path / "man.json"),
            "--expected",
            str(expected),
            "--out",
            str(out),
            "--sets",
            "v2",
        ]
    )
    assert code == 1
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert report["sets"]["v2"]["status"] == "regression"


def test_main_pass_and_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = tmp_path / "gold_set_v2.jsonl"
    gold.write_text(
        json.dumps(
            {"id": "1", "query": "q", "relevant_sources": ["hit.bsl"]}
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(eval_suite.GOLD_FILES, "v2", gold)
    monkeypatch.setattr(
        eval_suite,
        "default_searcher",
        lambda query, *, top_k, mode, collection: [_cand("hit.bsl")],
    )
    monkeypatch.setattr(eval_suite, "validate_manifest", lambda *_a, **_k: [])
    man = tmp_path / "man.json"
    man.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        eval_suite, "load_manifest", lambda _p: type("M", (), {})()
    )
    expected = tmp_path / "expected.json"
    out = tmp_path / "out"
    code = eval_suite.main(
        [
            "--manifest",
            str(man),
            "--expected",
            str(expected),
            "--out",
            str(out),
            "--sets",
            "v2",
            "--record",
        ]
    )
    assert code == 0
    assert expected.is_file()
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["status"] in {"ok", "partial"}
    assert report["sets"]["v2"]["status"] == "ok"


def test_missing_gold_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(eval_suite.GOLD_FILES, "v2", tmp_path / "nope.jsonl")
    man = tmp_path / "man.json"
    man.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(eval_suite, "load_manifest", lambda _p: type("M", (), {})())
    monkeypatch.setattr(eval_suite, "validate_manifest", lambda *_a, **_k: [])
    out = tmp_path / "out"
    code = eval_suite.main(
        [
            "--manifest",
            str(man),
            "--expected",
            str(tmp_path / "exp.json"),
            "--out",
            str(out),
            "--sets",
            "v2",
        ]
    )
    assert code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "partial"
    assert report["sets"]["v2"]["status"] == "skipped"


def test_manifest_mismatch_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp_1c_rag.startup import ValidationIssue

    man = tmp_path / "man.json"
    man.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(eval_suite, "load_manifest", lambda _p: type("M", (), {})())
    monkeypatch.setattr(
        eval_suite,
        "validate_manifest",
        lambda *_a, **_k: [
            ValidationIssue(
                level="error",
                code="manifest_count_mismatch",
                message="count",
                actual="x",
            )
        ],
    )
    code = eval_suite.main(
        [
            "--manifest",
            str(man),
            "--expected",
            str(tmp_path / "e.json"),
            "--out",
            str(tmp_path / "o"),
            "--sets",
            "v2",
        ]
    )
    assert code == 2
