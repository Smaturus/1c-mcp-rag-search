# Changelog

Все заметные изменения проекта. Формат — [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
версии — [SemVer](https://semver.org/lang/ru/).

## [0.2.1] — 2026-10-21

### Fixed

- **Методика eval:** nDCG@10 больше не учитывает повторные чанки одного файла
  как отдельные попадания. Добавлен `unique_sources`, регрессионные тесты;
  метрики пересчитаны (hybrid nDCG@10 0.378 → 0.199). См. ADR 0004.
- Обновлены все опубликованные цифры в README, ADR, презентации и тексте
  выступления — после внешнего ревью.

### Added

- ADR 0004 (схлопывание дубликатов в метриках), 6 регрессионных тестов (39 всего).
- `scripts/diag/README.md`: диагностические скрипты явно помечены как
  исследовательские артефакты, не входящие в пайплайн.

### Changed

- Проект переименован: `mcp-1c-homework` → **`1c-mcp-rag-search`** (репозиторий,
  pyproject, имя MCP-сервера, ссылки). Import-пакет `mcp_1c_homework` сохранён
  для обратной совместимости.
- README переписан в продуктовом тоне: результаты, архитектура, решения,
  ограничения и roadmap вместо «учебных» формулировок.

## [0.2.0] — 2026-10-21

### Added

- Глобальный BM25-индекс (`bm25.py`: `Bm25Index`) по всем чанкам коллекции:
  инвертированный индекс, компактное хранение (npz + json), сборка `scripts/build_bm25.py`.
- Reciprocal Rank Fusion (`rrf.py`) и гибридный поиск `hybrid_search`.
- Cross-encoder reranker (`rerank.py`, flashrank + `ms-marco-MultiBERT-L-12`),
  режим `hybrid_rerank`.
- Единый MCP-tool `search(query, top_k, mode, collection)` с режимами
  `vector` / `hybrid` / `hybrid_rerank` / `pearson`.
- Eval-контур: `eval_metrics.py` (file-level метрики), `scripts/make_goldset.py`,
  `scripts/eval_pipeline.py`, отчёты `data/eval/report_pipeline.{md,json}`.
- Бенчмарк латентности `scripts/bench.py` → `data/eval/bench.md`.
- Оценка воспроизводимости: `scripts/eval_stability.py` (mean ± std по
  нескольким независимым прогонам eval).
- Документация: `docs/ARCHITECTURE.md`, ADR 0001–0003, обновлённый README
  (пайплайн, evaluation, Decision Log).
- Инженерная обвязка: `ruff`, `mypy`, `pytest` (33 теста), GitHub Actions CI,
  `LICENSE`, метаданные пакета.

### Changed

- Токенизация BM25: сохраняются аббревиатуры и дефисные коды 1С (`НФ-001`).
- `Candidate` вынесен в лёгкий `models.py`; тяжёлые импорты (Chroma,
  sentence-transformers, flashrank) стали ленивыми — unit-тесты и CI не требуют ML-стека.
- Дефолтный режим поиска — `hybrid` (по результатам замеров).

### Fixed

- Восстановлен индекс после повреждения HNSW (chroma#7090): пересборка из sqlite,
  защита `hnsw:sync_threshold`, бэкапы.
- Метрики eval переведены с чанков на файлы (см. ADR 0003).

## [0.1.0] — 2026-10-18

### Added

- Первый MCP-сервер: `codesearch`, `metadatasearch`, `evaluate_retrieval`.
- Индексация `.bsl` и отчёта конфигурации, Chroma + e5-small.
- Интерактивный `evaluate_retrieval` (Hit@k / Recall@k / MRR / null-tail).
- Взвешенный гибрид cosine + BM25 + Pearson (режим `pearson`).
