# Changelog

Все заметные изменения проекта. Формат — [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
версии — [SemVer](https://semver.org/lang/ru/).

## [Unreleased]

_(пусто)_

## [0.5.0] — 2026-10-23

### Added

- **Независимый hold-out v3** (`data/eval/gold_set_v3_holdout.jsonl`): 100 запросов
  из 23 доменов, file-level разметка с полями `scope` / `answerable` / `confidence`;
  71 code + 5 metadata + 24 out-of-scope (платформенные механизмы, регламентированная
  отчётность, РИБ, давальческое сырьё). Набор **не использовался для настройки**.
- 10 лексических якорей (`data/eval/anchors_v3.jsonl`) — короткие точные фразы,
  которые обязаны находить конкретный файл в top-3; сейчас 10/10.
- `scripts/validate_goldset_v3.py`, `scripts/eval_anchors.py`,
  `scripts/eval_router_v3.py`; в `eval_pipeline.py` — фильтры `--scope` и
  `--answerable-only`; отчёты `data/eval/gold_set_v3_report.md`,
  `data/eval/v3_report.md`, `data/eval/v3_router.md`, `data/eval/v3_anchors.md`.
- **ADR 0010: CamelCase-aware токенизация BM25.** Токенизатор разбивает
  CamelCase/snake_case, сохраняя исходный токен; глобальный BM25 пересобран
  (179 586 документов, 236 320 термов, 81 МБ).

### Changed

- **Исправлен дефект BM25-токенизатора** (ADR 0010): запрос «заказ покупателя»
  теперь совпадает с `ЗаказПокупателя`. Лексика на v2 выросла с 0.116 до 0.426
  nDCG@10, hybrid — с 0.301 до **0.442 ± 0.012**; vector 0.331 ± 0.028.
  Paired vector−hybrid: Δ = −0.112, CI [−0.184; −0.038], p = **0.0037**.
- v3 hold-out, 3 прогона: lexical 0.218, vector 0.208 ± 0.020, pearson
  0.229 ± 0.019, hybrid 0.217 ± 0.015; Δ(vector−hybrid) = −0.0097,
  CI [−0.049; +0.030], p = 0.635 — не значимо, hybrid не хуже vector.
- **ADR 0009 отклонён**: «сдвиг лидера hybrid→vector» был артефактом дефектного
  токенизатора, а не свойством процедурной нарезки. Дефолт `hybrid` подтверждён
  и не меняется.
- Роутер на v3: **58/100 (58 %)**, основная ошибка `lexical→hybrid` (21 случай);
  правила не распознают естественные имена в кавычках. На hold-out не тюнинговали.
- Metadata-scope v3 (5 запросов, 3 прогона): `hybrid` 0.767 ± 0.058,
  `lexical` 0.677, `vector` 0.533 ± 0.115 (индикативно, n=5).

## [0.4.0] — 2026-10-23

### Added

- **`scripts/eval_paired.py`** — парное сравнение режимов по per-query значениям:
  bootstrap 95 % CI, paired t-test, Wilcoxon только справочно (33/70 ничьих,
  отбрасывание ничьих занижает p). `--runs-glob` усредняет несколько прогонов,
  `--report` оставляет одиночный прогон. Отчёт — `data/eval/code_proc/report_paired.md`.
- **Ablation 2×2 «выгрузка × нарезка»** (`data/eval/ablation_2x2.md`): клетка
  «текущий экспорт × символьная нарезка» измерена заново (4 прогона:
  vector 0.165 ± 0.013, hybrid 0.205 ± 0.007); старая клетка — исторические
  агрегаты (0.176 / 0.209); клетка «старый экспорт × процедурная» **не измерима**
  (дамп утрачен). Вывод: сдвиг лидера объясняется **нарезкой** (эффект +0.174 / +0.096),
  эффект экспорта — в пределах шума; interaction не оценён.
- **English presentation kit** (`docs/presentation/en/`): 12-слайдовый deck (PPTX/PDF),
  речь на 10 минут (`SPEECH_EN.md`), one-pager (`ONE_PAGER_EN.pdf`),
  fallback-скриншоты демо и RU-шпаргалка (`NOTES_RU.md`).

### Changed

- **Корпус пересобран:** 179 586 чанков (154 560 процедурных BSL + 25 026 XML-метаданных);
  MCP переключён на `data/chroma_xml` + `data/bm25_all`, коллекции `code_proc` / `metadata_xml`.
  Метрики gold-set v2 (4 прогона): `vector` **0.339 ± 0.023**, `hybrid` 0.301 ± 0.017,
  `pearson` 0.337 ± 0.030; лидер сменился с hybrid на vector.
- **ADR 0009 — proposed:** дефолт `MCP_1C_SEARCH_MODE=hybrid` не меняется.
  Парный тест: Δ(vector−hybrid) = +0.037, 95 % CI [−0.015; +0.086], p = 0.158 —
  не значимо. Отчёт — `data/eval/code_proc/report_paired.md`.
- README, ARCHITECTURE и ADR 0009 обновлены под новый корпус и ablation.
  English README — отдельным шагом.
- Инженерные метрики: **72** unit-теста, mypy **22** модуля, complexity
  **143 блока / средняя 4.37 (A) / MI 23.0**; CI зелёный.

### Fixed

- Из git убраны локальные артефакты `data/bm25_all` (60 МБ), `data/pytest_tmp`
  и `.coverage`; шаблоны добавлены в `.gitignore`.
- README больше не описывает старый корпус (124 253 чанка, hybrid 0.209) как текущий.

## [0.3.0] — 2026-10-22

- **Метаданные из XML-выгрузки** (`parse_config_xml.py`, `build_metadata_index.py`, ADR 0008): 5 911 объектов, 25 026 чанков (объект/реквизит/ТЧ/измерение/форма/команда/значение), словарь 11 850 имён. A/B на 300 кейсах: BM25 Hit@3 93,0 % → **99,7 %**. HNSW sync threshold (`MCP_1C_HNSW_SYNC_THRESHOLD`) обходит chroma#7090.

### Added

- **Контракт маршрутизации** (`router.py`, ADR 0006): `route_query` и три
  MCP-инструмента `search_1c_metadata` (BM25 lexical), `search_knowledge_base`
  (vector), `search_hybrid`. Режим `search(mode="lexical")`. 24 размеченных
  routing-кейса — 100 %; end-to-end nDCG@10 не хуже фиксированного hybrid.
- **Chunking BSL по процедурам** (`chunking_procedures.py`, ADR 0007):
  путь и имя процедуры в тексте чанка, части для длинных процедур, модули без
  процедур — fallback. Смоук: покрытие кода 99,6 %, сверх лимита 512 токенов 0,3 %.
- `scripts/build_procedure_index.py` — сборка отдельной коллекции `code_proc`
  (отдельная база, чекпоинт, возобновление); `scripts/eval_chunking.py`,
  `scripts/smoke_procedure_chunking.py`.
- `scripts/check_embedding_tokens.py` — измерение лимита e5 на полном индексе:
  32,1 % чанков длиннее 512 токенов (метаданные — 78 %).
- `scripts/check_complexity.py` + CI-гейт сложности (radon) и покрытие с floor 25 %.
- `scripts/eval_router.py`, `data/eval/routing_cases.jsonl`.
- **Роутер, ревизия 2:** feature-based признаки (точное имя, реквизит, лексика,
  сильная семантика, диагностика, отрицание, маркеры требований), правила
  «точное имя + смысл/диагностика → hybrid», «отрицание отсутствия → hybrid»,
  «одиночное имя после реквизит/опция/поле → hybrid». Регрессионный набор
  68 кейсов — 100 %; blind Part-3 (20 кейсов, SHA-256 зафиксирован) до правок —
  75 %, после — 20/20 (переведён в dev). Закрыты провалы «вылетает ошибка»,
  «документ без движений», «не трогай X, объясни…». ADR 0006 обновлён.

### Changed

- README: разделы «Маршрутизация запросов», «Chunking по процедурам»,
  «Embedding model», «Quality gates»; позиционирование контактов —
  Technology Executive | Enterprise AI & RAG Architecture.
- `build_bm25.py`: параметр `--id-prefix` для отдельной коллекции.

## [0.2.2] — 2026-10-21

### Added

- **Gold-set v2** (`scripts/make_goldset_v2.py`): 70 запросов в естественных
  формулировках (30 курируемых доменных + 40 по реальным объектам конфигурации).
  Отчёты: `data/eval/v2/`.
- ADR 0005: natural-набор, сравнение с v1 и ограничения авточасти.

### Changed

- Абсолютные метрики выросли (vector nDCG@10 0.143 → 0.176), а разрыв hybrid над
  vector честно сократился: +39 % → **+19 %** (0.209 против 0.176). Выводы о
  выборе hybrid и о проигрыше reranker не изменились.
- README, презентация и PDF обновлены под gold-set v2.

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

- Проект переименован: `mcp-1c-homework` → **`1c-mcp-rag-search`**. Изменены
  репозиторий, distribution и import-пакет (`mcp_1c_homework` → **`mcp_1c_rag`**),
  console-скрипты (`1c-rag-search`, `1c-rag-build-index`), имя MCP-сервера,
  все импорты, документация и ссылки.
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
