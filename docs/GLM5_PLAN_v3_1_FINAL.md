# План v3.1: алгоритмические улучшения retrieval (GLM5, доведённый)

- **Версия:** 3.1 (final после ревью v3.0)
- **Дата:** 2026-11-06
- **Статус:** готов к выполнению с Task 1 после отдельного «да». Пока ничего не запускалось.
- **Основание:** фаза B INCONCLUSIVE, measurement stability из `algo_baseline.md`.
- **Не обещает** прирост nDCG: цель — стабильный harness, покрытие источников, отсутствие регрессии.

## 0. Факты и baseline

| Набор | n | hybrid nDCG@10 | vector nDCG@10 | δ |
|---|---|---|---|---|
| v2 dev | 70 | 0.4209–0.4396 | 0.3247–0.3284 | 0.028 |
| v3_code hold-out | 64 | 0.2641 | 0.2564 | 0.020 |
| v4_code hold-out | 179 | 0.5562 | 0.4383 | 0.020 |

- Полоса межсессионного шума: **±0.02 nDCG@10** (HNSW-пересборка).
- B (BM25 code-only): **INCONCLUSIVE** — `algo_b_bm25_code.md`; B2 не начинать.
- Источники baseline: `algo_baseline.md`, `live_hybrid_confirm.md`, `state.json`.

## 1. Цели и границы

**Цели:** same-process harness; рост `unique_sources@10`; non-inferiority по nDCG/recall.
**Не цели:** прирост nDCG; смена embedding-модели; новая gold-разметка; BM25 code-only (закрыт).

**Защищать:** `data/chroma_xml/`, `data/bm25_all/`, `data/bm25_code_proc_live/` (не пересобирать), gold-sets.
**Можно писать:** `data/local_notes/algo_glm5/`, `data/eval/_runs_algo_glm5/`, `data/lexicon/`.
**Live MCP/yml/src/scripts:** не менять до PASS и отдельного «да».

## 2. Task 1 — same-process regression harness

### 2.1. Конструкция

1. Загрузить модель один раз (`embed_query`), Chroma-клиент один раз.
2. Загрузить два BM25-индекса: `Bm25Index.load(baseline_dir)` и `Bm25Index.load(candidate_dir)`.
3. Для каждого запроса:
   - один раз посчитать `qvec = embed_query(query)`;
   - один раз получить vector top-N из Chroma (`collection.query(query_embeddings=[qvec], n_results=pool, include=["documents","metadatas","distances"])`);
   - собрать `vector_candidates: list[Candidate]`;
   - для каждого BM25-индекса: `scores = index.scores(query)`; применить prefix-mask (`bsl_proc:` для `code_proc`); взять top-N id; получить документы/метаданные для id, которых нет среди vector-кандидатов (`collection.get(ids=..., include=["documents","metadatas"])`);
   - `fused = rrf_fuse([vector_ids, bm25_ids], k=RRF_K)`; `order = rrf_order(fused)`;
   - построить `Candidate` с итоговым `score=fused[id]`;
   - применить dedup (если задан `max_per_source`);
   - взять top-k.
4. Для каждой метрики отдельно накапливать `baseline_values[metric]` и `candidate_values[metric]`.
5. Paired bootstrap 95 % CI (10 000, seed 42) и paired t-test по per-query значениям; W/T/L.

Важно: **не использовать** `search_modes.search`/`bm25_search` в harness — они читают модульные глобалы `BM25_DIR`/`_bm25` и не позволят переключать baseline/candidate в одном процессе. Harness работает с `Bm25Index`, Chroma и `rrf_fuse` напрямую.

### 2.2. CLI (единый для Task 1/3/5)

```text
python scripts/eval_regression_harness.py \
  --golden <path> \
  --chroma-dir <path> --collection code_proc \
  [--scope code] [--embedding-model intfloat/multilingual-e5-small] \
  --baseline-bm25-dir <path> --candidate-bm25-dir <path> \
  [--baseline-max-per-source 0] [--candidate-max-per-source 0] \
  [--metrics ndcg@10,recall@10,unique_sources@10] [--k 10] \
  [--delta-floor <float>] --out-json <path>
```

- `--delta-floor` опционален; иначе δ по имени gold-файла: v2→0.028, v3/v4→0.020; неизвестный файл → ошибка.
- Все обязательные аргументы обязательны; никаких `--baseline-args`/`--candidate-args`.
- `--scope` фильтрует строки gold по полю `scope`. В **v2 поля `scope` нет** (есть `type`), поэтому для v2 `--scope` не применяется — всегда все 70; для v3/v4 `--scope code` даёт 64/179.

### 2.3. Гейты Task 1

| Проверка | Критерий |
|---|---|
| sanity baseline vs baseline | Δ ≈ 0, CI ∋ 0, status PASS |
| vector sanity | vector-метрики baseline/candidate совпадают (Δ=0, CI=[0,0]) |
| per-metric arrays | Δ/CI считаются отдельно по каждой метрике |
| scope | `--scope code` даёт 64 запроса для v3, 179 для v4 |

### 2.4. Тесты harness

- `test_baseline_vs_baseline_pass`: одинаковые dirs/max → `status=pass`, `|Δ|<0.001`, CI ∋ 0.
- `test_delta_floor`: v2→0.028, v3→0.020, v4→0.020.
- `test_scope_code_v3`: 64.
- `test_metrics_separate`: две метрики не смешиваются (важно: в v3.0 был баг).

Оценка: 4–5 ч.

## 3. Task 2 — source dedup + iterative backfill

### 3.1. Семантика (исправленная)

- `dedup_by_source(candidates, max_per_source)` — **только cap**, возвращает ≤ len(candidates), может быть < top_k.
- `search(...)` с `max_per_source>0`:
  1. запросить пул `top_k * pool_multiplier` (например, 100);
  2. `capped = dedup_by_source(pool, max_per_source)`;
  3. если `len(capped) < top_k` — **увеличить пул** (×2) и повторить, пока не наберётся top_k или пул не покроет всех кандидатов (`top_k` не может превысить число документов);
  4. если уникальных source всё равно меньше top_k — вернуть сколько есть; **нельзя** молча добирать дубли, ломая diversity.
- Возврат: ровно `min(top_k, доступно)`; тесты проверяют оба случая.

### 3.2. API и интеграция

```python
def search(query, top_k=10, mode=SEARCH_DEFAULT_MODE, collection="code",
           max_per_source=MAX_PER_SOURCE, pool_multiplier=DEDUP_START_MULTIPLIER) -> list[Candidate]:
    if max_per_source <= 0:
        return _search_impl(query, top_k, mode, collection)
    total = collection_count(collection)  # или collection.count() из Chroma
    pool = min(top_k * pool_multiplier, total)
    while True:
        candidates = _search_impl(query, pool, mode, collection)
        capped = dedup_by_source(candidates, max_per_source)
        if len(capped) >= top_k or len(candidates) < pool or pool >= total:
            return capped[:top_k]
        pool = min(pool * 2, total)
```

- `_search_impl` — текущее тело `search()` в `src/mcp_1c_rag/search_modes.py` (строка 305+), вынести без изменения поведения. `server.py` остаётся вызывающей стороной и вызывает `search()`.
- `len(candidates) < pool` означает, что пул исчерпан; `pool >= total` — дошли до всех документов. Отдельная `_max_available` не нужна.
- `Candidate.source` — property; использовать `c.source`, не `c["source"]`.
- `config.py`: `MAX_PER_SOURCE = int(os.getenv("MCP_1C_MAX_PER_SOURCE", "2"))`; **default 2**, `MCP_1C_MAX_PER_SOURCE=0` отключает dedup.
- `MAX_PER_SOURCE > 0` пробрасывать в MCP `search` и smoke.

### 3.3. Тесты (исправленные)

```python
def make(id, source, score):
    return Candidate(id=id, metadata={"source": source}, score=score)

def test_dedup_cap_only():
    cands = [make(f"c{i}", "file1", 1.0 - i * 0.01) for i in range(7)] + \
            [make(f"d{i}", "file2", 0.9 - i * 0.01) for i in range(3)]
    capped = dedup_by_source(cands, max_per_source=2)
    assert len(capped) == 4
    assert sum(c.source == "file1" for c in capped) == 2

def test_search_backfills_to_top_k():
    # Мок _search_impl: 20 кандидатов, 10 разных source, top_k=10
    # -> ровно 10, без дублей сверх max_per_source
    ...

def test_cap_preserved_in_backfill():
    # 3 source, много чанков, max_per_source=2, top_k=10
    # -> len<=10, ни у одного source не больше 2
    ...

def test_search_returns_top_k_with_enough_sources():
    # 20 source по 1 кандидату -> search(top_k=10, max_per_source=2) == 10
```

Оценка: 4–5 ч.

## 4. Task 3 — парный eval dedup

Команды (все три набора):

```text
python scripts/eval_regression_harness.py --golden data/eval/gold_set_v2.jsonl \
  --chroma-dir data/chroma_xml --collection code_proc \
  --baseline-bm25-dir data/bm25_all --candidate-bm25-dir data/bm25_all \
  --baseline-max-per-source 0 --candidate-max-per-source 2 \
  --metrics ndcg@10,recall@10,unique_sources@10 \
  --out-json data/local_notes/algo_glm5/dedup_v2.json
```

Аналогично v3 (`--scope code`) и v4 (`--scope code`).

Гейты:

| Метрика | Критерий |
|---|---|
| `unique_sources@10` | candidate mean > baseline mean (ворота) |
| `nDCG@10` | CI_lower > −δ |
| `Recall@10` | CI_lower > −δ |
| Latency | candidate ≤ baseline × 2 (в harness, без cold start) |

Если backfill не даёт роста unique_sources — это FAIL, default остаётся 0.

Оценка: 1 ч.

## 5. Task 4 — query expansion (опционально)

- Словарь: `scripts/build_identifier_lexicon.py --chroma-dir data/chroma_xml --out data/lexicon/identifiers.json` (из metadata-карточек, автоматически).
- Расширение **только на стороне запроса**: в `Bm25Index.scores` добавить опцию `query_tokens` или отдельный `scores_expanded`; **не** менять `tokenize` для документов при сборке индекса.
- Флаг `MCP_1C_QUERY_EXPANSION=0` (default off).
- Гейты: exact-срез hit@3 улучшается **и** общий nDCG CI_lower > −δ.
- Тесты: camelCase-разбиение, dictionary lookup, проверка, что документы не расширяются.

Оценка: 4–6 ч.

## 6. Task 5 — итоговый eval и решение

- Итоговый прогон harness по трём наборам с финальной конфигурацией (все аргументы явно, как в Task 3).
- `final_decision.md`: что проверено, что принято/отклонено, δ, CI, latency, ссылки на JSON.
- Кодовые изменения только при PASS: `MAX_PER_SOURCE` **default 2** (`MCP_1C_MAX_PER_SOURCE=0` отключает); `QUERY_EXPANSION` default 0; глобальный `MCP_1C_BM25_DIR` не менять.
- Критерии релиза: unique_sources растёт; nDCG/recall CI_lower > −δ на всех наборах; latency ≤150 мс; live-смоук после отдельного «да».

Оценка: 1–2 ч.

## 7. Порядок и оценки

| Task | Цель | Время |
|---|---|---|
| 1 | harness | 4–5 ч |
| 2 | dedup + backfill | 4–5 ч |
| 3 | eval dedup | 1 ч |
| 4 | expansion (опц.) | 4–6 ч |
| 5 | final decision | 1–2 ч |
| **Итого** | | **14–19 ч (2–3 дня)** |

Порядок: Task 1 → 2 → 3 → 5; Task 4 — после 1–3, опционально.

## 8. Definition of Done

**DoD-min (1–3):** harness PASS (sanity, vector, per-metric); dedup с cap/backfill, `len<=top_k` и diversity не ломается; unique_sources растёт; nDCG/recall CI_lower > −δ.
**DoD-full (1–5):** DoD-min; expansion проверен (если запускался); `final_decision.md`; README/ADR при изменении дефолтов.

## 9. Риски

| Риск | Митигация |
|---|---|
| Harness не повторяет реальный поиск | vector-кандидаты один раз, BM25/ RRF воспроизводят `hybrid_search`; smoke-проверка на одном запросе |
| Dedup ухудшает recall | backfill без нарушения cap; до eval default=0, после Task 5 — **default=2** (`MCP_1C_MAX_PER_SOURCE=0` отключает) |
| CI ∋ 0 ≠ нет ухудшения | только non-inferiority `CI_lower > −δ` |
| Expansion добавляет шум | два гейта: exact + overall |
| Latency | ≤150 мс и ≤2× baseline в harness |

## 10. Отличия от v3.0

| v3.0 | v3.1 |
|---|---|
| `get_embedder`, `vector_search_cached`, `hybrid_from_vector` не существуют | явно: `embed_query`, Chroma query, `Bm25Index`, `rrf_fuse` |
| Метрики смешивались в один список | отдельные массивы по каждой метрике |
| dedup-тест ожидал 4 при top_k=10 | разделены cap-only и search-backfill; тесты согласованы |
| `Candidate(source=...)` | `Candidate(metadata={"source": ...})` |
| backfill мог вернуть <top_k / ломал cap | iterative deepening, cap сохраняется |
| `_search_impl` не определён | явно: вынести текущее тело `search()` |
| expansion мог попасть в документы | только query-side |
| CLI местами расходился | единый интерфейс Task 1/3/5 |

## 11. Старт

После «да на Task 1»: реализовать harness + тесты, прогнать sanity (baseline vs baseline, vector sanity), записать `harness_v2.json`. Task 2 — только после PASS Task 1 и отдельного «да».
