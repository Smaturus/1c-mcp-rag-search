# ADR 0011: Нормализация скоров при merge коллекций

- **Статус:** принято
- **Дата:** 2026-11-06
- **Контекст:** `search(collection="all")` объединяет `code_proc` и
  `metadata_xml`. Сырые скоры разных коллекций несравнимы по масштабу.
  Исторически в `server._merged_search` использовалась эвристика
  `score / max_score` внутри коллекции без замера и без ADR. Presentation
  (`format_candidates`) и merge жили в чужих модулях.
- **Варианты:**
  1. **max** — `score / max_score` по коллекции (текущее поведение);
  2. **rrf** — Reciprocal Rank Fusion по рангам внутри каждой коллекции
     (`k=RRF_K`);
  3. **minmax** — min-max нормализация скоров внутри коллекции.
- **Замер:** `data/local_notes/merge_bench.py`, mode=`hybrid`, pool top_k=20,
  eval k=10, paired bootstrap n=10 000, seed=42. Δ = alt − max по nDCG@10.

  | set | n | max nDCG@10 | rrf Δ [CI] p | minmax Δ [CI] p |
  |---|---|---|---|---|
  | v2 | 70 | 0.357 | −0.004 [−0.019; +0.011] p=0.59 | −0.003 [−0.013; +0.003] p=0.50 |
  | v3 hold-out | 100 | 0.145 | −0.008 [−0.019; +0.003] p=0.14 | +0.001 [−0.001; +0.003] p=0.79 |
  | v4 hold-out | 185 | 0.494 | **+0.013 [+0.002; +0.024] p=0.023** | +0.003 [−0.002; +0.007] p=0.26 |

  Metadata-scope в v3: n=5 — недостаточно для отдельного вывода.
  Латентность merge пренебрежимо мала относительно fetch двух коллекций
  (~120–510 мс/запрос на CPU).
- **Решение:** default `normalization="max"` сохраняется. RRF значимо лучше
  только на v4; на v2/v3 не лучше (CI пересекает 0 или Δ ≤ 0). Minmax нигде
  не значим. Смена дефолта потребовала бы стабильного выигрыша на всех
  крупных наборах. Реализация вынесена в `merge.merged_search` /
  `formatting.format_candidates`; параметр `normalization` доступен для
  экспериментов без смены tool surface.
- **Последствия / долг:** при росте metadata-gold или смене корпуса —
  переснять `merge_bench` и при необходимости пересмотреть default
  (кандидат — `rrf`). Tool surface / router / MAX_PER_SOURCE не менялись.
- **Проверка:** `pytest` (formatting/merge/server_tools),
  `data/local_notes/merge_bench.md` + `merge_bench.json`.
