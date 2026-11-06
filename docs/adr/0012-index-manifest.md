# ADR 0012: Индекс-манифест и границы startup/eval проверок

- **Статус:** принято
- **Дата:** 2026-11-06
- **Контекст:** live-индекс (`chroma_xml` + `bm25_all`) не имел машиночитаемого
  описания модели/корпуса/checksum. Startup проверял пути и counts через Chroma
  client, но не фиксировал «какой именно индекс» ожидается. Полный
  fingerprint по `embedding_id` на каждый старт MCP дорог (сотни тысяч id).
- **Решение:**
  1. Манифест JSON (`data/manifests/chroma_xml.json`, gitignored) — schema,
     embedding_model, counts + fingerprint коллекций, BM25 n_docs/k1/b +
     fingerprint, chunker/hnsw, версии пакетов.
  2. **Startup (быстрый путь):** если файл есть — сверка `embedding_model`,
     counts коллекций (sqlite COUNT), BM25 `n_docs`/`k1`/`b`. Fingerprint
     **не** пересчитывается. Нет файла → warning (совместимость).
  3. **build_manifest / eval_suite:** полный fingerprint; eval_suite до
     прогона сверяет быстрый путь (опционально `--check-fingerprints`).
  4. Ожидаемые nDCG@10 — локальный `eval_expected.json` (не в git); gold
     приватные.
- **Почему не fingerprint на старте:** сортировка/хэш ~180k id замедляет
  cold-start и дублирует работу collect; counts+model+bm25 ловят типичные
  ошибки пути/пересборки; полный hash — nightly/CLI.
- **Последствия:** `MCP_1C_MANIFEST_PATH`; scripts/build_manifest.py,
  scripts/eval_suite.py; ADR не меняет tool surface и scoring.
