# Исследовательские артефакты (не входят в пайплайн)

Скрипты в этом каталоге — **диагностика и разовые эксперименты**, а не рабочие
инструменты проекта. Они оставлены как история инженерных решений и как
воспроизводимые доказательства для отчётов (см. `data/eval/`).

## Что здесь есть

| Скрипт | Назначение |
|---|---|
| `mcp_stdio_smoke.py` | проверка протокола MCP без клиента: initialize / tools/list / tools/call |
| `collections_health.py` | открываются ли коллекции Chroma, count по коллекциям |
| `hnsw_selfmatch.py` | полнота HNSW: находит ли индекс собственный чанк |
| `peek_chroma_schema.py` | схема sqlite Chroma (таблицы, индексы) |
| `pickle_and_dbstat_probe.py` | что лежит в `index_metadata.pickle` и в sqlite |
| `rebuild_from_sqlite.py` | пере-эмбеддинг документов из sqlite в новую базу (восстановление после сбоя HNSW) |
| `hybrid_smoke.py` | быстрый смоук гибридного поиска |
| `collections_health.py`, `peek_rebuilt_db.py`, `rebuild_chroma_from_chroma.py` | диагностика конкретного инцидента (chroma#7090) |

## Как пользоваться

Скрипты рассчитаны на ручной запуск из корня репозитория и, как правило, читают
`data/chroma/`:

```powershell
python scripts\diag\mcp_stdio_smoke.py
python scripts\diag\collections_health.py
python scripts\diag\hnsw_selfmatch.py
```

## Что не является контрактом

- Форматы вывода и аргументы могут меняться без предупреждения.
- Часть скриптов писалась под конкретный инцидент и ожидает его состояние данных.
- Они не покрыты тестами и не входят в CI.

Рабочий пайплайн — модули `src/mcp_1c_homework/` и скрипты `scripts/*.py`
(`build_bm25.py`, `make_goldset.py`, `eval_pipeline.py`, `eval_stability.py`, `bench.py`).
