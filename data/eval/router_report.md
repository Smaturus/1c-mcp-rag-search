# Оценка роутера запросов

- кейсов: 68 из `routing_cases.jsonl`
- **точность маршрута: 100.0%** (68/68)

## Матрица маршрутов (expected → actual)

| Ожидался | lexical | vector | hybrid |
|---|---|---|---|
| `lexical` | 22 | 0 | 0 |
| `vector` | 0 | 15 | 0 |
| `hybrid` | 0 | 0 | 31 |

## End-to-end качество на gold-set v2 (коллекция `code`, k=10)

_Не считалось: запуск с `--routing-only`._
