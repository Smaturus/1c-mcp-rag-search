# Оценка роутера запросов

- кейсов: 20 из `routing_cases_part3_blind.jsonl`
- **точность маршрута: 100.0%** (20/20)

## Матрица маршрутов (expected → actual)

| Ожидался | lexical | vector | hybrid |
|---|---|---|---|
| `lexical` | 7 | 0 | 0 |
| `vector` | 0 | 6 | 0 |
| `hybrid` | 0 | 0 | 7 |

## End-to-end качество на gold-set v2 (коллекция `code`, k=10)

_Не считалось: запуск с `--routing-only`._
