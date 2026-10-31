# Отчёты и наборы оценки

Каталог содержит gold-set, разметку маршрутизации и отчёты замеров. Чтобы
отделить актуальные числа от архива, у каждого файла указан статус:

- **актуально** — соответствует текущему корпусу, нарезке и токенизатору;
- **историческое** — снято на прежнем корпусе или до исправления методики;
  сохранено для воспроизводимости и истории;
- **superseded** — вывод отменён более поздним решением (указано каким).

Методика и текущие сводные значения — в [docs/EVALUATION.md](../../docs/EVALUATION.md).
Команды перезамера — там же.

## Наборы

| Файл | Что внутри | Статус |
|---|---|---|
| [`gold_set.jsonl`](gold_set.jsonl) | v1: 30 запросов, шаблонные формулировки | историческое |
| [`gold_set_v2.jsonl`](gold_set_v2.jsonl) | v2: 70 запросов в естественных формулировках; dev-набор | актуально (dev) |
| [`gold_set_v3_holdout.jsonl`](gold_set_v3_holdout.jsonl) | v3: 100 независимых запросов (71 код, 5 метаданных, 24 вне корпуса) | актуально (hold-out) |
| [`anchors_v3.jsonl`](anchors_v3.jsonl) | 10 коротких лексических якорей для регрессии точного поиска | актуально |
| [`routing_cases.jsonl`](routing_cases.jsonl) | 68 регрессионных кейсов маршрутизации | актуально (контракт) |
| [`routing_cases_part3_blind.jsonl`](routing_cases_part3_blind.jsonl) | 20 кейсов blind Part-3; после правок стал dev-набором | историческое |
| [`golden_queries.example.json`](golden_queries.example.json) | пример формата запроса для своего gold-set | актуально |

Правило: **v2 — для разработки, v3 — для финальной проверки.** v3 нельзя
использовать для настройки правил или выбора режима.

## Актуальные отчёты

| Файл | Что показывает |
|---|---|
| [`code_proc/report_stability.md`](code_proc/report_stability.md) | v2, 4 прогона: mean ± std по режимам, латентность |
| [`code_proc/report_paired.md`](code_proc/report_paired.md) | парный тест `vector` vs `hybrid`: Δ, bootstrap CI, p-value |
| [`code_proc/report_pipeline.md`](code_proc/report_pipeline.md) | один прогон v2; для агрегата использовать `report_stability.md` |
| [`v3_code/report_stability.md`](v3_code/report_stability.md) | v3, 71 code-запрос, 3 прогона |
| [`v3_code/report_paired.md`](v3_code/report_paired.md) | v3, парный тест `vector` vs `hybrid` |
| [`v3_metadata/report_stability.md`](v3_metadata/report_stability.md) | v3, 5 metadata-запросов (индикативно, n = 5) |
| [`v3_report.md`](v3_report.md) | итог v3 и решение по ADR 0009/0010 |
| [`gold_set_v3_report.md`](gold_set_v3_report.md) | состав hold-out v3, baseline, ограничения |
| [`v3_router.md`](v3_router.md) | точность роутера на v3: 58/100 |
| [`v3_anchors.md`](v3_anchors.md) | лексические якоря: 10/10 в top-3 |
| [`metadata_ab_report.md`](metadata_ab_report.md) | A/B источника метаданных: текстовый отчёт vs XML |
| [`metadata_xml_report.md`](metadata_xml_report.md) | качество XML-метаданных по типам объектов |
| [`procedure_chunking_smoke.md`](procedure_chunking_smoke.md) | смоук процедурной нарезки: покрытие и токены |
| [`router_report.md`](router_report.md) | регрессия маршрутизации: 68/68 (проверка контракта, не обобщения) |
| [`rrf_experiments_2026-10-23.md`](rrf_experiments_2026-10-23.md) | отрицательный результат: веса RRF, preprocessing, глубина каналов |

Часть отчётов продублирована в `.json` с теми же данными в машинном формате.

## История и отменённые выводы

| Файл | Что показывает | Статус |
|---|---|---|
| [`ablation_2x2.md`](ablation_2x2.md) | 2×2 «выгрузка × нарезка» | superseded: снято дефектным BM25, см. ADR 0010 |
| [`symbolic_ablation/report_stability.md`](symbolic_ablation/report_stability.md) | клетка «текущая выгрузка × символьные окна», 4 прогона | историческое, требует перезамера |
| [`part3_blind_report.md`](part3_blind_report.md) | blind Part-3 до правок: 75 % | историческое |
| [`part3_blind_report_postpatch.md`](part3_blind_report_postpatch.md) | blind Part-3 после правок: 20/20, набор стал dev | историческое |
| [`report_pipeline.md`](report_pipeline.md) | v1, 4 прогона на прежнем корпусе | историческое |
| [`report_stability.json`](report_stability.json) | устойчивость v1 на прежнем корпусе (JSON) | историческое |
| [`v2/report_pipeline.md`](v2/report_pipeline.md) | v2 на прежнем корпусе (символьные окна) | историческое |
| [`embedding_tokens.md`](embedding_tokens.md) | доля чанков сверх 512 токенов на старом корпусе | историческое (аргумент за ADR 0007) |
| [`bench.md`](bench.md) | латентность на старом корпусе (124 тыс. чанков) | историческое |
| [`proc_smoke_progress.json`](proc_smoke_progress.json) | служебный прогресс смоук-сборки | служебный артефакт |

Хронология решений и причины изменения чисел — в [docs/HISTORY.md](../../docs/HISTORY.md).
