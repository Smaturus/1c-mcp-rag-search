# План работ после embedding e5-base

- **Версия:** 2.5 (правки по вопросу о gold-наборах и факту F.0 от 2026-11-05)
- **Статус:** к согласованию. Ничего не запускать без отдельного «да» на фазу.
- **Исходная точка:** DoD-min достигнут (D0 PASS, D PASS, E PASS); embedding DONE — `data/local_notes/embedding_e5_base/state.json` (`phase: C_done`).
- **Закрытый план сборки:** `docs/PLAN_EMBEDDING_E5_BASE.md`.
- **Справочно:** `docs/not_actual_PLAN_EMBEDDING_BASELINE.md` / `docs/not-actual_PLAN_E5_LARGE_FULL_CONFIG.md` — старые цифры; SHA опубликованного v2 gold в репо **нет**.
- **Изменения v2.5:** явно закреплён перечень наборов для F — **только** v2/v3/v4 (curated, legacy, anchors, routing, веб-кандидаты исключены); добавлен факт F.0-проверки 2026-11-05 о расхождении top-10 у кандидата (F остановлен до решения); веб-кандидаты — отдельная задача после `decision.md`.
- **Изменения v2.4:** добавлен F.0 «HNSW warm/persist + детерминизм» — после E обнаружено, что cold/warm top-1 `code_proc` расходятся: `hnsw:sync_threshold=1000000` держит индекс не в pickle, и первый `PersistentClient` перестраивает граф из WAL (HNSW-сегменты ~0.31 МБ, mtime — время E). Без прогрева первый счётный прогон F может систематически отличаться и сместить mean/CI. F.4/F.6 запускаются только после PASS F.0; при нестабильности — стоп, без подбора параметров.
- **Изменения v2.3:** восстановлены подфазы G с разрешениями/бэкапами/критериями отката (в v2.2 они были «как в v2.1» без текста); добавлен pre-F гейт покрытия gold-источников; F сохраняет JSON/MD парных отчётов и считает latency-ratio; в E добавлен in-process замер query-latency; в G заданы поиск PID и определение ready; чистый out-dir F проверяется командой; файловый sha заменён логическим fingerprint (см. §0.4).
- **Живой MCP сейчас** (`.dsh-bundles/1c-mcp-rag-search/cordis.patch.yml`):
  - `CHROMA_DIR=data/chroma_xml`, `BM25_DIR=data/bm25_all`
  - `SEARCH_MODE=vector`, `CODE_COLLECTION=code_proc`, `METADATA_COLLECTION=metadata_xml`

## 0. Защита посчитанных данных

### 0.1 Что неприкосновенно (логический инвариант)

**Инвариант священного индекса:** набор коллекций `{code_proc, config_xml}` и counts **153990 / 237437**.  
**Не** инвариант: байты/`mtime`/`sha256` файла `chroma.sqlite3` — обычный `PersistentClient` (даже `list_collections`+`count` без записи документов) меняет служебные метаданные/WAL. Проверено 2026-11-05: sha сменился, counts нет.

| Путь | Когда священный | Защищать |
| --- | --- | --- |
| `data/chroma_e5_base/` | сразу | коллекции+counts; не удалять каталог |
| `data/checkpoints/e5_base_bsl.json`, `e5_base_xml.json` | сразу | не удалять/не переписывать |
| `data/local_notes/embedding_e5_base/` | сразу | не удалять |
| `data/bm25_e5_base/` | после PASS D | не удалять |
| `data/eval/_runs_e5_base/` | после прогонов F | не удалять `*_run*.json`; чистить только `_tmp_*` по протоколу F |

**Запрещено** без явной фразы с путём: удаление каталогов; `--reset`; `delete_collection`; повторный embedding в тот же chroma; создание новой коллекции (в т.ч. пустой `metadata_xml`); `git clean` по `data/`.

### 0.2 Рабочий контур e5-small (не сносить)

`data/chroma_xml` (154560 `code_proc` + 25026 `metadata_xml`), `data/bm25_all` (179586 доков), `data/chroma_xml_eval` — контрольная копия. При `SEARCH_MODE=vector` BM25 в unified search не участвует, но нужен lexical-тулам и hybrid-режиму.

### 0.3 Куда этой волне можно писать

| Назначение | Путь |
| --- | --- |
| BM25 кандидата | `data/bm25_e5_base/` |
| Throwaway BM25-смоуки | `data/bm25_e5_base_smoke_bsl/`, `data/bm25_e5_base_smoke_xml/` (удалять только по «да») |
| Заметки / отчёты / decision / бэкап yml / src_backup / одноразовые RO-скрипты | `data/local_notes/after_e5_base/` |
| Eval-прогоны F | `data/eval/_runs_e5_base/` |
| Бэкап Chroma (H) | `data/chroma_e5_base_backup_YYYY-MM-DD/` |
| MCP env | `.dsh-bundles/1c-mcp-rag-search/cordis.patch.yml` — только G-yml после «да» |
| Код сервера (G1/G2) | `src/mcp_1c_rag/server.py` (+ затронутые модули) — только G-code после отдельного «да» |
| Bundle второго MCP (G3) | `.dsh-bundles/…` — только G3-bundle после «да» |
| `.gitignore` | корень репо — D0 |

Иначе писать в `src/` и `scripts/` **запрещено**. Одноразовые проверки — только в `after_e5_base/` с diff до «да».

### 0.4 `get_or_create` и fingerprint

1. До/после клиента на `chroma_e5_base`: список коллекций + counts.
2. **Устойчивый fingerprint (обязателен для гейтов E/F/G):** sqlite `mode=ro`, sha256 по отсортированному списку `embedding_id` (при желании — с `source`). Хранить в `preflight.md`/`state.json`.
3. Файловый `sha256(chroma.sqlite3)` — **только справочно** в D0 и только до первого `PersistentClient`; после E/F/G файловый sha **не** гейтить.
4. Запрещено `MCP_1C_METADATA_COLLECTION=metadata_xml` при `CHROMA_DIR=data/chroma_e5_base`, пока коллекции нет.

### 0.5 Чеклист перед командой

1. Destructive на священное? → стоп.
2. `PersistentClient` на `chroma_e5_base`? → только **E / F / G** + пост-проверка «коллекции + counts + fingerprint» (не file-sha).
3. Сомнение → показать команду.

### 0.6 gitignore (D0)

- `data/chroma_e5_base/`, `data/chroma_e5_base_backup_*/`, `data/checkpoints/`
- `data/bm25_e5_base/`, `data/bm25_e5_base_smoke_bsl/`, `data/bm25_e5_base_smoke_xml/`
- `data/eval/_runs_e5_base/`

`after_e5_base` отдельно не дублировать (покрыт `data/local_notes/`). `.dsh-bundles/` уже ignored.

---

## 1. Цель и порядок

DoD-min = D0+D+E. DoD-full = +F+G.  
**G без F запрещена**, кроме явного «G без F».  
Порядок: **D0 → D → E → (F) → (G) → (H)**. Каждая фаза — своё «да».

---

## 2. Фазы

### Фаза D0 — ignore + baseline

1. `.gitignore` (§0.6) + `git check-ignore` на каждый путь.
2. Sqlite RO: коллекции и counts; **логический fingerprint** (sha256 отсортированных `embedding_id`) → `after_e5_base/preflight.md`; размер каталога `chroma_e5_base`.
3. Справочно один раз: файловый sha256 `chroma.sqlite3` → `preflight.md` с пометкой «до любого PersistentClient; после клиента не сравнивать».
4. SHA-256 трёх gold-файлов → `data/local_notes/after_e5_base/gold_sha.json` (зафиксированный снимок волны). Stop-rule: расхождение SHA файла с этим снимком позже — стоп.

### Фаза D — BM25

Клиент Chroma не открывается. Сверка с D0: логический fingerprint + counts (файловый sha — справочно).

**D-smoke (два каталога, оба в ignore):**

```text
py scripts/build_bm25.py --chroma-dir data/chroma_e5_base --out data/bm25_e5_base_smoke_bsl --limit 2000 --id-prefix bsl_proc:
py scripts/build_bm25.py --chroma-dir data/chroma_e5_base --out data/bm25_e5_base_smoke_xml --limit 2000 --id-prefix xml:
```

**Полная сборка:**

```text
py scripts/build_bm25.py --chroma-dir data/chroma_e5_base --out data/bm25_e5_base
```

**Гейты D:**

- fingerprint + counts без изменений; коллекций по-прежнему две;
- `doc_ids` BM25 = id `bsl_proc:` ∪ `xml:`; 153990 + 237437; пустых пропусков 0;
- `verify_bm25_ids.py` as-is **не использовать** (только `bsl_proc:%`). Параметризовать оба префикса или одноразовый RO-скрипт в `after_e5_base/`; diff до «да»;
- `data/bm25_all` и `chroma_xml*` не изменились (counts/fingerprint);
- после PASS: `data/bm25_e5_base/` входит в §0.1.

**Не делать:** `--out data/bm25_all`.

### Фаза E — смоук

Только `code_proc` / `config_xml`; `smoke_search.py` не RO (открывает `PersistentClient`), поэтому пост-гейт: коллекции + counts + fingerprint.

**Разделить startup и query latency:**

1. Cold-прогон (факт, не FAIL): `Measure-Command { py scripts/smoke_search.py ... }` — включает загрузку модели и Chroma; при холодном кэше открытие клиента наблюдалось >120 с.
2. Предпочтительно in-process замер: notes-скрипт `after_e5_base/warm_smoke.py`: импорт `search`, 1 прогревочный вызов, затем замер **второго** вызова и вывод ms. Это и есть query-latency; писать только в `after_e5_base/`.
3. Fallback: два подряд `Measure-Command`; второй писать как «warm reload + query», не как чистую query-latency.
4. Cold startup записать в `smoke.md` отдельно; порог «мягкого алерта» для прогретого замера согласовать после первого факта.

```text
$env:MCP_1C_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
Measure-Command { py scripts/smoke_search.py --chroma-dir data/chroma_e5_base --bm25-dir data/bm25_e5_base --collection code_proc --query "как начислить НДС с авансов полученных" }
Measure-Command { py scripts/smoke_search.py --chroma-dir data/chroma_e5_base --bm25-dir data/bm25_e5_base --collection config_xml --query "ДолжностиОрганизаций" }
```

PASS: results≥1 vector+hybrid; префиксы id `bsl_proc:`/`xml:`; source непустой; коллекции+counts+fingerprint без изменений; `smoke.md` записан.  
**Наблюдение E (2026-11-05):** cold vs warm vector-first у `code_proc` может отличаться (`bsl_proc:c5e7…:58` vs `bsl_proc:d0ade4…:134`), hybrid-first и `config_xml` — совпали. Причина — перестройка HNSW при первом клиенте (`hnsw:sync_threshold=1e6`). Для F это закрывается гейтом F.0.

### Фаза F — парный замер

#### F.0 HNSW warm/persist + детерминизм (обязательно до F.1)

**Почему:** `hnsw:sync_threshold=1000000` (workaround chroma#7090) держит HNSW-индекс вне pickle; первый `PersistentClient` после простоя перестраивает граф из WAL. В E это проявилось: cold `code_proc` vector-first `bsl_proc:c5e7…:58` ≠ warm `bsl_proc:d0ade4…:134` (hybrid-first и `config_xml` совпали). Если первый счётный прогон F попадёт на перестройку, он будет систематически отличаться; при n=3–4 это сместит mean и CI.

1. Зафиксировать в `after_e5_base/f0_hnsw.md`: размер/mtime файлов `data_level0.bin` в сегментах `data/chroma_e5_base` и `data/chroma_xml_eval`.
2. **Warm/persist:** по одному процессу на контур — vector-запрос по `code_proc` (`smoke_search.py --collection code_proc` или `warm_smoke.py`), дождаться завершения; проверить, что mtime `data_level0.bin` обновился (или записать, что состояние уже стабильно).
3. **Гейт детерминизма:** один и тот же vector-запрос в **двух разных процессах после прогрева** → top-10 id должны совпасть для `code_proc` на **обоих** контурах (`chroma_e5_base` и `chroma_xml_eval`). Записать id/score в `f0_hnsw.md`.
4. **STOP, если не совпали:** параметры (`hnsw:sync_threshold`, `hnsw:space`, переиндексация) не подбирать; отдельное решение пользователя.
5. Только после **F.0 PASS** запускать F.4. Если в F.0 выяснится, что первый прогон всё равно выбивается, допускается `n+1` прогон с исключением первого — **только** с явной записью в `decision.md` (какие файлы исключены и почему), не молча.

**Факт проверки 2026-11-05 (сессия F):** `f0_stab_cand_1.json` first id `bsl_proc:364b819ba6f301f1:119` (`CommonModules/…Kafka/…`), `f0_stab_cand_2.json` first id `bsl_proc:c5e7bd15cdee747f:58` (`Documents/…/ObjectModule.bsl`) — top-10 различаются. Если оба прогона были **после** прогрева (а не один warm-up), это **FAIL F.0**: F остановлен до отдельного решения пользователя по HNSW (`sync_threshold`/персист); параметры не подбирать, переиндексацию не запускать.

#### F.1 Честность

Контроль: символьный контур + e5-small (dim 384) + `metadata_xml` (карточки).  
Кандидат: токенный + e5-base (dim 768) + `config_xml` (сырые XML).  
Δ = связка модель+нарезка+состав, не чистый «base vs small». Ворота только **code**/`code_proc`.

#### F.2 Gold и SHA

n: v2=70 (без `--scope`), v3 code=64, v4 code=179; `--answerable-only` не нужен.  
SHA: сравнить с `after_e5_base/gold_sha.json` (D0); при расхождении — стоп.

**Использовать только эти три набора.** Не использовать: `data/eval/gold_set.jsonl` (legacy, заменён v2), `data/eval/v2_curated.jsonl` (подмножество v2; нет baseline/порогов), `data/eval/anchors_v3.jsonl`, `data/eval/routing_cases*.jsonl`, `docs/GOLD_CANDIDATES_WEB_2026-11-02.md` (черновик, не gold-set; разбор — отдельная задача после `decision.md`). В `decision.md` — явно, что другие наборы не использовались.

#### F.3 Pre-F гейты (до прогонов)

1. **Чистый out-dir:** `Get-ChildItem data/eval/_runs_e5_base -Filter *_run*.json` → если есть `old_*`/`new_*` — стоп, показать список (иначе `run_eval_runs` молча пропустит существующие).
2. **Покрытие gold-источников:** одноразовый RO-скрипт `after_e5_base/check_gold_sources.py` (diff до «да») читает `source` из `code_proc` через sqlite `mode=ro` и сверяет с `relevant_sources` трёх gold. Проверить **оба** контура: `chroma_e5_base` и `chroma_xml_eval`. Ожидание: missing=0. Иначе F не запускать — Δ будет искажён.
3. **Оценка времени:** `run_eval_runs.py` запускает отдельный процесс на прогон (HNSW/model). Процессов ≈ 3 набора × (4+3+4) × 2 стороны ≈ 22 + 2 процесса прогрева F.0. Прогноз = `N_proc × cold_start_sec (из E) + queries × latency`; записать в `decision.md` до старта. Стоп, если прогноз > согласованного потолка часов.

#### F.4 Контроль (e5-small) — до кандидата

```text
py scripts/run_eval_runs.py --golden data/eval/gold_set_v2.jsonl --chroma-dir data/chroma_xml_eval --bm25-dir data/bm25_all --collection code_proc --embedding-model intfloat/multilingual-e5-small --n-runs 4 --prefix old_v2 --out-dir data/eval/_runs_e5_base

py scripts/run_eval_runs.py --golden data/eval/gold_set_v3_holdout.jsonl --chroma-dir data/chroma_xml_eval --bm25-dir data/bm25_all --collection code_proc --embedding-model intfloat/multilingual-e5-small --scope code --n-runs 3 --prefix old_v3_code --out-dir data/eval/_runs_e5_base

py scripts/run_eval_runs.py --golden data/eval/gold_set_v4_holdout.jsonl --chroma-dir data/chroma_xml_eval --bm25-dir data/bm25_all --collection code_proc --embedding-model intfloat/multilingual-e5-small --scope code --n-runs 4 --prefix old_v4_code --out-dir data/eval/_runs_e5_base
```

**Перед этими командами обязателен F.0 PASS** (warm/persist + детерминизм на `chroma_xml_eval`). Если F.0 не пройден — контроль F не запускать.

#### F.5 δ vector (до кандидата)

```text
py scripts/record_control_delta.py --runs "data/eval/_runs_e5_base/old_v2_run*.json" --mode vector --metric ndcg@10 --floor 0.028 --key delta_v2 --state data/local_notes/after_e5_base/state.json

py scripts/record_control_delta.py --runs "data/eval/_runs_e5_base/old_v3_code_run*.json" --mode vector --metric ndcg@10 --floor 0.020 --key delta_v3 --state data/local_notes/after_e5_base/state.json

py scripts/record_control_delta.py --runs "data/eval/_runs_e5_base/old_v4_code_run*.json" --mode vector --metric ndcg@10 --floor 0.020 --key delta_v4 --state data/local_notes/after_e5_base/state.json
```

- Полы: v2 **0.028** (vector σ из baseline-дока), v3 **0.020** (vector), v4 **0.020** — пол по аналогии (опубликованной σ v4 нет); факт δ = `max(floor, std)` текущих прогонов.
- В `decision.md`: δ — разброс **между прогонами** (pstdev агрегатов nDCG), **не** SE/bootstrap по запросам из `eval_report_pair`; CI(Δ) в решении — парный bootstrap по запросам.

#### F.6 Кандидат и пара

```text
py scripts/run_eval_runs.py --golden data/eval/gold_set_v2.jsonl --chroma-dir data/chroma_e5_base --bm25-dir data/bm25_e5_base --collection code_proc --embedding-model intfloat/multilingual-e5-base --n-runs 4 --prefix new_v2 --out-dir data/eval/_runs_e5_base

py scripts/run_eval_runs.py --golden data/eval/gold_set_v3_holdout.jsonl --chroma-dir data/chroma_e5_base --bm25-dir data/bm25_e5_base --collection code_proc --embedding-model intfloat/multilingual-e5-base --scope code --n-runs 3 --prefix new_v3_code --out-dir data/eval/_runs_e5_base

py scripts/run_eval_runs.py --golden data/eval/gold_set_v4_holdout.jsonl --chroma-dir data/chroma_e5_base --bm25-dir data/bm25_e5_base --collection code_proc --embedding-model intfloat/multilingual-e5-base --scope code --n-runs 4 --prefix new_v4_code --out-dir data/eval/_runs_e5_base
```

**Кандидат запускать только после F.0 PASS** (warm/persist и детерминизм на `chroma_e5_base`); иначе первый прогон может быть смещён перестройкой HNSW.

Парные отчёты (для каждого набора; пример v2):

```text
py scripts/eval_report_pair.py --old "data/eval/_runs_e5_base/old_v2_run*.json" --new "data/eval/_runs_e5_base/new_v2_run*.json" --mode vector --metric ndcg@10 --out-json data/local_notes/after_e5_base/pair_v2_vector.json --out-md data/local_notes/after_e5_base/pair_v2_vector.md
py scripts/eval_report_pair.py --old "data/eval/_runs_e5_base/old_v2_run*.json" --new "data/eval/_runs_e5_base/new_v2_run*.json" --mode hybrid --metric ndcg@10 --out-json data/local_notes/after_e5_base/pair_v2_hybrid.json --out-md data/local_notes/after_e5_base/pair_v2_hybrid.md
```

4. **Latency-ratio (ворота ≤3× для vector):** `eval_report_pair` этого не считает. Нужен notes-скрипт `after_e5_base/latency_ratio.py` (diff до «да»): берёт `results.vector.latency_ms` (среднее) из `old_*`/`new_*` `run_*.json` по каждому набору, пишет old/new mean и отношение в `decision.md`; gate — max ratio ≤ 3.0.
5. После F: пост-проверка коллекций+counts+fingerprint; удалить только `data/eval/_runs_e5_base/_tmp_*`; `*_run*.json` сохранить (§0.1).

#### F.7 Решение

| Исход | Vector nDCG@10, три code-набора |
| --- | --- |
| Принять → G с `SEARCH_MODE=vector` | CI(Δ) lower > 0 на всех **или** (явный выбор) lower > −δ |
| Отклонить | upper < −δ хотя бы на одном |
| Неразличимо | иначе; G только по письменному риску |

Оговорка multiple comparisons → `decision.md`.  
Hybrid в G — только после такого же протокола с `--mode hybrid` (floors hybrid: v2 0.012, v3/v4 0.015; δ=max(floor,std)).

### Фаза G — MCP

#### G.0 Подфазы и разрешения

| Подфаза | Пишет | Нужно «да» |
| --- | --- | --- |
| G-decide | notes: выбор G1/G2/G3, `SEARCH_MODE`, metadata-контракт | вместе с «да на G» или отдельно |
| G-code | `src/mcp_1c_rag/server.py` (+ затронутые модули) | **отдельное** «да на G-code» (только G1/G2) |
| G-yml | `cordis.patch.yml` + backup | «да на правку yml» |
| G3-bundle | второй entry в `.dsh-bundles` | «да на G3» |
| G-verify | notes/замеры | после yml |
| G-rollback | восстановление yml (+ `server.py`) | по критерию |

**Бэкапы до правок:** `after_e5_base/cordis.patch.yml.backup` + sha256; при G-code — `after_e5_base/src_backup/` с копиями **всех** изменяемых файлов (с относительными путями).

**Варианты metadata (выбрать один в G-decide):**

| Вариант | Суть |
| --- | --- |
| **G1** | `METADATA_COLLECTION=config_xml` + G-code: описания тулов говорят «сырые XML-чанки (`xml:`)», merged-выдача помечает префикс; metadata-tool возвращает `xml:` id |
| **G2** | G-code: metadata-тулы и merged-пути не регистрируются/заглушены; yml без `metadata_xml`; вызов metadata-tool = ожидаемое отсутствие/ошибка |
| **G3** | без правки main `server.py`: второй MCP entry (уникальный `serverName`/id) на `chroma_xml`+small+`metadata_xml`; основной entry → e5-base+`code_proc` (+`config_xml` по решению) |

Создавать `metadata_xml` в `chroma_e5_base` — **запрещено**.

#### G.1 До yml

1. Backup yml + sha256; при G1/G2 — backup `server.py` (и всех изменяемых модулей).
2. Полный diff пользователю (yml и, если есть, код).
3. `SEARCH_MODE=vector` по умолчанию (согласован с F). Hybrid — только если пройден hybrid-гейт F.
4. G3: выяснить и записать, как DSH/Cursor различает два MCP (`serverName`, префикс в UI/именах тулов); второй entry — с **уникальным** именем.

#### G.2 После «да»

1. Сначала G-code/G3-bundle (если выбраны), потом yml.
2. Перезагрузка DSH/Cursor MCP — шаги пользователя; агент не объявляет успех без смоука.
3. **Startup/RAM:** найти PID stdio-процесса:
   ```text
   Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
     Where-Object { $_.CommandLine -like '*mcp_1c_rag*' } |
     Select-Object ProcessId,ParentProcessId,@{n='RSS_MB';e={[math]::Round($_.WorkingSetSize/1MB)}}
   ```
   Брать дочерний процесс с ненулевым RSS. Ready = timestamp первого успешного tool response; startup = ready − `CreationDate` процесса (или timestamp первой строки лога). RSS после загрузки — `Get-Process -Id <pid>`. **FAIL:** ready или первый запрос > 240 с (`toolCallTimeoutMs`) → откат; записать числа в notes.
4. **Пост-гейты:** коллекции+counts+fingerprint без изменений (не file-sha); поиск по `code_proc` с префиксами `bsl_proc:`; metadata-путь по выбранному G1/G2/G3; список tool names зафиксирован; нет чужих коллекций.

#### G.3 Откат

Критерии: ошибки metadata/search; ready/первый запрос > 240 с; появилась новая коллекция; counts/fingerprint не совпали; явное «откати».  
Действие: yml из backup; при G1/G2 — `server.py` (+изменённые модули) из `src_backup/`; кандидатские data **не** удалять.  
**Пост-откат обязателен:** смоук старого контура (`chroma_xml` + e5-small, codesearch/vector) — PASS.

### Фаза H — бэкап Chroma (опц.)

1. Свободно ≥ size×1.1 + 10 ГБ (перепроверить).
2. Нет `data/index.lock`.
3. Copy (не move) → `data/chroma_e5_base_backup_YYYY-MM-DD/`.
4. Верификация: counts + размер каталога + (манифест sha256 **всех** файлов каталога **или** контрольный vector-запрос к копии с отдельным `--chroma-dir` без создания новых коллекций). Одного sha sqlite мало.
5. Путь в `.gitignore` (§0.6).

---

## 3. Вне объёма

Повторный embedding / e5-large; удаление старых `chroma*`/`bm25*`; создание `metadata_xml` в `chroma_e5_base`; карточки metadata в кандидат без отдельного плана; git commit/push без просьбы; правки `src/` вне G-code; правки `scripts/` в этой волне (одноразовые проверки — только в `after_e5_base/`).  
Разбор веб-кандидатов (`docs/GOLD_CANDIDATES_WEB_2026-11-02.md`) и создание нового gold-set (новый SHA, отдельные пороги, новая методика сравнения) — отдельная задача после `decision.md`; v2/v3/v4 **не пополнять**.

## 4. Definition of Done

### DoD-min

1. D0: `.gitignore` + `preflight.md` (коллекции, counts, fingerprint, gold SHA, размер каталога).
2. D: полный `bm25_e5_base` + гейты; путь в §0.1.
3. E: `smoke.md` PASS (vector+hybrid, префиксы, latency cold/warm записана); инвариант без изменений.
4. `state.json`/отчёт в `after_e5_base/`.
5. MCP на старом контуре — норма для DoD-min.

### DoD-full

1. DoD-min.
2. F: F.0 PASS (warm/persist + детерминизм top-10 на обоих контурах); SHA gold сверены с `gold_sha.json`; pre-F gold-coverage PASS; δ записаны **до** кандидата; парные JSON/MD сохранены; latency-ratio посчитан; `decision.md` (выбор, CI, δ, multiple comparisons).
3. G: выбранный вариант; отдельные «да» на G-code/G3/yml; бэкапы; startup/RAM записаны; пост-гейты PASS; при откате — пост-откат смоук.
4. `state.json` дополнен: gold SHA, δ v2/v3/v4 (+hybrid при наличии), исход F, результаты G; при необходимости ссылка в `docs/INDEX.md`.

## 5. Старт для другого чата

Читать: этот файл (v2.5) + `data/local_notes/embedding_e5_base/state.json` + текущий `cordis.patch.yml`.  
Старт: **«да на D0»**. D → E только после D0. F — только после отдельного «да»; первым шагом F обязателен F.0 (HNSW warm/persist + детерминизм). G — отдельное «да»; hybrid в G — только после hybrid-гейта F.
