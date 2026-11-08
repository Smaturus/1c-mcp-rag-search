# План: embedding `.bsl` + `.xml` (e5-base)

- **Дата:** 2026-11-02
- **Статус:** план к согласованию. Сборка не начата.
- **Корпус:** `D:\MCP_1С_homework\1C-Corpus\1C_Corpus`
- **Модель:** `intfloat/multilingual-e5-base` (dim 768)
- **Машина:** та же (CPU, без дискретной GPU)
- **Batch эмбеддинга:** `8` (фиксировано; без отдельного ресурсного прогона)
- **Бюджет нарезки по умолчанию:** `max_tokens=500` (при гейте >512 — см. §5.A)

## 1. Цель

Посчитать и сохранить векторный индекс по **всем** файлам:

| Тип | Корень | Коллекция Chroma | Префикс id |
| --- | --- | --- | --- |
| `.bsl` | `1C-Corpus\1C_Corpus` | `code_proc` | `bsl_proc:` |
| `.xml` | тот же | `config_xml` | `xml:` |

Снимок корпуса (подсчёт 2026-11-02): **3444** `.bsl`, **16385** `.xml`.

После embedding: поиск / BM25 / замер качества — **отдельное решение**, в этот план не входят.

## 2. Размещение

Новый контур — **только новые пути**. Писать только сюда:

| Назначение | Путь |
| --- | --- |
| Chroma (полная сборка) | `data/chroma_e5_base` |
| Чекпоинт BSL | `data/checkpoints/e5_base_bsl.json` |
| Чекпоинт XML | `data/checkpoints/e5_base_xml.json` |
| Логи / state | `data/local_notes/embedding_e5_base/` |

В командах этой волны не подменять `MCP_1C_CHROMA_DIR` на чужие каталоги индексов.

## 3. Нарезка и эмбеддинг (уже в проекте)

Не изобретать новую схему. Использовать токенный контур, согласованный с `embed_passages` (`passage:` + префикс файла + тело).

| | BSL | XML |
| --- | --- | --- |
| Модуль | `src/mcp_1c_rag/chunking_tokens.py` | `src/mcp_1c_rag/chunking_xml_tokens.py` |
| API | `chunk_bsl_file_tokens` | `chunk_xml_file_tokens` |
| Токенизатор | `get_tokenizer(model_name)` — **тот же** `intfloat/multilingual-e5-base` | то же |
| Бюджет | `max_tokens=500`, `overlap=64`, `min_tokens=32`, `max_chars=2000` | те же константы |
| Границы | процедуры/функции BSL; длинные — по строкам/токенам | по строкам/тегам; без proc |
| Скрипт сборки | `scripts/build_token_index.py` | `scripts/build_xml_index.py` |
| Сухой прогон (без векторов) | `scripts/chunk_corpus_tokens.py` | `scripts/chunk_xml_corpus_tokens.py` |
| Id чанка | `bsl_proc:<sha1(source)[:16]>:<chunk_index>` | `xml:<sha1(source)[:16]>:<chunk_index>` |

Эмбеддинг: `src/mcp_1c_rag/embeddings.py` — `embed_passages` / `embed_query`, префиксы `passage:` / `query:`, `normalize_embeddings=True`.

Символьный `chunk_bsl_file` / старый indexer — **не** использовать в этой волне.

## 4. Сохранность прогресса

1. **Resume и идемпотентность.** Повторный запуск той же команды продолжает с чекпоинта. Запись в Chroma — `upsert` по детерминированным id (`bsl_proc:…` / `xml:…`): повторная обработка того же файла не создаёт дубликатов id.
2. **Чекпоинт и resume.** В текущих сборочных скриптах JSON — это `{"processed": N, "chunks": M}` (атомарно: temp + `replace`). Списка путей в файле **нет**: resume делает `sorted(root.rglob(...))` и продолжает с `files[N:]`. Контракт: тот же `--root`, тот же шаблон glob, стабильный порядок сортировки; не менять состав/имена файлов «левее» N между запусками. Поле `chunks` — для гейта count, не для выбора файлов. Upsert по детерминированным id страхует от дублей, если один и тот же файл попадёт в прогон повторно. Точную семантику подтвердить в §5.A.3 по коду (не по памяти).
3. Два писателя в один Chroma-каталог запрещены (`data/index.lock`, без `--no-lock`).
4. BSL и XML — строго последовательно: сначала `code_proc`, затем `config_xml`.
5. **Изоляция коллекций.** BSL и XML живут в одном `data/chroma_e5_base`, но в **разных** коллекциях. Сборка XML пишет только в `config_xml` и **не должна** удалять, пересоздавать, очищать или иным способом портить уже собранный `code_proc`. Чекпоинт XML — отдельный файл; он не заменяет и не обнуляет чекпоинт BSL. Перед стартом XML: `code_proc` на месте (ненулевой count, dim 768). После XML: count `code_proc` не уменьшился. Это **пост-гейты**; доверие к изоляции опирается на разбор кода в фазе A (§5.A), а не только на count после факта.
6. В этой волне флаг сброса коллекции/чекпоинта у сборочных скриптов **не использовать**.

## 5. Фазы

### Фаза A — подготовка (без записи в `data/chroma_e5_base`)

1. Свободное место на `D:` ≥ 30 ГБ.
2. Модель `intfloat/multilingual-e5-base` доступна в кэше HF (если нет — скачать до длинного прогона).
3. **Разбор кода (изоляция, запись, resume)** — зафиксировать в `data/local_notes/embedding_e5_base/code_review.md`:
   - `scripts/build_xml_index.py`: открывает только коллекцию из `--collection` (по умолчанию `config_xml`) через `get_collection` → `get_or_create_collection`; при обычном прогоне **не** вызывает `client.reset()`, **не** делает `delete_collection` для чужих имён, **не** удаляет каталог `--chroma-dir`; пишет `embed_passages` + `upsert` только в эту коллекцию.
   - `scripts/build_token_index.py`: то же для `code_proc`; **не** трогает `config_xml`.
   - Оба пути создания коллекции задают одну метрику: в `store.get_collection` — `hnsw:space=cosine` (нужно для нормализованных e5). Подтвердить, что оба скрипта ходят только через этот `get_collection`, без своего клиента с другой space.
   - Оба вызывают `embed_passages` (не сырой `model.encode`); префиксы и нормализация — в `embeddings.py`.
   - **Resume:** как именно `_read_checkpoint` / цикл по файлам используют `processed`; есть ли в JSON пути/хэши или только смещение в `sorted(rglob)`; достаточно ли этого для безопасного продолжения при неизменном корпусе; роль поля `chunks` только для гейта. Если окажется, что resume без списка путей хрупок при любом изменении дерева — стоп и отдельное решение до B (в этой волне скрипты не переписывать «заодно»).
4. **Сухой прогон нарезки** → `chunk_dry_bsl.json` / `chunk_dry_xml.json`. Обязательные поля:
   - `n_chunks`, `token_stats`: min, p50, p95, p99, max;
   - `n_over_512` — если > 0 → **стоп** (развилка ниже);
   - `files_without_chunks` / число файлов с нулём чанков;
   - число чанков с длиной &lt; `min_tokens`;
   - если скрипт ещё не пишет `min` и/или below-`min_tokens` — **короткий патч dry-run скриптов до старта B** (в команды полной сборки это не смешивать).
5. **Развилка при `n_over_512` > 0 (решение по умолчанию, без импровизации):**  
   снизить `max_tokens` с 500 до **480** в командах сухого прогона и полной сборки, перепрогнать dry-run. Если после 480 всё ещё есть чанки >512 — стоп и отдельное решение (чинить чанкер; `max_seq_length` модели в этой волне не трогать). Менять бюджет «на глаз» в момент аварии нельзя.
6. Итоги фазы A → `data/local_notes/embedding_e5_base/`. Без пунктов 3–5 фаза B не стартует.

### Фаза B — полная сборка embedding

1. BSL → `data/chroma_e5_base` / `code_proc` / `data/checkpoints/e5_base_bsl.json`
2. XML → тот же Chroma / `config_xml` / `data/checkpoints/e5_base_xml.json`  
   (только эта коллекция; готовый `code_proc` не трогать — см. §4 п.5)

Общее: `--root 1C-Corpus/1C_Corpus`, `--embedding-model intfloat/multilingual-e5-base`, `--batch-size 8`, `--max-tokens` как утверждено в A (500 или 480), без `--no-lock`.

Сбой / перезагрузка: повторить ту же команду — resume + upsert по тем же id.

Гейты после каждой коллекции:

- dim = 768;
- `collection.count()` согласован с полем `chunks` чекпоинта;
- id с ожидаемым префиксом — **сэмпл** (~100 случайных id), не полный перебор на 100k+ записях;
- пустых документов нет;
- после XML: count `code_proc` не меньше, чем был до старта XML.

### Фаза C — фиксация

`build.md` + `state.json`: модель, rev, counts, batch, max_tokens, факт wall-clock, пути.

Поиск / BM25 / eval — не в этом плане.

## 6. Команды (шаблон; запуск только после отдельного «да» на фазу)

Каталог запуска: `D:\MCP_1С_homework` — в имени каталога проекта буква **«С» кириллическая** (`MCP_1С_…`); в относительном корне корпуса — латиница: `1C-Corpus\1C_Corpus`. При копировании путей не смешивать.  
Интерпретатор: `D:\MCP_1С_homework\.venv\Scripts\python.exe` (ниже `py`).

Сухой прогон (`--max-tokens` явный, чтобы бюджет не «плавал» с дефолтом скрипта):

```text
py scripts/chunk_corpus_tokens.py --root 1C-Corpus/1C_Corpus --embedding-model intfloat/multilingual-e5-base --max-tokens 500 --out data/local_notes/embedding_e5_base/chunk_dry_bsl.json
py scripts/chunk_xml_corpus_tokens.py --root 1C-Corpus/1C_Corpus --embedding-model intfloat/multilingual-e5-base --max-tokens 500 --out data/local_notes/embedding_e5_base/chunk_dry_xml.json
```

Полная сборка:

```text
py scripts/build_token_index.py --root 1C-Corpus/1C_Corpus --chroma-dir data/chroma_e5_base --collection code_proc --checkpoint data/checkpoints/e5_base_bsl.json --embedding-model intfloat/multilingual-e5-base --batch-size 8 --max-tokens 500
py scripts/build_xml_index.py --root 1C-Corpus/1C_Corpus --chroma-dir data/chroma_e5_base --collection config_xml --checkpoint data/checkpoints/e5_base_xml.json --embedding-model intfloat/multilingual-e5-base --batch-size 8 --max-tokens 500
```

При срабатывании §5.A.5 (`n_over_512` > 0): во **всех четырёх** командах выше заменить `--max-tokens 500` на `--max-tokens 480` и перепрогнать dry-run, затем сборку с тем же бюджетом.

## 7. Риски

| Риск | Митигация |
| --- | --- |
| Порча `code_proc` при сборке XML | Разбор кода в A + пост-гейты count |
| Чанк >512 у лимита E5 | Стоп; по умолчанию max_tokens=480 и повтор dry-run |
| Потеря прогресса при обрыве | Атомарный чекпоинт; resume по смещению в sorted rglob + upsert по id; контракт корпуса не менять |
| Чужая метрика HNSW | Подтвердить `hnsw:space=cosine` в A |
| Расхождение нарезка/encode | Подтвердить `embed_passages` в A |
| OOM на CPU | batch 8; при падении — стоп и отдельное решение |
| Пустые / BOM-only файлы | Нуль чанков — нормально (`files_without_chunks` в dry-run) |

## 8. Вне объёма

- Ресурсный / smoke-прогон для подбора batch
- Переключение MCP / `cordis.patch.yml`
- BM25, hybrid, gold/nDCG
- Карточки `metadata_xml`
- `1C_Corpus_Report\ОтчетПоКонфигурации.txt`
- Удаление существующих индексов
- git commit / push

## 9. Done

1. В `data/chroma_e5_base` — непустые `code_proc` и `config_xml`, dim 768, `hnsw:space=cosine`.
2. Чекпоинты BSL/XML закрывают корпус; поле `chunks` согласовано с count.
3. Запись шла только в пути §2; после XML `code_proc` не уменьшился.
4. Есть `state.json` / `build.md` с фактом.

## 10. Старт работ

Отдельные «да» на фазы: **A** → **B** → **C**.  
Утверждение плана ≠ запуск embedding.
