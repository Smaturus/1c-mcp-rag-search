# Механизм тиков прогресса индексации (для новых чатов)

**Проект:** `D:\MCP_1С_homework`  
**Назначение:** каждые ~10 минут агент **пишет прогресс в чат** (не «смотрите файл»).  
**Основание:** Cursor skill **Loop** → *Monitored shell output (local)* → Fixed schedule.

Правило Cursor (подхватывается автоматически): `.cursor/rules/index-progress-ticks.mdc` (`alwaysApply: true`).

---

## Владелец чата (только чат переиндексации)

Stop-hook раньше слал `INDEX_CHAT_TICK` в **любой** чат workspace, где остановился агент. Теперь followup только если `conversation_id` совпадает с владельцем.

| Роль | Команда / артефакт |
| ---- | ------------------ |
| Запомнить id чата | `beforeSubmitPrompt` → `.cursor/hooks/index_chat_capture_id.py` → `data/index_chat_last_conversation.json` |
| Привязать тики к чату | `python scripts/index_tick_ctl.py claim` (в чате переиндексации, после любого сообщения пользователя) |
| Снять привязку | `python scripts/index_tick_ctl.py unclaim` |
| Файл владельца | `data/index_chat_owner.json` |

Без `claim` stop-hook **молчит** во всех чатах. Чужие чаты не постят прогресс и не делают arm.

**Важно:** в `index_chat_owner.json` должен быть **реальный** `conversation_id` из Cursor (после сообщения в чате переиндексации → `claim`). Заглушка `reindex-chat-uuid` followup **не** открывает — иначе инъекции прогресса сыпались бы во все чаты.

---

## Канон (что запускать) — one-shot + stop-hook

Вечный `index_tick_loop.py` **устарел** (умирает с сессией агента). Рабочая схема:


| Роль | Команда / артефакт |
| ---- | ------------------ |
| One-shot arm (10 мин → sentinel → exit) | `…\python.exe -u scripts\index_chat_arm_once.py --interval 600` |
| Notify | `notify_on_output.pattern` = `^AGENT_LOOP_TICK_index` |
| Страховка | `.cursor/hooks.json` → `stop` → `.cursor/hooks/index_chat_stop.py` (`followup_message`, только owner) |
| Проверка | `python scripts/index_tick_ctl.py status` |
| Claim | `python scripts/index_tick_ctl.py claim` |
| Стоп | `python scripts/index_tick_ctl.py stop` |
| Сразу тик | `python -u scripts/index_chat_arm_once.py --now` |


Перед arm: `status` → если `arm_alive=True`, второй **не** запускать. Arm только из чата-владельца.

На каждый wake: пост `summary_ru` в чат → снова один arm.

---



## Как агент стартует (Cursor Shell) — в чате переиндексации

1. `python scripts/index_tick_ctl.py claim` (если ещё нет owner / `this_submit_is_owner=False`)
2. `python scripts/index_tick_ctl.py status`
3. Сразу `summary_ru` в чат (`--now` или лог)
4. Если индекс не finished и arm не жив — **один** background Shell arm + notify
5. «Следующий автотик ~10 мин»
6. **Не** использовать: вечный loop, PowerShell `while`, `Start-Process -Hidden`

Почему one-shot: процесс короткий; после wake агент обязан переarm. Stop-hook поднимает **только** чат-владелец, если arm умер или тик просрочен.

---



## Формат wake (sentinel)

Строка stdout (начало строки = sentinel):

```text
AGENT_LOOP_TICK_index {"prompt":"...","summary_ru":"17:42 Индекс: code …","finished":false,...}

`summary_ru` всегда начинается с локального времени `hh:mm`.
```

При завершении индексации дополнительно:

```text
AGENT_LOOP_TICK_index_DONE
```



### На каждый wake агент обязан

1. Взять `summary_ru` из JSON (или `data/index_status.txt`)
2. **Написать эту строку в чат пользователю** (строка начинается с точного локального времени `hh:mm`)
3. Не отправлять пользователя в файлы как основной канал
4. Если `finished=true`: `index_tick_ctl.py stop`, обновить `progress.md`, сообщить «готово»
5. Пока луп жив — не создавать второй луп

---



## Файлы механизма


| Путь                         | Назначение                                             |
| ---------------------------- | ------------------------------------------------------ |
| `scripts/index_tick_loop.py` | Единственный луп (sleep → emit) + PID lock             |
| `scripts/index_progress.py`  | Разбор `data/index_full.log`, ETA, stall, `summary_ru` |
| `scripts/index_tick_ctl.py`  | `status` / `stop`                                      |
| `data/index_tick.pid`        | Lock одного процесса                                   |
| `data/index_tick_state.json` | Последний тик / delta                                  |
| `data/index_status.txt`      | Текстовый статус                                       |
| `data/INDEX_PROGRESS.md`     | Зеркало (запасное, не вместо чата)                     |
| `data/index_full.log`        | Лог индексатора                                        |


**Устарело (не использовать):** `arm_index_tick.py`, `index_tick_daemon.py`, `Get-Content -Wait` на wake-log.

---



## Индексация (фон, отдельно от тиков)

```powershell
cd D:\MCP_1С_homework
$env:PYTHONUNBUFFERED='1'
.\.venv\Scripts\python.exe -u -m mcp_1c_homework.indexer --reset >> data\index_full.log 2>&1
```

Лог: `data/index_full.log`. Тики только читают лог и пишут в чат.

---



## Чеклист для нового чата

- [ ] Прочитан этот файл и `.cursor/rules/index-progress-ticks.mdc`
- [ ] В чате переиндексации: `index_tick_ctl.py claim` → `this_submit_is_owner=True`
- [ ] `index_tick_ctl.py status` — arm жив или запущен один раз правильно
- [ ] Пользователь получает прогресс **только в этом чате** на каждый notify / stop-hook
- [ ] Другие чаты проекта не получают `INDEX_CHAT_TICK`
- [ ] Нет дублей лупов / Hidden / PS while
- [ ] По `finished` — stop + `unclaim` + запись в `progress.md`

---



## Краткий контекст проекта

- Свой MCP без Docker: `mcp-1c-homework` (stdio) в `mcp.json`
- Индекс: Chroma + `intfloat/multilingual-e5-small` в `data/chroma/`
- Источники: локальная выгрузка `.bsl` и отчёт конфигурации (пути через env)
- Выгрузку 1С и `data/chroma/` в публичный git не коммитить

