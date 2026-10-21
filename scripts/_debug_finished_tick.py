import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import index_tick_loop as m


def fake():
    return {
        "finished": True,
        "summary_ru": "16:00 Индексация ЗАВЕРШЕНА. чанков кода=1, чанков метаданных=1.",
        "phase": "done",
        "code_done": 1,
        "code_total": 1,
    }


m.collect_progress = fake  # type: ignore
m.write_status_files = lambda x: None  # type: ignore
sys.argv = ["x", "--interval", "1"]
raise SystemExit(m.main())
