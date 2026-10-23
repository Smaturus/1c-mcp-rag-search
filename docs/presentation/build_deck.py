"""Собрать презентацию проекта (12 слайдов) в docs/presentation/.

Запуск: python docs/presentation/build_deck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

OUT = Path(__file__).resolve().parent / "1c-mcp-rag-search-slides.pptx"

NAVY = RGBColor(0x0F, 0x3D, 0x5C)
ACCENT = RGBColor(0x1B, 0x7F, 0xAE)
LIGHT = RGBColor(0xEA, 0xF2, 0xF7)
INK = RGBColor(0x1F, 0x2A, 0x33)
GRAY = RGBColor(0x5B, 0x67, 0x70)
GREEN = RGBColor(0x2E, 0x7D, 0x4F)
RED = RGBColor(0xA8, 0x3A, 0x3A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W = Inches(13.333)
H = Inches(7.5)
MARGIN = Inches(0.55)


def new_slide(prs: Presentation) -> object:
    return prs.slides.add_slide(prs.slide_layouts[6])


def text(
    slide,
    x,
    y,
    w,
    h,
    content: str,
    *,
    size: int = 18,
    bold: bool = False,
    color: RGBColor = INK,
    align=PP_ALIGN.LEFT,
    anchor=MSO_ANCHOR.TOP,
    font: str = "Calibri",
    line_spacing: float = 1.0,
) -> object:
    box = slide.shapes.add_textbox(x, y, w, h)
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    para = frame.paragraphs[0]
    para.alignment = align
    para.line_spacing = line_spacing
    run = para.add_run()
    run.text = content
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = font
    run.font.color.rgb = color
    return box


def _hanging(para, *, mar: int = 260000, indent: int = -260000) -> None:
    """Hanging indent so wrapped bullet lines align under the text, not the dot."""
    p_pr = para._p.get_or_add_pPr()
    p_pr.set("marL", str(mar))
    p_pr.set("indent", str(indent))


def add_para(box, content: str, *, size: int = 18, bold: bool = False,
             color: RGBColor = INK, bullet: bool = True, space_before: int = 6,
             line_spacing: float = 1.0) -> None:
    frame = box.text_frame
    para = frame.add_paragraph()
    para.space_before = Pt(space_before)
    para.space_after = Pt(4)
    para.line_spacing = line_spacing
    if bullet:
        _hanging(para)
    run = para.add_run()
    run.text = ("•  " + content) if bullet else content
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = "Calibri"
    run.font.color.rgb = color


def slide_frame(prs: Presentation, title: str, number: int) -> object:
    slide = new_slide(prs)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(0.16))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    text(slide, MARGIN, Inches(0.38), W - 2 * MARGIN, Inches(0.8), title,
         size=30, bold=True, color=NAVY)
    footer = text(slide, MARGIN, H - Inches(0.5), Inches(9), Inches(0.3),
                  "MCP + RAG for 1C  ·  Sergey Matyukhin  ·  2026",
                  size=10, color=GRAY)
    footer.left = MARGIN
    pag = text(slide, W - MARGIN - Inches(0.8), H - Inches(0.5), Inches(0.8), Inches(0.3),
               str(number), size=10, color=GRAY, align=PP_ALIGN.RIGHT)
    pag.left = W - MARGIN - Inches(0.8)
    return slide


def notes(slide, content: str) -> None:
    slide.notes_slide.notes_text_frame.text = content


def card(slide, x, y, w, h, value: str, label: str) -> None:
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = RGBColor(0xD3, 0xE2, 0xEC)
    box.adjustments[0] = 0.08
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.18)
    tf.margin_right = Inches(0.18)
    first = tf.paragraphs[0]
    first.alignment = PP_ALIGN.LEFT
    r1 = first.add_run()
    r1.text = value
    r1.font.size = Pt(34)
    r1.font.bold = True
    r1.font.name = "Calibri"
    r1.font.color.rgb = NAVY
    second = tf.add_paragraph()
    second.space_before = Pt(2)
    r2 = second.add_run()
    r2.text = label
    r2.font.size = Pt(13)
    r2.font.name = "Calibri"
    r2.font.color.rgb = GRAY


def stage(slide, x, y, w, h, label: str, *, fill=LIGHT, line=ACCENT,
          fg=INK, size: int = 13, bold: bool = False) -> None:
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = line
    box.line.width = Pt(1.0)
    box.adjustments[0] = 0.12
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for index, line_text in enumerate(label.split("\n")):
        para = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        para.alignment = PP_ALIGN.CENTER
        run = para.add_run()
        run.text = line_text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = "Calibri"
        run.font.color.rgb = fg


def arrow(slide, x, y, w, h, *, color=ACCENT, direction=MSO_SHAPE.RIGHT_ARROW) -> None:
    shp = slide.shapes.add_shape(direction, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()


def table(slide, rows, x, y, w, h, *, col_widths=None, header=True, size=13) -> None:
    shape = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, h)
    tbl = shape.table
    if col_widths:
        total = sum(col_widths)
        for idx, cw in enumerate(col_widths):
            tbl.columns[idx].width = Emu(int(w * cw / total))
    for r_i, row in enumerate(rows):
        for c_i, value in enumerate(row):
            cell = tbl.cell(r_i, c_i)
            cell.text = ""
            para = cell.text_frame.paragraphs[0]
            run = para.add_run()
            run.text = str(value)
            run.font.size = Pt(size)
            run.font.name = "Calibri"
            run.font.bold = header and r_i == 0
            run.font.color.rgb = WHITE if (header and r_i == 0) else INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if (header and r_i == 0) else (
                LIGHT if r_i % 2 == 0 else WHITE
            )
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE


def build() -> Path:
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H

    # 1. Title
    slide = new_slide(prs)
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    text(slide, MARGIN, Inches(1.9), W - 2 * MARGIN, Inches(1.4),
         "MCP + RAG для 1С", size=50, bold=True, color=WHITE)
    text(slide, MARGIN, Inches(3.15), W - 2 * MARGIN, Inches(1.2),
         "Локальный поиск по BSL-коду и метаданным:\nгибрид BM25 + cosine через RRF и честная оценка качества",
         size=22, color=RGBColor(0xCF, 0xE4, 0xEF), line_spacing=1.15)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, Inches(4.75), Inches(2.2), Inches(0.05))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(0x6F, 0xB6, 0xD6)
    line.line.fill.background()
    text(slide, MARGIN, Inches(5.05), Inches(8), Inches(0.4),
         "Sergey Matyukhin", size=20, bold=True, color=WHITE)
    text(slide, MARGIN, Inches(5.5), Inches(9), Inches(1.2),
         "github.com/Smaturus/1c-mcp-rag-search\nMCP Python SDK  ·  Chroma  ·  multilingual-e5-small  ·  BM25  ·  RRF  ·  flashrank",
         size=13, color=RGBColor(0xB9, 0xD7, 0xE6), line_spacing=1.2)
    notes(slide, "Учебный и портфолио-проект: свой MCP-сервер и RAG-контур для поиска по выгрузке 1С. "
                 "Главная идея: не просто «поиск работает», а измеренное качество и документированные решения.")

    # 2. Problem / solution
    slide = slide_frame(prs, "Задача и что получилось", 2)
    box = text(slide, MARGIN, Inches(1.35), Inches(5.9), Inches(0.5),
               "Проблема", size=20, bold=True, color=NAVY)
    for item in [
        "Выгрузка 1С — тысячи .bsl-модулей и отчёт на десятки мегабайт",
        "Обычный grep не отвечает на смысловые вопросы",
        "Готовые поисковые решения требуют развёртывания и внешних API",
        "LLM-агенту нужен инструмент с предсказуемым качеством",
    ]:
        add_para(box, item, size=15)
    box2 = text(slide, Inches(6.85), Inches(1.35), Inches(5.9), Inches(0.5),
                "Решение", size=20, bold=True, color=NAVY)
    for item in [
        "Свой MCP-сервер (stdio): 8 инструментов, включая маршрутизацию запросов",
        "Полностью локальный контур: e5-small + Chroma, работает на CPU",
        "Гибридный поиск: cosine + BM25 через RRF; метаданные — из XML-выгрузки",
        "Измеримость: gold-set, file-level метрики, разброс по прогонам",
    ]:
        add_para(box2, item, size=15)
    notes(slide, "Формулирую задачу как инженерную: нужен инструмент для агента и способ доказать, "
                 "что он ищет хорошо. Поэтому в проекте две равные части — MCP и retrieval-пайплайн с eval.")

    # 3. Numbers
    slide = slide_frame(prs, "Проект в цифрах", 3)
    cards = [
        ("179 586", "чанков в индексе"),
        ("0.339", "nDCG@10 у vector"),
        ("68/68", "кейсов маршрутизации"),
        ("99,7 %", "BM25 Hit@3 метаданных"),
    ]
    x = MARGIN
    for value, label in cards:
        card(slide, x, Inches(1.5), Inches(2.85), Inches(1.85), value, label)
        x += Inches(3.05)
    box = text(slide, MARGIN, Inches(3.85), W - 2 * MARGIN, Inches(2.6),
               "Стек", size=20, bold=True, color=NAVY)
    for item in [
        "Эмбеддинги: intfloat/multilingual-e5-small (384d, префиксы query: / passage:)",
        "Хранение: Chroma 1.5.9, HNSW cosine; лексика: собственный инвертированный индекс",
        "Гибрид: Reciprocal Rank Fusion (k = 60); reranker: flashrank + ms-marco-MultiBERT-L-12",
        "Интерфейс: MCP stdio (Cursor / ZCode), 8 инструментов; eval: 70 запросов, 4 конфигурации",
    ]:
        add_para(box, item, size=15)
    notes(slide, "Ключевые числа проекта: 179 586 чанков — 154 560 кода (процедурная нарезка) и 25 026 метаданных (XML). Лидер на новом корпусе — vector (ADR 0009).")

    # 4. Architecture
    slide = slide_frame(prs, "Архитектура: два контура", 4)
    text(slide, MARGIN, Inches(1.02), Inches(6), Inches(0.4),
         "Индексация (offline)", size=15, bold=True, color=ACCENT)
    idx = [
        ("Выгрузка 1С\n.bsl + отчёт", 2.5),
        ("chunking", 1.7),
        ("e5\n(passage:)", 1.7),
        ("Chroma\ncosine", 1.7),
        ("BM25 индекс", 1.9),
    ]
    x = MARGIN
    y = Inches(1.45)
    for i, (label, width) in enumerate(idx):
        stage(slide, x, y, Inches(width), Inches(0.85), label, size=12)
        if i < len(idx) - 1:
            arrow(slide, x + Inches(width) + Inches(0.07), y + Inches(0.3),
                  Inches(0.25), Inches(0.25))
        x += Inches(width) + Inches(0.4)

    text(slide, MARGIN, Inches(2.85), Inches(6), Inches(0.4),
         "Поиск (online, MCP stdio)", size=15, bold=True, color=ACCENT)
    y2 = Inches(3.28)
    stages = [
        ("Запрос", 1.5),
        ("e5\n(query:)", 1.5),
        ("Chroma\ncosine top-50", 2.1),
        ("RRF\nk = 60", 1.4),
        ("rerank\n(опция)", 1.6),
        ("top-k", 1.3),
    ]
    x = MARGIN
    for i, (label, width) in enumerate(stages):
        stage(slide, x, y2, Inches(width), Inches(0.9), label, size=12,
              fill=RGBColor(0xE3, 0xF0, 0xF6))
        if i < len(stages) - 1:
            arrow(slide, x + Inches(width) + Inches(0.06), y2 + Inches(0.32),
                  Inches(0.24), Inches(0.24))
        x += Inches(width) + Inches(0.36)

    stage(slide, Inches(4.55), Inches(4.75), Inches(2.1), Inches(0.8),
          "BM25 top-50", size=12, fill=RGBColor(0xE3, 0xF0, 0xF6))
    arrow(slide, Inches(5.44), Inches(4.58), Inches(0.28), Inches(0.3),
          direction=MSO_SHAPE.UP_ARROW)
    text(slide, MARGIN, Inches(5.75), W - 2 * MARGIN, Inches(1.3),
         "Сверху — офлайн-подготовка индексов; ниже — онлайн-путь запроса. RRF объединяет два независимых ранжирования, "
         "reranker подключается опционально. Все данные и вычисления локальные: внешних API нет.",
         size=13, color=GRAY, line_spacing=1.15)
    notes(slide, "Два контура: офлайн индексация и онлайн поиск. Важно: BM25 — глобальный индекс по всем чанкам, "
                 "а не переранжирование кандидатов cosine, иначе RRF был бы бессмысленным.")

    # 5. Indexes
    slide = slide_frame(prs, "Индексы: векторы и лексика", 5)
    text(slide, MARGIN, Inches(1.25), Inches(6), Inches(0.5),
         "Chroma (векторный)", size=20, bold=True, color=NAVY)
    left = text(slide, MARGIN, Inches(1.75), Inches(5.9), Inches(4.8), "заголовок", size=14)
    left.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "179 586 чанков: code 154 560 (процедуры), metadata 25 026 (XML)",
        "e5-small: 384 измерения, лимит 512 токенов на чанк",
        "Проверка: 32,1 % чанков были длиннее лимита — исправлено нарезкой по процедурам (ADR 0007)",
        "Префиксы e5 сохранены: passage: для документов, query: для запросов",
        "HNSW cosine; sync_threshold = 1e6 — защита от бага chroma#7090",
        "Размер базы 2.3 ГБ; старт пересобирает HNSW из WAL (~60–70 с)",
    ]:
        add_para(left, item, size=14)

    text(slide, Inches(6.9), Inches(1.25), Inches(6), Inches(0.5),
         "BM25 (лексический)", size=20, bold=True, color=NAVY)
    right = text(slide, Inches(6.9), Inches(1.75), Inches(5.9), Inches(4.8), "заголовок", size=14)
    right.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "Инвертированный индекс: 219 968 термов, 6,6 млн постингов",
        "Сборка 22 с, 58 МБ на диске, читает документы прямо из sqlite",
        "Токенизация под 1С: кириллица, цифры, коды через дефис (НФ-001)",
        "Скоринг только по постингам запроса — без полного перебора корпуса",
        "Используется как второй источник ранжирования в RRF",
    ]:
        add_para(right, item, size=14)
    notes(slide, "Два независимых индекса: семантика и лексика. BM25 собирается отдельным скриптом, "
                 "не входит в git и пересобирается за 22 секунды.")

    # 6. MCP tools
    slide = slide_frame(prs, "MCP-сервер: инструменты для агента", 6)
    table(
        slide,
        [
            ["Инструмент", "Что делает"],
            ["search_1c_metadata(query, limit)", "grep-подобный BM25: точные имена, проводки, движения"],
            ["search_knowledge_base(query, limit)", "vector: «как сделать», «почему не работает», инструкции"],
            ["search_hybrid(query, limit)", "RRF: лучший для exact-имён; на новом корпусе vector впереди"],
            ["route_query(query)", "объясняет выбор: маршрут, причина, уверенность"],
            ["search / index_stats / evaluate_retrieval", "поиск, состав индекса и метрики по запросу"],
        ],
        MARGIN,
        Inches(1.3),
        W - 2 * MARGIN,
        Inches(3.0),
        col_widths=[5.6, 8.6],
        size=13,
    )
    box = text(slide, MARGIN, Inches(4.5), W - 2 * MARGIN, Inches(2.0),
               "Подключение", size=18, bold=True, color=NAVY)
    for item in [
        "stdio-транспорт: сервер поднимается клиентом (Cursor mcp.json, ZCode mcp.servers)",
        "Маршрутизация — код с тестами: 68 кейсов, 100 %; blind 20/20 после правок",
        "Пример: route_query(\"проводки документа РеализацияТоваровУслуг\") → search_1c_metadata",
    ]:
        add_para(box, item, size=14)
    notes(slide, "Восемь инструментов. Ключевое отличие от обычного MCP-сервера — явный контракт "
                 "маршрутизации: выбор инструмента объясним и проверяется тестами.")

    # 7. RRF
    slide = slide_frame(prs, "Гибридный поиск: Reciprocal Rank Fusion", 7)
    text(slide, MARGIN, Inches(1.3), W - 2 * MARGIN, Inches(0.8),
         "score(d) = Σ 1 / (k + rank_i(d)),   k = 60",
         size=26, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    box = text(slide, MARGIN, Inches(2.3), Inches(6.2), Inches(4.2),
               "Почему RRF", size=18, bold=True, color=NAVY)
    for item in [
        "Cosine и BM25 дают скоры в несравнимых шкалах",
        "RRF использует только ранги — калибровка не нужна",
        "Точные имена ловит BM25, смысл — cosine, RRF складывает оба",
        "k = 60 сглаживает вклад первых позиций",
    ]:
        add_para(box, item, size=15)
    res = text(slide, Inches(7.0), Inches(2.3), Inches(5.8), Inches(4.2),
               "Что дало (nDCG@10)", size=18, bold=True, color=NAVY)
    for item in [
        "Прежний корпус: hybrid 0.209 (+19 % к vector)",
        "Новый корпус: vector 0.339, hybrid 0.301",
        "Причина: процедурные чанки ослабили BM25 (ADR 0009)",
        "Латентность: 50–120 мс/запрос",
    ]:
        add_para(res, item, size=15, color=INK)
    strip = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, MARGIN, Inches(4.62),
                                   W - 2 * MARGIN, Inches(1.0))
    strip.fill.solid()
    strip.fill.fore_color.rgb = LIGHT
    strip.line.color.rgb = RGBColor(0xD3, 0xE2, 0xEC)
    tf = strip.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.2)
    tf.margin_right = Inches(0.2)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para = tf.paragraphs[0]
    para.alignment = PP_ALIGN.LEFT
    run = para.add_run()
    run.text = (
        "Пример. Запрос «ПередЗаписью»: BM25 поднимает модули с точным именем обработчика, "
        "cosine — смысловые совпадения. RRF складывает оба ранжирования без подбора весов."
    )
    run.font.size = Pt(14)
    run.font.name = "Calibri"
    run.font.color.rgb = INK
    notes(slide, "RRF — центральное решение проекта. Складываем не скоры, а позиции; поэтому не нужно подбирать веса. "
                 "На прежнем корпусе прирост был +19 %, на новом лидер сменился — это зафиксировано в ADR 0009.")

    # 8. Reranker
    slide = slide_frame(prs, "Reranker: отрицательный результат, оставленный в отчёте", 8)
    box = text(slide, MARGIN, Inches(1.3), Inches(6.2), Inches(4.4),
               "Как реализован", size=18, bold=True, color=NAVY)
    for item in [
        "flashrank 0.2.10, ONNX на CPU, без PyTorch",
        "ms-marco-MultiBERT-L-12, max_length = 128",
        "Вход: 20 кандидатов RRF; пассаж = путь файла + код",
        "Ленивая загрузка модели, кэш на диске",
    ]:
        add_para(box, item, size=15)
    res = text(slide, Inches(7.0), Inches(1.3), Inches(5.8), Inches(4.4),
               "Результат", size=18, bold=True, color=NAVY)
    add_para(res, "nDCG@10: 0.301 → 0.117 (−61 %)", size=15, color=RED, bullet=True)
    add_para(res, "Латентность: ~2,4 с/запрос (×20 к vector)", size=15, bullet=True)
    add_para(res, "Гипотеза: модель обучена на веб-пассажах, а BSL — длинный повторяющийся код",
             size=14, bullet=True)
    add_para(res, "Решение: режим оставлен опцией; дефолт пересматривается (ADR 0002, 0009)", size=14,
             bullet=True, color=NAVY)
    add_para(res, "Пайплайн не «крутился» под результат: проигравшая конфигурация осталась в таблице",
             size=13, bullet=True, color=GRAY)
    notes(slide, "Это честный отрицательный результат: reranker ухудшает качество на нашем домене. "
                 "Именно он показывает инженерную зрелость: измеряем, фиксируем, не подгоняем.")

    # 9. Evaluation methodology
    slide = slide_frame(prs, "Как измеряли качество", 9)
    box = text(slide, MARGIN, Inches(1.3), Inches(6.2), Inches(4.6),
               "Данные и метрики", size=18, bold=True, color=NAVY)
    for item in [
        "gold-set v2: 70 запросов (30 курируемых + 40 по объектам конфигурации)",
        "Три типа: exact (точное имя), semantic (смысл), trap (ловушка)",
        "Релевантность — на уровне файла-источника, а не отдельного чанка",
        "Метрики: Hit@1, MRR@10, nDCG@10, Recall@10",
        "Каждый прогон — новый процесс: HNSW пересобирается; 4 прогона, mean ± std",
    ]:
        add_para(box, item, size=14)
    right = text(slide, Inches(7.0), Inches(1.3), Inches(5.8), Inches(4.6),
                 "Почему так", size=18, bold=True, color=NAVY)
    for item in [
        "Чанковые эталоны давали ложно-нулевые метрики: правильный файл «не засчитывался»",
        "Файловая релевантность устойчива к смене chunk_size",
        "Recall@k нормирован на min(|relevant|, k)",
        "Разброс ±0.023 у vector — вывод воспроизводим, а не случайная удача",
    ]:
        add_para(right, item, size=14)
    notes(slide, "Методика важнее цифр: мы показываем, как посчитаны метрики и какой у них разброс. "
                 "Первая версия gold-set была с дефектом — исправили и задокументировали в ADR 0003.")

    # 10. Results
    slide = slide_frame(prs, "Результаты на gold-set v2 (70 запросов, k = 10)", 10)
    table(
        slide,
        [
            ["Конфигурация", "Hit@1", "MRR@10", "nDCG@10", "Recall@10", "мс/запрос"],
            ["vector", "0.150 ± 0.007", "0.268 ± 0.019", "0.339 ± 0.023", "0.523 ± 0.030", "~60"],
            ["pearson", "0.179 ± 0.021", "0.270 ± 0.024", "0.337 ± 0.030", "0.511 ± 0.048", "~90"],
            ["hybrid (RRF)", "0.129 ± 0.010", "0.240 ± 0.013", "0.301 ± 0.017", "0.482 ± 0.034", "~70"],
            ["hybrid_rerank", "0.086", "0.102", "0.117", "0.157", "~2400"],
        ],
        MARGIN,
        Inches(1.3),
        W - 2 * MARGIN,
        Inches(2.35),
        col_widths=[3.0, 2.2, 2.2, 2.3, 2.3, 2.2],
        size=12,
    )
    chart_data = CategoryChartData()
    chart_data.categories = ["vector", "pearson", "hybrid", "hybrid_rerank"]
    chart_data.add_series("nDCG@10", (0.339, 0.337, 0.301, 0.117))
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        MARGIN,
        Inches(3.95),
        Inches(8.2),
        Inches(2.9),
        chart_data,
    ).chart
    chart.has_legend = False
    chart.has_title = True
    chart.chart_title.text_frame.text = "nDCG@10 по конфигурациям"
    plot = chart.plots[0]
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.number_format = "0.000"
    labels.number_format_is_linked = False
    labels.position = XL_LABEL_POSITION.OUTSIDE_END
    labels.font.size = Pt(11)
    series = plot.series[0]
    series.format.fill.solid()
    series.format.fill.fore_color.rgb = ACCENT
    text(slide, Inches(9.0), Inches(4.0), Inches(3.8), Inches(2.8),
         "Источник: data/eval/code_proc/report_pipeline.md\n\nvector — лидер на новом корпусе.\nhybrid лучший для exact-имён\n(0.408 против 0.250 по nDCG@10).\n\nReranker ухудшает качество.",
         size=12, color=GRAY, line_spacing=1.15)
    notes(slide, "Таблица и график по новому корпусу. Vector впереди; hybrid остаётся лучшим для exact-запросов; reranker проигрывает.")

    # 11. Decisions and limitations
    slide = slide_frame(prs, "Решения и ограничения", 11)
    text(slide, MARGIN, Inches(1.2), Inches(6.2), Inches(0.5),
         "Decision Log (ADR)", size=18, bold=True, color=NAVY)
    box = text(slide, MARGIN, Inches(1.65), Inches(6.2), Inches(4.6), "x", size=14)
    box.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "0001: RRF вместо взвешенного Pearson — выше метрики, без калибровки весов",
        "0002: reranker оставлен опцией — ухудшает качество на BSL",
        "0003: релевантность на уровне файла — устойчивые и воспроизводимые метрики",
        "0006: маршрутизация запросов — код с тестами вместо текста промпта",
        "0007: нарезка по процедурам — устранён лимит 512 токенов",
        "0008: метаданные из XML — BM25 Hit@3 93,0 → 99,7 %",
        "0009: смена корпуса — лидер hybrid → vector, зафиксировано",
    ]:
        add_para(box, item, size=14)
    text(slide, Inches(7.0), Inches(1.2), Inches(5.8), Inches(0.5),
         "Ограничения", size=18, bold=True, color=NAVY)
    right = text(slide, Inches(7.0), Inches(1.65), Inches(5.8), Inches(4.6), "x", size=14)
    right.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "HNSW не персистится на диск: старт процесса ~60–70 с (защита от chroma#7090)",
        "Gold-set частично синтетический, абсолютные метрики скромные",
        "Один писатель индекса; параллельная индексация не поддерживается",
        "Нет инкрементальной индексации: полная пересборка — часы",
        "Reranker для кода — следующий контур (модель на коде или fine-tune)",
    ]:
        add_para(right, item, size=14)
    notes(
        slide,
        "Открытость про ограничения — плюс на защите: видно, что мы понимаем границы решения.",
    )

    # 12. Takeaways
    slide = slide_frame(prs, "Итоги и развитие", 12)
    box = text(slide, MARGIN, Inches(1.3), W - 2 * MARGIN, Inches(3.4),
               "Что сделано", size=18, bold=True, color=NAVY)
    for item in [
        "MCP-сервер с 8 инструментами и локальным индексом на 179 тыс. чанков",
        "Гибридный retrieval: глобальный BM25 + RRF + опциональный cross-encoder",
        "Маршрутизация: контракт с тестами, 68 кейсов + blind 20/20",
        "Eval-контур: gold-set v2 (70 запросов), file-level метрики, mean ± std",
        "Инженерная обвязка: 72 теста, ruff, mypy, complexity-гейт, CI, ADR, CHANGELOG",
    ]:
        add_para(box, item, size=15)
    box2 = text(slide, MARGIN, Inches(4.55), W - 2 * MARGIN, Inches(1.6),
                "Развитие", size=18, bold=True, color=NAVY)
    for item in [
        "Ablation нарезок и веса RRF; затем смена дефолта на vector",
        "Инкрементальная индексация и структурные инструменты (get_metadata, find_callers)",
        "Код-ориентированные эмбеддинги; reranker на коде; ручная разметка 100+ запросов",
    ]:
        add_para(box2, item, size=15)
    text(slide, MARGIN, Inches(6.35), W - 2 * MARGIN, Inches(0.6),
         "github.com/Smaturus/1c-mcp-rag-search  ·  smaturus@gmail.com",
         size=14, bold=True, color=ACCENT)
    notes(slide, "Финальный слайд: что сделано, что дальше, где посмотреть код. "
                 "Главный вывод — измеримость: каждое решение подтверждено цифрами и записано в ADR.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


def main() -> int:
    path = build()
    print("saved:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
