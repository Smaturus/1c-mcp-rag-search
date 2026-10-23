"""Render the three fallback demo screenshots from the LIVE MCP server capture.

Run:  python docs/presentation/en/generate_screenshots.py
Writes docs/presentation/en/screenshots/:
  01_tools_list.png, 02_search_hybrid.png, 03_index_stats.png

The text below is copied verbatim from a real capture made on 2026-10-23
against the current corpus (see screenshots/live_capture_2026-10-23.txt):
  MCP_1C_CHROMA_DIR=<copy of data/chroma_xml>
  MCP_1C_BM25_DIR=data/bm25_all
  MCP_1C_CODE_COLLECTION=code_proc
  MCP_1C_METADATA_COLLECTION=metadata_xml
Only long chunk excerpts are shortened (marked with an ellipsis) and absolute
filesystem paths are redacted, as labelled inside each image.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent / "screenshots"

FONT_DIR = Path(r"C:\Windows\Fonts")
FONT_REG = FONT_DIR / "consola.ttf"
FONT_BOLD = FONT_DIR / "consolab.ttf"

BG = (13, 20, 29)
BAR = (24, 35, 48)
EDGE = (48, 64, 82)
TEXT = (226, 232, 240)
DIM = (118, 137, 156)
GREEN = (126, 214, 141)
CYAN = (106, 190, 230)
YELLOW = (238, 208, 122)

SIZE = 21
LINE_H = 34
PAD_X = 34
PAD_TOP = 24
BAR_H = 56
FOOT_H = 44


def font(size: int = SIZE, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REG), size)


def line(*spans, bold=False):
    """One text line as (spans, bold) where a span is (text, colour)."""
    return ([span for span in spans], bold)


def draw_spans(draw, x, y, spans, fnt) -> int:
    for text, color in spans:
        draw.text((x, y), text, font=fnt, fill=color)
        x += int(fnt.getlength(text))
    return x


def render(name: str, title: str, rows, footer: str, *,
           cols: list[int] | None = None, size: int = SIZE) -> Path:
    """rows: flowing lines, or cell lines when `cols` (absolute x per cell) is given."""
    bold = font(size, True)
    regular = font(size)
    line_h = int(size * LINE_H / SIZE)

    def row_width(item) -> int:
        spans, bold_used = item
        fnt = bold if bold_used else regular
        if cols is not None and len(spans) == len(cols):
            return cols[-1] + int(fnt.getlength(spans[-1][0]))
        return sum(int(fnt.getlength(text)) for text, _ in spans)

    width = max(row_width(item) for item in rows) + PAD_X + 18
    width = max(width, 1180)
    height = BAR_H + PAD_TOP + len(rows) * line_h + FOOT_H

    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=16, fill=BG,
                           outline=EDGE, width=1)
    draw.rounded_rectangle([0, 0, width - 1, BAR_H], radius=16, fill=BAR)
    draw.rectangle([0, BAR_H - 16, width - 1, BAR_H], fill=BAR)
    draw.line([0, BAR_H, width, BAR_H], fill=EDGE, width=1)

    for index, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        cx = 26 + index * 24
        draw.ellipse([cx - 7, BAR_H // 2 - 7, cx + 7, BAR_H // 2 + 7], fill=color)

    draw.text((110, BAR_H // 2), title, font=font(size - 2), fill=DIM, anchor="lm")

    y = BAR_H + PAD_TOP
    for spans, bold_used in rows:
        fnt = bold if bold_used else regular
        if cols is not None and len(spans) == len(cols):
            for col_x, (text, color) in zip(cols, spans):
                draw.text((col_x, y), text, font=fnt, fill=color)
        else:
            draw_spans(draw, PAD_X, y, spans, fnt)
        y += line_h

    draw.line([PAD_X, height - FOOT_H, width - PAD_X, height - FOOT_H],
              fill=EDGE, width=1)
    draw.text((PAD_X, height - FOOT_H + 2), footer, font=font(size - 3), fill=DIM)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    img.save(path)
    return path


# --------------------------------------------------------------------- 1. tools
TOOLS = [
    ("codesearch", "query, limit",
     "Search 1C BSL code in object/form/common modules."),
    ("metadatasearch", "query, limit",
     "Search 1C metadata objects, fields and types in the configuration report."),
    ("evaluate_retrieval", "query, k, collection, relevant_sources, null_pairs",
     "Evaluate retrieval quality for one query (quality metrics)."),
    ("search", "query, top_k, mode, collection",
     "Unified retrieval: vector | hybrid (RRF) | hybrid_rerank | pearson (baseline)."),
    ("search_1c_metadata", "query, limit",
     "Grep-like lexical search for exact 1C names, handlers and relations."),
    ("search_knowledge_base", "query, limit",
     'Semantic (vector) search for "how to" and troubleshooting questions.'),
    ("search_hybrid", "query, limit",
     "Hybrid search (cosine + BM25 via RRF) for mixed or requirement queries."),
    ("route_query", "query",
     "Explain which search tool to use for a query, and why."),
    ("index_stats", "-",
     "Report which index the server is actually serving (paths, counts, model)."),
]


def build_tools_list() -> Path:
    size = 19
    cw = font(size).getlength("M")
    name_x = PAD_X
    args_x = name_x + int(cw * (max(len(t[0]) for t in TOOLS) + 2))
    desc_x = args_x + int(cw * (max(len(t[1]) for t in TOOLS) + 2))
    cols = [name_x, args_x, desc_x]

    lines = [
        line(("$ ", GREEN), ("mcp-cli tools/list --server 1c-mcp-rag-search", TEXT)),
        line(("", TEXT)),
        ([("tool", CYAN), ("arguments", CYAN), ("description", CYAN)], True),
    ]
    for name, args, description in TOOLS:
        lines.append(([(name, YELLOW), (args, TEXT), (description, DIM)], False))
    lines.append(line(("", TEXT)))
    lines.append(line(("9 tools", CYAN),
                      ("  |  transport: stdio  |  live server, 23 Oct 2026", DIM)))
    return render(
        "01_tools_list.png",
        "mcp-cli — tools/list · live MCP server · 9 tools",
        lines,
        "Live tools/list output. Arguments are the real inputSchema property names.",
        cols=cols,
        size=size,
    )


# -------------------------------------------------------- 2. route + hybrid search
def build_search() -> Path:
    query = "как начислить НДС с авансов полученных"
    lines = [
        line(("$ ", GREEN), ("route_query(", TEXT), (f'"{query}"', YELLOW), (")", TEXT)),
        line(("route=hybrid", TEXT)),
        line(("tool=", DIM), ("search_hybrid", TEXT)),
        line(("confidence=", DIM), ("0.4", TEXT)),
        line(("reason=", DIM),
             ("недостаточно признаков; по правилу «сомневаешься — hybrid»", DIM)),
        line(("", TEXT)),
        line(("$ ", GREEN), ("search_hybrid(", TEXT), (f'"{query}"', YELLOW),
             (", limit=3)", TEXT)),
        line(("", TEXT)),
        line(("[1] ", DIM),
             ("source=Documents/ФормированиеЗаписейКнигиПродаж/Ext/ObjectModule.bsl", CYAN),
             ("  chunk=20  ", DIM), ("score=0.0164", GREEN),
             ("  vector_rank=1.0  bm25_rank=0.0", DIM)),
        line(("    // file: Documents/ФормированиеЗаписейКнигиПродаж/Ext/ObjectModule.bsl"
              " · proc: ЗаполнитьРазделАвансы", DIM)),
        line(("    Процедура ЗаполнитьРазделАвансы() Экспорт", TEXT)),
        line(("    Таблица_НДСcАвансов = ЗаполнитьНДСНачисленныйДаннымиПоАвансам();", TEXT)),
        line(("    …", DIM)),
        line(("---", EDGE)),
        line(("[2] ", DIM),
             ("source=Documents/СчетФактураВыданный/Ext/ObjectModule.bsl", CYAN),
             ("  chunk=157  ", DIM), ("score=0.0164", GREEN),
             ("  vector_rank=0.0  bm25_rank=1.0", DIM)),
        line(("    // file: Documents/СчетФактураВыданный/Ext/ObjectModule.bsl"
              " · proc: ДвиженияПоРегистрам", DIM)),
        line(("    Процедура ДвиженияПоРегистрам(СтруктураШапкиДокумента,", TEXT)),
        line(("        ТаблицаПоДокументамОснованиям, Отказ, Заголовок)", TEXT)),
        line(("        Если НаАванс Тогда", TEXT)),
        line(("            ДвиженияПоРегиструНДСсАвансов(...);", TEXT)),
        line(("    …", DIM)),
        line(("---", EDGE)),
        line(("[3] ", DIM), ("source=Document/АвансовыйОтчет.xml", CYAN),
             ("  chunk=?  ", DIM), ("score=0.0164", GREEN),
             ("  vector_rank=1.0  bm25_rank=0.0", DIM)),
        line(("    Документ · АвансовыйОтчет", TEXT)),
        line(("    Реквизиты: ВалютаДокумента, Комментарий, КурсДокумента,", TEXT)),
        line(("        КратностьДокумента, Организация, … (metadata chunk)", TEXT)),
        line(("", TEXT)),
        line(("3 of 3 results", CYAN),
             ("  |  code_proc + metadata_xml  |  cold call 9.4 s incl. model load", DIM)),
    ]
    return render(
        "02_search_hybrid.png",
        "mcp-cli — route_query + search_hybrid · live index",
        lines,
        "Live capture, 23 Oct 2026. Excerpts shortened with …; scores are raw RRF values (1/(60+rank)).",
    )


# ------------------------------------------------------------- 3. index stats
def build_index_stats() -> Path:
    rows = [
        ("chroma_dir", "data/chroma_xml", TEXT),
        ("bm25_dir", "data/bm25_all", TEXT),
        ("embedding_model", "intfloat/multilingual-e5-small", YELLOW),
        ("default_mode", "hybrid", YELLOW),
        ("collection[code_proc]", "154560", CYAN),
        ("collection[metadata_xml]", "25026", CYAN),
        ("collection[code]", "0", DIM),
        ("bm25_index_present", "True", GREEN),
    ]
    lines = [line(("$ ", GREEN), ("index_stats()", TEXT)), line(("", TEXT))]
    for key, value, color in rows:
        lines.append(line((f"{key}=", DIM), (value, color)))
    lines.append(line(("", TEXT)))
    lines.append(line(("# live capture 2026-10-23; absolute paths redacted", DIM)))
    lines.append(line(("# collection[code]=0 is a leftover empty collection in this database", DIM)))
    lines.append(line(("# cold open 69.0 s (HNSW rebuilt from WAL)  |  179,586 chunks", DIM)))
    return render(
        "03_index_stats.png",
        "mcp-cli — index_stats · live index · current corpus",
        lines,
        "Live index_stats() from the running server. The server reports the index it actually serves.",
    )


def main() -> int:
    paths = [build_tools_list(), build_search(), build_index_stats()]
    for path in paths:
        print("saved:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
