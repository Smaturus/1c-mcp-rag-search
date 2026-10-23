"""Build the English presentation deck (12 slides) for the 1C MCP + RAG project.

Run:  python docs/presentation/en/build_deck_en.py
Writes: docs/presentation/en/1c-mcp-rag-search-slides-en.pptx

Visual language is ported from the Russian deck (docs/presentation/build_deck.py):
16:9, NAVY/ACCENT palette, slide_frame / notes / card / table helpers, same footer.
All text is written in English for a 10-minute talk (~40-50 s per slide).
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

OUT = Path(__file__).resolve().parent / "1c-mcp-rag-search-slides-en.pptx"

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


def _hanging(para, *, mar: int = 240000, indent: int = -240000) -> None:
    """Hanging indent so wrapped bullet lines align under the text, not the dot."""
    p_pr = para._p.get_or_add_pPr()
    p_pr.set("marL", str(mar))
    p_pr.set("indent", str(indent))


def add_para(
    box,
    content: str,
    *,
    size: int = 18,
    bold: bool = False,
    color: RGBColor = INK,
    bullet: bool = True,
    space_before: int = 5,
    line_spacing: float = 1.0,
) -> None:
    frame = box.text_frame
    para = frame.add_paragraph()
    para.space_before = Pt(space_before)
    para.space_after = Pt(3)
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
         size=28, bold=True, color=NAVY)
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


def card(slide, x, y, w, h, value: str, label: str, *, value_size: int = 32) -> None:
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
    r1.font.size = Pt(value_size)
    r1.font.bold = True
    r1.font.name = "Calibri"
    r1.font.color.rgb = NAVY
    second = tf.add_paragraph()
    second.space_before = Pt(2)
    r2 = second.add_run()
    r2.text = label
    r2.font.size = Pt(12)
    r2.font.name = "Calibri"
    r2.font.color.rgb = GRAY


def stage(slide, x, y, w, h, label: str, *, fill=LIGHT, line=ACCENT,
          fg=INK, size: int = 12, bold: bool = False) -> None:
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


def table(slide, rows, x, y, w, h, *, col_widths=None, header=True, size=13,
          row_heights=None) -> None:
    shape = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, h)
    tbl = shape.table
    if col_widths:
        total = sum(col_widths)
        for idx, cw in enumerate(col_widths):
            tbl.columns[idx].width = Emu(int(w * cw / total))
    if row_heights:
        for idx, rh in enumerate(row_heights):
            tbl.rows[idx].height = Inches(rh)
    for r_i, row in enumerate(rows):
        for c_i, value in enumerate(row):
            cell = tbl.cell(r_i, c_i)
            cell.text = ""
            tf = cell.text_frame
            tf.margin_left = Inches(0.08)
            tf.margin_right = Inches(0.08)
            tf.margin_top = Inches(0.02)
            tf.margin_bottom = Inches(0.02)
            para = tf.paragraphs[0]
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


def strip(slide, y, h, content, *, size=13, color=INK) -> None:
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, MARGIN, y,
                                 W - 2 * MARGIN, h)
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = RGBColor(0xD3, 0xE2, 0xEC)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.2)
    tf.margin_right = Inches(0.2)
    tf.margin_top = Inches(0.06)
    tf.margin_bottom = Inches(0.06)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.paragraphs[0].alignment = PP_ALIGN.LEFT
    run = tf.paragraphs[0].add_run()
    run.text = content
    run.font.size = Pt(size)
    run.font.name = "Calibri"
    run.font.color.rgb = color


def divider(slide, x, y, h) -> None:
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Emu(9525), h)
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(0xD3, 0xE2, 0xEC)
    line.line.fill.background()


def build() -> Path:
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H

    # ---------------------------------------------------------------- 1. Title
    slide = new_slide(prs)
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    text(slide, MARGIN, Inches(1.75), W - 2 * MARGIN, Inches(1.4),
         "MCP + RAG for 1C", size=50, bold=True, color=WHITE)
    text(slide, MARGIN, Inches(3.0), W - 2 * MARGIN, Inches(1.3),
         "Local-first search over BSL business logic and configuration metadata:\n"
         "hybrid BM25 + cosine via RRF, and an honest evaluation",
         size=22, color=RGBColor(0xCF, 0xE4, 0xEF), line_spacing=1.15)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, Inches(4.62),
                                  Inches(2.2), Inches(0.05))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(0x6F, 0xB6, 0xD6)
    line.line.fill.background()
    text(slide, MARGIN, Inches(4.9), Inches(8), Inches(0.4),
         "Sergey Matyukhin", size=20, bold=True, color=WHITE)
    text(slide, MARGIN, Inches(5.35), Inches(11.5), Inches(1.2),
         "github.com/Smaturus/1c-mcp-rag-search\n"
         "MCP Python SDK  ·  Chroma  ·  multilingual-e5-small  ·  BM25  ·  RRF  ·  flashrank",
         size=13, color=RGBColor(0xB9, 0xD7, 0xE6), line_spacing=1.2)
    text(slide, MARGIN, Inches(6.45), Inches(11.5), Inches(0.5),
         "English talk · 10 minutes + Q&A",
         size=13, color=RGBColor(0x8F, 0xB8, 0xCF), line_spacing=1.2)
    notes(slide,
          "Thirty seconds: who I am, what the artefact is. "
          "One sentence to plant the thesis - this is not only 'search works', "
          "it is measured quality plus documented decisions. "
          "Duration is on screen so I keep the pace.")

    # ------------------------------------------------- 2. Problem / what I built
    slide = slide_frame(prs, "The problem, and what I built", 2)
    text(slide, MARGIN, Inches(1.2), Inches(5.9), Inches(0.5),
         "Problem", size=20, bold=True, color=NAVY)
    left = text(slide, MARGIN, Inches(1.7), Inches(5.85), Inches(4.6), "x", size=16,
                anchor=MSO_ANCHOR.MIDDLE)
    left.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "A 1C configuration dump is thousands of .bsl modules plus a multi-megabyte metadata report",
        "grep matches text, not meaning; an exact handler name and an intent question need different tools",
        "Hosted search stacks mean deployment work and data leaving the machine",
        "An LLM agent needs a tool whose quality is measured, not assumed",
    ]:
        add_para(left, item, size=16)
    divider(slide, Inches(6.65), Inches(1.3), Inches(4.95))
    text(slide, Inches(6.95), Inches(1.2), Inches(5.85), Inches(0.5),
         "What I built", size=20, bold=True, color=NAVY)
    right = text(slide, Inches(6.95), Inches(1.7), Inches(5.85), Inches(4.6), "x", size=16,
                 anchor=MSO_ANCHOR.MIDDLE)
    right.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "A local MCP server over stdio: 9 tools, including an explainable router",
        "Fully local retrieval: e5-small embeddings plus Chroma, CPU only, no external APIs",
        "Hybrid search: cosine and a global BM25 index, fused with Reciprocal Rank Fusion",
        "Measured: gold-set v2, 70 queries, file-level relevance, 4 independent runs",
    ]:
        add_para(right, item, size=16)
    notes(slide,
          "Frame the task as engineering, not as a demo. There are two equal halves: "
          "the MCP server an agent can call, and the retrieval pipeline with an evaluation "
          "contour. The second half is what the talk defends.")

    # ---------------------------------------------------------- 3. Project facts
    slide = slide_frame(prs, "The project in numbers", 3)
    cards = [
        ("179,586", "chunks in the index"),
        ("0.339", "nDCG@10, vector (leader)"),
        ("68/68", "router regression cases"),
        ("99.7 %", "metadata BM25 Hit@3"),
    ]
    x = MARGIN
    for value, label in cards:
        card(slide, x, Inches(1.3), Inches(2.9), Inches(1.65), value, label,
             value_size=30)
        x += Inches(3.08)
    text(slide, MARGIN, Inches(3.3), W - 2 * MARGIN, Inches(0.4),
         "Stack", size=20, bold=True, color=NAVY)
    box = text(slide, MARGIN, Inches(3.8), W - 2 * MARGIN, Inches(2.9), "x", size=15,
               anchor=MSO_ANCHOR.MIDDLE)
    box.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "Embeddings: intfloat/multilingual-e5-small, 384 dimensions, 512-token limit, query:/passage: prefixes",
        "Storage: Chroma with HNSW cosine; lexical side: a custom BM25 inverted index over all chunks",
        "Fusion: Reciprocal Rank Fusion, k = 60; optional flashrank cross-encoder, off by default",
        "Interface: MCP over stdio, 9 tools; evaluation: 70 queries x 4 configurations, 4 runs each",
        "Engineering: Python 3.11/3.12, 72 unit tests, ruff clean, mypy on 22 modules, complexity gate 143 blocks (avg 4.37, A); 40-60 ms/query",
    ]:
        add_para(box, item, size=15, space_before=9)
    notes(slide,
          "Four numbers to anchor the talk: corpus size, quality of the current leader, "
          "router coverage, metadata hit rate. Then the stack in one breath. "
          "The counts are exact: 154,560 code chunks plus 25,026 metadata chunks.")

    # ----------------------------------------------------------- 4. Architecture
    slide = slide_frame(prs, "Architecture: two loops", 4)
    text(slide, MARGIN, Inches(1.0), Inches(6), Inches(0.4),
         "Offline index build", size=15, bold=True, color=ACCENT)
    idx = [
        ("1C configuration\ndump", 2.35),
        ("chunking\nprocedures", 2.05),
        ("e5\npassage:", 1.45),
        ("Chroma\ncosine", 1.55),
        ("BM25\ninverted index", 1.85),
    ]
    x = MARGIN
    y = Inches(1.42)
    for i, (label, width) in enumerate(idx):
        stage(slide, x, y, Inches(width), Inches(0.85), label, size=12)
        if i < len(idx) - 1:
            arrow(slide, x + Inches(width) + Inches(0.08), y + Inches(0.3),
                  Inches(0.24), Inches(0.25))
        x += Inches(width) + Inches(0.4)

    text(slide, MARGIN, Inches(2.75), Inches(7), Inches(0.4),
         "Online query path (MCP over stdio)", size=15, bold=True, color=ACCENT)
    y2 = Inches(3.18)
    stages = [
        ("query", 1.2),
        ("e5\nquery:", 1.35),
        ("cosine\ntop-50", 1.75),
        ("RRF\nk = 60", 1.35),
        ("rerank\n(optional)", 1.5),
        ("top-k", 1.1),
    ]
    x = MARGIN
    for i, (label, width) in enumerate(stages):
        stage(slide, x, y2, Inches(width), Inches(0.9), label, size=12,
              fill=RGBColor(0xE3, 0xF0, 0xF6))
        if i < len(stages) - 1:
            arrow(slide, x + Inches(width) + Inches(0.06), y2 + Inches(0.32),
                  Inches(0.22), Inches(0.24))
        x += Inches(width) + Inches(0.34)

    stage(slide, Inches(4.25), Inches(4.5), Inches(2.0), Inches(0.75),
          "BM25 top-50", size=12, fill=RGBColor(0xE3, 0xF0, 0xF6))
    arrow(slide, Inches(5.08), Inches(4.36), Inches(0.26), Inches(0.28),
          direction=MSO_SHAPE.UP_ARROW)
    text(slide, MARGIN, Inches(5.55), W - 2 * MARGIN, Inches(1.3),
         "Top: the offline index build. Bottom: the online query path. RRF fuses two independent rankings; "
         "the cross-encoder is an optional extra stage. Everything runs locally: no external APIs, "
         "no data leaves the machine.",
         size=13, color=GRAY, line_spacing=1.15)
    notes(slide,
          "Two loops. The important design point is that BM25 is a global index over all "
          "chunks, not a reranker over cosine candidates - otherwise RRF would be meaningless. "
          "The reranker sits after fusion and is off by default.")

    # ------------------------------------------------------- 5. Corpus / chunking
    slide = slide_frame(prs, "Corpus and chunking", 5)
    text(slide, MARGIN, Inches(1.2), Inches(6.1), Inches(0.5),
         "Procedure-level chunking", size=19, bold=True, color=NAVY)
    left = text(slide, MARGIN, Inches(1.7), Inches(6.0), Inches(4.6), "x", size=14)
    left.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "154,560 code chunks: one procedure or function per chunk (ADR 0007)",
        "Before (symbolic 1200/200 windows): procedures cut mid-body, 32.1% of chunks over the 512-token limit",
        "25,026 metadata chunks parsed from the configuration XML dump",
        "Total: 179,586 chunks in the index",
        "Current gates: 99.6% code coverage; 0.3% over 512 tokens; tokens p50/p99/max = 179 / 472 / 554",
    ]:
        add_para(left, item, size=14)

    divider(slide, Inches(6.75), Inches(1.3), Inches(4.75))
    text(slide, Inches(7.05), Inches(1.2), Inches(5.4), Inches(0.5),
         "Metadata from XML", size=19, bold=True, color=NAVY)
    right = text(slide, Inches(7.05), Inches(1.7), Inches(5.4), Inches(2.6), "x", size=14)
    right.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "25,026 chunks over 5,911 objects: attributes, types and form names",
        "BM25 Hit@3 99.7% against 93.0% for the old text-report source (ADR 0008)",
        "Structured source keeps field names and types inside the chunk text",
    ]:
        add_para(right, item, size=14)
    card(slide, Inches(7.05), Inches(3.9), Inches(5.15), Inches(1.55),
         "93.0% → 99.7%", "BM25 Hit@3 on metadata, text report → XML", value_size=26)
    text(slide, Inches(7.05), Inches(5.65), Inches(5.2), Inches(0.7),
         "Parsed once into JSONL, then indexed like code.",
         size=12, color=GRAY)
    notes(slide,
          "The corpus is the change that moved the metrics. Procedure chunks are coherent "
          "units, so cosine improved and file-level recall went from 0.310 to 0.523. "
          "The same repetition of procedure names and paths weakened the lexical signal. "
          "Metadata came from XML, which lifted BM25 Hit@3 from 93.0% to 99.7%.")

    # ------------------------------------------------------------ 6. MCP tools
    slide = slide_frame(prs, "MCP server: tools for an agent", 6)
    table(
        slide,
        [
            ["Tool", "What it returns / when to use it"],
            ["codesearch(query, limit)", "cosine baseline over BSL code modules"],
            ["metadatasearch(query, limit)", "cosine baseline over configuration metadata"],
            ["search(query, top_k, mode, collection)", "unified entry: vector | hybrid | hybrid_rerank | pearson"],
            ["evaluate_retrieval(query, k, ...)", "Hit@k, MRR, nDCG and a null-tail check for one query"],
            ["search_1c_metadata(query, limit)", "BM25: exact names, handlers, calls, \"who writes to this register\""],
            ["search_knowledge_base(query, limit)", "vector: \"how to\", \"why it fails\", instructions"],
            ["search_hybrid(query, limit)", "RRF fusion for mixed queries; the safe default"],
            ["route_query(query)", "explains route, tool, confidence and reason"],
            ["index_stats()", "diagnostic: which index is served, collection counts, model, default mode"],
        ],
        MARGIN,
        Inches(1.18),
        W - 2 * MARGIN,
        Inches(3.3),
        col_widths=[4.7, 7.5],
        size=11.5,
    )
    text(slide, MARGIN, Inches(4.9), W - 2 * MARGIN, Inches(0.35),
         "How it connects", size=16, bold=True, color=NAVY)
    box = text(slide, MARGIN, Inches(5.27), W - 2 * MARGIN, Inches(1.6), "x", size=12.5)
    box.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "stdio transport: the client launches the server, there is no service to deploy",
        "Routing is code with tests, not instructions inside a prompt: 68/68 on the dev regression set",
        "Blind Part-3 set (20 cases, hash fixed): 75% before patches, published as below the 80% gate; 20/20 after, but that set is dev now - a new hold-out is needed",
    ]:
        add_para(box, item, size=13)
    notes(slide,
          "Nine tools, one contract. The interesting part is not the count, it is that tool "
          "choice is explicit and testable. Be honest about the router: the blind set scored 75% "
          "before the patches, which is below my own gate, so I published it. After the patches "
          "it is 20/20 - but on a set that is no longer blind.")

    # ------------------------------------------------------------- 7. RRF slide
    slide = slide_frame(prs, "Hybrid retrieval: Reciprocal Rank Fusion", 7)
    text(slide, MARGIN, Inches(1.15), W - 2 * MARGIN, Inches(0.8),
         "score(d) = Σ 1 / (k + rank_i(d)),   k = 60",
         size=25, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    text(slide, MARGIN, Inches(2.15), Inches(3.1), Inches(0.4),
         "Why RRF", size=18, bold=True, color=NAVY)
    left = text(slide, MARGIN, Inches(2.6), Inches(5.9), Inches(2.9), "x", size=14)
    left.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "cosine and BM25 produce scores on incomparable scales",
        "RRF uses ranks only, so there is nothing to calibrate",
        "BM25 catches exact names, cosine catches meaning; fusion keeps both",
        "k = 60 damps the first positions so one source cannot dominate",
    ]:
        add_para(left, item, size=14)
    divider(slide, Inches(6.65), Inches(2.2), Inches(3.2))
    text(slide, Inches(6.95), Inches(2.15), Inches(5.8), Inches(0.4),
         "What it bought us (nDCG@10)", size=18, bold=True, color=NAVY)
    right = text(slide, Inches(6.95), Inches(2.6), Inches(5.8), Inches(2.9), "x", size=14)
    right.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "Old corpus: hybrid 0.209, vector 0.176 - hybrid was the leader",
        "Current corpus: vector 0.339, hybrid 0.301",
        "A weighted Pearson variant reaches 0.337, but needs tuned weights",
        "Latency stays at 40-60 ms per query on CPU",
    ]:
        add_para(right, item, size=14)
    strip(slide, Inches(4.95), Inches(1.05),
          "Example: query \"ПередЗаписью\". BM25 lifts modules that contain the exact handler name, "
          "cosine adds related meaning, and RRF merges the two rankings without a weight search.",
          size=13)
    notes(slide,
          "RRF is the central past design decision. We fuse positions, not scores, so no weights "
          "are tuned. On the old corpus this gave a +19% gain over vector. On the current corpus "
          "the order flipped - that is slide 10, and it is recorded in ADR 0009.")

    # ------------------------------------------------------ 8. Reranker (negative)
    slide = slide_frame(prs, "Negative result: the cross-encoder reranker", 8)
    text(slide, MARGIN, Inches(1.2), Inches(6.0), Inches(0.5),
         "What I built", size=19, bold=True, color=NAVY)
    left = text(slide, MARGIN, Inches(1.7), Inches(5.9), Inches(4.4), "x", size=14)
    left.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "flashrank 0.2.10 with an ONNX cross-encoder on CPU, no PyTorch",
        "ms-marco-MultiBERT-L-12, max_length = 128",
        "Input: top-20 RRF candidates; a passage is the file path plus its code",
        "Lazy model load, cache on disk, one extra stage after fusion",
    ]:
        add_para(left, item, size=14)

    divider(slide, Inches(6.65), Inches(1.3), Inches(4.9))
    text(slide, Inches(6.95), Inches(1.2), Inches(5.8), Inches(0.5),
         "What it did to quality", size=19, bold=True, color=NAVY)
    right = text(slide, Inches(6.95), Inches(1.7), Inches(5.8), Inches(4.4), "x", size=14)
    right.text_frame.paragraphs[0].runs[0].text = ""
    add_para(right, "Old corpus: 0.209 → 0.069", size=14, color=RED, bullet=True)
    add_para(right, "Current corpus: 0.117 against 0.301 for plain hybrid", size=14,
             color=RED, bullet=True)
    add_para(right, "Latency ~2.4 s per query on CPU", size=14, bullet=True)
    add_para(right, "Hypothesis: trained on web passages; BSL is long, repetitive, "
                    "identifier-heavy code", size=13, bullet=True)
    add_para(right, "Decision: kept as an option, not the default (ADR 0002)", size=14,
             bullet=True, color=NAVY)
    strip(slide, Inches(4.95), Inches(0.95),
          "The losing configuration stays in the report. Nothing was tuned until the numbers looked good.",
          size=13, color=NAVY)
    notes(slide,
          "This is the honest negative result. The reranker makes quality worse on this domain "
          "and is much slower, so it stays behind a flag. Keeping the losing row in the table is "
          "the point: the pipeline was not massaged until it looked good.")

    # -------------------------------------------------------- 9. Methodology
    slide = slide_frame(prs, "How quality was measured", 9)
    text(slide, MARGIN, Inches(1.2), Inches(6.1), Inches(0.5),
         "Gold-set v2", size=19, bold=True, color=NAVY)
    left = text(slide, MARGIN, Inches(1.7), Inches(6.0), Inches(4.0), "x", size=15)
    left.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "70 natural-language queries: 30 curated plus 40 generated over configuration objects",
        "Three query types: exact, semantic, trap",
        "Relevance is scored at the source-file level, not per chunk",
        "Duplicate chunks from one file are collapsed before scoring (ADR 0004)",
        "Metrics: Hit@1, MRR@10, nDCG@10, Recall@10; four independent runs, a fresh process each",
    ]:
        add_para(left, item, size=15)
    divider(slide, Inches(6.75), Inches(1.3), Inches(4.75))
    text(slide, Inches(7.05), Inches(1.2), Inches(5.4), Inches(0.5),
         "Why this way", size=19, bold=True, color=NAVY)
    right = text(slide, Inches(7.05), Inches(1.7), Inches(5.4), Inches(4.0), "x", size=15)
    right.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "Chunk-level ground truth scored a correct file as a miss: false zeros",
        "File-level relevance stays stable when the chunking changes",
        "Recall@k is normalised by min(|relevant|, k)",
        "std of 0.023 for vector means the result is reproducible, not a lucky run",
        "Each run reloads the index: the numbers are means, not a best-of selection",
    ]:
        add_para(right, item, size=15)
    notes(slide,
          "The methodology matters more than the numbers. The first gold-set version had a defect "
          "in how relevance was scored, it was fixed and written down in ADR 0003. "
          "The spread across four runs is reported everywhere, so nobody has to trust a single run.")
    strip(slide, Inches(5.75), Inches(0.85),
          "Every number in this talk is a mean of four independent runs, reported together with its standard deviation.",
          size=13)

    # ------------------------------------------------------------- 10. Results
    slide = slide_frame(prs, "Results: gold-set v2, 70 queries, k = 10", 10)
    table(
        slide,
        [
            ["Configuration", "Hit@1", "MRR@10", "nDCG@10", "Recall@10"],
            ["vector", "0.150 ± 0.007", "0.268 ± 0.019", "0.339 ± 0.023", "0.523 ± 0.030"],
            ["pearson", "0.179 ± 0.021", "0.270 ± 0.024", "0.337 ± 0.030", "0.511 ± 0.048"],
            ["hybrid (RRF)", "0.129 ± 0.010", "0.240 ± 0.013", "0.301 ± 0.017", "0.482 ± 0.034"],
            ["hybrid + rerank", "0.086", "0.102", "0.117", "0.157"],
        ],
        MARGIN,
        Inches(1.15),
        W - 2 * MARGIN,
        Inches(2.05),
        col_widths=[3.1, 2.3, 2.3, 2.4, 2.3],
        size=12,
    )
    chart_data = CategoryChartData()
    chart_data.categories = ["vector", "pearson", "hybrid", "hybrid_rerank"]
    chart_data.add_series("nDCG@10", (0.339, 0.337, 0.301, 0.117))
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        MARGIN,
        Inches(3.4),
        Inches(5.7),
        Inches(2.45),
        chart_data,
    ).chart
    chart.has_legend = False
    chart.has_title = True
    chart.chart_title.text_frame.text = "nDCG@10 on the current corpus"
    chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    chart.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
    chart.chart_title.text_frame.paragraphs[0].runs[0].font.name = "Calibri"
    plot = chart.plots[0]
    plot.gap_width = 90
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.number_format = "0.000"
    labels.number_format_is_linked = False
    labels.position = XL_LABEL_POSITION.OUTSIDE_END
    labels.font.size = Pt(10)
    labels.font.name = "Calibri"
    series = plot.series[0]
    series.format.fill.solid()
    series.format.fill.fore_color.rgb = ACCENT
    chart.value_axis.minimum_scale = 0.0
    chart.value_axis.maximum_scale = 0.40
    chart.value_axis.has_major_gridlines = False
    chart.value_axis.major_unit = 0.1
    chart.value_axis.tick_labels.font.size = Pt(9)
    chart.value_axis.tick_labels.font.name = "Calibri"
    chart.category_axis.tick_labels.font.size = Pt(10)
    chart.category_axis.tick_labels.font.name = "Calibri"

    text(slide, Inches(6.6), Inches(3.3), Inches(6.2), Inches(0.3),
         "Validation: v2 and independent v3 hold-out", size=12, bold=True, color=NAVY)
    text(slide, Inches(6.6), Inches(3.58), Inches(6.2), Inches(0.9),
         "v2: Δ +0.037, CI [-0.015; +0.086], p = 0.154\n"
         "v3 (71 code queries, 3 runs): vector 0.205 ± 0.014 / hybrid 0.183 ± 0.005\n"
         "v3: Δ +0.023, CI [-0.019; +0.061], p = 0.275 — not significant\n"
         "Both not significant: default hybrid unchanged, ADR 0009 stays proposed.",
         size=9.6, color=INK, line_spacing=1.0)
    text(slide, Inches(6.6), Inches(4.5), Inches(6.2), Inches(0.3),
         "2×2 ablation (nDCG@10, vector / hybrid)", size=12, bold=True, color=NAVY)
    table(
        slide,
        [
            ["2×2 nDCG@10", "vector", "hybrid"],
            ["old export · symbolic", "0.176", "0.209"],
            ["old export · procedural", "—", "—"],
            ["current export · symbolic", "0.165 ± 0.013", "0.205 ± 0.007"],
            ["current export · procedural", "0.339 ± 0.023", "0.301 ± 0.017"],
        ],
        Inches(6.6),
        Inches(4.78),
        Inches(6.18),
        Inches(1.12),
        col_widths=[3.1, 1.55, 1.55],
        size=9.6,
    )
    text(slide, Inches(6.6), Inches(6.02), Inches(6.18), Inches(0.75),
         "Chunking explains the flip: hybrid leads under symbolic chunking, vector leads under "
         "procedural. Export effect at symbolic chunking: −0.012 / −0.004, inside run noise.",
         size=9.7, color=INK, line_spacing=1.05)
    text(slide, Inches(6.6), Inches(6.5), Inches(6.18), Inches(0.3),
         "Interaction cell missing (old dump gone); additivity assumed.",
         size=9.4, color=GRAY)
    text(slide, MARGIN, Inches(5.95), Inches(5.8), Inches(0.35),
         "All CPU modes: 40-60 ms per query. Cross-encoder rerank: ~2.4 s.",
         size=10, color=GRAY)
    notes(slide,
          "Read the table top to bottom: vector 0.339, pearson 0.337, hybrid 0.301, "
          "rerank 0.117. The paired test on v2 gives +0.037 with an interval that crosses zero "
          "(p = 0.154). I then ran an independent 100-query v3 hold-out that never touched tuning: "
          "on 71 code queries over three runs vector is 0.205 and hybrid 0.183, delta +0.023, "
          "p = 0.275, again not significant. So the default stays hybrid and ADR 0009 remains proposed. "
          "The 2x2 ablation explains why the leader moved: symbolic chunking favours hybrid, procedural "
          "chunking favours vector, and the export effect is inside run noise.")

    # ------------------------------------------------ 11. Decisions & limitations
    slide = slide_frame(prs, "Decisions and limitations", 11)
    text(slide, MARGIN, Inches(1.15), Inches(6.2), Inches(0.5),
         "Decision log (ADR)", size=18, bold=True, color=NAVY)
    left = text(slide, MARGIN, Inches(1.62), Inches(6.1), Inches(3.9), "x", size=14)
    left.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "0001: RRF over weighted Pearson - better ranking, no weight calibration",
        "0002: reranker stays optional - it hurts quality on BSL",
        "0003: file-level relevance - stable, reproducible metrics",
        "0004: deduplicated metrics - one file counted once",
        "0005: gold-set v2 - natural-language queries, not identifier templates",
        "0006: routing as tested code - tool choice is explainable",
        "0007: procedure chunking - the 512-token overflow is gone",
        "0008: XML metadata - BM25 Hit@3 93.0% → 99.7%",
        "0009: corpus change - leader moved from hybrid to vector (proposed)",
    ]:
        add_para(left, item, size=14)
    divider(slide, Inches(6.75), Inches(1.3), Inches(4.8))
    text(slide, Inches(7.05), Inches(1.15), Inches(5.4), Inches(0.5),
         "Limitations", size=18, bold=True, color=NAVY)
    right = text(slide, Inches(7.05), Inches(1.62), Inches(5.4), Inches(3.9), "x", size=14)
    right.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "HNSW is rebuilt on start: ~60-70 s cold start, a workaround for a known Chroma issue",
        "Gold-set is partly synthetic; absolute metrics are modest",
        "A single writer; no parallel indexing",
        "No incremental indexing: a full rebuild takes hours",
        "v3 hold-out: router 58/100; quoted-name lexical cases fall back to hybrid",
    ]:
        add_para(right, item, size=14)
    notes(slide,
          "Nine ADRs, five honest limitations. As a portfolio piece this is a strength: it shows where "
          "the solution stops. The cold start and the missing incremental indexing are the two "
          "things a production user would hit first.")
    strip(slide, Inches(5.35), Inches(0.85),
          "ADR 0009 is deliberately still proposed: the default changes only after an independent sample confirms the shift.",
          size=13)

    # ------------------------------------------------------------ 12. Takeaways
    slide = slide_frame(prs, "What is done, and what comes next", 12)
    text(slide, MARGIN, Inches(1.15), W - 2 * MARGIN, Inches(0.5),
         "What is done", size=18, bold=True, color=NAVY)
    box = text(slide, MARGIN, Inches(1.6), W - 2 * MARGIN, Inches(2.75), "x", size=15)
    box.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "MCP server with 9 tools over a 179,586-chunk local index",
        "Retrieval: e5 cosine + global BM25 + RRF + an optional cross-encoder",
        "Router as a tested contract: 68/68 on the dev regression set",
        "Evaluation loop: gold-set v2, file-level metrics, 4 runs, mean ± std",
        "Engineering: 72 tests, ruff + mypy (22 modules) + complexity gate (avg 4.37 A), ADR log, CI green",
    ]:
        add_para(box, item, size=15)
    text(slide, MARGIN, Inches(4.45), W - 2 * MARGIN, Inches(0.5),
         "What comes next", size=18, bold=True, color=NAVY)
    box2 = text(slide, MARGIN, Inches(4.9), W - 2 * MARGIN, Inches(1.6), "x", size=15)
    box2.text_frame.paragraphs[0].runs[0].text = ""
    for item in [
        "Finish the 2x2 ablation, then decide the default mode (ADR 0009)",
        "A new hold-out for the router; incremental indexing; get_metadata and find_callers",
        "Code-oriented embeddings; a code reranker; 100+ manually labelled queries",
    ]:
        add_para(box2, item, size=15)
    text(slide, MARGIN, Inches(6.5), W - 2 * MARGIN, Inches(0.5),
         "github.com/Smaturus/1c-mcp-rag-search  ·  smaturus@gmail.com",
         size=14, bold=True, color=ACCENT)
    notes(slide,
          "Close on the thesis: measurable retrieval plus decisions that are written down. "
          "Then the next steps, and where to find the code. Hand over to questions.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


def main() -> int:
    path = build()
    print("saved:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
