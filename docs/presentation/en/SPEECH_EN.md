# English talk script — MCP + RAG for 1C

**Talk:** 10 minutes + Q&A
**Pace:** ~140 words per minute. Spoken text is ~1,340 words for the talk plus a ~160-word fallback (1,498 words in total, inside 1,300–1,500).
**Use:** the slide time cues are targets, not a script to race. If a slide runs long, cut the sentence marked "optional" in the deck notes, never the numbers.

---

## Slide 1 — 0:00–0:30 · Title

Good morning. My name is Sergey Matyukhin.

This is a local-first MCP server and retrieval layer for a 1C configuration dump. It searches BSL business logic and configuration metadata. Nothing leaves the machine.

I will show what it does, how I measured it, and what I got wrong.

## Slide 2 — 0:30–1:20 · The problem, and what I built

The problem is a 1C configuration dump: thousands of `.bsl` modules, plus a metadata report that is tens of megabytes.

`grep` finds text. It does not find meaning. And an exact handler name and a question like "how is VAT on advances calculated" need two different search strategies.

Hosted search stacks mean deployment work and sending the code elsewhere. This project goes the other way.

I built a local MCP server over stdio. Nine tools. All retrieval runs on CPU, with e5-small embeddings and Chroma. There is no external API call in the path.

The second half is the part I want to defend: an evaluation contour. Seventy queries, file-level relevance, four independent runs. When I say "search works", I can show the number and the error bar.

## Slide 3 — 1:20–2:00 · The project in numbers

One hundred seventy-nine thousand five hundred eighty-six chunks in the index.

nDCG@10 of zero point three three nine for the vector mode, the current leader.

Sixty-eight out of sixty-eight on the router regression set.

And ninety-nine point seven percent Hit@3 for metadata BM25.

The stack is deliberately boring. e5-small at three hundred eighty-four dimensions. Chroma with HNSW cosine. A hand-written BM25 inverted index. Reciprocal Rank Fusion with k equals sixty. An optional cross-encoder. Python, forty to sixty milliseconds per query on CPU.

## Slide 4 — 2:00–2:50 · Architecture

Two loops.

The top one is offline. I take the dump, chunk it, embed the chunks with the passage prefix, and write two indexes: Chroma for vectors, and a BM25 inverted index for terms.

The bottom loop is the query path. The query gets the query prefix. Cosine returns the top fifty, BM25 returns its top fifty, and RRF merges the two rankings. The cross-encoder can rerank after that, but it is off by default.

BM25 is a global index over every chunk. It is not a reranker over the cosine candidates. If it were, RRF would just be reordering one list twice.

## Slide 5 — 2:50–3:45 · Corpus and chunking

The code side has one hundred fifty-four thousand five hundred sixty chunks, one procedure per chunk. Before that — the old symbolic chunking — I used fixed windows of twelve hundred characters. They cut procedures in the middle, and thirty-two percent of chunks were longer than the model's five-hundred-twelve-token limit.

After procedure chunking, ninety-nine point six percent of the code is covered, and only zero point three percent of chunks are over the limit. Token percentiles are one seventy-nine, four seventy-two, five fifty-four.

The metadata side has twenty-five thousand twenty-six chunks, parsed from the configuration XML. That moved BM25 Hit@3 from ninety-three to ninety-nine point seven percent.

## Slide 6 — 3:45–4:30 · MCP tools

There are nine tools.

`codesearch` and `metadatasearch` are plain cosine. `search` is the unified entry point with a mode switch. `evaluate_retrieval` returns Hit@k, MRR and nDCG for one query.

`search_1c_metadata` is lexical: exact names, handlers, register questions. `search_knowledge_base` is vector, for how-to questions. `search_hybrid` is RRF. `route_query` explains which tool it would pick, and why. And `index_stats` is the diagnostic: it reports the collection sizes, the model and the default mode.

The router is code with tests, not a paragraph inside a prompt. Sixty-eight of sixty-eight on the dev set. The blind set of twenty was seventy-five percent before the patches. I published it: it was below my own eighty percent gate. After the patches it is twenty of twenty, but that set is no longer blind. So a new hold-out is needed.

## Slide 7 — 4:30–5:20 · RRF

Reciprocal Rank Fusion is the central retrieval decision.

You take each ranking, replace the score with one over k plus the rank, and add them up. k equals sixty.

Why this and not a weighted sum? Cosine and BM25 scores live on different scales. RRF uses positions only, so there is nothing to calibrate. BM25 catches exact names. Cosine catches meaning. Fusion keeps both.

On the old corpus this paid off: hybrid zero point two zero nine, against vector zero point one seven six.

Then the corpus changed, and the order flipped.

## Slide 8 — 5:20–6:05 · Reranker

I also built a cross-encoder reranker. flashrank, ONNX, CPU, no PyTorch. ms-marco-MultiBERT-L-12, one hundred twenty-eight tokens. It takes the top twenty RRF candidates.

It made quality worse. On the old corpus, nDCG@10 went from zero point two zero nine to zero point zero six nine. On the current corpus it is zero point one one seven, against zero point three zero one for plain hybrid. And it costs about two point four seconds per query.

My hypothesis: the model is trained on web passages. BSL is long, repetitive, identifier-heavy code.

So the reranker stays as an option, not the default. The losing row stays in the report.

## Slide 9 — 6:05–6:50 · Measurement

On measurement.

The gold set has seventy natural-language queries: thirty curated, forty generated over configuration objects. Three types: exact, semantic, trap. Relevance is scored at the source-file level, not per chunk. Duplicate chunks from the same file are collapsed first.

Why file level? Because chunk-level ground truth marked a correct file as a miss when the right chunk was not the one I had labelled. False zeros.

Each run starts fresh: four runs, mean and standard deviation. Every number in this talk has an error bar.

## Slide 10 — 6:50–7:55 · Results

Here is the current table. Vector leads with nDCG@10 zero point three three nine, plus or minus zero point zero two three. Pearson is zero point three three seven. Hybrid is zero point three zero one. Rerank is zero point one one seven.

I ran a paired comparison of vector and hybrid: delta plus zero point zero three seven, bootstrap ninety-five percent confidence interval from minus zero point zero one five to plus zero point zero eight six, p equals zero point one five four.

That interval crosses zero. So the difference is not statistically significant. I did not flip the default. ADR zero zero zero nine stays proposed.

The two-by-two ablation explains it. With symbolic chunking, hybrid leads: zero point two zero five against zero point one six five. With procedural chunking, vector leads: zero point three three nine against zero point three zero one. The export effect is minus zero point zero one two for vector and minus zero point zero zero four for hybrid — inside run noise. So chunking explains the flip, not the export. One cell is missing: the old dump is gone.

Recall@10 went from zero point three one zero to zero point five two three, so the vector signal got stronger. The lexical signal got weaker, because procedure names and file paths repeat across chunks.

## Slide 11 — 7:55–8:40 · Decisions and limitations

Nine decisions are written down as ADRs. RRF over weighted Pearson. Reranker optional. File-level relevance. Deduplicated metrics. Gold-set v2. Routing as tested code. Procedure chunking. XML metadata. And the corpus change, still proposed.

HNSW is rebuilt on every start, so cold start is about sixty to seventy seconds. The gold set is partly synthetic, so the absolute numbers are modest. There is a single writer, no parallel indexing. No incremental indexing; a full rebuild takes hours. And the routing gain is inside the noise, so it is not proven.

## Slide 12 — 8:40–9:40 · What is done, and what comes next

To sum up.

There is an MCP server with nine tools over a local index of one hundred seventy-nine thousand chunks. Retrieval is e5 cosine plus global BM25, fused with RRF, with an optional cross-encoder. The evaluation loop reports file-level metrics over four runs. And the engineering side has seventy-two tests, ruff clean, mypy over twenty-two modules, and a complexity gate at one hundred forty-three blocks, average four point three seven.

Next: finish the ablation, then decide the default. Build a new hold-out for the router. Add incremental indexing. Try code-oriented embeddings and a code reranker. Label a hundred more queries by hand.

A retrieval tool is only as good as the evidence behind it. Every decision in this project has a number, an error bar, or an ADR next to it.

Thank you. I am happy to take questions.

---

## 60-second fallback summary - use if the demo or laptop fails

I built a local MCP server and retrieval layer for a 1C configuration dump. The index holds one hundred seventy-nine thousand five hundred eighty-six chunks: BSL procedures and metadata objects.

Search fuses e5-small cosine and a global BM25 index with Reciprocal Rank Fusion at k equals sixty. An optional cross-encoder hurt quality, so it is off by default.

Quality is measured on seventy natural-language queries, file-level, four runs. Vector leads at nDCG@10 zero point three three nine; hybrid is zero point three zero one. The paired difference is plus zero point zero three seven, but the interval crosses zero, so the default is unchanged and the decision is still proposed.

Nine MCP tools, a tested router, seventy-two tests, ruff clean, mypy over twenty-two modules and a complexity gate in CI. Forty to sixty milliseconds per query on CPU, and everything runs locally.

The interesting part is knowing, with an error bar, when a change is real and when it is noise.