# MP3 Reflection

**Author:** Haimanty Biswas  
**Project:** Week 12 — Grounded Q&A (MP3)  
**Corpus:** Five Sherlock Holmes story summaries (`corpus/*.txt`), same texts as Week 9 MP2, indexed to **73 chunks** in Qdrant collection `mp3_sherlock`.

**Graded artifact:** `mp3_eval.txt` is the full stdout of `python mp3_grounded_qa.py eval` (harness header + per-question table + four summary rates). Regenerate with `scripts/run_eval_for_submission.ps1` after any code or threshold change.

---

## Grounding & abstention

The assistant must not answer beyond retrieved excerpts. I use **`RETRIEVAL_THRESHOLD=0.30`**: `should_abstain()` returns true when there are no chunks or the top cosine score is below 0.30, and the pipeline skips generation. Otherwise the model may answer, but if it still prints the exact abstain phrase, the harness treats that as abstention too.

Manual `ask` checks: in-corpus questions (e.g. whether Clay accepts arrest, Blue Carbuncle / Ryder) produced cited answers; out-of-corpus prompts (e.g. who Carol was, where Helen studied, generic “duties of a hydraulic engineer” beyond the story) correctly returned *"I don't know based on the provided documents."* Predefined abstain items (Moriarty, boiling point) and learner abstain **q7_abstain_lestrade** all abstained in `eval`.

**Evidence from `eval`:**

- Moriarty: abstained = **yes**
- Boiling point: abstained = **yes**
- `q7_abstain_lestrade`: abstained = **yes**

---

## Citations

`format_context()` labels chunks `[1]..[n]`; the system prompt requires inline `[n]` markers. `validate_citations()` rejects out-of-range markers (covered in **`python mp3_grounded_qa.py selftest`** — offline, no API keys).

Live runs: answerable rows in `eval` showed **grounded Y** (valid citations and cited `source` matching `expected_source`). Example live question: *Does Clay accept the arrest?* — answer grounded in `01_red_headed_league.txt` with `[n]` markers.

---

## Reading your eval table

Nine questions total: six answerable (four predefined + two learner) and three should-abstain (two predefined + one learner).

| Metric | My value |
|--------|----------|
| Retrieval hit rate | **1.0** |
| Grounded & cited rate | **1.0** |
| Abstention accuracy | **1.0** |
| Hallucination rate | **0.0** |

**Weakest metric during tuning:** grounded rate. Retrieval often placed the correct **file** in top-K while the **decisive passage** (e.g. swamp adder in Speckled Band, Ryder confession in Blue Carbuncle) sat lower in the ranking. Hit rate could look fine while the model abstained or answered without a matching citation.

**Fix:** increased **`MP3_RETRIEVAL_K` from 3 to 8** (default in `mp3_grounded_qa.py`, overridable in `.env`). That pulled explanation chunks into the prompt more reliably. I kept **`RETRIEVAL_THRESHOLD=0.30`** so Moriarty/Lestrade-style questions still abstain when scores are weak.

**Chunks & latency:** ingest creates **73** chunks (13+13+13+16+18 across the five `.txt` files; ~500-char target, 80-char overlap). Each `ask`/`eval` step retrieves **8** chunks, numbers them for the prompt, and prints **`latency_ms`** (query embed + Qdrant + chat on grounded paths; **no chat** on threshold abstain). Cited answers are typically **~1–3+ s** wall-clock depending on API/network; abstains are often faster.

**If extended:** hybrid lexical+dense retrieval, rerank-then-truncate (e.g. 20→5), section-aware chunking, MMR for diverse top-K, query/answer caching, adaptive K by question type, and co-located Qdrant region.

These numbers are on this fixed nine-question harness only; broader generalization would need more held-out questions.

---

## One surprise

**Selftest** was cheap insurance before API spend: it forced correct handling of `should_abstain` and `expected_source: null` in `evaluate()`. The bigger surprise in live RAG was the gap between **retrieval hit** and **grounded & cited** — fixing retrieval depth (K) mattered as much as prompting for citations.

---

## Demo (2–3 min)

**Recording:** `demo/Recording_MP3_final.mp4` in this repository (also uploaded on the course portal with the zip). **Read aloud from:** `docs/mp3_demo_speak.md`.

1. **Cited (learner):** Blue Carbuncle — where the stone was found and who hid it in the goose (`03_blue_carbuncle.txt`).
2. **Multi-hop (learner):** Speckled Band — bedroom fixtures + animal that killed Julia (`02_speckled_band.txt`).
3. **Abstain:** Inspector Lestrade (learner abstain); mention Moriarty on eval only.
4. Show `mp3_eval.txt` header + four summary rates; call out **73 indexed / 8 sent** and **`Latency`** on `ask`.

---

## Implementation note

Models: **`openai/gpt-4o-mini`** and **`openai/text-embedding-3-small`** via an OpenAI-compatible client (**OpenRouter**) and **Qdrant Cloud**; environment loaded by `mp3_env.py` from `Week12/mp3/.env`. No LangChain — only `openai` and `qdrant-client` as in the brief.