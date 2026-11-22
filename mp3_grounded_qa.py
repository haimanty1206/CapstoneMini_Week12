"""MP3 · Grounded Q&A — Starter
================================

The load → chunk → embed → store → retrieve → answer pipeline is PROVIDED and
working (it's the MP2 loop). Your job is the three things that make it
*trustworthy*. Fill in the FOUR functions marked TODO:

    format_context     (citations)   validate_citations (citations)
    should_abstain     (abstention)  evaluate           (evaluation)

Everything else — the pipeline, the prompt, `answer()`, the table renderer, the
CLI — is done for you. Once your four functions are in, `answer()` and `eval`
will work end to end.

Check your work offline (no API key needed):
    python mp3_grounded_qa.py selftest      # goes green once your 4 functions are correct

Under the hood this is the MP2 Mini-RAG pipeline extended into a *trustworthy*
assistant:

  1. CITATIONS  — the answer cites each claim inline with [n], and every [n]
                  is validated against an actually-retrieved chunk.
  2. ABSTENTION — when retrieval finds no supporting chunk, the assistant says
                  "I don't know based on the provided documents" instead of
                  guessing.
  3. EVALUATION — a small harness measures retrieval hit rate, grounded rate,
                  and abstention accuracy over a question set, and prints a
                  table (this is the graded artifact).

The load → chunk → embed → store → retrieve loop is carried from MP2. The three
capabilities above are the MP3 additions.

Run:
    pip install -r requirements.txt
    source .env
    python mp3_grounded_qa.py ingest      # build the collection (once)
    python mp3_grounded_qa.py ask         # interactive Q&A with citations + abstention
    python mp3_grounded_qa.py eval        # the retrieval-quality evaluation table
    python mp3_grounded_qa.py selftest    # OFFLINE checks of the new logic (no keys)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

# ─── Configuration ──────────────────────────────────────────────────────

DATA_DIR           = Path(__file__).parent / "data"
COLLECTION_NAME    = "mp3_sherlock"
EMBEDDING_MODEL    = "text-embedding-3-small"
EMBEDDING_DIM      = 1536
CHAT_MODEL         = "gpt-4o-mini"
TARGET_CHUNK_SIZE  = 500    # characters
CHUNK_OVERLAP      = 80     # characters
RETRIEVAL_THRESHOLD = float(os.environ.get("RETRIEVAL_THRESHOLD", "0.30"))  # abstain if top chunk scores below this (cosine); tune via .env
RETRIEVAL_K        = int(os.environ.get("MP3_RETRIEVAL_K", "8"))  # top-K chunks; 8 helps Julia-style Qs reach the swamp-adder section in 02

ABSTAIN_MESSAGE = "I don't know based on the provided documents."

ROOT = Path(__file__).resolve().parent


def _ok_fail(cond: bool) -> str:
    return "OK" if cond else "FAIL"


def _yn(value: bool) -> str:
    return "Y" if value else "N"


# ─── Clients (lazy; selftest imports with no creds) ────────────────────────
import mp3_env


def corpus_dir() -> Path:
    return mp3_env.resolve_corpus_dir()


oai: Any = None
qdrant: Any = None


def _require_clients() -> None:
    global oai, qdrant
    if oai is not None and qdrant is not None:
        return
    q_issues = mp3_env.qdrant_url_issues()
    if q_issues:
        sys.exit("Qdrant config:\n  - " + "\n  - ".join(q_issues))
    dns_err = mp3_env.check_qdrant_dns()
    if dns_err:
        sys.exit(dns_err)
    oai = mp3_env.make_openai_client()
    qdrant = mp3_env.make_qdrant_client()
    if oai is None or qdrant is None:
        sys.exit(
            "This command needs QDRANT_URL and OPENAI_API_KEY or OPENROUTER_API_KEY. "
            "Copy Week12/capstone/.env to Week12/mp3/.env or run scripts/sync_openrouter_env.ps1. "
            "(The `selftest` command needs neither.)"
        )


# ─── Pipeline (carried from MP2) ────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read every .txt file in the corpus directory (pure — no network)."""
    docs = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        title = next((ln.strip() for ln in text.splitlines() if ln.strip()), path.stem)
        docs.append({"source": path.name, "title": title, "text": text})
    return docs


def _is_header(line: str) -> bool:
    """Heuristic: short lines with no terminal punctuation are section headers."""
    s = line.strip()
    return 0 < len(s) < 60 and not s.endswith((".", "!", "?", ":", ",", ";"))


def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Paragraph-aware chunking with heuristic section-header tracking (pure)."""
    chunks: list[dict[str, Any]] = []
    section = doc["title"]
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append({"source": doc["source"], "title": doc["title"],
                           "section": section, "text": buf.strip()})
        buf = ""

    for para in doc["text"].split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if _is_header(para):
            flush()
            section = para
            continue
        if len(buf) + len(para) + 1 > TARGET_CHUNK_SIZE:
            flush()
            # carry a little overlap from the previous chunk
            if chunks and CHUNK_OVERLAP:
                buf = chunks[-1]["text"][-CHUNK_OVERLAP:] + " "
        buf += para + " "
    flush()
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = oai.embeddings.create(model=mp3_env.embedding_model(), input=texts)
    return [d.embedding for d in resp.data]


def setup_collection() -> None:
    from qdrant_client.models import Distance, VectorParams

    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    from qdrant_client.models import PointStruct

    vectors = embed_texts([c["text"] for c in chunks])
    points = [PointStruct(id=str(uuid.uuid4()), vector=v, payload=c)
              for c, v in zip(chunks, vectors)]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def retrieve(query: str, k: int = RETRIEVAL_K) -> list[dict[str, Any]]:
    qvec = embed_texts([query])[0]
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=k,
    ).points
    out = []
    for h in hits:
        chunk = dict(h.payload or {})
        chunk["score"] = round(float(h.score or 0.0), 3)
        out.append(chunk)
    return out


# ─── NEW 1 · Citations ──────────────────────────────────────────────────

def format_context(chunks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Number the retrieved chunks [1..n] for the prompt and return a parallel
    source map the answer's [n] markers can be validated against."""
    parts: list[str] = []
    sources: list[dict[str, Any]] = []
    for n, ch in enumerate(chunks, start=1):
        title = ch.get("title", "")
        section = ch.get("section", "")
        text = ch.get("text", "")
        parts.append(f'[{n}] (from "{title}" — {section})\n{text}')
        sources.append(
            {
                "n": n,
                "source": ch["source"],
                "title": title,
                "section": section,
                "score": ch.get("score", 0.0),
            }
        )
    return "\n\n".join(parts), sources


def validate_citations(answer_text: str, n_sources: int) -> dict[str, Any]:
    """Parse [n] markers from the answer and check each maps to a real source.
    A fabricated citation (a number with no matching source) is the tell we
    care about."""
    nums = [int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", answer_text)]
    used = sorted({n for n in nums if 1 <= n <= n_sources})
    invalid = sorted({n for n in nums if n < 1 or n > n_sources})
    return {"used": used, "invalid": invalid, "has_citation": bool(used)}


# ─── NEW 2 · Abstention ─────────────────────────────────────────────────

def should_abstain(chunks: list[dict[str, Any]], threshold: float = RETRIEVAL_THRESHOLD) -> bool:
    """Abstain when there is no sufficiently-relevant chunk to ground an answer."""
    if not chunks:
        return True
    best = max(float(c.get("score", 0.0)) for c in chunks)
    return best < threshold


SYSTEM_PROMPT = """You answer questions about a small collection of Sherlock Holmes stories.
Use ONLY the numbered excerpts provided below. After each claim, cite the excerpt it
came from using its number in square brackets, e.g. [1]. If the excerpts do not contain
the answer, reply with exactly: "I don't know based on the provided documents." Do not
use any outside knowledge."""


def answer(question: str, k: int = RETRIEVAL_K, threshold: float = RETRIEVAL_THRESHOLD) -> dict[str, Any]:
    """Retrieve → (abstain?) → cite-formatted generation → validate citations."""
    t0 = time.time()
    chunks = retrieve(question, k=k)
    retrieved_sources = [{"source": c["source"], "score": c.get("score", 0.0)} for c in chunks]

    if should_abstain(chunks, threshold):
        return {"question": question, "answer": ABSTAIN_MESSAGE, "citations": [],
                "retrieved_sources": retrieved_sources, "abstained": True,
                "citation_check": {"used": [], "invalid": [], "has_citation": False},
                "latency_ms": int((time.time() - t0) * 1000)}

    context, sources = format_context(chunks)
    resp = oai.chat.completions.create(
        model=mp3_env.chat_model(), temperature=0,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"}],
    )
    text = resp.choices[0].message.content.strip()

    # A model can ignore instructions and abstain in prose — respect that.
    if ABSTAIN_MESSAGE.lower() in text.lower():
        return {"question": question, "answer": ABSTAIN_MESSAGE, "citations": [],
                "retrieved_sources": retrieved_sources, "abstained": True,
                "citation_check": {"used": [], "invalid": [], "has_citation": False},
                "latency_ms": int((time.time() - t0) * 1000)}

    check = validate_citations(text, len(sources))
    cited = [s for s in sources if s["n"] in check["used"]]
    return {"question": question, "answer": text, "citations": cited,
            "retrieved_sources": retrieved_sources, "abstained": False,
            "citation_check": check, "latency_ms": int((time.time() - t0) * 1000)}


# ─── NEW 3 · Evaluation harness ─────────────────────────────────────────

def evaluate(questions: list[dict[str, Any]],
             answer_fn: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """Score an assistant over a question set. Each question is either
    answerable (has `expected_source`) or a should-abstain question
    (`should_abstain: true` or `expected_source: null`).

    Metrics (demonstrating grounded-response and hallucination KPIs):
      - retrieval hit rate    : answerable Qs whose expected source is in top-K
      - grounded & cited rate  : answerable Qs answered, citing the right source, no bad [n]
      - abstention accuracy    : should-abstain Qs the assistant correctly refused
      - hallucination rate     : should-abstain Qs the assistant answered anyway
    """
    rows: list[dict[str, Any]] = []
    n_answerable = 0
    n_abstain = 0
    hits = 0
    grounded_n = 0
    abst_ok = 0
    hallucinated = 0

    for q in questions:
        qid = q.get("id", q["question"][:24])
        result = answer_fn(q["question"])
        should_abs = bool(q.get("should_abstain")) or q.get("expected_source") in (
            None,
            "",
            "ABSTAIN",
        )

        if should_abs:
            n_abstain += 1
            abstained = bool(result.get("abstained"))
            if abstained:
                abst_ok += 1
            else:
                hallucinated += 1
            rows.append(
                {
                    "id": qid,
                    "type": "abstain",
                    "hit": False,
                    "abstained": abstained,
                    "grounded": False,
                    "ok": abstained,
                }
            )
            continue

        n_answerable += 1
        expected = q["expected_source"]
        retrieved = result.get("retrieved_sources") or []
        hit = any(s.get("source") == expected for s in retrieved)
        if hit:
            hits += 1

        check = result.get("citation_check") or {}
        cited_sources = {c.get("source") for c in result.get("citations") or []}
        grounded = (
            not result.get("abstained")
            and bool(check.get("has_citation"))
            and not check.get("invalid")
            and expected in cited_sources
        )
        if grounded:
            grounded_n += 1

        rows.append(
            {
                "id": qid,
                "type": "answer",
                "hit": hit,
                "abstained": bool(result.get("abstained")),
                "grounded": grounded,
                "ok": grounded,
            }
        )

    ans_denom = n_answerable or 1
    abs_denom = n_abstain or 1
    summary = {
        "n_answerable": n_answerable,
        "n_abstain": n_abstain,
        "retrieval_hit_rate": hits / ans_denom,
        "grounded_rate": grounded_n / ans_denom,
        "abstention_accuracy": abst_ok / abs_denom,
        "hallucination_rate": hallucinated / abs_denom,
    }
    return {"rows": rows, "summary": summary}


def _dedupe_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        qid = r.get("id", "")
        if qid in seen:
            continue
        seen.add(qid)
        out.append(r)
    return out


_ROW_RE = re.compile(
    r"^\s{2}(\S+)\s+(answer|abstain)\s+([YN])\s+([YN])\s+([YN])\s+([YN])\s*$"
)


def dedupe_mp3_eval_text(text: str) -> str:
    """Remove repeated table rows / appended eval runs from mp3_eval.txt."""
    blocks = text.split("MP3 evaluation harness")
    blocks = [b for b in blocks if b.strip()]
    if not blocks:
        return text.strip() + "\n"

    # One harness block; dedupe repeated rows inside it (e.g. from Tee-Object append).
    chunk = blocks[-1]
    lines = chunk.splitlines()
    header: list[str] = ["MP3 evaluation harness"]
    table_rows: dict[str, str] = {}
    row_order: list[str] = []
    footer: list[str] = []
    in_table = False
    for line in lines:
        m = _ROW_RE.match(line)
        if m:
            in_table = True
            qid = m.group(1)
            if qid not in table_rows:
                table_rows[qid] = line
                row_order.append(qid)
            continue
        if line.strip().startswith("Retrieval hit rate"):
            footer = [line]
        elif line.strip().startswith(("Grounded", "Abstention", "Hallucination")):
            if footer:
                footer.append(line)
            continue
        if not in_table and line.strip():
            header.append(line)

    if not row_order:
        return text.strip() + "\n"

    while header and not header[-1].strip():
        header.pop()
    out: list[str] = list(header)
    if not any("id" in ln and "type" in ln for ln in out):
        out.extend(
            [
                "  id                              type     hit   abstained  grounded  ok",
                "  " + "-" * 74,
            ]
        )
    for qid in row_order:
        out.append(table_rows[qid])
    out.append("  " + "-" * 74)
    out.append("")
    out.extend(footer)
    return "\n".join(out).strip() + "\n"


def format_eval_table(result: dict[str, Any]) -> str:
    """Render the evaluation as the submittable table (ASCII-safe for graders)."""
    def cell(v: bool) -> str:
        return _yn(v)

    rows = _dedupe_eval_rows(result["rows"])
    lines = ["", "  id                              type     hit   abstained  grounded  ok",
             "  " + "-" * 74]
    for r in rows:
        lines.append(f"  {r['id']:<31} {r['type']:<8} {cell(r['hit']):<5} "
                     f"{cell(r['abstained']):<10} {cell(r['grounded']):<9} {cell(r['ok'])}")
    s = result["summary"]
    lines += [
        "  " + "-" * 74, "",
        f"  Retrieval hit rate    : {s['retrieval_hit_rate']}   "
        f"({s['n_answerable']} answerable questions)",
        f"  Grounded & cited rate : {s['grounded_rate']}   (KPI: grounded response rate)",
        f"  Abstention accuracy   : {s['abstention_accuracy']}   "
        f"({s['n_abstain']} should-abstain questions)",
        f"  Hallucination rate    : {s['hallucination_rate']}   (KPI: hallucination rate)",
        "",
    ]
    return "\n".join(lines)


# ─── Commands ───────────────────────────────────────────────────────────

def cmd_ingest() -> None:
    _require_clients()
    print(">> Loading corpus...")
    cdir = corpus_dir()
    print(f">> Corpus: {cdir}")
    docs = load_corpus(cdir)
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        cs = chunk_document(doc)
        all_chunks.extend(cs)
        print(f"  {doc['source']}: {len(cs)} chunks")
    print(f">> {len(all_chunks)} chunks | setup collection | ingest...")
    setup_collection()
    ingest_chunks(all_chunks)
    print("Done. Try: python mp3_grounded_qa.py ask")


def _print_result(r: dict[str, Any]) -> None:
    print(f"\n{r['answer']}\n")
    if r["abstained"]:
        print("  (abstained — no supporting excerpt)")
    else:
        print("  Sources:")
        for c in r["citations"]:
            print(f"    [{c['n']}] {c['title']} — {c['section']}  (score {c['score']})")
        if r["citation_check"]["invalid"]:
            print(f"  WARN fabricated citation(s): {r['citation_check']['invalid']}")
    print(f"  Latency: {r['latency_ms']}ms\n")


def cmd_ask() -> None:
    _require_clients()
    print("Grounded Q&A over the Sherlock corpus. Type a question, then Enter. Empty line to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if not q:
            return
        _print_result(answer(q))


def _load_eval_questions() -> list[dict[str, Any]]:
    path = DATA_DIR / "predefined_questions.jsonl"
    questions = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    learner = DATA_DIR / "learner_questions.jsonl"
    if learner.exists():
        extra = [
            json.loads(line)
            for line in learner.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if extra and not str(extra[0]["question"]).startswith("Replace this"):
            questions += extra
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for q in questions:
        qid = q.get("id") or q["question"][:24]
        if qid in seen:
            continue
        seen.add(qid)
        unique.append(q)
    return unique


def _eval_report_body(questions: list[dict[str, Any]]) -> str:
    from datetime import datetime, timezone

    header = [
        "MP3 evaluation harness",
        f"Command: python mp3_grounded_qa.py eval",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Corpus dir: {corpus_dir()}",
        f"Collection: {COLLECTION_NAME}",
        f"Chat model: {mp3_env.chat_model()}",
        f"Embed model: {mp3_env.embedding_model()}",
        f"RETRIEVAL_K={RETRIEVAL_K}  RETRIEVAL_THRESHOLD={RETRIEVAL_THRESHOLD}",
        f"Questions: {len(questions)} "
        f"(predefined + learner; includes should-abstain rows)",
        "",
    ]
    return "\n".join(header) + format_eval_table(evaluate(questions, answer))


def cmd_eval(save_path: Path | None = None) -> None:
    _require_clients()
    questions = _load_eval_questions()
    body = _eval_report_body(questions)
    print(body)
    out = save_path or ROOT / "mp3_eval.txt"
    if save_path is not False:
        clean = dedupe_mp3_eval_text(body)
        out.write_text(clean, encoding="utf-8")
        print(f"Wrote {out}")


def cmd_selftest() -> None:
    """Offline checks of the MP3 logic — no OpenAI, no Qdrant, no keys."""
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{_ok_fail(cond)}] {name}")
        ok = ok and cond

    docs = load_corpus(corpus_dir())
    check(f"load_corpus reads the corpus ({len(docs)} docs)", len(docs) >= 1)
    if docs:
        cs = chunk_document(docs[0])
        check(f"chunk_document splits doc 0 into chunks ({len(cs)})",
              len(cs) >= 1 and all({"source", "title", "section", "text"} <= c.keys() for c in cs))

    fake = [{"source": "a.txt", "title": "A", "section": "S1", "text": "alpha", "score": 0.9},
            {"source": "b.txt", "title": "B", "section": "S2", "text": "beta", "score": 0.5}]
    ctx, srcs = format_context(fake)
    check("format_context numbers chunks [1]/[2]", "[1]" in ctx and "[2]" in ctx and len(srcs) == 2)

    v_ok = validate_citations("The adder [1] came via the ventilator [2].", 2)
    v_bad = validate_citations("An outside fact [3].", 2)
    check("validate_citations accepts real markers", v_ok["used"] == [1, 2] and not v_ok["invalid"])
    check("validate_citations flags a fabricated marker", v_bad["invalid"] == [3])

    check("should_abstain: low score -> abstain", should_abstain([{"score": 0.1}], 0.3) is True)
    check("should_abstain: good score -> answer", should_abstain([{"score": 0.8}], 0.3) is False)
    check("should_abstain: no chunks -> abstain", should_abstain([], 0.3) is True)

    # evaluate() with a deterministic mock answer_fn (no network)
    qs = [
        {"id": "ans_hit", "question": "q1", "expected_source": "01.txt"},
        {"id": "ans_miss", "question": "q2", "expected_source": "02.txt"},
        {"id": "abstain_ok", "question": "q3", "should_abstain": True},
    ]
    def mock(q):
        if q == "q1":  # correct source retrieved + cited
            return {"answer": "A [1]", "abstained": False, "retrieved_sources": [{"source": "01.txt", "score": 0.9}],
                    "citations": [{"source": "01.txt"}], "citation_check": {"used": [1], "invalid": [], "has_citation": True}}
        if q == "q2":  # wrong source retrieved -> miss + not grounded
            return {"answer": "B [1]", "abstained": False, "retrieved_sources": [{"source": "99.txt", "score": 0.9}],
                    "citations": [{"source": "99.txt"}], "citation_check": {"used": [1], "invalid": [], "has_citation": True}}
        return {"answer": ABSTAIN_MESSAGE, "abstained": True, "retrieved_sources": [{"source": "x", "score": 0.1}],
                "citations": [], "citation_check": {"used": [], "invalid": [], "has_citation": False}}
    res = evaluate(qs, mock)
    s = res["summary"]
    check("evaluate: hit rate 1/2 answerable", s["retrieval_hit_rate"] == 0.5)
    check("evaluate: grounded rate 1/2 answerable", s["grounded_rate"] == 0.5)
    check("evaluate: abstention accuracy 1/1", s["abstention_accuracy"] == 1.0)
    check("evaluate: hallucination rate 0/1", s["hallucination_rate"] == 0.0)
    check("format_eval_table renders a table", "Retrieval hit rate" in format_eval_table(res))

    print("\nselftest passed" if ok else "\nselftest FAILED")
    sys.exit(0 if ok else 1)


def main() -> None:
    cmds = {"ingest": cmd_ingest, "ask": cmd_ask, "eval": cmd_eval, "selftest": cmd_selftest}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__); sys.exit(0 if len(sys.argv) < 2 else 1)
    if sys.argv[1] == "eval":
        save = ROOT / "mp3_eval.txt"
        if len(sys.argv) > 2 and sys.argv[2] == "--no-save":
            cmd_eval(save_path=False)
        elif len(sys.argv) > 3 and sys.argv[2] == "--save":
            cmd_eval(save_path=Path(sys.argv[3]))
        else:
            cmd_eval(save_path=save)
        return
    cmds[sys.argv[1]]()


if __name__ == "__main__":
    main()
