# Week 12 MP3 — Grading rubric map (100 pts)

Use this while tuning and before upload. Targets **Excellent** on each row.

## 1. Assistant works with real citations (30 pts)

**What graders look for**

- Answerable questions return text with **`[1]`, `[2]`, …** matching `format_context`.
- `validate_citations` would flag fake `[99]` (you can mention selftest or a live misfire).
- **`grounded_rate`** in eval: answerable rows show `grounded ✓` — citation valid + **cited source file** matches `expected_source`.

**Excellent checklist**

- [ ] Run `ask` on q2 (Julia Stoner) — answer contains `[n]` and matches retrieved chunk.
- [ ] Eval: predefined answerable rows mostly **grounded ✓** (not only hit ✓).
- [ ] Reflection names at least one **citation** behaviour (valid marker vs fabricated, or grounded vs hit).

**Risk → Satisfactory / Zero**

- Model answers with **no `[n]`** → low grounded rate.
- Model invents facts with valid-looking `[1]` but wrong story file → hit ✓, grounded ✗.

**Quick fix**

- `temperature=0` (already set). Re-ask. Increase `k=5` for multi-hop learner question if hit ✓ but grounded ✗.

---

## 2. Abstention works (25 pts)

**What graders look for**

- `abstain_moriarty` and `abstain_out_of_domain` → exact **"I don't know based on the provided documents."**
- **`q7_abstain_lestrade`** (your custom) also **abstained ✓** in eval table.
- Reflection explains **threshold** (e.g. 0.30): why that value, what you’d change if Moriarty answered.

**Excellent checklist**

- [ ] Eval: both predefined abstain rows **abstained ✓**; **hallucination rate 0.0** (or 0/N).
- [ ] Eval: custom abstain row **abstained ✓**.
- [ ] Reflection: concrete threshold sentence (not only “abstention is important”).

**Risk → Satisfactory**

- One OOD question answered from general knowledge (boiling point, Moriarty lore) → raise `RETRIEVAL_THRESHOLD` (try **0.35–0.45**) and re-ingest if needed.

**Quick test**

```text
? Who is Professor Moriarty, and how does he meet his end?
? What is the boiling point of water in degrees Celsius?
? What role does Inspector Lestrade play in solving The Red-Headed League case?
```

All three must print the abstain message **before** calling the LLM (`should_abstain` true) or model must echo abstain phrase.

---

## 3. Evaluation table (25 pts)

**What graders look for**

- `python mp3_grounded_qa.py eval` prints **all four rates**:
  - retrieval hit rate  
  - grounded & cited rate  
  - abstention accuracy  
  - hallucination rate  
- Table includes **predefined (6) + your 3 learner questions** (9 rows if learner file filled).
- Reflection cites **weakest metric** and **one change** you’d try.

**Excellent checklist**

- [ ] `mp3_eval.txt` includes the **MP3 evaluation harness** header + table + four rates (`python mp3_grounded_qa.py eval` writes it automatically).
- [ ] `evaluate()` treats `should_abstain` and `expected_source: null` (your code does).
- [ ] Reflection pastes four numbers and interprets them.

**Risk → Satisfactory**

- Abstain questions counted as hits → bug in `evaluate()` classification (fixed in your implementation).
- `learner_questions.jsonl` still has “Replace this” → eval skips custom rows.

---

## 4. Reflection and demo (20 pts)

**What graders look for**

| Section | Excellent |
|---------|-----------|
| Threshold | Named value + trade-off (false abstain vs false answer) |
| Citations | Real example or explicit “none in live runs because …” |
| Eval numbers | Weakest rate + improvement |
| Demo | **One cited answer** + **one correct “I don’t know”** |

**Demo script (record or live)**

1. Answerable with `[1]` visible (e.g. Speckled Band / carbuncle).
2. Moriarty or Lestrade → abstain message only, no biography.

---

## Score self-audit (before zip)

| Criterion | Max | Your target evidence |
|-----------|-----|----------------------|
| Real citations | 30 | Eval grounded ✓ on most answerable; demo shows `[n]` |
| Abstention | 25 | 3/3 abstain questions ✓; threshold explained |
| Eval table | 25 | `mp3_eval.txt` + 9 rows + 4 rates |
| Reflection + demo | 20 | Specific reflection + 2–3 min demo |

**Total:** 100 pts

---

## Commands (submission-quality)

```powershell
cd Week12\mp3
python mp3_grounded_qa.py selftest
python mp3_grounded_qa.py ingest
python mp3_grounded_qa.py eval
# or: scripts/run_eval_for_submission.ps1
```

Copy the four summary lines from the bottom of `mp3_eval.txt` into `mp3_reflection.md`.