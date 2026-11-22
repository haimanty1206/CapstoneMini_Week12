# Week 12 MP3 submission checklist

Matches **Week 12_Graded Mini Project** / `AI-RAG_MP3_Brief.md`.

## Required files in zip

| File | Your copy |
|------|-----------|
| `mp3_grounded_qa.py` | `Week12/mp3/mp3_grounded_qa.py` |
| `mp3_eval.txt` | Generate via `scripts/run_eval_for_submission.ps1` |
| `data/learner_questions.jsonl` | 2 answerable + 1 abstain (edited) |
| `mp3_reflection.md` | Complete name + paste eval metrics |
| **Demo** | 2–3 min: cited answers + one abstention |

## Zip name

`Week 12_Graded Mini Project_[Your Last Name].zip`

```powershell
cd "d:\IIT Madras\Labs\AI_Practice\Week12\mp3"
powershell -File scripts\build_submission_folder.ps1 -LastName Biswas
```

## Before submit

- [ ] `python mp3_grounded_qa.py selftest` → `✓ selftest passed`
- [ ] `.env` uses **OpenAI** (`OPENAI_API_KEY`), not only OpenRouter
- [ ] `ingest` + `eval` run once; real table in `mp3_eval.txt`
- [ ] Moriarty question abstains in `ask`
- [ ] Reflection includes **your** eval numbers and threshold rationale
- [ ] Personal GitHub repo URL submitted per course portal

## Grading learning outcomes (self-check)

- [ ] Inline `[n]` citations in answers  
- [ ] `validate_citations` catches out-of-range markers  
- [ ] Abstention when evidence insufficient  
- [ ] Eval table reports hit, grounded, abstention, hallucination rates  