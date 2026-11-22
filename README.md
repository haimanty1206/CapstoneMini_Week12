# Week 12 · MP3 Grounded Q&A (Haimanty Biswas)

Trustworthy RAG over five Sherlock Holmes summaries: inline citations `[n]`, abstention when evidence is weak, and a nine-question evaluation harness.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # add OPENAI_API_KEY or OPENROUTER_API_KEY + QDRANT_URL
python mp3_grounded_qa.py selftest
python mp3_grounded_qa.py ingest
python mp3_grounded_qa.py eval
python mp3_grounded_qa.py ask
```

## Course deliverables

| Artifact | Location |
|----------|----------|
| Assistant | `mp3_grounded_qa.py`, `mp3_env.py` |
| Eval output | `mp3_eval.txt` |
| Learner questions | `data/learner_questions.jsonl` |
| Reflection | `mp3_reflection.md` |
| Canvas zip | `submission/Week 12_Graded Mini Project_Biswas.zip` |
| Demo video | `demo/Recording_MP3_final.mp4` (see `demo/README.md`) |

## Eval summary (harness)

- **Retrieval hit rate:** 1.0  
- **Grounded & cited rate:** 1.0  
- **Abstention accuracy:** 1.0  
- **Hallucination rate:** 0.0  
- **Settings:** `RETRIEVAL_K=8`, `RETRIEVAL_THRESHOLD=0.30`

## Layout

- `corpus/` — five story summaries  
- `data/` — predefined + learner JSONL for `eval`  
- `docs/` — checklist, FAQ, rubric notes, demo script  

IIT Madras · AI Practice · Week 12 graded mini project (separate from the integrated capstone repo).