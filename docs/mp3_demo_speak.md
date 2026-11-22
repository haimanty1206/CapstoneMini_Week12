# MP3 demo — words to speak only

Read this while you record. Paste each **question** into `ask` when indicated.

---

## Opening (before typing anything)

Hi, my name is Haimanty Biswas. This is my Week 12 mini project, Grounded Q&A — a cited assistant over five Sherlock summaries. At ingest I split the five stories into 73 chunks — about 500 characters each with a little overlap — and embed them into Qdrant. Each question only sends the top eight retrieved chunks to the model, not the whole index. I use gpt-4o-mini and text-embedding-3-small, same stack as Week 9 MP2. For this recording I use ask on three learner questions; eval covers nine questions total.

---

## Demo 1 — Blue Carbuncle (type the question, then read)

**Type:**

Where was the Blue Carbuncle found, and who admitted to hiding it inside the goose?

**Then say:**

This is learner q5. Eight chunks were retrieved and numbered for the prompt; the Sources list shows what the model actually cited from the Blue Carbuncle file. You should see bracket numbers in the answer — James Ryder and the stone in the goose's crop. You can also view At the bottom, Latency in ms - for end-to-end execution: embed the question, search Qdrant, then generate. On my runs that is usually a few seconds for a cited answer. That is grounded and cited in the eval table.

---

## Demo 2 — Speckled Band multi-hop (type the question, then read)

**Type:**

In The Speckled Band, what unusual bedroom fixtures did Holmes notice, and what animal ultimately caused Julia Stoner's death?

**Then say:**

This is q6, multi-hop. That is why I raised retrieval K to eight — so more than one Speckled Band chunk can land in the same prompt. The model combines fixtures like the dummy bell-rope and ventilator with the swamp adder. Temperature zero keeps it on the eight excerpts only.

---

## Demo 3 — Abstention (type the question, then read)

**Type:**

What role does Inspector Lestrade play in solving The Red-Headed League case?

**Then say:**

This is learner abstain q7. We still retrieve candidates, but the top score is below threshold so we skip the chat call. Lestrade is not in these summaries. You should see the exact abstain sentence and lower latency than the cited demos — mostly embed and search, no generation.

---

## Demo 4 — Eval file (show mp3_eval.txt on screen, then read)

**Then say:**

For submission I ran python mp3_grounded_qa.py eval. The file lists nine rows — six answerable and three should-abstain — then four rates: retrieval hit rate, grounded and cited rate, abstention accuracy, and hallucination rate. On my run all four were at one hundred percent hit and grounded, one hundred percent abstention accuracy, and zero hallucination. That artifact is mp3_eval.txt.

---

## After eval (optional — improvement tips, ~20 sec)

**Then say:**

If I extended this, I would try hybrid keyword plus vector search for names, rerank twenty candidates then send fewer chunks to cut tokens and latency, section-aware chunking so K does not have to be so large, and query or answer caching for repeat evals. For now, tuning K and threshold got me to one hundred percent on this harness. Details are in mp3_reflection.

---

## Closing

This demo shows a cited in-corpus answer and a correct abstention, with chunk counts and latency on screen. The zip includes the script, eval output, learner questions, and reflection. Thank you.