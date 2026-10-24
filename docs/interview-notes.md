# DocuMind – Interview Notes

A study guide that grows with the project: one section per build phase.
Read it out loud. If you can explain each section without looking, you're ready.

---

## Phase 1 – Project Setup & Configuration

### The 30-second answer
> "Before writing any AI code, I set up the project so it's safe and easy to change.
> All settings — which LLM to use, the model name, how many chunks to retrieve,
> and the API key — live in a `.env` file that is never committed to Git.
> One module, `config.py`, reads that file, fills in sensible defaults, checks the
> values, and gives the rest of the app a single read-only `settings` object."

### What each file does
| File | Purpose | Why it matters |
|---|---|---|
| `requirements.txt` | Lists every library with a minimum version, grouped by phase | Anyone can recreate the environment with one command |
| `.env.example` | Template showing which settings exist, with no real secrets | Safe to commit; a new developer copies it to `.env` |
| `.env` (you create it) | Your real API key and choices | Git-ignored, so the key never reaches GitHub |
| `.gitignore` | Excludes `.env`, `venv/`, the vector DB and logs | Keeps secrets and large generated files out of the repo |
| `src/__init__.py` | Makes `src` a Python package | Lets us write `from src.config import settings` |
| `src/config.py` | Loads, defaults and validates settings | Single source of truth for configuration |
| `venv/` | Isolated Python environment | Project libraries don't clash with other projects |

### How `config.py` works, step by step
1. **Find the project root** with `Path(__file__).resolve().parent.parent`, so paths
   like `data/chroma` work no matter which folder you run the app from.
2. **`load_dotenv()`** copies the lines of `.env` into environment variables.
   `override=False` means a real environment variable wins — useful when deploying,
   where secrets are set in the hosting dashboard rather than in a file.
3. **`load_settings()`** reads each variable with `os.getenv`, applies defaults
   (e.g. `TOP_K=4`, model chosen per provider), and **validates** them:
   provider must be `groq` or `ollama`, `TOP_K` must be a whole number ≥ 1.
4. Returns a **frozen dataclass** `Settings` — frozen means no code can change a
   setting by accident while the app runs.
5. `settings = load_settings()` runs **once** at import; every module shares it.
6. `masked_api_key()` prints `********ABCD`, so the key never appears in logs or screenshots.

### Design decisions you should be able to defend
- **Why `.env` instead of hard-coding the key?** Hard-coded keys get pushed to GitHub
  and leaked (bots scan GitHub for keys within minutes). `.env` + `.gitignore` prevents that,
  and lets each person/environment use different values without changing code.
- **Why one config module?** If five files each called `os.getenv("TOP_K")`, they could
  disagree on defaults. Centralising means one place to change and one place to validate.
- **Why validate early ("fail fast")?** A typo like `TOP_K=abc` fails at startup with
  a clear message, instead of a confusing crash deep inside the search code later.
- **Why a provider switch (Groq vs Ollama)?** Groq is a fast, free cloud API; Ollama runs
  models locally with no internet and full privacy. Switching is a one-line `.env` change —
  the rest of the code doesn't care which one is used.
- **Why separate chunk size / embedding model from `.env` settings?** Those must stay the
  same as when documents were indexed — changing the embedding model would make old
  vectors incompatible. So they're fixed in code, not user-tunable.
- **Why `langchain-text-splitters` instead of full `langchain`?** I only need the text
  splitter, so I install just that small package — fewer dependencies, faster install.

### If the interviewer asks…
- **"What is a virtual environment?"** — An isolated folder with its own Python and
  packages, so this project's library versions don't conflict with other projects.
- **"What happens if the API key is missing?"** — Config still loads (you might be
  using Ollama), but it prints a warning; later the UI will show a friendly error
  instead of crashing.
- **"How would you deploy this with secrets?"** — Set `GROQ_API_KEY` as an environment
  variable / secret on the host (e.g. Streamlit Cloud secrets). Because real env vars
  override `.env`, no code changes are needed.
- **"What's a dataclass?"** — A Python class that auto-generates `__init__` and a readable
  `__repr__` from type-annotated fields; `frozen=True` makes instances immutable.

### How to run / test
```bash
venv\Scripts\activate
copy .env.example .env
python -m src.config
```
Tested scenarios: no `.env` (defaults + warning), Ollama with `TOP_K=6`, a masked
key, `TOP_K=abc` (clear error), and `LLM_PROVIDER=openai` (clear error).

---

## Phase 2 – Ingestion (Load → Chunk → Embed → Store)

### The 30-second answer
> "Ingestion turns raw files into something searchable. I read PDFs page by page
> with `pypdf`, keeping the file name and page number for every page. I split each
> page into ~800-character chunks with 150 characters of overlap, turn every chunk
> into a 384-dimensional vector with the `all-MiniLM-L6-v2` sentence-transformer,
> and store text, metadata and vector in a persistent ChromaDB collection that uses
> cosine distance. A SHA-256 hash of each file prevents duplicates, and I can delete
> a document by its file name."

### The pipeline
```
file (path or uploaded bytes)
  └─ ingest.py: SHA-256 hash ── already stored? → "skipped"
       └─ loader.py   → pages  [{text, source, page}]
       └─ chunker.py  → chunks [{id, text, metadata{source, page, chunk_index, doc_hash}}]
       └─ embedder.py → vectors (384 numbers each, normalised to length 1)
       └─ vector_store.py → ChromaDB upsert  (data/chroma/)
```

### What each file does
| File | Purpose | Key detail to mention |
|---|---|---|
| `src/loader.py` | PDF/TXT/MD → list of pages with `source` + `page` | Accepts a path **or** bytes, so the CLI and Streamlit upload use the same code; skips empty pages; TXT/MD = page 1 |
| `src/chunker.py` | Pages → overlapping chunks with metadata | Splits **per page**, so every chunk has exactly one correct page number |
| `src/embedder.py` | Text → vector | Model loaded once (`lru_cache`); batch encoding; `normalize_embeddings=True` |
| `src/vector_store.py` | Every ChromaDB call: add, delete, list, count, clear | The only file that knows about ChromaDB, so it's easy to swap the DB later |
| `src/ingest.py` | Runs the whole pipeline for one file + duplicate logic | Not in the original plan: added so the UI, scripts and the future Quiz Generator share one ingestion function |
| `scripts/make_sample_docs.py` | Builds 2 demo PDFs + 1 Markdown file | Original content with specific facts per page, so Phase 5 can test page-accurate retrieval |
| `scripts/ingest_samples.py` | Phase 2 test: ingest samples, print chunk counts | Run twice to see the duplicate check |

### Key concepts — explain in your own words
- **Chunking**: one embedding for a whole document would blur many topics into one
  average. Small chunks each hold one idea, so search can find the exact passage, and
  the LLM gets a short, focused prompt (cheaper, faster, less distraction).
- **Chunk size 800 / overlap 150**: 800 characters ≈ 1–2 paragraphs ≈ 150–200 tokens,
  which is enough context to stay meaningful but small enough to stay specific. Overlap
  repeats the end of one chunk at the start of the next so a sentence on the boundary
  isn't cut in half and lost. (150 is the *maximum*; the splitter overlaps on whole
  words/sentences, so the real overlap is often less, e.g. 91 characters in our test.)
- **RecursiveCharacterTextSplitter**: it tries the "nicest" separator first, i.e.
  paragraphs (`\n\n`), then lines, then sentences (`. `), then words, and only cuts
  mid-word as a last resort.
- **Embeddings**: a neural network maps text to a vector so that texts with similar
  *meaning* end up close together, even with different words ("sick leave" ≈ "medical
  absence"). MiniLM gives 384 numbers per text, is free, and runs fast on a CPU.
- **Cosine similarity**: the angle between two vectors. 1 means the same direction (same
  meaning) and about 0 means unrelated. Because the vectors are normalised to length 1,
  cosine similarity is just the dot product. Chroma returns *distance* = 1 − similarity.
- **Why ChromaDB?** Free, runs locally with no server, persists to disk, stores
  text + metadata + vectors together, and supports metadata filters (`where={"source": ...}`),
  which is how delete and list work. It uses an HNSW index for fast approximate
  nearest-neighbour search.
- **Duplicate handling with SHA-256**: the hash is a fingerprint of the file's bytes.
  Same bytes → same hash → skip, even under a different name. Same name with
  different bytes → delete the old chunks, then add the new ones, so no stale text
  remains. Chunk ids (`<hash12>-p<page>-c<index>`) are deterministic, and `upsert`
  means re-adding never creates copies.
- **Safe ordering**: the new file is loaded and chunked *before* the old version is
  deleted, so a broken upload can't wipe a good document.

### If the interviewer asks…
- **"Why not split across pages?"** Then one chunk could span pages 3–4 and the
  citation would be wrong. Accurate citations matter more to me than the rare sentence
  split by a page break.
- **"What about scanned PDFs?"** `pypdf` only reads the text layer. A scanned PDF has
  none, so I raise a clear error. The fix would be OCR (e.g. Tesseract), which I list as
  future work.
- **"Why compute embeddings yourself instead of letting Chroma do it?"** It gives me
  explicit control over the model, and the query must be embedded with the *same* model
  as the documents. Keeping both in `embedder.py` guarantees that.
- **"What if you change the embedding model?"** Old vectors become incompatible
  (different vector space, maybe a different size), so you must clear the DB and
  re-ingest. That's why the model name is fixed in `config.py`, not in `.env`.
- **"How does it scale?"** Batch embedding (32 at a time) and HNSW indexing handle
  thousands of chunks on a laptop. For millions I'd use a GPU for embeddings and a
  hosted vector DB.

### How to run / test
```bash
python scripts/make_sample_docs.py
python scripts/ingest_samples.py
python scripts/ingest_samples.py
```
Results: 3 files → **16 chunks** (leave policy 3 pages / 6 chunks, ML notes 4 / 8,
Markdown 1 / 2). Second run: all 3 skipped. Also tested: upload as bytes, replace on a
changed file, a renamed duplicate skipped, delete, `.png` and empty files rejected
with clear errors, and a test search ("How many days of sick leave…?") whose top
result was `company_leave_policy.pdf` page 2 with similarity 0.72.

---

## Phase 3 – Retrieval + Generation (the RAG pipeline)

### The 30-second answer
> "When a question comes in, I embed it with the same model as the documents, ask
> ChromaDB for the 4 most similar chunks, and check the best similarity score. If
> it's below 0.25, the question is unrelated, so I answer 'not found' without
> calling the LLM at all. Otherwise I build a prompt with strict rules (answer only
> from this context, cite `[file, p.N]`, or say the exact not-found sentence), send it
> to the LLM through one `generate()` function that works with Groq or Ollama, and
> return the answer, the sources with file, page, snippet and score, and the time taken."

### The pipeline in simple words (an "open-book exam" analogy)
1. **Understand the question.** Turn it into a vector (`embed_query`).
2. **Find the right pages in the book.** Ask ChromaDB for the top-K closest chunks
   (`vector_store.search`), each with a similarity score from 0 to 1.
3. **Is this question even about the book?** If the best score is below 0.25, stop and
   answer "I couldn't find this in the uploaded documents." This is fast and free.
4. **Hand the student only the relevant pages.** Build the prompt: system rules, then
   each chunk labelled `[file.pdf, p.3]`, then the question. Chunks below the threshold
   are dropped so they can't distract the model.
5. **The student answers using only those pages.** `llm.generate()` with temperature 0.1.
6. **Check the answer.** If the LLM said "not found", return no sources (showing sources
   would be misleading). Otherwise return the sources and mark which ones were actually cited.

### What each file does
| File | Purpose | Key detail to mention |
|---|---|---|
| `vector_store.search()` | Top-K nearest chunks + scores | Converts Chroma's cosine *distance* into *similarity* = 1 − distance |
| `src/llm.py` | One `generate(prompt, system_prompt)` for Groq **or** Ollama | Provider chosen in `.env`; every failure becomes a friendly `LLMError`; strips `<think>` blocks from reasoning models |
| `src/rag_pipeline.py` | `answer_question()`: retrieve → guard → prompt → generate → sources | Two anti-hallucination layers; returns a plain dict the UI can display |
| `scripts/ask_questions.py` | CLI test: 3 in-doc questions + 1 out-of-doc | Also accepts your own questions as arguments |

### How I reduce hallucination (they WILL ask this)
1. **Retrieval grounding.** The model only sees text from your documents.
2. **Similarity threshold (0.25).** Unrelated questions never reach the LLM.
   *How I chose 0.25:* I measured real scores. Valid questions scored 0.34–0.72 and
   unrelated ones (FIFA, cake, capitals) scored 0.08–0.14, so 0.25 sits safely between them.
3. **Strict system prompt.** "Answer ONLY from the context", "cite every fact", and an
   *exact* refusal sentence, which I can detect in code.
4. **Low temperature (0.1).** Less creative, more factual and repeatable.
5. **Citations + snippets shown to the user.** Anyone can verify the answer.

Layers 2 and 3 cover different cases. "What is the salary of a software engineer?" scores
**0.45**: it's on-topic (company, employees), so it passes the threshold, but the LLM
correctly says "not found". The threshold alone can't catch that; the prompt can.

### Design decisions you should be able to defend
- **Why a system prompt *and* a user prompt?** The system message holds standing rules
  the model treats as higher priority; the user message holds this request's data
  (context + question). Keeping them separate makes the rules harder to override.
- **Why an exact refusal sentence?** So the code can detect it reliably (`_is_not_found`)
  and hide the sources, and so the UI and evaluation can count "not found" answers.
- **Why one `generate()` function?** This is the *adapter* idea. The pipeline doesn't care
  whether Groq or Ollama answers, so switching is one `.env` line and adding OpenAI
  would be one new function.
- **Why did the model change from Llama 3?** When I built it, Groq had retired its
  Llama 3 chat models. I listed the models my key could use, tested `gpt-oss-20b` and
  `qwen3.8-27b` on the same RAG prompt, and picked Qwen because it followed the citation
  format most reliably. Because the model is just a `.env` setting, it was a one-line
  change. *(This is a good real-world story: APIs change, so design for it.)*
- **Why Top-K = 4?** About 4 × 800 characters is ≈ 3,200 characters of context: enough
  to cover an answer that spans two chunks, small enough to stay fast and focused.
  Phase 5's hit rate is how I'd tune it.
- **Why return `used_llm`, `best_score` and `time_taken`?** For debugging, the logs
  (Phase 4) and evaluation (Phase 5): you can't improve what you don't measure.

### If the interviewer asks…
- **"What is RAG?"** Retrieval-Augmented Generation. Instead of relying on what the LLM
  memorised during training, you *retrieve* relevant text from your own documents and
  *augment* the prompt with it, so the LLM *generates* an answer grounded in that text.
  This gives up-to-date, private, verifiable answers without retraining the model.
- **"Why not fine-tune instead?"** Fine-tuning is expensive, slow to update, and still
  hallucinates without citing anything. With RAG, adding a document takes seconds and
  every answer shows its source.
- **"What if the answer is spread across many chunks?"** Increase Top-K, or add re-ranking
  (retrieve 20, use a cross-encoder to keep the best 4). Both are listed as future work.
- **"What does temperature do?"** It controls randomness when choosing the next token.
  Near 0 the model picks the most likely tokens: factual and repeatable.
- **"What happens if Groq is down?"** `LLMError` gives a clear message, and switching
  `LLM_PROVIDER=ollama` runs everything locally.

### How to run / test
```bash
python scripts/ask_questions.py
python scripts/ask_questions.py "What is the salary of a software engineer at Northwind?"
```
Results (Qwen3 on Groq): sick leave → p.2 ✅, precision vs recall → p.4 ✅,
overfitting → p.3 ✅, FIFA → "not found" in 0.02 s with no LLM call ✅.
Extra checks: salary and gradient descent → the LLM refused ✅; "Can I work from home?"
(the lowest valid score, 0.34) → answered from p.3 ✅. Missing key, wrong key and
Ollama not running each show a clear one-line error ✅. Typical answer time: 0.8–3.6 s.

---

## Phase 4 – Streamlit UI + Query Logs

### The 30-second answer
> "The UI is a Streamlit app with a sidebar and two tabs. The sidebar handles the
> knowledge base: multi-file upload, a document list with a delete button for each file,
> a 'Clear all' that asks for confirmation, and a Top-K slider. The Chat tab keeps the
> conversation in `st.session_state` and shows each answer with its response time, best
> score and an expandable Sources section. Every question is appended to a JSONL log,
> and the Query Logs tab shows the history plus metrics like average response time.
> The UI only *calls* the pipeline: `app.py` has no RAG logic of its own."

### What each file does
| File | Purpose | Key detail to mention |
|---|---|---|
| `app.py` | All UI code: sidebar, Chat tab, Query Logs tab | Calls `ingest_file`, `answer_question` and `vector_store`, so there's no RAG logic in the UI |
| `src/logger.py` | `log_query`, `read_logs`, `clear_logs` | JSON Lines, append-only; damaged lines are skipped instead of crashing |
| `.claude/launch.json` | Dev-server config used to run and test the app | Not needed by end users |

### How Streamlit works (the #1 thing to understand)
- **The whole script re-runs from top to bottom on every interaction.** Clicking a button,
  typing a message or moving a slider all trigger a fresh run.
- So anything that must survive (the chat history) lives in **`st.session_state`**, a
  dictionary kept per browser session. Without it, every new question would erase the old ones.
- Heavy objects are loaded **once per server process**: the embedding model and the
  ChromaDB client are wrapped in `@lru_cache`, and Python keeps imported modules in memory
  between re-runs. That's why the first question takes ~3 s and later ones ~0.5–0.8 s.
- `st.rerun()` forces an immediate re-run, which I use after deleting a document so
  the sidebar list updates straight away.

### UI design decisions you should be able to defend
- **An "Add to knowledge base" button instead of auto-ingesting on upload.** Because the
  script re-runs constantly, auto-processing would re-hash the files on *every* click.
  The button makes ingestion happen once. After it runs, I change the uploader's `key`,
  which is a standard trick to reset a Streamlit widget.
- **Two-step "Clear all".** It's destructive and can't be undone, so the first click only
  shows a warning with "Yes, delete" and "Cancel". A flag in `session_state` remembers
  that we're in the "confirming" state across re-runs.
- **Chat input below the messages.** I reserve a `st.container()` for the conversation
  *before* drawing the input box, then write new messages into that container. *(I first
  had the input above the messages, spotted it while testing in the browser, and fixed it.)*
- **Friendly errors, not stack traces.** A missing API key shows a red explanation and
  *disables* the chat box. LLM failures (no internet, rate limit) are caught as `LLMError`
  and shown as a ⚠️ message, and they're logged too, so errors show up in the Logs tab.
- **Sources under each answer.** File, page, similarity score, a 300-character snippet,
  and ✅ on chunks the answer actually cites. This is what makes the answers *verifiable*.
- **"LLM skipped" caption.** It makes the threshold guard from Phase 3 visible: you can see
  that an off-topic question was answered in 0.03 s without calling the LLM.
- **The Top-K slider is passed into `answer_question(top_k=...)`.** The pipeline already
  took a `top_k` argument, so the UI could add this feature without changing the Phase 3 logic.

### Why JSON Lines for logs?
- **Append-only**: writing a log entry is one line added at the end, so it's fast and an
  app crash can't corrupt earlier entries (unlike rewriting one big JSON array).
- **Easy to analyse**: `pandas.read_json("logs/queries.jsonl", lines=True)` loads it straight away.
- **Small**: I store file, page and score for the sources, not the full snippets.
- **Git-ignored**: logs can contain users' questions, which may be private.
- In production I'd use a database (SQLite/Postgres) or a logging service, but for one
  user JSONL is the simplest thing that works.

### If the interviewer asks…
- **"What is `st.session_state`?"** — A per-user dictionary that survives Streamlit's
  re-runs. I keep the chat history, the uploader key and the "confirm clear" flag there.
- **"Why is the UI separate from the pipeline?"** — Separation of concerns. The same
  `answer_question()` is used by the CLI script and the web app, and it will be reused by
  the Quiz Generator. I could swap Streamlit for FastAPI + React without touching `src/`.
- **"How would multiple users work?"** — Each browser tab gets its own `session_state`
  (separate chats), but they share one ChromaDB, so everyone sees the same documents.
  For real multi-user use I'd add login and store a `user_id` in each chunk's metadata,
  then filter searches with `where={"user_id": ...}`.
- **"How did you test the UI?"** — Manually in a browser (upload, duplicate upload,
  delete, Top-K, history, logs), plus Streamlit's `AppTest` for the destructive and
  config cases: "Clear all → Yes" and a missing API key.
- **"What was hard?"** — Streamlit's re-run model. Things like the uploader re-processing
  files, or the input box appearing above messages, only make sense once you understand
  that the script runs top to bottom on every click. *(Also a real story: the app first
  crashed because the laptop was out of memory. I diagnosed it with the Windows memory
  counters: 0.5 GB of commit memory was free and torch needs about 1 GB.)*

### How to run / test
```bash
streamlit run app.py
```
Tested in the browser: 3 sample docs listed; sick-leave question → cites p.2 with 4
sources, then 1 source at Top-K = 1; overfitting → p.3; FIFA → "not found", LLM skipped,
0.03 s; uploaded `cafeteria_notice.txt` → answered from it; same content under another name
→ "Already ingested"; deleted it → the same question now says "not found"; Clear all →
Cancel keeps everything; Query Logs → 6 questions, avg 0.82 s, 4/6 found, and it survives
a restart. With `AppTest`: missing key → red error + disabled chat; Clear all → Yes →
16 chunks removed and the empty-state hint shown.

---

## Phase 5 – Evaluation + Polish

### The 30-second answer
> "I didn't want to say 'it seems to work', so I built an evaluation. I wrote 15 test
> questions: 12 with a known answer location (file and page) and 3 that are not in the
> documents. The script measures retrieval separately from generation: the retrieval hit
> rate is whether the right page is in the top-K chunks, and citation accuracy is whether
> the final answer is right and cites that page. It also checks the 'not found' rate and
> response time. On my sample set everything scored 100% with a 1.35 s average, and I'm
> honest that this set is small, so the next step is real-world questions from the query logs."

### What each file does
| File | Purpose | Key detail to mention |
|---|---|---|
| `eval/test_questions.json` | 15 questions with `expected_source` + `expected_page` | Questions are **reworded**, not copied from the docs; 3 have `null` as the expected answer |
| `eval/evaluate.py` | Runs retrieval-only *and* the full pipeline per question | Uses the same `answer_question()` as the app, so it tests what users get |
| `eval/results.md` | Auto-generated report: summary, per-question table, failures | Includes the config (model, Top-K, threshold), so results are reproducible |
| `README.md` | Overview, features, Mermaid diagram, setup, results, model story, future work | It's the project's first impression on GitHub |

### The metrics, in simple words
- **Retrieval hit rate @K**: "Did the search bring back the page that holds the answer?"
  12/12. If this fails, nothing after it can succeed, so I measure it **separately**.
- **MRR (Mean Reciprocal Rank)**: "How high up was the correct page?" Rank 1 → 1.0,
  rank 2 → 0.5, rank 4 → 0.25, then averaged. We got 1.00, meaning it was always ranked first.
- **Citation accuracy**: the end-to-end check, meaning the answer was found *and* it contains
  `[expected_file, p.N]`. 12/12.
- **Correct "not found" rate**: of the out-of-docs questions, how many were refused instead
  of answered with a guess? 3/3. This measures **hallucination resistance**.
- **Response time**: 1.35 s on average. Normal answers take 0.6–1.9 s; LLM refusals took
  about 5 s, which is interesting because the model "thinks" longer before refusing. The
  off-topic question took 0.02 s because the threshold skipped the LLM.

### Why I designed the test set this way
- **The 3 not-found questions cover both safety layers**: FIFA (off-topic → threshold),
  salary (on-topic company question → the LLM must refuse), and gradient descent (an ML
  topic the notes don't cover → the LLM must refuse).
- **Reworded questions** ("paid vacation days" instead of "annual leave") test *meaning*-based
  search, which is the whole point of embeddings.
- **Honest limitation**: there are only 16 chunks and I wrote both the documents and the questions,
  so 100% is optimistic. Saying this in an interview shows maturity. The fix is a bigger
  test set built from real PDFs and real user questions from `logs/queries.jsonl`.

### Fresh-install check
I copied only the files Git would include into a clean folder (no `venv`, no `.env`, no
database), added a fresh `.env`, ingested the samples into a brand-new ChromaDB, ran the
CLI and the evaluation (100% again), started `streamlit run app.py`, and asked a chat
question through Streamlit's `AppTest`. This proves the repo doesn't secretly depend on
files that only exist on my laptop. A from-scratch `pip install -r requirements.txt` in a
new venv couldn't be completed that day because PyPI was unreachable from my network (every
other site worked), so it was re-run later. *(Interview tip: say what you could and
couldn't verify. Don't claim a test you didn't run.)*

### If the interviewer asks…
- **"How do you know your RAG system works?"** I measure retrieval and generation
  separately (hit rate, MRR, citation accuracy), plus a not-found rate for hallucination
  and latency, and save a reproducible report.
- **"Why measure retrieval separately?"** To locate failures. A low hit rate means fix
  chunking, embeddings or Top-K; a high hit rate but low citation accuracy means fix the prompt or LLM.
- **"How would you evaluate at scale?"** A larger labelled set; LLM-as-a-judge for
  answer correctness and faithfulness (tools like RAGAS); and tracking metrics on every change in CI.

---

## 🎯 Final Interview Cheat Sheet (PLAN.md section 6)

**1. What is RAG and why use it?**
Retrieval-Augmented Generation: *retrieve* relevant chunks from your own documents,
*augment* the prompt with them, and let the LLM *generate* an answer grounded in that text.
LLMs don't know your private or recent documents and can hallucinate. RAG gives up-to-date,
private, **verifiable** answers (with citations) without retraining the model. It's
cheaper and faster to update than fine-tuning: adding a document takes seconds.

**2. Why does chunk overlap matter?**
Chunks are cut every ~800 characters. Without overlap, a sentence on a boundary is split
across two chunks and neither one holds the full idea. A 150-character overlap repeats the end of one
chunk at the start of the next, so boundary information isn't lost. Too much overlap
just wastes storage and returns near-duplicate chunks.

**3. What are embeddings and cosine similarity?**
An embedding is a list of numbers (384 for MiniLM) that represents the *meaning* of a text;
similar meanings give vectors pointing in similar directions, even with different words.
Cosine similarity measures the angle between two vectors: 1 = same meaning, ~0 = unrelated.
I normalise the vectors, so it's just a dot product. Chroma returns distance = 1 − similarity.

**4. Why ChromaDB?**
Free, open-source, runs locally with no server, persists to disk, stores text + metadata +
vectors together, supports metadata filters (used for delete/list by file), and uses an
HNSW index for fast nearest-neighbour search. For production scale I'd consider pgvector,
Qdrant or Pinecone, and only `vector_store.py` would change.

**5. How do I reduce hallucination?**
Five layers: (1) answers are grounded in retrieved text; (2) a similarity threshold of 0.25,
**chosen by measuring** real scores, skips the LLM for off-topic questions; (3) a strict
prompt: only use the context, cite every fact, and use an exact refusal sentence that the code detects;
(4) temperature 0.1; (5) visible sources so users can verify. Measured: 3/3 correct refusals.

**6. What does the hit rate mean?**
The share of test questions where the chunk containing the answer was among the top-K
retrieved. It measures **retrieval quality only**, separate from the LLM. Mine: 100% @4,
with MRR 1.00 (always ranked first) on the sample set.

**7. How would I improve the project?**
- **Re-ranking**: retrieve 20 chunks, re-score them with a cross-encoder (which reads the question and
  chunk *together*, so it's more accurate but slower), and keep the top 4.
- **Hybrid search**: add BM25 keyword search to catch exact names, codes and numbers that
  embeddings can miss, then merge the two result lists.
- **Bigger or better models**: stronger embeddings (bge, e5) and larger LLMs, with the evaluation
  script deciding whether each change actually helps.
- Plus OCR, streaming answers, multi-user support, CI-run evaluation, and the **AI Exam & Quiz
  Generator**, which reuses `src/` unchanged.

**Bonus story: "Tell me about a problem you faced."**
Groq retired Llama 3 mid-project. I listed the available models through the API, tested two
candidates on my real RAG prompt, picked Qwen3 because it followed the citation format
best, and since the model was a `.env` setting it was a one-line change. Lesson: design
for external APIs changing.
