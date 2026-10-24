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
