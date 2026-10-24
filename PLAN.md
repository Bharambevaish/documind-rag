# Project Plan: DocuMind – RAG Document Q&A Assistant

## Instructions for Claude Code
- I am a final-year IT student building this for my AI/ML resume and job interviews.
- Build it **one phase at a time**. After each phase, stop, explain what you built in simple words, and tell me how to run and test it before moving on.
- Keep the code simple, readable and well-commented. I must be able to explain every file in an interview.
- Do not hard-code API keys. Use a `.env` file and add it to `.gitignore`.
- Use Python 3.10+ and a virtual environment (`venv`).

---

## 1. Goal
A web app where a user uploads documents (PDF, TXT, MD) and asks questions about them. The app finds the most relevant parts of the documents and uses an LLM to answer, **showing the source file and page number** for each answer. If the answer isn't in the documents, it should say so instead of making something up.

Later this pipeline will be reused for a second project: an AI Exam & Quiz Generator (generate MCQs from uploaded notes). So keep the RAG logic in reusable modules, separate from the UI.

## 2. Tech Stack
| Part | Tool |
|---|---|
| Language | Python |
| Document loading | `pypdf` (PDF), plain read for TXT/MD |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (chunk ~800 chars, overlap ~150) |
| Embeddings | `sentence-transformers` – `all-MiniLM-L6-v2` (free, runs on CPU) |
| Vector database | ChromaDB (persistent, stored locally in `data/chroma/`) |
| LLM | Groq API (free tier, Llama 3 model) as default; Ollama as optional local alternative, chosen via `.env` |
| UI | Streamlit |
| Config | `python-dotenv` |

## 3. Folder Structure
```
documind-rag/
├── app.py                  # Streamlit UI
├── src/
│   ├── __init__.py
│   ├── config.py           # settings loaded from .env
│   ├── loader.py           # read PDF/TXT/MD → text + metadata (file name, page)
│   ├── chunker.py          # split text into chunks with metadata
│   ├── embedder.py         # sentence-transformers wrapper
│   ├── vector_store.py     # ChromaDB: add, search, delete, list documents
│   ├── llm.py              # Groq / Ollama client behind one function
│   ├── rag_pipeline.py     # retrieve → build prompt → generate → return answer + sources
│   └── logger.py           # save each query, answer, sources, time taken to logs/queries.jsonl
├── eval/
│   ├── test_questions.json # ~15 questions with expected source file/page
│   └── evaluate.py         # measures retrieval hit rate and avg response time
├── data/
│   ├── sample_docs/        # 2–3 sample PDFs for demo
│   └── chroma/             # vector DB (git-ignored)
├── logs/                   # git-ignored
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 4. Build Phases

### Phase 1 – Setup (Day 1)
- Create folder structure, `venv`, `requirements.txt`, `.gitignore`, `.env.example` (`GROQ_API_KEY`, `LLM_PROVIDER=groq|ollama`, `MODEL_NAME`, `TOP_K=4`).
- `config.py` loads settings.
- Test: a small script prints the loaded config.

### Phase 2 – Ingestion (Day 1–2)
- `loader.py`: load PDF page by page, keep `{source, page}` metadata. Load TXT/MD as page 1.
- `chunker.py`: split into chunks, keep metadata, give each chunk a unique id.
- `embedder.py` + `vector_store.py`: embed chunks and store in ChromaDB. Avoid duplicates if the same file is uploaded twice. Support deleting a document.
- Test: ingest the sample PDFs and print the number of chunks stored.

### Phase 3 – Retrieval + Generation (Day 2–3)
- `vector_store.search(query, top_k)` returns chunks with similarity scores.
- `llm.py`: one `generate(prompt)` function that works with Groq or Ollama.
- `rag_pipeline.py`: build a prompt that tells the LLM to answer **only from the given context**, cite sources like `[file.pdf, p.3]`, and reply "I couldn't find this in the uploaded documents" when context is not relevant. Skip the LLM call if the best similarity score is below a threshold.
- Return: answer, list of sources (file, page, snippet, score), time taken.
- Test: ask 3 questions from the command line.

### Phase 4 – Streamlit UI (Day 3–4)
- Sidebar: upload files, list ingested documents with delete buttons, "Clear all" button, Top-K slider.
- Main area: chat interface (`st.chat_message`) with chat history in session state.
- Under each answer: expandable "Sources" section showing file, page and the text snippet used.
- Spinner while answering; friendly error if API key is missing.
- `logger.py` saves every query. Add a "Query Logs" tab showing past questions and response times.

### Phase 5 – Evaluation + Polish (Day 4–5)
- `eval/evaluate.py`: for each test question, check if the expected source appears in the top-K retrieved chunks → print **retrieval hit rate** and **average response time**. Save results to `eval/results.md`.
- README: project overview, architecture diagram (Mermaid: Upload → Chunk → Embed → ChromaDB → Retrieve → LLM → Answer + Sources), setup steps, screenshots placeholder, evaluation results, future work (Exam & Quiz Generator, hybrid search, re-ranking).
- Clean code, add docstrings, make sure `pip install -r requirements.txt` + `streamlit run app.py` works from scratch.

## 5. Done When
- [ ] Can upload multiple PDFs and ask questions about them
- [ ] Every answer shows correct source file + page
- [ ] Says "not found" for questions outside the documents
- [ ] Works with Groq (and Ollama if installed)
- [ ] Evaluation script prints hit rate and response time
- [ ] Clean README, pushed to GitHub

## 6. Interview Prep (explain these after the project is done)
After finishing, give me a short explanation of: what RAG is and why it's used, why chunk overlap matters, what embeddings and cosine similarity are, why ChromaDB, how I reduce hallucination, what the hit rate means, and how I'd improve the project (re-ranking, hybrid search, bigger models).
