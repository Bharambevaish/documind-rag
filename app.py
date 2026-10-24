"""
app.py - the Streamlit web interface for DocuMind.

This file contains ONLY user-interface code. All the real work (loading,
chunking, embedding, searching, answering) lives in `src/`, so the same
pipeline can be reused by other apps such as the planned Quiz Generator.

How Streamlit works (important for understanding this file):
  * The whole script re-runs from top to bottom on EVERY click or message.
  * Anything that must survive a re-run (the chat history) is kept in
    `st.session_state`, a dictionary that lives for the user's browser session.

Run:  streamlit run app.py
"""

import streamlit as st

from src import vector_store
from src.config import settings
from src.ingest import ingest_file
from src.llm import LLMError
from src.logger import clear_logs, log_query, read_logs
from src.rag_pipeline import answer_question

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Session state: values that must survive Streamlit's re-runs
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []        # chat history: [{"role", "content", "result"}]
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0     # changing it empties the file uploader
if "upload_results" not in st.session_state:
    st.session_state.upload_results = []  # messages from the last upload
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False


# ---------------------------------------------------------------------------
# Helper: show an answer's sources
# ---------------------------------------------------------------------------
def render_sources(result: dict) -> None:
    """Show timing info and an expandable list of the chunks used for the answer."""
    best = f"{result['best_score']:.2f}" if result.get("best_score") is not None else "n/a"
    note = "" if result.get("used_llm", True) else " · LLM skipped (no relevant text)"
    st.caption(f"⏱ {result['time_taken']}s · best match score {best}{note}")

    if not result["sources"]:
        return
    with st.expander(f"📚 Sources ({len(result['sources'])})"):
        for s in result["sources"]:
            cited = " · ✅ cited" if s["cited"] else ""
            st.markdown(f"**{s['source']}**, page {s['page']} · score {s['score']:.2f}{cited}")
            st.caption(s["snippet"])


# ---------------------------------------------------------------------------
# Sidebar: upload, document list, settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 DocuMind")
    st.caption("Ask questions about your documents.")

    # --- Upload ---
    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT or MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    # A button (instead of processing straight away) means files are only
    # ingested once, not again on every re-run of the script.
    if st.button("Add to knowledge base", disabled=not uploaded_files, type="primary",
                 width="stretch"):
        results = []
        with st.spinner("Reading, chunking and embedding..."):
            for file in uploaded_files:
                try:
                    r = ingest_file(file.getvalue(), file.name)
                    results.append(("info" if r["status"] == "skipped" else "success",
                                    f"{file.name}: {r['message']}"))
                except Exception as e:  # one bad file shouldn't stop the others
                    results.append(("error", f"{file.name}: {e}"))
        st.session_state.upload_results = results
        st.session_state.uploader_key += 1  # empty the uploader for the next batch
        st.rerun()

    for kind, message in st.session_state.upload_results:
        getattr(st, kind)(message)  # st.success / st.info / st.error

    # --- Document list with delete buttons ---
    st.subheader("Your documents")
    documents = vector_store.list_documents()
    if not documents:
        st.caption("No documents yet.")
    for doc in documents:
        name_col, delete_col = st.columns([5, 1])
        pages = f"{doc['pages']} page" + ("s" if doc["pages"] != 1 else "")
        chunks = f"{doc['chunks']} chunk" + ("s" if doc["chunks"] != 1 else "")
        name_col.markdown(f"**{doc['source']}**  \n{pages} · {chunks}")
        if delete_col.button("🗑️", key=f"delete_{doc['source']}", help=f"Delete {doc['source']}"):
            vector_store.delete_document(doc["source"])
            st.session_state.upload_results = [("info", f"Deleted {doc['source']}.")]
            st.rerun()

    # --- Clear all (asks for confirmation first, because it can't be undone) ---
    if documents:
        if not st.session_state.confirm_clear:
            if st.button("Clear all documents", width="stretch"):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("Delete ALL documents? This can't be undone.")
            yes_col, no_col = st.columns(2)
            if yes_col.button("Yes, delete", type="primary", width="stretch"):
                removed = vector_store.clear_all()
                st.session_state.confirm_clear = False
                st.session_state.upload_results = [("info", f"Cleared {removed} chunks.")]
                st.rerun()
            if no_col.button("Cancel", width="stretch"):
                st.session_state.confirm_clear = False
                st.rerun()

    # --- Settings ---
    st.subheader("Settings")
    top_k = st.slider("Chunks to retrieve (Top-K)", min_value=1, max_value=10, value=settings.top_k,
                      help="More chunks = more context for the LLM, but slower and noisier.")
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"LLM: {settings.llm_provider} · {settings.model_name}")


# ---------------------------------------------------------------------------
# Main area: Chat tab and Query Logs tab
# ---------------------------------------------------------------------------
chat_tab, logs_tab = st.tabs(["💬 Chat", "📜 Query Logs"])

with chat_tab:
    # Friendly warnings instead of crashes.
    missing_key = settings.llm_provider == "groq" and not settings.groq_api_key
    if missing_key:
        st.error("**GROQ_API_KEY is missing.** Copy `.env.example` to `.env`, add your free key "
                 "from https://console.groq.com/keys, then restart the app. "
                 "(Or set `LLM_PROVIDER=ollama` to run a local model.)")
    elif not documents:
        st.info("👈 Upload a PDF, TXT or MD file in the sidebar to get started.")

    # Reserve space for the conversation FIRST, then draw the input box below it.
    # New messages are written into this container, so they appear above the input.
    conversation = st.container()
    question = st.chat_input("Ask a question about your documents...", disabled=missing_key)

    with conversation:
        # Re-draw the whole chat history (the script re-runs on every message).
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("result"):
                    render_sources(message["result"])

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching your documents..."):
                    try:
                        result = answer_question(question, top_k=top_k)
                        log_query(result, top_k)
                        reply = {"role": "assistant", "content": result["answer"], "result": result}
                    except (LLMError, ValueError) as e:
                        log_query({"question": question}, top_k, error=str(e))
                        reply = {"role": "assistant", "content": f"⚠️ {e}"}
                st.markdown(reply["content"])
                if reply.get("result"):
                    render_sources(reply["result"])
            st.session_state.messages.append(reply)

with logs_tab:
    logs = read_logs()
    if not logs:
        st.info("No questions asked yet. Your questions and response times will appear here.")
    else:
        answered = [e for e in logs if not e.get("error")]
        times = [e["time_taken"] for e in answered if e.get("time_taken") is not None]
        found = sum(1 for e in answered if e.get("found"))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Questions asked", len(logs))
        c2.metric("Avg response time", f"{sum(times) / len(times):.2f}s" if times else "n/a")
        c3.metric("Answered from docs", f"{found}/{len(answered)}")
        c4.metric("Errors", len(logs) - len(answered))

        st.dataframe(
            [
                {
                    "Time": e["timestamp"].replace("T", " "),
                    "Question": e["question"],
                    "Found": "✅" if e.get("found") else ("⚠️" if e.get("error") else "❌"),
                    "Seconds": e.get("time_taken"),
                    "Best score": e.get("best_score"),
                    "Sources": ", ".join(f"{s['source']} p.{s['page']}" for s in e.get("sources", [])),
                    "Model": e.get("model"),
                    "Error": e.get("error") or "",
                }
                for e in logs
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "Seconds": st.column_config.NumberColumn(format="%.2f"),
                "Best score": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        if st.button("Clear logs"):
            clear_logs()
            st.rerun()
