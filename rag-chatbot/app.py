# app.py

import os
import re
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from prompts import MODE_INSTR, PERSONA_MAP  # optional, for future UI use
from rag_tools import summarize_uploaded_papers, llm_complete
from graph_config import AppState, rag_graph
from proposal_graph_config import ProposalState, proposal_graph #anna

# -------- env + Langfuse -------- #

load_dotenv()

LANGFUSE_HANDLER = LangfuseCallbackHandler()

# -------- Streamlit setup -------- #

st.set_page_config(page_title="Proposify", layout="wide")
col1, col2 = st.columns([1, 1])
with col1:
    st.image("assets/logo.png", width=500)
with col2:
    st.markdown(" ")

# -------- session state -------- #

if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "recent_sources" not in st.session_state:
    st.session_state.recent_sources = []
if "mode" not in st.session_state:
    st.session_state.mode = "Research question helper"
if "persona" not in st.session_state:
    st.session_state.persona = "Helper"
if "upload_collection_name" not in st.session_state:
    import uuid
    # kept for backward compatibility / tracing
    st.session_state.upload_collection_name = f"user_uploads_{uuid.uuid4().hex[:8]}"
# NEW: store summaries of uploaded papers
if "paper_summaries" not in st.session_state:
    st.session_state.paper_summaries: Dict[str, str] = {}
if "summarized_paper_count" not in st.session_state:
    st.session_state.summarized_paper_count = 0
if "last_task" not in st.session_state:
    st.session_state.last_task = "(none)"

# -------- summary utils -------- #

SELF_REF_RE = re.compile(
    r"\[Self-Reflection Checklist\].*?(?:\Z|\n{2,})",
    re.IGNORECASE | re.DOTALL,
)
PROMPTY_RE = re.compile(
    r"\[Write (?:the )?updated summary below\]\s*",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = SELF_REF_RE.sub("", text)
    text = PROMPTY_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def update_summary_ephemeral(history, current_summary: str) -> str:
    chat_turns = [h for h in history if h.get("kind", "chat") == "chat"]
    pairs = []
    for item in chat_turns:
        u = _clean_text(item.get("frage", ""))
        a = _clean_text(item.get("antwort", ""))
        if u or a:
            pairs.append(f"USER: {u}\nASSISTANT: {a}")
    recent_text = "\n\n".join(pairs) if pairs else "None"
    prompt = f"""You are a thesis-proposal assistant's memory summarizer.

Rules:
- Output at most 5 bullet points.
- No checklists, no meta-instructions, no quotes from the chat.
- Keep only: current research question, scope/constraints, chosen methods/datasets, deadlines/supervisor notes, key decisions & open TODOs.
- Omit rhetorical prompts and any 'self-reflection' text.

[Existing summary]
{_clean_text(current_summary)}

[Conversation turns to merge]
{recent_text}

[Write the updated summary as 3–5 bullets only, no header:]
"""
    try:
        new_summary = llm_complete(prompt, max_tokens=180, temperature=0.0)
        return _clean_text(new_summary) or current_summary
    except Exception:
        return current_summary


# -------- wrapper: call LangGraph -------- #

def answer_with_rag_and_memory(question: str) -> Dict[str, Any]:
    recent_qas_text = "\n\n".join(
        [
            f"Q: {h.get('frage', '')}\nA: {h.get('antwort', '')}"
            for h in st.session_state.history
            if h.get("kind", "chat") == "chat"
        ]
    ) or "None"

    mode = st.session_state.mode

    # ---------------------------
    # 1) Proposal refinement mode
    # ---------------------------
    if mode == "Proposal refinement assistant":
        initial_state: ProposalState = {
            "question": question,
            "mode": mode,
            "persona": st.session_state.persona,
            "summary": st.session_state.summary,
            "recent_qas": recent_qas_text,
            "task": "proposal_refine",
            "answer": "",
        }

        final_state = proposal_graph.invoke(
            initial_state,
            config={
                "callbacks": [LANGFUSE_HANDLER],
                "metadata": {
                    "session_id": st.session_state.get(
                        "upload_collection_name", "unknown"
                    ),
                    "mode": mode,
                    "persona": st.session_state.persona,
                },
            },
        )

    # ---------------------------
    # 2) Research question mode
    # ---------------------------
    else:
        initial_state: AppState = {
            "question": question,
            "mode": mode,
            "persona": st.session_state.persona,
            "summary": st.session_state.summary,
            "recent_qas": recent_qas_text,
            "task": "structure_question",  # router will overwrite
            "upload_collection_name": st.session_state.upload_collection_name,
            "context_docs": [],
            "selected_titles": [],
            "metadatas": [],
            "answer": "",
            # gap pipeline fields
            "gap_paper_docs": [],
            "gap_paper_metas": [],
            "gap_paper_summaries": "",
            "gap_guides": "",
            "gap_candidates": "",
            "rq_candidates": "",
            # methods pipeline fields
            "methods_task": "critique_design",
            "methods_guides": "",
        }

        final_state = rag_graph.invoke(
            initial_state,
            config={
                "callbacks": [LANGFUSE_HANDLER],
                "metadata": {
                    "session_id": st.session_state.get(
                        "upload_collection_name", "unknown"
                    ),
                    "mode": mode,
                    "persona": st.session_state.persona,
                },
            },
        )

    st.session_state.last_task = final_state.get("task", "?")
    st.session_state.recent_sources = final_state.get("metadatas", [])

    return {
        "antwort": final_state["answer"],
        "quellen": final_state.get("metadatas", []),
    }



# -------- UI: sidebar -------- #

st.sidebar.header("Session Controls")

st.sidebar.markdown(
    f"**Upload collection ID:** `{st.session_state.upload_collection_name}`"
)
st.sidebar.markdown(f"**Last agent decision:** `{st.session_state.last_task}`")
st.sidebar.markdown(
    f"**Summarized papers stored:** {st.session_state.summarized_paper_count}"
)

st.sidebar.subheader("Upload your papers")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs (papers, articles, etc.)",
    type=["pdf"],
    accept_multiple_files=True,
)
if uploaded_files and st.sidebar.button("Summarize uploaded papers"):
    summaries = summarize_uploaded_papers(uploaded_files)
    # Merge with existing ones (so you can add more later)
    st.session_state.paper_summaries.update(summaries)
    st.session_state.summarized_paper_count = len(st.session_state.paper_summaries)

    st.sidebar.success(
        f"Summarized {len(summaries)} paper(s). "
        f"Total stored: {st.session_state.summarized_paper_count}"
    )

    if summaries:
        st.sidebar.markdown("**Newly summarized:**")
        for title in summaries.keys():
            st.sidebar.markdown(f"- {title}")

st.sidebar.markdown(
    "💡 If your question is about **one specific paper**, mention its file name in the chat, "
    'e.g. _\"In **review.pdf**, what is the paper about?\"_. '
    "Otherwise, the assistant will consider all summarized papers."
)

st.sidebar.subheader("Assistant Mode")
modes = ["Research question helper", "Proposal refinement assistant"]
current_index = (
    modes.index(st.session_state.mode) if st.session_state.mode in modes else 0
)
st.session_state.mode = st.sidebar.radio(
    "Select a mode",
    modes,
    index=current_index,
    label_visibility="collapsed",
    key="mode_select",
)

st.sidebar.subheader("Answer style")
st.session_state.persona = st.sidebar.radio(
    "Choose style",
    ["Supervisor", "Helper", "Creative"],
    index=["Supervisor", "Helper", "Creative"].index(st.session_state.persona),
)

if st.sidebar.button("Summarize conversation"):
    summary_text = st.session_state.summary or "No summary yet - start chatting!"
    st.sidebar.markdown("### Session Summary")
    st.sidebar.write(summary_text)

if st.sidebar.button("Reset session"):
    st.session_state.history = []
    st.session_state.summary = ""
    st.session_state.recent_sources = []
    st.session_state.paper_summaries = {}
    st.session_state.summarized_paper_count = 0
    st.experimental_rerun()

# -------- UI: main chat -------- #

# replay history
for item in st.session_state.history:
    if item.get("kind", "chat") != "chat":
        continue
    with st.chat_message("user"):
        st.markdown(item.get("frage", ""))
    with st.chat_message("assistant"):
        st.markdown(item.get("antwort", ""))
        if item.get("quellen"):
            seen = set()
            uniq_titles = []
            for meta in item["quellen"]:
                title = (meta.get("quelle") or meta.get("title") or "Untitled").strip()
                if title not in seen:
                    seen.add(title)
                    uniq_titles.append(title)
            if uniq_titles:
                st.markdown("Sources (ephemeral):")
                for title in uniq_titles:
                    st.markdown(f"- {title}")
        st.markdown("---")

frage = st.chat_input("Ask a question...")

if frage:
    with st.spinner("Thinking..."):
        result = answer_with_rag_and_memory(frage)
        st.session_state.history.append(
            {
                "kind": "chat",
                "frage": frage,
                "antwort": result["antwort"],
                "quellen": result["quellen"],
            }
        )
        st.session_state.summary = update_summary_ephemeral(
            history=st.session_state.history,
            current_summary=st.session_state.summary,
        )
        st.rerun()
