# app.py

import base64
import html
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langfuse.types import TraceContext
import tiktoken

from prompts import MODE_INSTR, PERSONA_MAP  # optional, for future UI use
from rag_tools import summarize_uploaded_papers, llm_complete
from graph_config import AppState, rag_graph
from proposal_graph_config import ProposalState, proposal_graph #anna

# -------- env + Langfuse -------- #

load_dotenv()

LANGFUSE_HANDLER = LangfuseCallbackHandler()
LANGFUSE_CLIENT = None
if (
    os.getenv("LANGFUSE_PUBLIC_KEY")
    and os.getenv("LANGFUSE_SECRET_KEY")
    and os.getenv("LANGFUSE_BASE_URL")
):
    try:
        LANGFUSE_CLIENT = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            base_url=os.getenv("LANGFUSE_BASE_URL"),
        )
    except Exception as exc:
        print(f"Langfuse client disabled: {exc}")
        LANGFUSE_CLIENT = None

# -------- Streamlit setup -------- #

st.set_page_config(page_title="Proposify", layout="wide")


def inject_custom_css():
    css_path = Path("styles/custom.css")
    css_chunks = []
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_chunks.append(f.read())

    bg_path = Path("assets/bcg.png")
    if bg_path.exists():
        encoded = base64.b64encode(bg_path.read_bytes()).decode()
        css_chunks.append(
            ".stApp {"
            f"background-image: url('data:image/png;base64,{encoded}');"
            "background-size: cover;"
            "background-position: center;"
            "background-attachment: fixed;"
            "}"
        )

    if css_chunks:
        st.markdown(f"<style>{''.join(css_chunks)}</style>", unsafe_allow_html=True)


inject_custom_css()
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
if "langfuse_trace_id" not in st.session_state:
    st.session_state.langfuse_trace_id = None

MEMORY_MAX_TOKENS = 1800
ENCODING = tiktoken.get_encoding("cl100k_base")

BANNED_WRITE_PHRASES = [
    "write a thesis proposal for me",
    "write me a thesis proposal",
    "write thesis proposal for me",
    "write my thesis proposal",
    "write the proposal for me",
    "create a full proposal for me",
    "create my thesis proposal",
    "create a thesis proposal for me",
    "prepare my thesis proposal",
    "prepare a thesis proposal for me",
    "complete the proposal for me",
    "do my thesis proposal",
    "draft the entire proposal",
    "draft my thesis proposal",
    "draft the thesis proposal for me",
    "draft me a thesis proposal",
    "finish the proposal for me",
]

WRITE_COMMAND_RE = re.compile(
    r"^(?:please\s+|kindly\s+)?"
    r"(?:(?:can|could|would|will)\s+you\s+|i\s+need\s+you\s+to\s+|i['’]d\s+like\s+you\s+to\s+)?"
    r"(?:help\s+me\s+)?"
    r"(?:write|draft|create|prepare|complete|finish|produce|compose)\b"
    r".*?\bthesis proposal\b",
    re.IGNORECASE | re.DOTALL,
)

HERO_EXAMPLES = [
    {
        "title": "1. Assistant modes",
        "body": "Choose your desired Assistant - Research question helper for ... or Proposal refinement assistant if you need help or have any questions regarding your thesis proposal.",
    },
    {
        "title": "2. Answer styles",
        "body": "Choose one of three answer styles for better answers! Supervisor if you need critique, Helper if you need some advice regarding your proposal or Creative if you want some crazy ideas!",
    },
    {
        "title": "3. Upload your files",
        "body": "Upload your notes if you have them and let our smart assistant craft a professional proposal for you."
    }
]

HERO_NOTE = (
    "Simply tell us your topic or research interests - upload your notes if you have them "
    "and let our assistant refine your thesis proposal."
)

USER_AVATAR = "assets/user.png"
ASSISTANT_AVATAR = "assets/chat.png"

# -------- hero + styling helpers -------- #

def _time_greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


def render_hero_section():
    greeting = _time_greeting()
    st.markdown(
        f"<div class='hero-title'>{html.escape(greeting)}, how can I help you today?</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='hero-subtitle'>Tell me your thesis topic, research question, or general area of interest.</p>",
        unsafe_allow_html=True,
    )

    cards_html = "".join(
        f"<div class='hero-card'><strong>{html.escape(item['title'])}</strong>{html.escape(item['body'])}</div>"
        for item in HERO_EXAMPLES
    )
    st.markdown(f"<div class='hero-cards'>{cards_html}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<p class='hero-subtitle' style='margin-top:1.2rem;'>{html.escape(HERO_NOTE)}</p>",
        unsafe_allow_html=True,
    )

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


def build_recent_history(chat_history: List[Dict[str, Any]], max_tokens: int) -> str:
    if not chat_history:
        return "None"

    tokens_used = 0
    segments: List[str] = []
    for item in reversed(chat_history):
        if item.get("kind", "chat") != "chat":
            continue
        user = _clean_text(item.get("frage", ""))
        assistant = _clean_text(item.get("antwort", ""))
        if not (user or assistant):
            continue
        segment = f"Q: {user}\nA: {assistant}"
        segment_tokens = len(ENCODING.encode(segment))
        if tokens_used + segment_tokens > max_tokens and segments:
            break
        segments.append(segment)
        tokens_used += segment_tokens
        if tokens_used >= max_tokens:
            break

    return "\n\n".join(reversed(segments)) if segments else "None"


def _is_full_proposal_request(question: str) -> bool:
    if not question:
        return False
    lowered = question.lower()
    for phrase in BANNED_WRITE_PHRASES:
        if phrase in lowered:
            return True
    normalized = lowered.strip()
    normalized = re.sub(r"^(?:hey|hi|hello|dear)(?: there)?[,\s]+", "", normalized)
    if WRITE_COMMAND_RE.match(normalized):
        return True
    return False


# -------- wrapper: call LangGraph -------- #

def answer_with_rag_and_memory(question: str) -> Dict[str, Any]:
    if _is_full_proposal_request(question):
        warning = (
            "I can't write the entire thesis proposal for you. "
            "Please ask to refine your own draft, structure sections "
            "or ask about research design details. "
            "Don't forget to send your own draft for the refinement."
        )
        st.session_state.last_task = "blocked_full_proposal"
        st.session_state.recent_sources = []
        return {"antwort": warning, "quellen": []}

    recent_qas_text = build_recent_history(
        [h for h in st.session_state.history if h.get("kind", "chat") == "chat"],
        max_tokens=MEMORY_MAX_TOKENS,
    )

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
            "paper_summaries": st.session_state.paper_summaries,
            "metadatas": [],
            "context_docs": [],
            "next_step": "",
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
            # summaries of uploaded PDFs (filled via sidebar button)
            "paper_summaries": st.session_state.paper_summaries,
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
    st.session_state.langfuse_trace_id = None
    st.rerun()

# -------- UI: main chat -------- #

render_hero_section()

# replay history
for item in st.session_state.history:
    if item.get("kind", "chat") != "chat":
        continue
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(item.get("frage", ""))
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown(item.get("antwort", ""), unsafe_allow_html=True)
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

frage = st.chat_input("Tell us your thesis topic, research question, or proposal challenge...")

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
        if LANGFUSE_CLIENT:
            try:
                if not st.session_state.langfuse_trace_id:
                    st.session_state.langfuse_trace_id = LANGFUSE_CLIENT.create_trace_id()
                trace_context = TraceContext(trace_id=st.session_state.langfuse_trace_id)
                with LANGFUSE_CLIENT.start_as_current_span(
                    trace_context=trace_context,
                    name="proposal-chat-turn",
                    input={"question": frage},
                    output={
                        "answer": result["antwort"],
                        "sources": result.get("quellen", []),
                    },
                    metadata={
                        "mode": st.session_state.mode,
                        "persona": st.session_state.persona,
                        "task": st.session_state.last_task,
                        "turn_index": len(st.session_state.history),
                    },
                ):
                    LANGFUSE_CLIENT.update_current_trace(
                        name="proposal-chat",
                        user_id=st.session_state.get("upload_collection_name", "unknown"),
                        session_id=st.session_state.get("upload_collection_name", "unknown"),
                        input={
                            "question": frage,
                            "recent_summary": st.session_state.summary,
                        },
                        output=result["antwort"],
                        metadata={
                            "mode": st.session_state.mode,
                            "persona": st.session_state.persona,
                            "task": st.session_state.last_task,
                            "history_length": len(st.session_state.history),
                        },
                        tags=[st.session_state.mode],
                    )
                LANGFUSE_CLIENT.flush()
            except Exception as exc:
                print(f"Langfuse logging failed: {exc}")
        st.rerun()
