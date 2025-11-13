import streamlit as st
import os
import requests
import chromadb
from dotenv import load_dotenv
import time
import re

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
assert TOGETHER_API_KEY, "Bitte TOGETHER_API_KEY als Umgebungsvariable setzen."

MIXTRAL_MODEL = os.getenv("LLM_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://ollama:11434/api/embeddings")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://ollama:11434")
EMBED_FALLBACKS = [m.strip() for m in os.getenv("EMBED_FALLBACKS", "mxbai-embed-large,all-minilm").split(",") if m.strip()]
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "gesetzestexte")

st.set_page_config(page_title="Proposify", layout="wide")
col1, col2 = st.columns([1, 1])
with col1:
    st.image("assets/logo.png", width=500)
with col2:
    st.markdown(" ")

if "history" not in st.session_state:
    st.session_state.history = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "recent_sources" not in st.session_state:
    st.session_state.recent_sources = []
if "mode" not in st.session_state:
    st.session_state.mode = "Research question helper"
if "persona" not in st.session_state:
    st.session_state.persona = "Helper"

def llm_complete(prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    resp = requests.post(
        "https://api.together.xyz/v1/completions",
        headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
        json={
            "model": MIXTRAL_MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"LLM error: {resp.status_code} — {resp.text}")
    return resp.json().get("choices", [{}])[0].get("text", "").strip()

SELF_REF_RE = re.compile(r"\[Self-Reflection Checklist\].*?(?:\Z|\n{2,})", re.IGNORECASE | re.DOTALL)
PROMPTY_RE = re.compile(r"\[Write (?:the )?updated summary below\]\s*", re.IGNORECASE)

def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = SELF_REF_RE.sub("", text)
    text = PROMPTY_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def update_summary_ephemeral(history, current_summary):
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

def ollama_pull(model_name: str):
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/pull", json={"name": model_name}, timeout=600)
        return r.status_code, r.text
    except Exception as e:
        return 0, str(e)

def ensure_embedding_model(model_name: str) -> str:
    status, _ = ollama_pull(model_name)
    if status in (200, 201):
        return model_name
    for alt in EMBED_FALLBACKS:
        s2, _ = ollama_pull(alt)
        if s2 in (200, 201):
            return alt
    return model_name

def embed_text_ollama(text: str):
    global EMBEDDING_MODEL
    def _embed(model):
        r = requests.post(EMBEDDING_URL, json={"model": model, "prompt": text}, timeout=120)
        return r
    r = _embed(EMBEDDING_MODEL)
    if r.status_code == 200:
        return r.json().get("embedding")
    if r.status_code == 404 and "not found" in r.text.lower():
        chosen = ensure_embedding_model(EMBEDDING_MODEL)
        r2 = _embed(chosen)
        if r2.status_code == 200:
            EMBEDDING_MODEL = chosen
            return r2.json().get("embedding")
        for alt in EMBED_FALLBACKS:
            r3 = _embed(alt)
            if r3.status_code == 200:
                EMBEDDING_MODEL = alt
                return r3.json().get("embedding")
    raise RuntimeError(f"Embedding error: {r.status_code} — {r.text}")

def chroma_client_retry(max_attempts=10, delay=2.0):
    for attempt in range(max_attempts):
        try:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            client.heartbeat()
            return client
        except Exception:
            if attempt == max_attempts - 1:
                raise
            time.sleep(delay)

def retrieve_context(question: str, n_results: int = 5):
    q_emb = embed_text_ollama(question)
    client = chroma_client_retry()
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    result = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    return docs, metas

def answer_with_rag_and_memory(question: str) -> dict:
    docs, metas = retrieve_context(question, n_results=5)
    context = "\n\n".join(docs)
    st.session_state.recent_sources = metas
    mode = st.session_state.mode
    mode_instr = {
        "Research question helper": "Help the student define or refine a precise, feasible research question; ask clarifying questions; propose concrete next steps.",
        "Proposal refinement assistant": "Guide refinement of structure, methods, datasets, ethics, and BFH compliance; provide step-by-step edits and a short improvement plan."
    }[mode]
    persona = st.session_state.persona
    persona_map = {
        "Supervisor": {
            "temp": 0.1,
            "instr": "Strict critique. Identify weaknesses, risks, missing operationalization, measurement issues, and BFH compliance gaps. Add a short self-reflection checklist at the end."
        },
        "Helper": {
            "temp": 0.2,
            "instr": "Guided drafting. Provide structure, step-by-step guidance, short examples, and concrete next actions."
        },
        "Creative": {
            "temp": 0.7,
            "instr": "Brainstorm innovative topics and angles. Diverge with multiple ideas, then converge to 2–3 concrete candidates with crisp research questions and feasibility notes."
        }
    }
    style = persona_map.get(persona, persona_map["Helper"])
    recent_qas_text = "\n\n".join(
        [f"Q: {h.get('frage', '')}\nA: {h.get('antwort', '')}" for h in st.session_state.history if h.get("kind", "chat") == "chat"]
    )
    summary = st.session_state.summary
    prompt = f"""You are a BFH thesis-proposal assistant.
Use the provided context responsibly and cite helpful source titles if available.
Mode: {mode}. Instruction: {mode_instr}
Style: {persona} — {style["instr"]}

[Conversation Summary]
{summary}

[Recent Q&A]
{recent_qas_text if recent_qas_text else 'None'}

[Retrieved Context]
{context}

[User Question]
{question}

[Assistant Answer]
Provide a helpful, concise answer grounded in the context. If the context is insufficient, state what is missing and propose next steps.
If you cite, use a lightweight inline form like (Source: <title>).
"""
    model_answer = llm_complete(prompt, max_tokens=900, temperature=style["temp"])
    return {"antwort": model_answer, "quellen": metas}

st.sidebar.header("Session Controls")
st.sidebar.subheader("Assistant Mode")
modes = ["Research question helper", "Proposal refinement assistant"]
current_index = modes.index(st.session_state.mode) if st.session_state.mode in modes else 0
st.session_state.mode = st.sidebar.radio(
    "Select a mode",
    modes,
    index=current_index,
    label_visibility="collapsed",
    key="mode_select",
)

st.sidebar.subheader("Answer style")
st.session_state.persona = st.sidebar.radio(
    "Choose style", ["Supervisor", "Helper", "Creative"], index=["Supervisor","Helper","Creative"].index(st.session_state.persona)
)

if st.sidebar.button("Summarize conversation"):
    summary_text = st.session_state.summary or "No summary yet - start chatting!"
    st.sidebar.markdown("### Session Summary")
    st.sidebar.write(summary_text)

if st.sidebar.button("Reset session"):
    st.session_state.history = []
    st.session_state.summary = ""
    st.session_state.recent_sources = []
    st.experimental_rerun()

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
        st.session_state.history.append({
            "kind": "chat",
            "frage": frage,
            "antwort": result["antwort"],
            "quellen": result["quellen"]
        })
        st.session_state.summary = update_summary_ephemeral(
            history=st.session_state.history, current_summary=st.session_state.summary
        )
        st.rerun()
