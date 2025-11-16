# rag_tools.py

import os
import time
from typing import List, Dict, Any

import requests
import chromadb
import fitz  # PyMuPDF
from dotenv import load_dotenv

from llm_service import llm_service  # <-- BFH LLM wrapper

load_dotenv()

# --------------------------------------------------------------------
# Environment + config
# --------------------------------------------------------------------

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("Bitte TOGETHER_API_KEY als Umgebungsvariable setzen.")

MIXTRAL_MODEL = os.getenv("LLM_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

# Embeddings for STATIC KB (Creswell / BFH docs)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://ollama:11434/api/embeddings")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://ollama:11434")
EMBED_FALLBACKS = [
    m.strip()
    for m in os.getenv("EMBED_FALLBACKS", "mxbai-embed-large,all-minilm").split(",")
    if m.strip()
]

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_KB_COLLECTION = os.getenv("CHROMA_COLLECTION", "gesetzestexte")


# --------------------------------------------------------------------
# LLM (Together / Mixtral) for router, methods & gap pipelines
# --------------------------------------------------------------------

def llm_complete(prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    """
    Call Together's completion endpoint with the configured MIXTRAL_MODEL.

    Used by the LangGraph pipelines (router, methods, gap, memory summariser).
    """
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


# --------------------------------------------------------------------
# Embeddings + Chroma (ONLY for static KB: Creswell / BFH docs)
# --------------------------------------------------------------------

def ollama_pull(model_name: str):
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/pull",
            json={"name": model_name},
            timeout=600,
        )
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


def embed_text_ollama(text: str) -> List[float]:
    """
    Embed text via the local Ollama embedding endpoint.
    Used for querying the static Creswell/BFH KB in Chroma.
    """
    global EMBEDDING_MODEL

    def _embed(model):
        r = requests.post(
            EMBEDDING_URL,
            json={"model": model, "prompt": text},
            timeout=120,
        )
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


# -------- Chroma helpers (static KB only) -------- #

def get_chroma_client(max_attempts: int = 10, delay: float = 2.0):
    """
    Connect to ChromaDB (used only for the static methods KB).
    """
    for attempt in range(max_attempts):
        try:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            client.heartbeat()
            return client
        except Exception:
            if attempt == max_attempts - 1:
                raise
            time.sleep(delay)


def retrieve_kb_context(question: str, n_results: int = 5):
    """
    Retrieve from the static Creswell / BFH methods knowledge base.

    This is vector-based (Chroma) but only for the fixed KB docs,
    not for user-uploaded PDFs.
    """
    q_emb = embed_text_ollama(question)
    client = get_chroma_client()
    collection = client.get_or_create_collection(CHROMA_KB_COLLECTION)
    result = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    return docs, metas


# --------------------------------------------------------------------
# NEW: Full-paper summarization with BFH LLM (gpt-oss:120b)
# --------------------------------------------------------------------

def _extract_full_text_from_pdf(data: bytes) -> str:
    """
    Read the entire PDF (bytes) and return plain text from all pages.

    A soft character cap is applied to keep within the BFH LLM context window.
    """
    with fitz.open(stream=data, filetype="pdf") as doc:
        pages = [page.get_text() for page in doc]

    full_text = "\n\n".join(pages).strip()
    if not full_text:
        return ""

    # Safety cap: if the PDF is huge, truncate.
    max_chars = int(os.getenv("SUMMARY_MAX_CHARS", "25000"))
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]
    return full_text


def summarize_single_paper_with_bfh_llm(title: str, full_text: str) -> str:
    """
    Use the BFH LLM (ollama/gpt-oss:120b) to produce a structured summary
    of one paper.

    The summary extracts:
    - Topic / problem
    - Research question(s)
    - Methodology
    - Data
    - Key findings
    - Limitations / gaps
    """
    if not full_text.strip():
        return f"## {title}\n\n(No readable text was extracted from this PDF.)"

    system_prompt = (
        "You are an assistant that reads full academic papers for a thesis "
        "proposal assistant. Your job is to produce a structured, honest "
        "summary for each paper.\n\n"
        "Rules:\n"
        "- Use only the information in the provided text.\n"
        "- Do NOT invent authors, years, sample sizes, or results that are "
        "  not clearly present.\n"
        "- If something is not mentioned, explicitly write "
        "  'not specified in text'.\n"
        "- Keep the summary concise but detailed enough for a student to "
        "  understand what the paper did."
    )

    user_prompt = f"""
Here is the full text of a research paper titled "{title}".

[Paper text]
{full_text}
[End of paper text]

TASK:
Write a markdown summary under the heading "## {title}" with the following subsections:

- Topic / problem
- Research question(s)
- Methodology (design, data collection, analysis)
- Data (sample, data source, size if given)
- Key findings
- Limitations / gaps

If a subsection is not covered in the text, write "not specified in text" for that subsection.
"""

    resp = llm_service.generate_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,  # deterministic for summaries
    )

    return resp["text"].strip()


def summarize_uploaded_papers(files) -> Dict[str, str]:
    """
    Summarize each uploaded PDF with the BFH LLM.

    Args:
        files: list of Streamlit UploadedFile-like objects.

    Returns:
        Dict mapping filename -> markdown summary string.
    """
    summaries: Dict[str, str] = {}

    for f in files:
        data = f.read()
        if not data:
            continue

        title = getattr(f, "name", "uploaded_paper.pdf")
        full_text = _extract_full_text_from_pdf(data)
        summary = summarize_single_paper_with_bfh_llm(title, full_text)
        summaries[title] = summary

    return summaries
