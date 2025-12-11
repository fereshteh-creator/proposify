# rag_tools.py

import os
import time
from typing import Any, Dict, List

import chromadb
import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv

from llm_service import llm_service  # Together LLM wrapper

load_dotenv()


# --------------------------------------------------------------------
# Environment + config
# --------------------------------------------------------------------

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

# Known BFH guideline sources in the KB (values of the `quelle` metadata field)
BFH_SOURCES = [
    "AI Policy BFH_EN",
    "AI Recommendations for Lecturers_DE_EN",
    "BScDigi_Proposal_Instruction_EN_v1",
    "Plagiarism_Guidelines_DE",
    "Plagiarism_Guidelines_EN",
    "Regulations written assignments_DE",
    "Regulations written assignments_EN",
]


# --------------------------------------------------------------------
# LLM helper for router, methods & gap pipelines
# --------------------------------------------------------------------


def llm_complete(prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    """
    Call the configured Together LLM via the shared `llm_service` wrapper.

    This keeps a simple text-in / text-out interface that is used by the
    LangGraph pipelines (router, methods, gap, memory summariser).

    The `max_tokens` parameter is kept for backwards compatibility with
    existing call sites; the underlying chat model currently controls the
    exact token count.
    """
    resp = llm_service.generate_completion(
        system_prompt=(
            "You are a helpful assistant in a thesis proposal RAG chatbot. "
            "Follow the instructions in the user message carefully and "
            "return only the answer text."
        ),
        user_prompt=prompt,
        temperature=temperature,
    )
    return resp.get("text", "").strip()


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

    def _embed(model: str):
        return requests.post(
            EMBEDDING_URL,
            json={"model": model, "prompt": text},
            timeout=120,
        )

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

    raise RuntimeError(f"Embedding error: {r.status_code} - {r.text}")


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


def retrieve_kb_context(question: str, n_results: int = 5, min_bfh: int = 0):
    """
    Retrieve from the static Creswell / BFH methods knowledge base.

    This is vector-based (Chroma) but only for the fixed KB docs,
    not for user-uploaded PDFs.

    If `min_bfh` > 0, we try to ensure that at least that many chunks come
    from BFH guideline documents (AI policy, proposal instructions,
    plagiarism/regulations), as long as they appear near the top of the
    similarity ranking.
    """
    q_emb = embed_text_ollama(question)
    client = get_chroma_client()
    collection = client.get_or_create_collection(CHROMA_KB_COLLECTION)

    # Ask Chroma for more results than we finally need so we can mix sources.
    raw_n = n_results if min_bfh <= 0 else max(n_results * 2, n_results + min_bfh)
    result = collection.query(
        query_embeddings=[q_emb],
        n_results=raw_n,
        include=["documents", "metadatas", "distances"],
    )
    docs_all = result.get("documents", [[]])[0]
    metas_all = result.get("metadatas", [[]])[0]

    if min_bfh <= 0:
        return docs_all[:n_results], metas_all[:n_results]

    bfh_docs: List[str] = []
    bfh_metas: List[Dict[str, Any]] = []
    other_docs: List[str] = []
    other_metas: List[Dict[str, Any]] = []

    for doc, meta in zip(docs_all, metas_all):
        quelle = (meta or {}).get("quelle", "")
        if quelle in BFH_SOURCES:
            bfh_docs.append(doc)
            bfh_metas.append(meta)
        else:
            other_docs.append(doc)
            other_metas.append(meta)

    final_docs: List[str] = []
    final_metas: List[Dict[str, Any]] = []

    # 1) Take up to `min_bfh` BFH guideline chunks first (if available).
    for doc, meta in zip(bfh_docs, bfh_metas):
        if len(final_docs) >= min_bfh or len(final_docs) >= n_results:
            break
        final_docs.append(doc)
        final_metas.append(meta)

    # 2) Fill remaining slots with the highest-ranked other chunks.
    for doc, meta in zip(other_docs, other_metas):
        if len(final_docs) >= n_results:
            break
        final_docs.append(doc)
        final_metas.append(meta)

    return final_docs, final_metas


# --------------------------------------------------------------------
# Full-paper summarization with the LLM backend
# --------------------------------------------------------------------


def _extract_full_text_from_pdf(data: bytes) -> str:
    """
    Read the entire PDF (bytes) and return plain text from all pages.

    A soft character cap is applied to keep within the LLM context window.
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


def _summarize_single_paper_with_llm(title: str, full_text: str) -> str:
    """
    Use the configured LLM backend to produce a structured summary of one paper.

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
    Summarize each uploaded PDF with the configured LLM backend.

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
        summary = _summarize_single_paper_with_llm(title, full_text)
        summaries[title] = summary

    return summaries
