import streamlit as st
import os
import requests
import chromadb
from dotenv import load_dotenv
import time

load_dotenv()

# Konfiguration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MIXTRAL_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_URL = "http://ollama:11434/api/embeddings"


assert TOGETHER_API_KEY, "Bitte TOGETHER_API_KEY als Umgebungsvariable setzen."

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Proposify - Ask me a question about my data ")

# Session-Initialisierung
if "history" not in st.session_state:
    st.session_state.history = []

def generate_antwort(frage):
    # 1. Frage-Embedding
    embed_response = requests.post(EMBEDDING_URL, json={
        "model": EMBEDDING_MODEL,
        "prompt": frage
    })

    if embed_response.status_code != 200:
        return {"error": f"Fehler beim Erzeugen des Embeddings: {embed_response.text}"}

    frage_embedding = embed_response.json()["embedding"]

    # 2. Chroma Retrieval
    for attempt in range(10):
        try:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            client.heartbeat()
            break
        except Exception as e:
            print(f"Attempt {attempt+1}/10: Could not reach Chroma - {e}")
            time.sleep(2)
    else:
        return {"error": "Chroma konnte nicht erreicht werden."}


    collection = client.get_or_create_collection("gesetzestexte")
    result = collection.query(
        query_embeddings=[frage_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    

    top_docs = result["documents"][0]
    kontext = "\n\n".join(top_docs)

    # 3. Prompt bauen
    prompt = f"""Answer the following question based on the given context.

Frage: {frage}

Kontext:
{kontext}

Antwort:"""

    # 4. Anfrage an LLM
    response = requests.post(
        "https://api.together.xyz/v1/completions",
        headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
        json={
            "model": MIXTRAL_MODEL,
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.1
        }
    )

    if response.status_code != 200:
        return {"error": f"Fehler bei LLM-Anfrage: {response.status_code} – {response.text}"}

    response_json = response.json()
    if "choices" not in response_json:
        return {"error": "Unerwartete Antwortstruktur."}

    antwort = response_json["choices"][0]["text"].strip()
    return {
        "antwort": antwort,
        "quellen": result["metadatas"][0]
    }

# Chatverlauf anzeigen
for eintrag in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(eintrag["frage"])

    with st.chat_message("assistant"):
        st.markdown(eintrag["antwort"])
        st.markdown("**Quellen:**")
        for meta in eintrag["quellen"]:
            quelle = meta.get("quelle", "Unbekannt")
            chunk_id = meta.get("chunk_id", "N/A")
            st.markdown(f"- {quelle}, Chunk-ID: `{chunk_id}`")
        st.markdown("---")

# Eingabefeld bleibt immer unten sichtbar
frage = st.chat_input("Ask a question...")

if frage:
    with st.spinner("Thinking..."):
        result = generate_antwort(frage)
        




        if "error" in result:
            st.error(result["error"])
        else:
            st.session_state.history.append({
                "frage": frage,
                "antwort": result["antwort"],
                "quellen": result["quellen"]
            })
            st.rerun()  # Lädt das Interface neu, damit neue Nachrichten sofort angezeigt werden

