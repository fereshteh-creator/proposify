import chromadb
import requests
from dotenv import load_dotenv

from llm_service import llm_service


load_dotenv()


def main() -> None:
    # === Frage abfragen ===
    question = input("Was möchtest du wissen?\n> ").strip()
    if not question:
        print("Keine Frage eingegeben.")
        return

    # === Embedding mit nomic-embed-text via Ollama ===
    print("Erzeuge Embedding für die Frage...")
    embed_response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": question},
        timeout=60,
    )
    if embed_response.status_code != 200:
        print("Fehler beim Embedding:", embed_response.text)
        return

    question_embedding = embed_response.json()["embedding"]

    # === ChromaDB: relevante Chunks abrufen ===
    print("Suche relevante Chunks in Chroma...")
    client = chromadb.HttpClient(host="localhost", port=8000)
    collection = client.get_or_create_collection("gesetzestexte")
    result = collection.query(
        query_embeddings=[question_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    # === Welche Dokumente & Abschnitte wurden verwendet? ===
    print("\n--- Verwendete Dokumente / Quellen ---")
    for i, meta in enumerate(metadatas):
        quelle = meta.get("quelle", "Unbekannt")
        chunk_id = meta.get("chunk_id", "N/A")
        print(f"{i + 1}. {quelle} - Chunk {chunk_id}")

    context = "\n\n".join(documents)

    # === Prompt für das LLM bauen ===
    user_prompt = f"""Beantworte die folgende Frage ausschließlich basierend auf dem gegebenen Kontext.

Frage: {question}

Kontext:
{context}

Antwort:"""

    # === Anfrage an das LLM senden ===
    print("Anfrage wird an das LLM gesendet...")
    system_prompt = (
        "Du bist ein hilfreicher Assistent in einem RAG-Chatbot. "
        "Beantworte Fragen ausschließlich basierend auf dem bereitgestellten Kontext."
    )
    try:
        response = llm_service.generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
        )
        answer = response["text"]
    except Exception as exc:
        print("Fehler bei LLM-Request:", exc)
        return

    # === Ausgabe ===
    print("\n--- Antwort ---")
    print(answer.strip())


if __name__ == "__main__":
    main()

