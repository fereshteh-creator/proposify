import json
import chromadb

# === Mit dem Chroma-Server verbinden ===
client = chromadb.HttpClient(host="chroma", port=8000)
collection = client.get_or_create_collection(name="gesetzestexte")

# === Embeddings laden ===
with open("data/embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# === In Chroma importieren ===
for entry in data:
    collection.add(
        ids=[entry["id"]],
        documents=[entry["text"]],
        embeddings=[entry["embedding"]],
        metadatas=[{
            "filename": entry["filename"],
            "chunk_id": entry["chunk_id"],
            "quelle": entry["quelle"]
        }]
    )
    print(f"✅ Importiert: {entry['filename']}")

print(f"\n Import abgeschlossen: {len(data)} Embeddings in Chroma (Server-Modus)")
