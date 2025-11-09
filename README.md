# proposify

This document describes how the RAG application in this repository must be set up correctly and what preparations must be made.

## Prerequisites
* In order to run the application with Docker, you need to install Docker Desktop on Windows systems. 
If you have a MacOS client or a Linux based system, you have to isntall Docker with the respective setups/installers.
* Please note that in order for Docker to run correctly on your system you need to activate a virtualization technology (WSL, Hyper-V etc.). 
* It is possible that you need to activate CPU virtualization in your BIOS!


## Setup Guide: RAG Chatbot
### 1. Clone the Repository
 
    git clone <REPOSITORY_URL>

    cd rag-chatbot
 
### 2. Add API Key
 
Rename the environment file and add your Together.ai API key:
 
    mv .env.apiKey .env
 
Then edit the .env file and insert your API key:
 
TOGETHER_API_KEY=your_actual_api_key_here

**The API Key is found in the report provided on Moodle.**


 
### 3. Build and Start the App
 
**Important:** Open first Docker Desktop App and wait until it shows "Docker is running"
Use Docker Compose to build and launch all services:
 
    docker-compose up --build
 
This sets up:
 
* ollama (local embedding model)
 
* chroma (vector DB)
 
* Streamlit UI
 
### 4. Import Embeddings
 
Once containers are running (see that the app is live on http://localhost:8501), open a new terminal tab and run:
 
    docker exec -it rag_chatbot_app python scripts/import_embeddings_to_chroma_server.py
  
And then:

    docker exec -it ollama ollama pull nomic-embed-text
 
You should see logs like:
 
✅ Importiert: Creswell_chunk_602.txt
✅ Importiert: Plagiarism_Guidelines_DE_chunk_001.txt

Import abgeschlossen: 640 Embeddings in Chroma (Server-Modus))

Those steps are importanant, otherwise the bot will not function correctly!
 
### 5. Use the App
 
Open http://localhost:8501 in your browser. 

You can now ask questions like:
    How can I use AI to write my thesis proposal?
 
ℹ️ Notes
 
You only need to import embeddings after rebuilding (--build) or clearing the Chroma volume.
 
Do not push data/chroma/ to Git – it’s excluded automatically.
 
To run the docker again run:
 
    docker-compose up
