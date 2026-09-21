# 🏥 Hospital AI Assistant — Advanced RAG

A beginner-friendly Hospital AI Assistant built using an
Advanced Retrieval-Augmented Generation (RAG) pipeline.

## Architecture

Hospital Data
→ Chunking
→ Metadata
→ BGE-M3 Embeddings
→ FAISS
→ BM25
→ Hybrid Retrieval
→ RRF
→ Cross-Encoder Re-Ranking
→ Context Assembly
→ Groq GPT-OSS 120B
→ Grounded Answer

## Main Features

- Semantic search using BGE-M3
- FAISS vector search
- BM25 keyword search
- Hybrid retrieval using RRF
- Cross-encoder re-ranking
- Context assembly
- RAG evaluation
- Groq GPT-OSS 120B
- Conversation memory
- Persistent sessions
- Streamlit chat interface
- Source display

## Project Structure

hospital_rag/
│
├── app/
│   ├── __init__.py
│   └── hospital_rag_backend.py
│
├── data/
│   ├── rag/
│   │   ├── hospital_chunks_with_metadata.json
│   │   ├── hospital_rag.index
│   │   └── faiss_mapping.json
│   │
│   └── sessions/
│
├── .streamlit/
│   └── config.toml
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore

## Run Locally

Install dependencies:

pip install -r requirements.txt

Set the Groq API key:

GROK_API_KEY=your_groq_api_key

Run:

streamlit run streamlit_app.py

## API Key

The Groq API key must NEVER be written directly inside Python
code or committed to GitHub.

For Streamlit Cloud, use Streamlit Secrets:

GROK_API_KEY = "your_groq_api_key"

## Important Data Security Note

Do not upload real patient data, medical records, or other
sensitive information to a public GitHub repository.

The data included in a public demo should be synthetic or
properly anonymized.

## Current Model

LLM:
Groq GPT-OSS 120B

Embedding:
BAAI/bge-m3

Reranker:
BAAI/bge-reranker-v2-m3

## Application

The main application is:

streamlit_app.py
