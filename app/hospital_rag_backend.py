
import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import faiss

from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# PATHS
# ============================================================

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = BASE_DIR / "data/rag/hospital_chunks_with_metadata.json"
FAISS_FILE = BASE_DIR / "data/rag/hospital_rag.index"
MAPPING_FILE = BASE_DIR / "data/rag/faiss_mapping.json"
SESSION_DIR = BASE_DIR / "data/sessions"

SESSION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL NAMES
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
GROQ_MODEL = "openai/gpt-oss-120b"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# ============================================================
# LOAD DATA
# ============================================================

def load_chunks():

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_faiss():

    index = faiss.read_index(str(FAISS_FILE))

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    return index, mapping


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):

    return re.findall(r"[a-z0-9]+", text.lower())


# ============================================================
# LOAD RESOURCES
# ============================================================

_resources = None


def load_resources():

    global _resources

    if _resources is not None:
        return _resources

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Hospital RAG resources...")
    print("Device:", device)

    chunks = load_chunks()

    faiss_index, faiss_mapping = load_faiss()

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device
    )

    reranker = CrossEncoder(
        RERANKER_MODEL,
        device=device
    )

    corpus = [
        chunk.get("text", "")
        for chunk in chunks
    ]

    tokenized_corpus = [
        tokenize(text)
        for text in corpus
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    _resources = {
        "chunks": chunks,
        "faiss_index": faiss_index,
        "faiss_mapping": faiss_mapping,
        "embedding_model": embedding_model,
        "reranker": reranker,
        "bm25": bm25,
        "device": device
    }

    return _resources


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():

    api_key = os.getenv("GROK_API_KEY")

    if not api_key:

        raise RuntimeError(
            "GROK_API_KEY was not found. "
            "Add it to the environment or Streamlit Secrets."
        )

    if not api_key.startswith("gsk"):

        raise RuntimeError(
            "GROK_API_KEY does not look like a Groq API key."
        )

    return OpenAI(
        api_key=api_key,
        base_url=GROQ_BASE_URL
    )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(query, top_k=20):

    r = load_resources()

    model = r["embedding_model"]
    index = r["faiss_index"]

    vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(
        vector,
        min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        chunk = r["chunks"][int(idx)]

        results.append({
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text", ""),
            "metadata": chunk.get("metadata", {}),
            "semantic_score": float(score)
        })

    return results


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(query, top_k=20):

    r = load_resources()

    bm25 = r["bm25"]
    chunks = r["chunks"]

    scores = bm25.get_scores(tokenize(query))

    order = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in order:

        if scores[idx] <= 0:
            continue

        chunk = chunks[int(idx)]

        results.append({
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text", ""),
            "metadata": chunk.get("metadata", {}),
            "bm25_score": float(scores[idx])
        })

    return results


# ============================================================
# RRF HYBRID SEARCH
# ============================================================

def hybrid_search(query, candidate_k=20, top_k=10):

    semantic = semantic_search(
        query,
        top_k=candidate_k
    )

    keyword = bm25_search(
        query,
        top_k=candidate_k
    )

    rrf_scores = {}

    items = {}

    k = 60

    for rank, item in enumerate(semantic, start=1):

        cid = item["chunk_id"]

        items[cid] = item

        rrf_scores[cid] = rrf_scores.get(cid, 0) + (
            1 / (k + rank)
        )

    for rank, item in enumerate(keyword, start=1):

        cid = item["chunk_id"]

        items[cid] = item

        rrf_scores[cid] = rrf_scores.get(cid, 0) + (
            1 / (k + rank)
        )

    ordered = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for cid, score in ordered[:top_k]:

        item = items[cid].copy()

        item["rrf_score"] = float(score)

        results.append(item)

    return results


# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def rerank(query, candidates, top_k=5):

    if not candidates:
        return []

    r = load_resources()

    reranker = r["reranker"]

    pairs = [
        [query, item["text"]]
        for item in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = []

    for item, score in zip(candidates, scores):

        new_item = item.copy()

        new_item["reranker_score"] = float(score)

        ranked.append(new_item)

    ranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return ranked[:top_k]


# ============================================================
# CONTEXT ASSEMBLY
# ============================================================

def assemble_context(results, max_chars=12000):

    blocks = []

    total = 0

    for item in results:

        text = item.get("text", "").strip()

        if not text:
            continue

        source = item.get(
            "chunk_id",
            "UNKNOWN"
        )

        block = (
            f"[Source: {source}]\n"
            f"{text}"
        )

        if total + len(block) > max_chars:

            remaining = max_chars - total

            if remaining > 100:
                blocks.append(
                    block[:remaining]
                )

            break

        blocks.append(block)

        total += len(block)

    return "\n\n".join(blocks)


# ============================================================
# QUERY REWRITE
# ============================================================

def rewrite_query(question, history):

    if not history:
        return question

    client = get_groq_client()

    recent = history[-6:]

    conversation = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in recent
    )

    prompt = f"""
Convert the user's latest question into a standalone search question.

Do NOT answer the question.

Keep the meaning exactly the same.

Conversation:
{conversation}

Latest question:
{question}

Standalone search question:
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You rewrite questions for a hospital "
                    "information retrieval system."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        reasoning_effort="low",
        temperature=0,
        max_completion_tokens=150
    )

    return response.choices[0].message.content.strip()


# ============================================================
# MAIN HOSPITAL AI FUNCTION
# ============================================================

def hospital_ai_answer(
    question,
    history=None,
    top_k=5,
    candidate_k=20
):

    if history is None:
        history = []

    search_question = rewrite_query(
        question,
        history
    )

    candidates = hybrid_search(
        search_question,
        candidate_k=candidate_k,
        top_k=candidate_k
    )

    reranked = rerank(
        search_question,
        candidates,
        top_k=top_k
    )

    context = assemble_context(
        reranked
    )

    client = get_groq_client()

    system_prompt = """
You are a Hospital AI Assistant.

Answer the user's question using ONLY the hospital
information provided in the context.

Important rules:

1. Do not invent hospital information.
2. If the context does not contain the answer, say:
   "I could not find this information in the hospital knowledge base."
3. Keep the answer simple and clear.
4. Use the source IDs provided in the context.
5. Do not provide medical diagnosis or treatment advice.
6. For clinical questions, explain that the information
   should be confirmed with qualified hospital staff.
"""

    user_prompt = f"""
Hospital context:

{context}

User question:

{question}

Answer using the hospital context.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        reasoning_effort="low",
        temperature=0.2,
        max_completion_tokens=700
    )

    answer = response.choices[0].message.content.strip()

    sources = []

    for item in reranked:

        sources.append({
            "chunk_id": item.get("chunk_id"),
            "score": item.get("reranker_score"),
            "metadata": item.get("metadata", {})
        })

    return {
        "answer": answer,
        "search_question": search_question,
        "sources": sources,
        "context": context
    }


# ============================================================
# SIMPLE SESSION FUNCTIONS
# ============================================================

def session_file(session_id):

    safe_id = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        session_id
    )

    return SESSION_DIR / f"{safe_id}.json"


def load_session(session_id):

    path = session_file(session_id)

    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("messages", [])


def save_session(session_id, messages):

    path = session_file(session_id)

    with open(path, "w", encoding="utf-8") as f:

        json.dump(
            {
                "session_id": session_id,
                "messages": messages
            },
            f,
            indent=2,
            ensure_ascii=False
        )


def clear_session(session_id):

    path = session_file(session_id)

    if path.exists():
        path.unlink()
