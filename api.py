import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =============================
# ENV & CONFIG
# =============================
load_dotenv()

VECTOR_DB_PATH = "vector_db"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "deepseek/deepseek-r1-0528:free"

TOP_K = 5
FETCH_K = 20
TEMPERATURE = 0.2
MAX_TOKENS = 450

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is missing")

# =============================
# MEDICAL SAFETY
# =============================
RED_FLAGS = [
    "chest pain",
    "difficulty breathing",
    "shortness of breath",
    "unconscious",
    "loss of consciousness",
    "severe bleeding",
    "seizure",
    "stroke",
    "confusion",
    "sudden vision loss",
]

# Pages we NEVER want to answer from
BAD_SOURCE_KEYWORDS = [
    "healthtopics",
    "druginformation",
    "encyclopedia",
    "spanish",
    "about",
    "news",
    "tools",
]

# =============================
# FASTAPI APP
# =============================
app = FastAPI(
    title="Medical RAG API",
    version="1.0.0",
    description="Evidence-based healthcare information assistant (educational use only)",
)

# =============================
# LOAD VECTOR DB (ON STARTUP)
# =============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)

# =============================
# REQUEST / RESPONSE MODELS
# =============================
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    disclaimer: str


# =============================
# SAFETY GUARD
# =============================
def emergency_guard(question: str):
    q = question.lower()
    for flag in RED_FLAGS:
        if flag in q:
            return (
                "🚨 **Emergency Warning**\n\n"
                "Your symptoms may indicate a medical emergency.\n"
                "Please seek **immediate medical care or call emergency services now.**"
            )
    return None


# =============================
# RETRIEVAL (MMR + POST FILTER)
# =============================
def retrieve_context(question: str):
    # Step 1: Broad retrieval from FAISS
    docs = db.max_marginal_relevance_search(
        question,
        k=FETCH_K,
        fetch_k=FETCH_K,
        filter={"dataset": "MedlinePlus"},
    )

    # Step 2: Remove category / junk pages
    clean_docs = []
    for doc in docs:
        src = doc.metadata.get("source", "").lower()
        if not any(bad in src for bad in BAD_SOURCE_KEYWORDS):
            clean_docs.append(doc)

    # Step 3: Take top K clean docs
    clean_docs = clean_docs[:TOP_K]

    if not clean_docs:
        return "", []

    context = "\n\n".join(doc.page_content for doc in clean_docs)
    sources = sorted({doc.metadata["source"] for doc in clean_docs})

    return context, sources


# =============================
# LLM CALL
# =============================
def call_openrouter(context: str, question: str):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medical information assistant.\n"
                    "Use the provided context as the primary source.\n"
                    "Explain common possible causes, related symptoms, and when to seek medical care.\n"
                    "Do NOT diagnose or prescribe medication.\n"
                    "Be specific, calm, and structured.\n"
                    "If symptoms are persistent, severe, or worsening, advise seeing a doctor."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )

    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# =============================
# API ENDPOINT
# =============================
@app.post("/ask", response_model=AskResponse)
def ask_question(data: AskRequest):
    question = data.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Emergency guard
    emergency = emergency_guard(question)
    if emergency:
        return AskResponse(
            answer=emergency,
            sources=[],
            disclaimer="Emergency response triggered",
        )

    # Retrieval
    context, sources = retrieve_context(question)

    if not context:
        return AskResponse(
            answer=(
                "I couldn’t find reliable medical information for this question. "
                "Please consult a healthcare professional."
            ),
            sources=[],
            disclaimer="Educational only. Not a medical diagnosis.",
        )

    # LLM synthesis
    answer = call_openrouter(context, question)

    return AskResponse(
        answer=answer,
        sources=sources,
        disclaimer="Educational only. Not a medical diagnosis.",
    )
