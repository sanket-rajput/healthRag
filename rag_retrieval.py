import os
import requests

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# =============================
# CONFIG
# =============================
VECTOR_DB_PATH = "vector_db"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "deepseek/deepseek-r1-0528:free"

TOP_K = 5
TEMPERATURE = 0.2
MAX_TOKENS = 400

# Emergency red-flag keywords
RED_FLAGS = [
    "chest pain",
    "difficulty breathing",
    "shortness of breath",
    "unconscious",
    "loss of consciousness",
    "severe bleeding",
    "seizure",
    "stroke",
]


# =============================
# LOAD VECTOR DB
# =============================
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db


# =============================
# SAFETY CHECK
# =============================
def emergency_guard(query: str):
    q = query.lower()
    for flag in RED_FLAGS:
        if flag in q:
            return (
                "🚨 **Emergency Alert**\n\n"
                "Your symptoms may indicate a medical emergency.\n"
                "Please seek **immediate medical help or call emergency services now.**"
            )
    return None


# =============================
# RETRIEVE CONTEXT
# =============================
def retrieve_context(db, query: str):
    docs = db.similarity_search(
        query,
        k=TOP_K,
        filter={"dataset": "MedlinePlus"}
    )

    context = "\n\n".join(
        f"- {doc.page_content}" for doc in docs
    )

    sources = list({doc.metadata["source"] for doc in docs})
    return context, sources


# =============================
# CALL OPENROUTER LLM
# =============================
def call_openrouter(context: str, question: str):
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set in environment variables")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medical information assistant.\n"
                    "Use ONLY the provided context.\n"
                    "Do NOT diagnose or prescribe medication.\n"
                    "If symptoms are serious or worsening, advise seeing a doctor.\n"
                    "Keep the answer clear, calm, and structured."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# =============================
# MAIN ASK FUNCTION
# =============================
def ask_medical_question(db, question: str):
    # 1️⃣ Emergency guard
    emergency = emergency_guard(question)
    if emergency:
        return emergency

    # 2️⃣ Retrieve relevant medical context
    context, sources = retrieve_context(db, question)

    if not context.strip():
        return (
            "I couldn't find reliable medical information for this question.\n"
            "Please consult a healthcare professional."
        )

    # 3️⃣ LLM synthesis
    answer = call_openrouter(context, question)

    # 4️⃣ Append disclaimer & sources
    final_answer = (
        f"{answer}\n\n"
        "---\n"
        "⚠️ **Disclaimer:** This information is educational only and not a medical diagnosis.\n"
        "📚 **Sources:**\n" + "\n".join(sources)
    )

    return final_answer


# =============================
# CLI ENTRY POINT
# =============================
def main():
    print("🧠 Medical RAG Assistant (CLI)")
    print("Type 'exit' to quit.\n")

    db = load_vector_db()

    while True:
        question = input("❓ Ask a health question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        try:
            response = ask_medical_question(db, question)
            print("\n💡 Answer:\n")
            print(response)
            print("\n" + "=" * 80 + "\n")
        except Exception as e:
            print("⚠️ Error:", e)


if __name__ == "__main__":
    main()
